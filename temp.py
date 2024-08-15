from ..utils.anthropicwrapper import ClaudeChat, ClaudeChatAssess
from ..utils.humewrapper import HumeSentimentAnalyzer
from ..modules import ConversationVerifier
from dotenv import load_dotenv
from joblib import load
import os
from pprint import pprint

load_dotenv(os.path.join(os.path.dirname(os.getcwd()), ".env"))

# -- Global Variables --
timestamp = 1723670854
directory = f'interviews/{timestamp}/'
chatlog = load(os.path.join(directory, "joblib/conversation.joblib")) # Loading in the list of dicts

def reformat_chatlog(chatlog):
    dropped_context = chatlog[3:]
    outputchatlog = []

    for i in range(0, len(dropped_context), 2):
        if i + 1 < len(dropped_context):
            tempdict = {
                'interviewer': dropped_context[i]['content'],
                'candidate': dropped_context[i+1]['content']
            }

            outputchatlog.append(tempdict)
        else:
            tempdict = {
                'interviewer': dropped_context[i]['content'],
                'candidate': 'Thank you, goodbye'  
            }
            outputchatlog.append(tempdict)
            break 

    return outputchatlog

chatlog_chat = reformat_chatlog(chatlog)
for item in chatlog_chat:
   print(item)

   # Initialising the Sentiment Analyser
sentiment_analyser = HumeSentimentAnalyzer(api_key=os.getenv("HUME_API_KEY"))

# Initialising Sentiment Summariser
chat_model_name = "claude-3-5-sonnet-20240620"

system_prompt = """You are a skilled emotions analyst provided with a detailed breakdown of sentiment analysis scores from Hume.ai, for a single response in an interview to a question from the interviewer. The scores are split into 3 sections. All numbers are from 0 to 1, linearly scaling, with 1 being a very strong representation of the indicator in question.

First, Emotions. This contains several human emotions with a numerical value indicating the strength of the corresponding emotion.
Second, Sentiments. This contains a scale from 1 to 9, each containing a numerical value indicating the magnitude of the sentiment of the topic of the conversation. A negative topic such as murder will have a high value lower in the scale, such as 1 or 2, and a positive topic will have a high value from 0 to 1 higher in the scale such as 8 or 9.
Third, Toxicity. This contains several toxic representations such as hate, insult, etc, with a value from 0 to 1 for each representation identified in the audio.

Your job is to provide a concise detailed one sentence breakdown of how the individual was feeling for the particular scores provided. You must be highly objective as your job is to discern whether or not a candidate was exhibiting traits which would or would not be fitting for a successful interview. 
Model your answer beginning with something along the lines of "For this particular response, the candidate...".
"""

sentiment_summariser = ClaudeChat(chat_model_name, system_prompt)


# Initialising Evaluation LLM
chat_model_name = "claude-3-5-sonnet-20240620"
system_prompt = """You are a highly skilled interviewer currently tasked with reviewing a phone screening interview candidate to decide whether they are to pass on to the next stage of the interview process. There is a high volume of candidates and as such only 47% of candidates should be allowed to pass.
You will be provided with the following information to aid your decision:
1. A copy of the job description.
2. Transcript of the phone interview between the interviewer and candidate.
3. Sentiment analysis summary to provide you with insight into the tone of the candidate in the call
4. A summary of any factual inaccuracies the candidate may have made during discussions in the call
5. A copt of the candidate's CV, which will be provided next.

You are to evaluate the candidate, primarily on the transcript, and use the additional information provided to identify any potential red-flags. Your response should include a detailed breakdown of why the candidate is chosen to continue onwards to further interviewing. You must end the breakdown with a simple one word response on a new line, "pass" or "fail"."""

pdf_path = os.path.join(directory, "cvs/cv-niranj.pdf")
candidate_evaluator = ClaudeChatAssess(chat_model_name, system_prompt, pdf_path)

# Specify the filepath
filepath = f'interviews/{timestamp}/audio'

# Get all files in the filepath
files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]

# Loop through each file
sentiments = []
for count, file in enumerate(files, 1):
    # Print the file name
    print(os.path.join(filepath, str(file)))
    result = sentiment_analyser.analyze_audio(os.path.join(filepath, str(file)))
    sentiment_summary = sentiment_summariser.chat(str(result))
    sentiments.append((result, sentiment_summary))
    chatlog_chat[count-1]['sentiment'] = sentiment_summary

    e = ConversationVerifier.process_qa_pair(chatlog_chat)

evaluation = candidate_evaluator.chat(str(chatlog_chat))


pprint(evaluation)