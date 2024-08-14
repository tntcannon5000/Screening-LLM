import os
from dotenv import load_dotenv
import fitz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from webscraper import WebScraper


def process_qa_pair(chat_log):
    # Load environment variables
    load_dotenv()

    # Set up OpenAI API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    # Extract Q&A from PDF

    # Decomposition template
    template = """You are a helpful assistant that generates multiple sub-queries related to an answer by an interview candidate. \n
    You will be provided the original question for context. The goal is to find the accuracy of the answer. For this
    you will break down the answer into a set of sub-problems / sub-queries that can be used as a search string in google to find the relevant information. \n
    Here is the original question: {question}
    Generate multiple search queries related to: {answer} \n
    Output (2 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # LLM setup
    llm = ChatOpenAI(temperature=0)
    documents = []

    # Generate queries and scrape data
    generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    for qa_pair in chat_log:
        question = qa_pair['interviewer']
        answer = qa_pair['candidate']
        queries = generate_queries_decomposition.invoke({"question": question, "answer": answer})
        for query in queries:
            scraper = WebScraper(query, 2)
            documents.extend(scraper.get_scraped_data())

    # Process and store documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Accuracy checking template
    accuracy_template = """
    Here is the question for the answer you need to check:

    \n --- \n {question} \n --- \n

    Here is the answer you need to check:

    \n --- \n {answer} \n --- \n

    Here is any available background question + answer + accuracy percentage + feedback:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context, your own knowledge and background question + answer + accuracy percentage + feedback
    on the subject matter to get the accuracy of the answer. A percentage accuracy score and also note down the 
    places that the answer was inaccurate and give feedback for those places. Please keep in mind do not check for this :
    1. Typograpical error as it is a audio interview and words can be misspelled when translated.
    2. Do not look into things the candidate did not answer if it was not asked in the question.
    3. Do not provide a summary at the end of each question answer pair.

    """
    accuracy_prompt = ChatPromptTemplate.from_template(accuracy_template)

    def format_qa_pair(question, answer, feedback):
        return f"Question: {question}\nAnswer: {answer}\nFeedback: {feedback}\n\n".strip()

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    q_a_pairs = ""
    rag_chain = (
        {"context": itemgetter("answer") | retriever, 
         "answer": itemgetter("answer"),
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} 
        | accuracy_prompt
        | llm
        | StrOutputParser()
    )

    for chat in chat_log:
        question = chat['interviewer']
        answer = chat['candidate']
        feedback = rag_chain.invoke({"answer": answer, "question": question, "q_a_pairs": q_a_pairs})
        chat['feedback'] = feedback
        q_a_pair = format_qa_pair(question, answer, feedback)
        q_a_pairs += "\n---\n" + q_a_pair

    return chat_log

#To Test
if __name__ == "__main__":
    chat_log = [
        {
            'interviewer': "Hello! It's nice to meet you too. Thank you for taking the time to speak with me today about the Entry-Level Machine Learning Engineer position at G-Research. To start off, could you tell me a bit about your experience with machine learning, particularly in areas like model training and optimization?",
            'candidate': " Yeah, sure. I'd be happy to. I've got some experience in model training and optimization. I've mainly acquired this experience throughout my time at university and through competing in various competitions on Kaggle. Some of those included regression tasks where I trained extra boost algorithms to predict the age of abelones, whatever that is. I've no idea what that is, but yeah. I mean, it didn't really matter to me for the competition, but yeah. Some other ones that I've been involved in is some other stuff, yeah. I get fired immediately. I get sent out."
        },
        {
            'interviewer': 'I see. Thank you for sharing that. Could you elaborate a bit more on your experience with specific machine learning libraries or frameworks like PyTorch or NumPy? How have you used these in your projects or competitions?',
            'candidate': " To be honest, I'm not very sure."
        },
        {
            'interviewer': "I understand. Let's move on to another aspect of the role. The position involves working with large datasets. Could you describe any experience you have in handling and processing large volumes of data, particularly in a machine learning context?",
            'candidate': " Um, I've never actually done that before."
        }
    ]

    result = process_qa_pair(chat_log)
    print(result)