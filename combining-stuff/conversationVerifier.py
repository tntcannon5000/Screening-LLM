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

def extract_qa_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    lines = text.split('\n')
    qa_dict = {}
    current_speaker = None
    current_message = []
    first_user_statement = True
    current_question = None

    for line in lines:
        if line.strip() == 'Assistant:':
            if current_speaker == 'User' and not first_user_statement:
                qa_dict[current_question] = ' '.join(current_message).strip()
            current_speaker = 'Assistant'
            current_message = []
        elif line.strip() == 'User:':
            if current_speaker == 'Assistant':
                current_question = ' '.join(current_message).strip()
            current_speaker = 'User'
            current_message = []
            if first_user_statement:
                first_user_statement = False
        elif line.strip():
            current_message.append(line.strip())

    if current_speaker == 'User' and not first_user_statement and current_question:
        qa_dict[current_question] = ' '.join(current_message).strip()
    qa_dict.pop(None, None)
    return qa_dict

def process_qa_pair(pdf_path):
    # Load environment variables
    load_dotenv()

    # Set up OpenAI API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    # Extract Q&A from PDF
    qa_dictionary = extract_qa_from_pdf(pdf_path)

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
    for question, answer in qa_dictionary.items():
        queries = generate_queries_decomposition.invoke({"question": question, "answer": answer})
        for query in queries:
            scraper = WebScraper(answer, 2)
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

    for question, answer in qa_dictionary.items():
        feedback = rag_chain.invoke({"answer": answer, "question": question, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(question, answer, feedback)
        q_a_pairs += "\n---\n" + q_a_pair

    return q_a_pairs

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ")
    result = process_qa_pair(pdf_path)
    print(result)