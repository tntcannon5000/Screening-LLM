import os
from getpass import getpass
import google.generativeai as genai
import pprint
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain

import logging
import dotenv
import re


class WebGemini:

    def __init__(self, model=None, api_key=None, cse_id=None, use_web=False, params=None, logging=True, env_file="env_file.env"):

        dotenv.load_dotenv(dotenv_path=env_file)

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        if cse_id:
            os.environ["GOOGLE_CSE_ID"] = cse_id

        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass("Please enter your Google API Key: ")
        if not os.environ.get("GOOGLE_CSE_ID"):
            os.environ["GOOGLE_CSE_ID"] = getpass("Please enter your Google CSE ID: ")
        
        if not os.environ.get("GOOGLE_API_KEY"):
            raise KeysNotProvidedError("API key not provided.") 
        if not os.environ.get("GOOGLE_CSE_ID"):
            raise KeysNotProvidedError("CSE key not provided.") 

        with open(env_file, 'w') as f:
            f.write(f"GOOGLE_API_KEY={os.environ['GOOGLE_API_KEY']}\n")
            f.write(f"GOOGLE_CSE_ID={os.environ['GOOGLE_CSE_ID']}\n")

        print("API Key: " + str(os.environ.get("GOOGLE_API_KEY")))
        print("CSE ID: " + str(os.environ.get("GOOGLE_CSE_ID")))

        self.modelslist = self.get_models()

        if model is None:
            print("No model provided. Using gemini-pro by default. Here are the available models:")
            print(self.modelslist)
            self.modelname = "gemini-pro"
        
        elif type(model) == int:
            if model not in self.modelslist.keys():
                print("Model not found. Using gemini-pro by default.")
                self.modelname = "gemini-pro"
            else:
                self.modelname = self.modelslist[model] 
        elif type(model) == str:
            if model not in self.modelslist.values():
                print("Model not found. Using gemini-pro by default.")
                self.modelname = "gemini-pro"
            else:
                self.modelname = model

        self.logging = logging        
        self.chat_model = None
        self.use_web = use_web

        self.init_model()    


    def get_models(self, print_models=False):
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        modelslist = ""
        for model in genai.list_models():
            modelslist += str(model) + "\n"
            if print_models:
                print(model)

        model_names = re.findall(r"name='models/(.+?)'", modelslist)
        thedict = {}
        for modelname, count in enumerate(model_names):
            thedict[modelname] = count

        return thedict
    
    def list_models(self):

        return self.modelslist

    def set_model(self, model):
        self.modelname = model
        self.init_model()

    def init_no_web(self):

        self.chat_model = ChatGoogleGenerativeAI(model=self.modelname, max_output_tokens=4096)

    def init_web(self):
        if self.logging:
            logging.basicConfig()
            logging.getLogger("web_research_mod").setLevel(logging.INFO)
  
        self.chat_model = ChatGoogleGenerativeAI(model=self.modelname, max_output_tokens=2048)
        self.vector_store = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory="./chroma_db_yeet")
        self.conversation_memory = ConversationSummaryBufferMemory(llm=self.chat_model, input_key='question', output_key='answer', return_messages=True)
        self.google_search = GoogleSearchAPIWrapper()
        self.web_research_retriever = WebResearchRetriever.from_llm(vectorstore=self.vector_store, llm=self.chat_model, search=self.google_search, num_search_results=4)
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.chat_model, retriever=self.web_research_retriever)

    def init_model(self):
        if self.use_web:
            self.init_web()
        else:
            self.init_no_web()

    def chat(self, question):
        if self.use_web:
            response = self.qa_chain({"question": question})

            print(response["answer"])
            print("Sources:" + str(response["sources"]))
            return response["answer"]
        else:
            response = self.chat_model.invoke(question).content
            print(response)
            return response