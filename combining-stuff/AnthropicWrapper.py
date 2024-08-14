import os
from anthropic import Client
from dotenv import load_dotenv


class ClaudeChat():

    def __init__(self, model, systemprompt):
        
        
        # Get the path to the immediate parent folder of the current working directory
        parent_folder_path = os.path.dirname(os.getcwd())
        # Construct the path to the .env file in the parent folder
        dotenv_path = os.path.join(parent_folder_path, ".env")
        # Load the .env file
        load_dotenv(dotenv_path)

        if os.getenv('ANTHROPIC_API_KEY') is None:
            os.environ["ANTHROPIC_API_KEY"] = input("Please enter your API key: ")

        self.client = Client()
        self.model = model
        self.systemprompt = systemprompt
        self.max_tokens = 512
        self.temperature = 0.5

    def chat(self, message, preformed_message=False):
        
        if preformed_message == False:
            formatted_message = [
                {
                    "role": "user",
                    "content": message
                }
            ]
        else:
            formatted_message = message
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.systemprompt,
            messages=formatted_message
        )

        content = ''.join(block.text for block in response.content)

        return content
    
    def set_model_temperature(self, temperature):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_model(self, model):
        self.model = model

    def set_system_prompt(self, systemprompt):
        self.systemprompt = systemprompt




class ClaudeChatHistory(ClaudeChat):
    
    def __init__(self, model, systemprompt):
        super().__init__(model, systemprompt)
        self.message_history = []


    def chat_with_history(self, message):

        formatted_message = {"role": "user", "content": message}
        self.message_history.append(formatted_message)
        formatted_messages = self.message_history

        content = super().chat(formatted_messages, preformed_message=True)

        self.message_history.append({"role": "assistant", "content": content})

        return content

    def clear_history(self):
        self.message_history = []

    def add_message_to_history(self, message):
        self.message_history.append(message)

    def set_message_history(self, message_history):
        self.message_history = message_history

    def get_message_history(self):
        return self.message_history
    
    def get_conversation(self):
        testlist = []
        for message in self.message_history:
            if message.get("role") == "user" and "The following is a document that I am providing you with." in message.get("content", ""):
                testlist.append("Document Included, please use ClaudeChatDocument.get_document() to view document\n")
                testlist.append("-" * 20 + "\n") 
            else:
                for key, value in message.items():
                    testlist.append(f"{key}: {value}\n")
                testlist.append("-" * 20 + "\n") 

        yeet = "".join(testlist)

        return yeet
    
    def get_conversation_dict(self):
        return self.message_history



class ClaudeChatDocument(ClaudeChatHistory):
    
    def __init__(self, model, systemprompt, document):
        
        super().__init__(model, systemprompt)
        self.document = self._process_pdf(document, force=True)
        self.documentprompt = "The following is a document that I am providing you with. You are to keep this document in the back of your mind and consider it or use it, should it be relevant to the discussion."
        documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
        self.message_history.append({"role": "user", "content": documentstr})
        self.message_history.append({"role": "assistant", "content": "Understood! I'll keep this in the back of my mind."})


    def chat_with_history_doc(self, message):

        content = super().chat_with_history(message)

        return content

    def _process_pdf(self, pdf_path, force=False):
        import fitz
        if force or self.document is None:
            doc = fitz.open(pdf_path)
            extracted_text = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                blocks = page.get_text("blocks")

                for block in blocks:
                    text = block[4].strip()
                    if text:
                        if block[0] < 200 and block[3] > 12: 
                            extracted_text += f"\n\n## {text}\n"
                        else:
                            extracted_text += f"{text} "

            return extracted_text
        else:
            return self.document
    
    def set_document(self, document):

        self.document = self._process_pdf(document, force=True)
        documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
        self.message_history[0] = {"role": "user", "content": documentstr}

    def get_document(self):
        return self.document
    
class ClaudeChatCV(ClaudeChatHistory):
    
    def __init__(self, model, systemprompt, document):
        
        super().__init__(model, systemprompt)
        self.document = self._process_pdf(document, force=True)
        self.documentprompt = "The following is a CV that I am providing you with. You are to keep this document in the back of your mind and consider it or use it, should it be relevant to the discussion."
        documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
        self.message_history.append({"role": "user", "content": documentstr})
        self.message_history.append({"role": "assistant", "content": "Understood! I'll keep this CV in the back of my mind and use it should it be relevant to the discussion."})


    def chat_with_history_doc(self, message):

        content = super().chat_with_history(message)

        return content

    def _process_pdf(self, pdf_path, force=False):
        import fitz
        if force or self.document is None:
            doc = fitz.open(pdf_path)
            extracted_text = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                blocks = page.get_text("blocks")

                for block in blocks:
                    text = block[4].strip()
                    if text:
                        if block[0] < 200 and block[3] > 12: 
                            extracted_text += f"\n\n## {text}\n"
                        else:
                            extracted_text += f"{text} "

            return extracted_text
        else:
            return self.document
    
    def set_document(self, document):

        self.document = self._process_pdf(document, force=True)
        documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
        self.message_history[0] = {"role": "user", "content": documentstr}

    def get_document(self):
        return self.document