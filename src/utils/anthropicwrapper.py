import os
from anthropic import Client
from dotenv import load_dotenv, find_dotenv
import fitz
class ClaudeChat():
    """Base class for interacting with Claude AI model."""
    def __init__(self, model, systemprompt):
        """
        Initialize the ClaudeChat instance.

        Args:
            model (str): The name of the Claude model to use.
            systemprompt (str): The system prompt to set the context for the AI.
        """
        try:
            load_dotenv(find_dotenv())

            if os.getenv('ANTHROPIC_API_KEY') is None:
                os.environ["ANTHROPIC_API_KEY"] = input("Please enter your API key: ")

            self.client = Client()
            self.model = model
            self.systemprompt = systemprompt
            self.max_tokens = 1024
            self.temperature = 0.5
        except Exception as e:
            print(f"Error initializing ClaudeChat: {str(e)}")
            raise

    def chat(self, message, preformed_message=False):
        """
        Send a message to the Claude AI and get a response.

        Args:
            message (str or list): The message to send to the AI.
            preformed_message (bool): Whether the message is already formatted.

        Returns:
            str: The AI's response.
        """
        try:
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
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return None
    
    def set_model_temperature(self, temperature):
        """Set the temperature for the model."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        """Set the maximum number of tokens for the model response."""
        self.max_tokens = max_tokens

    def set_model(self, model):
        """Set the model to be used."""
        self.model = model

    def set_system_prompt(self, systemprompt):
        """Set the system prompt for the model."""
        self.systemprompt = systemprompt




class ClaudeChatHistory(ClaudeChat):
    """ClaudeChat with conversation history management."""
    def __init__(self, model, systemprompt):
        """Initialize ClaudeChatHistory."""
        super().__init__(model, systemprompt)
        self.message_history = []


    def chat_with_history(self, message):
        """
        Chat with the AI, maintaining conversation history.

        Args:
            message (str): The message to send to the AI.

        Returns:
            str: The AI's response.
        """
        try:
            formatted_message = {"role": "user", "content": message}
            self.message_history.append(formatted_message)
            formatted_messages = self.message_history

            content = super().chat(formatted_messages, preformed_message=True)

            self.message_history.append({"role": "assistant", "content": content})

            return content
        except Exception as e:
            print(f"Error in chat_with_history: {str(e)}")
            return None

    def clear_history(self):
        """Clear the conversation history."""
        self.message_history = []

    def add_message_to_history(self, message):
        """Add a message to the conversation history."""
        self.message_history.append(message)

    def set_message_history(self, message_history):
        """Set the entire conversation history."""
        self.message_history = message_history

    def get_message_history(self):
        """Get the current conversation history."""
        return self.message_history
    
    def get_conversation(self):
        """
        Get a formatted string representation of the conversation.

        Returns:
            str: Formatted conversation history.
        """
        try:
            testlist = []
            for message in self.message_history:
                if message.get("role") == "user" and "The following is a document that I am providing you with." in message.get("content", ""):
                    testlist.append("Document Included, please use ClaudeChatDocument.get_document() to view document\n")
                    testlist.append("-" * 20 + "\n")
                else:
                    for key, value in message.items():
                        testlist.append(f"{key}: {value}\n")
                    testlist.append("-" * 20 + "\n")

            return "".join(testlist)
        except Exception as e:
            print(f"Error in get_conversation: {str(e)}")
            return ""
    
    def get_conversation_dict(self):
        """Get the conversation history as a dictionary."""
        return self.message_history



class ClaudeChatDocument(ClaudeChatHistory):
    """ClaudeChat with document processing capabilities."""
    def __init__(self, model, systemprompt, document):
        """
        Initialize ClaudeChatDocument.

        Args:
            model (str): The name of the Claude model to use.
            systemprompt (str): The system prompt to set the context for the AI.
            document (str): Path to the document file.
        """
        try:
            super().__init__(model, systemprompt)
            self.document = self._process_pdf(document, force=True)
            self.documentprompt = "The following is a document that I am providing you with. You are to keep this document in the back of your mind and consider it or use it, should it be relevant to the discussion."
            documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history.append({"role": "user", "content": documentstr})
            self.message_history.append({"role": "assistant", "content": "Understood! I'll keep this in the back of my mind."})
        except Exception as e:
            print(f"Error initializing ClaudeChatDocument: {str(e)}")
            raise


    def chat_with_history_doc(self, message):
        """Chat with history, considering the document context."""
        content = super().chat_with_history(message)

        return content

    def _process_pdf(self, pdf_path, force=False):
        """
        Process a PDF file and extract its text.

        Args:
            pdf_path (str): Path to the PDF file.
            force (bool): Whether to force reprocessing of the document.

        Returns:
            str: Extracted text from the PDF.
        """
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
        try:
            self.document = self._process_pdf(document, force=True)
            documentstr = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history[0] = {"role": "user", "content": documentstr}
        except Exception as e:
            print(f"Error setting document: {str(e)}")

    def get_document(self):
        """Get the current document text."""
        return self.document
    
class ClaudeChatCV(ClaudeChatHistory):
    """ClaudeChat specialized for handling CV (resume) documents."""
    def __init__(self, model, systemprompt, document):
        """
        Initialize ClaudeChatCV.

        Args:
            model (str): The name of the Claude model to use.
            systemprompt (str): The system prompt to set the context for the AI.
            document (str): Path to the CV document file.
        """
        try:
            super().__init__(model, systemprompt)
            self.document = self._process_pdf(document, force=True)
            self.documentprompt = "The following is a CV that I am providing you with. You are to keep this document in the back of your mind and consider it or use it, should it be relevant to the discussion."
            documentstr = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history.append({"role": "user", "content": documentstr})
            self.message_history.append({"role": "assistant", "content": "Understood! I'll keep this CV in the back of my mind and use it should it be relevant to the discussion."})
        except Exception as e:
            print(f"Error initializing ClaudeChatCV: {str(e)}")
            raise

    def chat_with_history_doc(self, message):
        """Chat with history, considering the CV context."""
        return super().chat_with_history(message)

    def _process_pdf(self, pdf_path, force=False):
        """
        Process a PDF file and extract its text.

        Args:
            pdf_path (str): Path to the PDF file.
            force (bool): Whether to force reprocessing of the document.

        Returns:
            str: Extracted text from the PDF.
        """
        import fitz
        try:
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
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""
    
    def set_document(self, document):

        """Set a new CV document and update the conversation history."""
        try:
            self.document = self._process_pdf(document, force=True)
            documentstr = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history[0] = {"role": "user", "content": documentstr}
        except Exception as e:
            print(f"Error setting document: {str(e)}")

    def get_document(self):
        """Get the current CV document text."""
        return self.document
    

class ClaudeChatAssess(ClaudeChatHistory):
    """ClaudeChat specialized for assessment tasks."""
    def __init__(self, model, systemprompt, document):
        """
        Initialize ClaudeChatAssess.

        Args:
            model (str): The name of the Claude model to use.
            systemprompt (str): The system prompt to set the context for the AI.
            document (str): Path to the document file for assessment.
        """
        try:
            super().__init__(model, systemprompt)
            self.document = self._process_pdf(document, force=True)
            self.documentprompt = """The following the candidate's CV that I am providing you with, in addition to the candidate's information which will be provided shortly. Respond confirming you understand, and we'll proceed. """
            documentstr  = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history.append({"role": "user", "content": documentstr})
            self.message_history.append({"role": "assistant", "content": "Understood!, provide me with the candidate's performance details and I'll proceed with the assessment."})
        except Exception as e:
            print(f"Error initializing ClaudeChatAssess: {str(e)}")
            raise
        


    def chat_with_history_doc(self, message):

        """Chat with history, considering the assessment context."""
        return super().chat_with_history(message)

    def _process_pdf(self, pdf_path, force=False):
        """
        Process a PDF file and extract its text.

        Args:
            pdf_path (str): Path to the PDF file.
            force (bool): Whether to force reprocessing of the document.

        Returns:
            str: Extracted text from the PDF.
        """
        try:
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
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""
        
    def set_document(self, document):
        """Set a new document for assessment and update the conversation history."""
        
        try:
            self.document = self._process_pdf(document, force=True)
            documentstr = self.documentprompt + "\n\n Document below: \n\n" + self.document
            self.message_history[0] = {"role": "user", "content": documentstr}
        except Exception as e:
            print(f"Error setting document: {str(e)}")

    def get_document(self):
        """Get the current assessment document text."""
        return self.document