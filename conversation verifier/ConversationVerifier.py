from WebScraper import WebScraper
from joblib import load

class ConversationVerifier:

    def __init__(self):
        self.webscraper = WebScraper()
        self.chatlog = []

    def verify_conversation(self, chatlog):
        self.chatlog = chatlog

        
        


