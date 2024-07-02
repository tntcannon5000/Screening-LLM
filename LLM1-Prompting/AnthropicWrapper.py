import os
from anthropic import Client, AI_PROMPT, HUMAN_PROMPT
from dotenv import load_dotenv

class ClaudeWrapper():

    def __init__(self, model, systemprompt):
        load_dotenv('.env')

        if os.getenv('ANTHROPIC_API_KEY') is None:
            os.environ["ANTHROPIC_API_KEY"] = input("Please enter your API key: ")

        self.client = Client()
        self.model = model
        self.systemprompt = systemprompt
        self.message_history = []

    def chat(self, message):

        formatted_message = [
            {
                "role": "user",
                "content": message
            }
        ]

        print(message)
        print("============================")
        print(formatted_message)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.systemprompt,
            messages=formatted_message
        )

        content = ''.join(block.text for block in response.content)

        return content

    def chat_with_history(self, message):

        formatted_message = {"role": "user", "content": message}
        self.message_history.append(formatted_message)
        formatted_messages = self.message_history

        print(message)
        print("============================")
        print(formatted_messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.systemprompt,
            messages=formatted_messages
        )

        content = ''.join(block.text for block in response.content)

        self.message_history.append({"role": "assistant", "content": content})

        return content

    def clear_history(self):
        self.message_history = []

    def add_message_to_history(self, message):
        self.message_history.append(message)
    
    def set_message_history(self, message_history):
        self.message_history = message_history
    
    def set_model_temperature(self, temperature):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def get_message_history(self):
        return self.message_history

    def set_model(self, model):
        self.model = model

    def set_system_prompt(self, systemprompt):
        self.systemprompt = systemprompt