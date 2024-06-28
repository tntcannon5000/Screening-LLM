import os
import anthropic
from dotenv import load_dotenv

class ClaudeWrapper():

    def __init__(self, model, systemprompt):
        load_dotenv('.env')

        if os.getenv('ANTHROPIC_API_KEY') is None:
            os.environ["ANTHROPIC_API_KEY"] = input("Please enter your API key: ")

        self.client = anthropic.Client()
        self.model = model
        self.systemprompt = systemprompt

    def chat(self, message):

        formatted_messages = [
            {
                "role": "user",
                "content": message
            }
        ]
        print(message)
        print("============================")
        print(formatted_messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.5,
            system=self.systemprompt,
            messages=formatted_messages
        )

        content = ''.join(block.text for block in response.content)

        return content