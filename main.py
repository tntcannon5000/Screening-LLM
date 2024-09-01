from src.modules.realtime_conversation import InterviewBot
from src.modules.post_conversation import PostConversationProcessor
from dotenv import find_dotenv, load_dotenv
import os

def main():
    # Load the environment variables
    load_dotenv(os.path.join(os.path.dirname(os.getcwd()), ".env"))
    load_dotenv(find_dotenv())
    # List of required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUME_API_KEY",
        "LANGCHAIN_API_KEY"
    ]

    # Check for missing environment variables and prompt user if needed
    for var in required_vars:
        if var not in os.environ:
            value = input(f"{var} is not set. Please enter its value: ")
            os.environ[var] = value


    """Main function to run the interview and post conversation processing"""
    try:
        bot = InterviewBot()
        timestamp, pass_rate = bot.main()
    except Exception as e:
        print(f"An error occurred during the interview: {e}")
        result = None
    else:
        postcon = PostConversationProcessor(timestamp, pass_rate)
        result = postcon.run()
    result = None 
    if result is not None:
        print(result)

    input("Press enter to exit")

def eval_candidate(timestamp, pass_rate):
    postcon = PostConversationProcessor(str(timestamp), pass_rate)
    result = postcon.run()
    print(result)

# Run the main.py file
if __name__ == '__main__':
    """Run the main function"""
    #main()
    eval_candidate(1725184560, 47)