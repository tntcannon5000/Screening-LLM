from src.modules.realtime_conversation import InterviewBot
from src.modules.post_conversation import PostConversationProcessor

def main():
    
    try:
        bot = InterviewBot()
        timestamp = bot.main()
    except Exception as e:
        print(f"An error occurred during the interview: {e}")
        result = None
    else:
        postcon = PostConversationProcessor(timestamp)
        result = postcon.run()

    if result is not None:
        print(result)

    input("Press enter to exit")

    
# Run the main.py file
if __name__ == '__main__':
    main()