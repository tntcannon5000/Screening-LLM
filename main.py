from src.modules.realtime_conversation import InterviewBot
from src.modules.post_conversation import PostConversationProcessor

def main():
    
    bot = InterviewBot()
    timestamp = bot.main()
    postcon = PostConversationProcessor(timestamp)
    result = postcon.run()

    print(result)
# Run the main.py file
if __name__ == '__main__':
    main()