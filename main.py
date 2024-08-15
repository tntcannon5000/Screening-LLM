import src.modules.realtime_conversation as realtime_conversation
from src.modules.post_conversation import PostConversationProcessor

def main():
    #timestamp = realtime_conversation.main()
    postcon = PostConversationProcessor(1723670854)
    result = postcon.run()

    print(result)
# Run the main.py file
if __name__ == '__main__':
    main()