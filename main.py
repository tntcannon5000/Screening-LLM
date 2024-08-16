import src.modules.realtime_conversation as realtime_conversation
from src.modules.post_conversation import PostConversationProcessor

def main():
    #timestamp = realtime_conversation.main()
    postcon = PostConversationProcessor(1723769368)
    result = postcon.run()

    print(result)

    input("Press enter to exit")

    
# Run the main.py file
if __name__ == '__main__':
    main()