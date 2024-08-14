from hume import HumeBatchClient
from hume.models.config import LanguageConfig
from collections import defaultdict
import json
import os
from dotenv import load_dotenv

class HumeSentimentAnalyzer:
    def __init__(self, api_key=os.getenv("HUME_API_KEY")):
        self.client = HumeBatchClient(api_key)

    def analyze_audio(self, file_path):
        config = LanguageConfig(
            granularity="conversational_turn",
            sentiment={},
            toxicity={}
        )
        
        job = self.client.submit_job(None, [config], files=[file_path])
        print("Analyzing audio...")
        job.await_complete()
        predictions = job.get_predictions()

        return self._process_predictions(predictions)

    def _process_predictions(self, predictions):
        pred = predictions[0]['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions'][0]
        
        result = {
            'emotions': {emotion['name']: emotion['score'] for emotion in pred['emotions']},
            'sentiments': {sentiment['name']: sentiment['score'] for sentiment in pred['sentiment']},
            'toxicity': {toxicity['name']: toxicity['score'] for toxicity in pred['toxicity']}
        }

        return result

    def print_analysis(self, analysis):
        print("Emotion Scores:")
        for name, score in analysis['emotions'].items():
            print(f"  {name}: {score}")

        print("\nSentiment Scores:")
        for name, score in analysis['sentiments'].items():
            print(f"  {name}: {score}")

        print("\nToxicity Scores:")
        for name, score in analysis['toxicity'].items():
            print(f"  {name}: {score}")


# Example usage
if __name__ == "__main__":
    
    # Get the absolute path of the currently executing script
    script_path = os.path.abspath(__file__)
    print("Script Path : "+script_path)
    # Get the directory containing the script
    script_directory = os.path.dirname(script_path)
    print("Script Directory : "+script_directory)
    # Navigate one folder up
    one_folder_up = os.path.dirname(script_directory)
    # Construct the path to the .env file in the parent folder
    parent_folder_path = os.path.dirname(os.getcwd())
    dotenv_path = os.path.join(parent_folder_path, ".env")
    # Load the .env file
    load_dotenv(dotenv_path)
    analyzer = HumeSentimentAnalyzer(os.getenv("HUME_API_KEY"))
    result = analyzer.analyze_audio(os.path.join(script_directory, "AllforOne.wav"))
    analyzer.print_analysis(result)