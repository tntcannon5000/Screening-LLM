from hume import HumeBatchClient
from hume.models.config import LanguageConfig
from collections import defaultdict
import json
import os
from dotenv import load_dotenv

class HumeSentimentAnalyzer:
    """Wrapper for Hume.ai sentiment analysis."""
    def __init__(self, api_key=os.getenv("HUME_API_KEY")):
        """
        Initialize the HumeSentimentAnalyzer.

        Args:
            api_key (str): API key for Hume.ai.

        Raises:
            ValueError: If the API key is not provided or invalid.
        """
        if not api_key:
            raise ValueError("API key is required")
        self.client = HumeBatchClient(api_key)

    def analyze_audio(self, file_path):
        """
        Asynchronously analyze audio file.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            dict: Analysis result from Hume.ai containing emotions, sentiments, and toxicity scores.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: For any other errors during analysis.
        """
        try:
            print(f"Analyzing audio file: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")

            config = LanguageConfig(
                granularity="conversational_turn",
                sentiment={},
                toxicity={}
            )
            
            file_path = os.path.abspath(file_path)

            job = self.client.submit_job(None, [config], files=[file_path])
            print("Job submitted successfully")
            job.await_complete()
            print("Job completed")
            predictions = job.get_predictions()
            return self._process_predictions(predictions)
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            raise

    def _process_predictions(self, predictions):
        """
        Process the predictions from Hume.ai.

        Args:
            predictions (list): Raw predictions from Hume.ai.

        Returns:
            dict: Processed analysis result containing emotions, sentiments, and toxicity scores.
        """
        print(predictions)
        pred = predictions[0]['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions'][0]
        
        result = {
            'emotions': {emotion['name']: emotion['score'] for emotion in pred['emotions']},
            'sentiments': {sentiment['name']: sentiment['score'] for sentiment in pred['sentiment']},
            'toxicity': {toxicity['name']: toxicity['score'] for toxicity in pred['toxicity']}
        }
        return result

    def print_analysis(self, analysis):
        """
        Print the analysis results.

        Args:
            analysis (dict): Processed analysis result containing emotions, sentiments, and toxicity scores.
        """
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
    result = analyzer.analyze_audio("D:/Kent/University Of Kent UK/Projects/Disso/Screening-LLM/data/interviews/1723836354/audio/audio_2_1723836354.wav")
    analyzer.print_analysis(result)