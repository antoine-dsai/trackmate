import requests
import os
from dotenv import load_dotenv
from typing import List, Dict
from requests.exceptions import RequestException

# Load environment variables from .env file
load_dotenv()

# Define the base URL of your FastAPI app
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class TrackMateSDK:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _get(self, url: str) -> Dict:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Error during GET request: {e}")
            return {"error": str(e)}

    def _post(self, url: str, data: Dict) -> Dict:
        try:
            print(f"Sending POST request to {url} with data: {data}")  # Debug print
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Error during POST request: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"error": "Unexpected error"}

    def create_experiment(self, name: str, description: str) -> Dict:
        url = f"{self.base_url}/experiments/"
        data = {"name": name, "description": description}
        print(f"Payload: {data}")  # Debugging the payload before sending it
        return self._post(url, data)

    def start_run(self, experiment_id: str, run_name: str, nested: bool = False) -> Dict:
        url = f"{self.base_url}/experiments/{experiment_id}/runs/"
        data = {"run_name": run_name, "nested": nested}
        return self._post(url, data)

    def log_param(self, run_id: str, key: str, value: str) -> Dict:
        url = f"{self.base_url}/runs/{run_id}/params/"
        data = {"key": key, "value": value}
        return self._post(url, data)

    def log_metric(self, run_id: str, key: str, value: float) -> Dict:
        url = f"{self.base_url}/runs/{run_id}/metrics/"
        data = {"key": key, "value": value}
        return self._post(url, data)

    def log_artifact(self, run_id: str, file_path: str) -> Dict:
        url = f"{self.base_url}/runs/{run_id}/artifacts/"

        # Open the file to upload as part of multipart form data
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file, 'application/octet-stream')}
            response = requests.post(url, files=files)

        try:
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Error during artifact upload: {e}")
            return {"error": str(e)}
