import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace
from dotenv import load_dotenv

load_dotenv()

class ModelEndpoints:
    def __init__() -> str:
        self.model_endpoint = os.environ["MODEL_ENDPOINT"]
        self.model_key = os.environ["MODEL_KEY"]

    class Response(TypedDict):
        query: str
        response: str

    @trace
    def __call__(self: Self, question: str) -> Response:
        # We can integrate with any model endpoint
        output = self.call_model_endpoint(question)
        return output

    def query(self: Self, endpoint: str, headers: str, payload: str) -> str:
        response = requests.post(url=endpoint, headers=headers, json=payload)
        return response.json()

    def call_model_endpoint(self: Self, question: str) -> Response:
        endpoint = self.model_endpoint
        key = self.model_key

        headers = {"Content-Type": "application/json", "api-key": key}
        payload = {"text": question}        
        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        answer = output["response"]

        return {"query": question, "response": answer}