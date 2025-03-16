import os
import google.generativeai as genai
from google.api_core import retry

class LLM():
    def __init__(self):
        self.document_mode = False
        self.llm = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    def __call__(self, input: str) -> str:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]
    
    def create_embedding(self, text):
        return self.llm.embed_content(text)
        