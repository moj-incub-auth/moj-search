from abc import abstractmethod
from typing import Protocol, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class EmbeddingAPI(Protocol):

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

class KServeEmbeddingAPI(EmbeddingAPI):
    deployment_url: str
    model: str
    openai_client: OpenAI

    def __init__(self, deployment_url: str, model: str):
        self.deployment_url = deployment_url
        self.model = model

    def is_healthy(self) -> bool:
        return self.openai_client is not None

    def connect(self):
        self.openai_client = OpenAI(
            base_url=f"{self.deployment_url}/openai/v1",
            api_key="empty"
        )

    def close(self):
        self.openai_client.close()
        self.openai_client = None

    def embed(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

class LocalModelEmbeddingAPI(EmbeddingAPI):
    embedding_model: str
    sentence_transformers_client: SentenceTransformer

    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model

    def is_healthy(self) -> bool:
        return self.sentence_transformers_client is not None

    def connect(self):
        # TODO: Add local model support
        self.sentence_transformers_client = SentenceTransformer(self.embedding_model, trust_remote_code=True)

    def close(self):
        self.sentence_transformers_client = None

    def embed(self, text: str) -> List[float]:
        return self.sentence_transformers_client.encode(text).tolist()