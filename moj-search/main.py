# Python imports
import os
import logging
from abc import abstractmethod
from typing import Protocol, List, Dict, Any
from contextlib import asynccontextmanager

# Third-party imports
from pydantic import BaseModel

# FastAPI imports
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi_health import health

# AI/ML imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
)


logger = logging.getLogger(f"uvicorn.{__name__}")


# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    message: str


class SearchComponent(BaseModel):
    title: str
    url: str
    description: str
    parent: str
    accessibility: str
    created_at: str | None
    updated_at: str | None
    has_research: bool
    views: int


class ScoredSearchComponent(SearchComponent):
    score: float


class SearchResponse(BaseModel):
    message: str
    components: list[SearchComponent]


class MilvusKnowledgeBase:
    collection_name: str
    host: str
    port: int
    collection: Collection

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        host: str = "localhost",
        port: int = 19530,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port

    def is_healthy(self) -> bool:
        return self.collection is not None

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)
        # Get collection and load it
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"Connected to Milvus collection: {self.collection_name}")

    def close(self):
        self.collection.release()
        self.collection = None
        connections.disconnect("default")
        logger.info(f"Disconnected from Milvus collection: {self.collection_name}")

    def search(
        self, embedded_query: List[float], limit: int = 10
    ) -> List[ScoredSearchComponent]:
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Perform search
        results = self.collection.search(
            data=[embedded_query],
            anns_field="content_embedding",
            param=search_params,
            limit=limit,
            output_fields=[
                "title",
                "description",
                "url",
                "parent",
                "accessibility",
                "has_research",
                "views",
            ],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = ScoredSearchComponent(
                    score=hit.score,
                    title=hit.entity.get("title"),
                    description=hit.entity.get("description"),
                    url=hit.entity.get("url"),
                    parent=hit.entity.get("parent"),
                    accessibility=hit.entity.get("accessibility"),
                    has_research=hit.entity.get("has_research"),
                    created_at=hit.entity.get("created_at"),
                    updated_at=hit.entity.get("updated_at"),
                    views=hit.entity.get("views"),
                )
                formatted_results.append(result)
        return formatted_results


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
            base_url=f"{self.deployment_url}/v1", api_key="empty"
        )
        logger.info(f"Connected to KServe embedding API: {self.deployment_url}")

    def close(self):
        self.openai_client.close()
        self.openai_client = None
        logger.info(f"Disconnected from KServe embedding API: {self.deployment_url}")

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
        self.sentence_transformers_client = SentenceTransformer(
            self.embedding_model, trust_remote_code=True
        )
        logger.info(f"Connected to local embedding model: {self.embedding_model}")

    def close(self):
        self.sentence_transformers_client = None
        logger.info(f"Disconnected from local embedding model: {self.embedding_model}")

    def embed(self, text: str) -> List[float]:
        return self.sentence_transformers_client.encode(text).tolist()


def create_knowledge_base() -> MilvusKnowledgeBase:
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    return MilvusKnowledgeBase(collection_name=collection_name, host=host, port=port)


def create_embedding_api() -> EmbeddingAPI:
    if os.environ.get("EMBEDDING_API_URL"):
        deployment_url = os.getenv("EMBEDDING_API_URL")
        model = os.getenv("EMBEDDING_API_MODEL", "qwen3-embedding")
        return KServeEmbeddingAPI(deployment_url=deployment_url, model=model)
    else:
        embedding_model = os.getenv(
            "EMBEDDING_API_MODEL", "nomic-ai/nomic-embed-text-v1.5"
        )
        return LocalModelEmbeddingAPI(embedding_model=embedding_model)


knowledge_base = create_knowledge_base()
embedding_api = create_embedding_api()


def knowledge_base_status() -> bool:
    return knowledge_base.is_healthy()


def embedding_api_status() -> bool:
    return embedding_api.is_healthy()


async def health_handler(**kwargs) -> Dict[str, Any]:
    is_success = all(kwargs.values())
    return {
        "status": "success" if is_success else "failure",
        "results": kwargs.items(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    knowledge_base.connect()
    embedding_api.connect()
    yield
    knowledge_base.close()
    embedding_api.close()


app = FastAPI(title="Vector Search", description="Vector Search API", lifespan=lifespan)
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)
app.add_api_route(
    "/health",
    health(
        [knowledge_base_status, embedding_api_status],
        success_handler=health_handler,
        failure_handler=health_handler,
    ),
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    embedded_query = embedding_api.embed(request.message)
    results = knowledge_base.search(embedded_query)
    return SearchResponse(message="Search successful", components=results)
