from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi_health import health
from contextlib import asynccontextmanager
from .knowledge_base import MilvusKnowledgeBase
from .embedding_api import KServeEmbeddingAPI, LocalModelEmbeddingAPI, EmbeddingAPI
from .models import SearchRequest, SearchResponse
import os
from typing import Dict


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


def is_application_healthy() -> Dict[str, bool]:
    return {
        "knowledge_base": knowledge_base.is_healthy(),
        "embedding_api": embedding_api.is_healthy(),
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
app.add_api_route("/health", health([is_application_healthy]))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    embedded_query = embedding_api.embed(request.message)
    results = knowledge_base.search(embedded_query)
    return SearchResponse(message="Search successful", components=results)
