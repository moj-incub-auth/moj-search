from typing import List
from .models import ScoredSearchComponent
from pymilvus import (
    connections,
    Collection,
)
import logging

logger = logging.getLogger(__name__)


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
        logging.info(f"Connected to Milvus collection: {self.collection_name}")

    def close(self):
        self.collection.release()
        self.collection = None
        connections.disconnect("default")
        logging.info(f"Disconnected from Milvus collection: {self.collection_name}")

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
