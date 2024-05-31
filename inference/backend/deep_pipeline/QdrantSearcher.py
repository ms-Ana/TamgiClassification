import numpy as np
from qdrant_client import QdrantClient, models


class QdrantSearcher:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, vector: np.ndarray):
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5,
        )

        return search_result
