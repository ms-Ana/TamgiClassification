import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def embeddings_upload(
    collection_name: str, embeddings_file: str, embeddings_dim: int = 512
):
    with open(embeddings_file, "r") as f:
        embeddings = json.load(f)
    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings_dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb_info["embedding"],
            payload={"label": emb_info["label"], "image_file": img},
        )
        for img, emb_info in embeddings.items()
    ]

    operation_info = client.upsert(
        collection_name=collection_name, wait=True, points=points
    )
    print(operation_info)
