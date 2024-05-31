from collections import Counter

from deep_pipeline.Pipeline import Pipeline
from deep_pipeline.QdrantSearcher import QdrantSearcher
from fastapi import APIRouter

router = APIRouter()
pipeline = Pipeline("./pipeline.yaml")
qdrant_searcher = QdrantSearcher("tamgi")


@router.get("/status")
def status():
    return {"status": "ok"}


def define_label(labels: list):
    count = Counter(labels)
    max_count = max(count.values())
    for element in lst:
        if count[element] == max_count:
            return element


@router.post("/classify")
def query(img_path: str):
    embedding = Pipeline(img_path)
    search_result = QdrantSearcher.search(embedding)

    labels = [emb["payload"]["label"] for emb in search_result]
    label = define_label(labels)

    img_files = [emb["payload"]["image_file"] for emb in search_result][:3]

    return {"class": label, "image_files": img_files}
