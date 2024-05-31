from argparse import Namespace
from typing import Dict

import cv2
from InferenceDetector import DetInferencer
from InferenceEmbedder import EmbeddingInferencer

from .utils.data_utils import normalize


class Pipeline:
    def __init__(self, config: Dict):
        self.params = Namespace(**config)
        self.det_inferencer = DetInferencer(
            model=self.params.detector["model"],
            weights=self.params.detector["weights"],
            device=self.params.device,
        )
        self.emb_inferencer = EmbeddingInferencer(
            model_ckpt=self.params.embedder["model_ckpt"],
            embedding_size=self.params.embedder["embedding_size"],
        )

    def __call__(self, img_path: str):
        det = self.det_inferencer(img_path)["predictions"][0]["bboxes"][0]
        det = list(map(int, det))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = image[det[1] : det[3], det[0] : det[2]]
        crop = cv2.resize(crop, (299, 299))
        crop = normalize(crop, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        embedding = self.emb_inferencer(crop.to(self.params.device))

        return embedding.numpy()
