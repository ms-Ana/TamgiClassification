import json
import os

import hydra
from model.LitModel import LitModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from embeddings.src.data.LitDataset import ImageDataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_model(cfg: DictConfig):
    model = LitModel(hparams=cfg)
    predictions, paths = [], []
    for batch in tqdm(
        DataLoader(
            ImageDataset(
                image_dir=cfg.data.image_dir,
                target_size=(cfg.data.width, cfg.data.height),
            ),
            batch_size=cfg.data.batch_size,
            shuffle=False,
        )
    ):
        x, xpath = batch
        batch = x.to(cfg.general.device)
        preds = model(x)
        predictions.extend(preds.detach().cpu().numpy().astype(float))
        paths.extend(xpath)
    with open(
        os.path.join(cfg.data.embeddings_save_path, f"{cfg.model.backbone.name}.json"),
        "w",
    ) as f:
        json.dump({path: list(emb) for path, emb in zip(paths, predictions)}, f)


if __name__ == "__main__":
    run_model()
