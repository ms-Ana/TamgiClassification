import os
from abc import ABC
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from utils.utils import load_obj


def normalize_cv2(img, mean, denominator):
    if mean.shape and len(mean) != 4 and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
    elif len(denominator) != 4 and denominator.shape != img.shape:
        denominator = np.array(
            denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64
        )

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img


def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)


class ImageDataset(Dataset):
    def __init__(self, **kwargs):
        self.image_dir = Path(kwargs["image_dir"])
        self.data = pd.read_csv(kwargs["data"])
        self.images = [
            os.path.join(idir, imgp)
            for idir in os.listdir(self.image_dir)
            for imgp in os.listdir(os.path.join(self.image_dir, idir))
        ]
        self.preprocess = kwargs.get("preprocess", None)
        self.params = kwargs

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.tensor]:
        img_path = self.image_dir / self.images[idx]
        label = torch.tensor([self.data.loc[idx, "label"]], dtype=torch.int64)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            img = self.preprocess(image=img)["image"]
        else:
            img = ToTensorV2(p=1.0)(image=img)["image"]
        return img, label


class LitDataset(pl.LightningDataModule, ABC):
    def __init__(self, data_params: DictConfig, aug_params: DictConfig):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.save_hyperparameters()
        self.params = data_params
        self.augmentations = self._get_augmentations(aug_params)

    def _get_augmentations(self, hparams_aug: DictConfig) -> dict:
        train_augs = Compose(
            [
                load_obj(aug["class_name"])(**aug["params"])
                for aug in hparams_aug["train"]["augs"]
            ]
        )
        val_augs = Compose(
            [
                load_obj(aug["class_name"])(**aug["params"])
                for aug in hparams_aug["val"]["augs"]
            ]
        )
        return {"train_augs": train_augs, "val_augs": val_augs}

    def prepare_data(self):
        self.train_dataset = ImageDataset(
            image_dir=self.params.image_dir,
            data=self.params.train_path,
            preprocess=self.augmentations["train_augs"],
        )

        self.val_dataset = ImageDataset(
            image_dir=self.params.image_dir,
            data=self.params.val_path,
            preprocess=self.augmentations["val_augs"],
        )
        self.test_dataset = ImageDataset(
            image_dir=self.params.image_dir,
            data=self.params.test_path,
            preprocess=self.augmentations["val_augs"],
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )
