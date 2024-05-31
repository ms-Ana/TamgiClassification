import hydra
import pytorch_lightning as pl
import torch
from data.LitDataset import LitDataset
from model.LitModel import LitModel
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import set_seed

torch.set_float32_matmul_precision("medium")
import os


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_testing(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)
    model = LitModel(hparams=cfg)
    data_module = LitDataset(data_params=cfg.data, aug_params=cfg.augmentation)

    cfg.general.resume_from_checkpoint = hydra.utils.to_absolute_path(
        cfg.general.resume_from_checkpoint
    )
    model = LitModel.load_from_checkpoint(cfg.general.resume_from_checkpoint)

    trainer = pl.Trainer(
        **cfg.trainer,
    )
    model.to("cuda")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    run_testing()
