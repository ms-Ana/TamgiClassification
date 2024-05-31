import hydra
import pytorch_lightning as pl
import torch
from data.LitDataset import LitDataset
from model.LitModel import LitModel
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import set_seed

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)
    model = LitModel(hparams=cfg)
    data_module = LitDataset(data_params=cfg.data, aug_params=cfg.augmentation)

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        **cfg.callbacks.model_checkpoint.params
    )

    # loggers
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)

    if cfg.general.resume_from_checkpoint is not None:
        cfg.general.resume_from_checkpoint = hydra.utils.to_absolute_path(
            cfg.general.resume_from_checkpoint
        )
        model = LitModel.load_from_checkpoint(cfg.general.resume_from_checkpoint)

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[early_stopping, model_checkpoint, lr_logger],
        **cfg.trainer,
    )
    model.to("cuda")
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    torch.save(model.state_dict(), cfg.model.backbone.name + ".pth")


if __name__ == "__main__":
    run_training()
