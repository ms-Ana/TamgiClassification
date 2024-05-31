import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import torchmetrics.retrieval
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.utils import get_model, load_obj


class LitModel(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.params = hparams
        self.save_hyperparameters()
        self.model = get_model(self.params)
        self.loss = load_obj(self.params.loss.class_name)(**self.params.loss.params)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.metric = torchmetrics.retrieval.RetrievalMAP(top_k=5)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_obj(self.params.optimizer.class_name)(
            self.model.parameters(), **self.params.optimizer.params
        )
        scheduler = load_obj(self.params.scheduler.class_name)(
            optimizer, **self.params.scheduler.params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.params.training.metric,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        y_pred = self(x)
        val_loss = self.loss(y_pred, y)
        self.log_dict(
            {"val_loss": val_loss}, on_step=True, on_epoch=True, prog_bar=True
        )
        self.validation_step_outputs.append({"y": y, "y_pred": y_pred})
        return {"y": y, "y_pred": y_pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        y_pred = self(x)
        test_loss = self.loss(y_pred, y)
        self.log_dict(
            {"test_loss": test_loss}, on_step=True, on_epoch=True, prog_bar=True
        )
        self.test_step_outputs.append({"y": y, "y_pred": y_pred})
        return {"y": y, "y_pred": y_pred}

    def _get_similarity(self, embedding: torch.tensor, others: torch.tensor, num: int):

        embedding_norm = embedding / torch.linalg.norm(embedding)
        other_embeddings_norm = others / torch.linalg.norm(others, dim=1, keepdim=True)

        similarities = torch.matmul(other_embeddings_norm, embedding_norm)

        top_similarities, top_indices = torch.topk(similarities, num)

        return top_indices, top_similarities

    def _get_similarities(self, ys, y_preds, num):
        indices = []
        target = []
        preds = []
        for i, pair in enumerate(zip(ys, y_preds)):
            y, pred = pair
            similarity_indices, similarites = self._get_similarity(
                pred, torch.cat([y_preds[:i], y_preds[i + 1 :]], dim=0), num
            )
            indices.extend([i] * num)
            preds.extend(similarites)
            target.extend([y == ys[j] for j in similarity_indices])
        return indices, target, preds

    def _get_pred_labels(self, ys, y_preds, num=5):
        target = ys
        preds = []
        for i, pred in enumerate(y_preds):
            similarity_indices, _ = self._get_similarity(
                pred, torch.cat([y_preds[:i], y_preds[i + 1 :]], dim=0), num
            )
            unique_labels, counts = torch.unique(
                ys[similarity_indices], return_counts=True
            )
            max_count = counts.max().item()
            modes = unique_labels[counts == max_count]
            pred_label = modes[0].item()
            preds.append(pred_label)

        return target, torch.tensor(preds).cuda()

    def on_validation_epoch_end(self):
        ys = (
            torch.cat([x["y"] for x in self.validation_step_outputs], dim=0)
            .type(torch.IntTensor)
            .cuda()
        )
        y_preds = torch.cat([x["y_pred"] for x in self.validation_step_outputs], dim=0)
        num = 15

        indices, target, preds = self._get_similarities(ys, y_preds, num)

        self.log_dict(
            {
                "val_rmap": self.metric(
                    torch.tensor(preds),
                    torch.tensor(target),
                    indexes=torch.tensor(indices),
                )
            },
            on_epoch=True,
            on_step=False,
        )

    def on_test_epoch_end(self):
        ys = (
            torch.cat([x["y"] for x in self.test_step_outputs], dim=0)
            .squeeze()
            .type(torch.IntTensor)
            .cuda()
        )
        y_preds = torch.cat(
            [x["y_pred"] for x in self.test_step_outputs], dim=0
        ).squeeze()

        num = 15

        indices, target, preds = self._get_similarities(ys, y_preds, num)

        for k in [1, 5, 10]:
            metric = torchmetrics.retrieval.RetrievalMAP(top_k=k)
            self.log_dict(
                {
                    f"rmap@{k}": metric(
                        torch.tensor(preds),
                        torch.tensor(target),
                        indexes=torch.tensor(indices),
                    )
                },
                on_epoch=True,
                on_step=False,
            )
        target, preds = self._get_pred_labels(ys, y_preds, 15)
        num_classes = torch.unique(ys)
        num_classes = num_classes.size(0)

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).cuda()
        self.log_dict({"test_acc": acc(preds, target)})
        for k in range(6):
            self.log_dict(
                {f"test_acc_{k}": acc(preds[target == k], target[target == k])}
            )
