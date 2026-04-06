from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
import pytorch_lightning as pl
from torch.optim.lr_scheduler import (
    SequentialLR,
    ConstantLR,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    LinearLR,
    ExponentialLR,
)
import logging
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class PretrainModule(pl.LightningModule):
    def __init__(
        self, model, loss, lr=0.001, eval_knn=False, eval_regressor=True, **kwargs
    ):
        """
        Batch must be a tuple of batches: (batch, augmented batch)
        Model output must be of shape (bsz,embeddings)
        Loss input is (embedding_batch_1, embedding_batch_2)
        Args:
            model (_type_): _description_
            loss (_type_): _description_

        """
        super().__init__()
        self.gradients_ = None
        self.lr = lr
        self.model = model
        self.loss = loss
        logging.debug("using learning rate {}".format(self.lr))
        self.init_model()
        self.collect_train_embs = None
        self.collect_train_labels = None
        self.collect_val_embs = None
        self.collect_val_labels = None
        self.eval_knn = eval_knn
        self.eval_regressor = eval_regressor
        self.learning_rate = kwargs["learning_rate"]
        self.warmup = 0

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def training_step(self, batch, batch_idx):
        embedding_1 = self.model(**batch[0])[
            :, 0, :
        ]  # torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        embedding_2 = self.model(**batch[1])[
            :, 0, :
        ]  # torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)

        # embedding_1  =torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        # embedding_2 = torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)
        loss_dict = self.loss(embedding_1, embedding_2)

        with torch.no_grad():
            for key, value in loss_dict.items():
                if "emb_corr" in key:
                    self.logger.experiment.add_histogram(key, value, self.global_step)
                elif "mean_" in key:
                    self.logger.experiment.add_histogram(key, value, self.global_step)

                elif "percent" in key:
                    self.log(
                        f"{key}", value, on_epoch=False, on_step=True, sync_dist=True
                    )
                else:
                    self.log(
                        f"loss_train/{key}",
                        value,
                        on_epoch=False,
                        on_step=True,
                        sync_dist=True,
                    )

        # self.log(f'Tmax_0',self.model.time_encoder.time_encoders[0].Tmax,on_step = True, sync_dist=True)
        return loss_dict["loss"]

    def gradfilter_ema(
        self,
        m: nn.Module,
        grads: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 0.95,
        lamb: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        if grads is None:
            grads = {
                n: p.grad.data.detach()
                for n, p in m.named_parameters()
                if p.requires_grad and p.grad is not None
            }

        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
                p.grad.data = p.grad.data + grads[n] * lamb

        return grads

    def on_after_backward(self) -> None:
        self.gradients = self.gradfilter_ema(m=self.model, grads=self.gradients_)

    def validation_step(self, batch, batch_idx):
        embedding_1 = self.model(**batch[0])[
            :, 0, :
        ]  # torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        embedding_2 = self.model(**batch[1])[
            :, 0, :
        ]  # torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)

        # embedding_1  =torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        # embedding_2 = torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)
        loss_dict = self.loss(embedding_1, embedding_2)
        with torch.no_grad():
            for key, value in loss_dict.items():
                if "emb_corr" not in key:
                    self.log(
                        f"loss_validation/{key}",
                        value,
                        on_epoch=True,
                        on_step=False,
                        add_dataloader_idx=False,
                    )
        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        # -------------------------
        # Parameter grouping
        # -------------------------
        no_decay_keywords = [
            "bias",
            "LayerNorm.weight",
            "LayerNorm.bias",
            "embedding",
            "token",
            "time_encoder",
        ]

        backbone_decay = []
        backbone_no_decay = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "classifier" in name:
                head_params.append(param)
            elif any(nd in name for nd in no_decay_keywords):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        # -------------------------
        # Optimizer
        # -------------------------
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_decay,
                    "lr": self.learning_rate,
                    "weight_decay": 1e-3,
                },
                {
                    "params": backbone_no_decay,
                    "lr": self.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": head_params,
                    "lr": self.learning_rate,  # best performance is  * 1e-2
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #    optimizer,
        #    start_factor=0.01,
        #    total_iters=1000,
        # )

        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer,
        #    T_max=100,
        #    eta_min=self.learning_rate * 0.01,
        # )
        return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": cosine_scheduler,
                "interval": "step",  # IMPORTANT for large datasets
                "frequency": 1,
            },
        }

        # return optimizer

    def get_real_classes_weights(self, labels):

        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.unique(labels)]
        )
        # print('class_sampler_count', class_sample_count)
        weight = 1.0 / class_sample_count
        uniques = np.unique(labels).astype(int)
        d = {key: value for key, value in zip(uniques, weight)}
        samples_weight = np.array([d[labels[i].item()] for i in range(len(labels))])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def get_regressor_eval(
        self,
    ):
        weights = self.get_real_classes_weights(torch.tensor(self.collect_train_labels))
        weights_dict = {float(i): weights[i] for i in range(len(weights))}

        std_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # z = (x - mean) / std
                (
                    "model",
                    LogisticRegression(
                        random_state=0,
                        max_iter=1000,
                        multi_class="ovr",
                        class_weight=weights_dict,
                    ),
                ),
            ]
        )
        std_pipeline.fit(self.collect_train_embs, self.collect_train_labels)
        val_preds = std_pipeline.predict(self.collect_val_embs)
        return val_preds

    def get_knn_eval(self):
        knn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # z = (x - mean) / std
                # ('pca', PCA()),
                ("model", KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ]
        )
        knn_pipeline.fit(self.collect_train_embs, self.collect_train_labels)
        val_preds = knn_pipeline.predict(self.collect_val_embs)
        return val_preds
