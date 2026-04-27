import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from typing import Dict, Optional
import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class PretrainModule(pl.LightningModule):
    def __init__(
        self, model, loss, eval_knn=False, eval_regressor=True, **kwargs
    ):
        """
        Batch must be a tuple of batches: (batch, augmented batch)
        Model output must be of shape (bsz,embeddings)
        Loss input is (embedding_batch_1, embedding_batch_2)
        Args:
            model: neural network model for embedding generation
            loss: loss function that takes two embeddings and returns a dict with at least 'loss' key

        """
        super().__init__()
        self.gradients_ = None
        self.model = model
        self.loss = loss
        self.learning_rate = kwargs["learning_rate"]
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.eta_min_factor = kwargs.get("eta_min_factor", 1e-2)
        logging.debug("using learning rate %s", self.learning_rate)
        self.init_model()
        self.collect_train_embs = None
        self.collect_train_labels = None
        self.collect_val_embs = None
        self.collect_val_labels = None
        self.eval_knn = eval_knn
        self.eval_regressor = eval_regressor
        self._histogram_keys = None
        self._scalar_keys = None

    def init_model(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def _get_embeddings(self, batch):
        """Extract CLS token embeddings from model output for both batch elements."""
        embedding_1 = self.model(**batch[0])[:, 0, :]
        embedding_2 = self.model(**batch[1])[:, 0, :]
        return embedding_1, embedding_2

    def training_step(self, batch, _batch_idx):
        embedding_1, embedding_2 = self._get_embeddings(batch)
        loss_dict = self.loss(embedding_1, embedding_2)

        # Precompute key categories on first call
        if self._histogram_keys is None:
            self._histogram_keys = set()
            self._scalar_keys = set()
            self._percent_keys = set()
            for key in loss_dict.keys():
                if "emb_corr" in key or "mean_" in key:
                    self._histogram_keys.add(key)
                elif "percent" in key:
                    self._percent_keys.add(key)
                else:
                    self._scalar_keys.add(key)

        with torch.no_grad():
            for key in self._histogram_keys & loss_dict.keys():
                self.logger.experiment.add_histogram(key, loss_dict[key], self.global_step)
            for key in self._percent_keys & loss_dict.keys():
                self.log(f"{key}", loss_dict[key], on_epoch=False, on_step=True, sync_dist=False)
            for key in self._scalar_keys & loss_dict.keys():
                self.log(f"loss_train/{key}", loss_dict[key], on_epoch=False, on_step=True, sync_dist=False)

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
        self.gradients_ = self.gradfilter_ema(m=self.model, grads=self.gradients_)

    def validation_step(self, batch, _batch_idx):
        embedding_1, embedding_2 = self._get_embeddings(batch)
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
            if "project" in name:
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
                    "lr": self.learning_rate,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # -------------------------
        # Two-stage scheduler
        # -------------------------


        return {
            "optimizer": optimizer,
           # "lr_scheduler": {
            #    "scheduler": sequential_scheduler,
             #   "interval": "step",
              #  "frequency": 1,
            #},
        }

    def get_real_classes_weights(self, labels):
        # Convert labels to numpy if needed
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels

        # Use bincount for O(n) instead of O(n_classes * n)
        class_sample_count = np.bincount(labels_np.astype(int))
        weight = 1.0 / class_sample_count

        # Direct indexing instead of dict lookup
        samples_weight = weight[labels_np.astype(int)]
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def get_regressor_eval(self):
        weights = self.get_real_classes_weights(self.collect_train_labels)
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
