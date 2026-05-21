import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score


def _ar_lr_lambda(step, warmup_steps, total_steps, eta_min=0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))


class PretrainARModule(pl.LightningModule):
    def __init__(
        self,
        model,
        embedding_size,
        num_bands,
        lr=1e-4,
        warmup_steps=1000,
        total_steps=100_000,
        eta_min_factor=0.1,
        weight_decay=1e-3,
        eval_probe=False,
        context_size=1,
        normalize_flux=False,
    ):
        super().__init__()
        self.model = model
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
        )
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min_factor = eta_min_factor
        self.weight_decay = weight_decay
        self.eval_probe = eval_probe
        self.num_bands = num_bands
        self.context_size = context_size
        self.normalize_flux = normalize_flux
        self.gradients_ = None

        nn.init.kaiming_uniform_(self.reconstruction_head[0].weight)
        nn.init.kaiming_uniform_(self.reconstruction_head[2].weight)

        # Probe eval buffers
        self.collect_train_embs = []
        self.collect_train_labels = []
        self.collect_val_embs = []
        self.collect_val_labels = []

    @staticmethod
    def denormalize_flux(normalized_flux, mean, std):
        """Denormalize flux: x_original = normalized * std + mean"""
        return normalized_flux * std + mean

    def training_step(self, batch, batch_idx):
        batch.pop("labels", None)

        data = batch["data"].float()  # (B, T, num_bands)
        time = batch["time"].float()
        mask = batch["mask"]  # (B, T, num_bands) — True = real obs, False = padding

        # Slice to configured number of bands
        data = data[..., :self.num_bands]
        time = time[..., :self.num_bands]
        mask = mask[..., :self.num_bands]


        # AR objective: predict observation t from last context_size observations
        # Input: observations 0..T-2 (all but last), Target: observations 1..T-1 (all but first)
        inp_data = data[:, :-1, :]  # (B, T-1, num_bands)
        inp_time = time[:, :-1, :]
        inp_mask = mask[:, :-1, :]
        target = data[:, 1:, :]      # (B, T-1, num_bands)
        target_mask = mask[:, 1:, :]

        # With context_size=N: only use last N observations, skip first T-1-N
        if self.context_size > 0:
            skip = max(0, inp_data.shape[1] - self.context_size)
            inp_data = inp_data[:, skip:, :]
            inp_time = inp_time[:, skip:, :]
            inp_mask = inp_mask[:, skip:, :]
            target = target[:, skip:, :]
            target_mask = target_mask[:, skip:, :]

        # Model outputs (B, (T-1)*num_bands, D) due to TimeHandler concatenation per band
        emb = self.model(inp_data, inp_time, inp_mask)

        B, T_minus_1, nb = target.shape
        # Reshape target to match TimeHandler's per-band concatenation order
        # TimeHandler concatenates as: [band0_t0..band0_tT, band1_t0..band1_tT, ...]
        # target shape is (B, T-1, num_bands), permute to (B, num_bands, T-1) then flatten
        target_flat = target.permute(0, 2, 1).reshape(B, -1)  # (B, (T-1)*num_bands)
        target_mask_flat = target_mask.permute(0, 2, 1).reshape(B, -1)   # (B, (T-1)*num_bands)

        # Reconstruction head predicts flux at each position
        # Input: (B, (T-1)*num_bands, D)
        # Output: (B, (T-1)*num_bands, 1)
        pred = self.reconstruction_head(emb).squeeze(-1)  # (B, (T-1)*num_bands)

        # Only compute loss on targets that are real (not padding)
        loss = F.mse_loss(pred[target_mask_flat], target_flat[target_mask_flat])

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop("labels", None)
        data = batch["data"].float()
        time = batch["time"].float()
        mask = batch["mask"]

        # Slice to configured number of bands
        data = data[..., :self.num_bands]
        time = time[..., :self.num_bands]
        mask = mask[..., :self.num_bands]


        inp_data, inp_time, inp_mask = data[:, :-1], time[:, :-1], mask[:, :-1]
        target = data[:, 1:]
        target_mask = mask[:, 1:]  # Which targets are real (not padding)

        # With context_size=N: only use last N observations, skip first T-1-N
        if self.context_size > 0:
            skip = max(0, inp_data.shape[1] - self.context_size)
            inp_data = inp_data[:, skip:, :]
            inp_time = inp_time[:, skip:, :]
            inp_mask = inp_mask[:, skip:, :]
            target = target[:, skip:, :]
            target_mask = target_mask[:, skip:, :]

        emb = self.model(inp_data, inp_time, inp_mask)

        B, T_minus_1, nb = target.shape
        # Reshape target to match TimeHandler's per-band concatenation order
        target_flat = target.permute(0, 2, 1).reshape(B, -1)  # (B, (T-1)*num_bands)
        target_mask_flat = target_mask.permute(0, 2, 1).reshape(B, -1)   # (B, (T-1)*num_bands)

        pred = self.reconstruction_head(emb).squeeze(-1)  # (B, (T-1)*num_bands)

        loss = F.mse_loss(pred[target_mask_flat], target_flat[target_mask_flat])

        # Only log reconstruction metrics on dataloader 0
        if dataloader_idx == 0:
            self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
            # Per-band loss
            for b in range(nb):
                # Get indices for band b: [b*T, b*T+1, ..., b*T+T-1]
                band_indices = torch.arange(b * T_minus_1, (b + 1) * T_minus_1, device=target_mask_flat.device)
                v_b = target_mask_flat[:, band_indices].reshape(-1)
                if v_b.any():
                    band_loss = F.mse_loss(pred[:, band_indices].reshape(-1)[v_b],
                                           target_flat[:, band_indices].reshape(-1)[v_b])
                    self.log(f"val/loss_band{b}", band_loss, add_dataloader_idx=False, sync_dist=True)

        # Probe embedding collection (masked mean-pool across sequence)
        if self.eval_probe and dataloader_idx in (1, 2) and labels is not None:
            # emb is (B, (T-1)*num_bands, D), reshape to (B, T-1, num_bands, D)
            # then mean-pool across both time and band dims, excluding padded positions
            B, seq_band, D = emb.shape
            T_minus_1 = data.shape[1] - 1
            num_bands_actual = data.shape[2]
            emb_reshaped = emb.reshape(B, T_minus_1, num_bands_actual, D)

            # Use target_mask for masked pooling (same mask we used to zero embeddings)
            mask_reshaped = target_mask.unsqueeze(-1).float()  # (B, T-1, num_bands_actual, 1)
            mask_sum = mask_reshaped.sum(dim=(1, 2), keepdim=False).clamp(min=1.0)  # (B, 1)
            masked_emb = emb_reshaped * mask_reshaped  # (B, T-1, num_bands_actual, D)
            pooled = masked_emb.sum(dim=(1, 2)) / mask_sum  # (B, D)

            if dataloader_idx == 1:
                self.collect_train_embs.append(pooled.detach().cpu().numpy())
                self.collect_train_labels.append(labels.cpu().numpy())
            else:
                self.collect_val_embs.append(pooled.detach().cpu().numpy())
                self.collect_val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_start(self):
        if self.eval_probe:
            self.collect_train_embs = []
            self.collect_train_labels = []
            self.collect_val_embs = []
            self.collect_val_labels = []

    def on_validation_epoch_end(self):
        if not self.eval_probe or not self.collect_train_embs or not self.collect_val_embs:
            return

        X_tr = np.concatenate(self.collect_train_embs)
        y_tr = np.concatenate(self.collect_train_labels)
        X_val = np.concatenate(self.collect_val_embs)
        y_val = np.concatenate(self.collect_val_labels)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        pipe.fit(X_tr, y_tr)
        acc = balanced_accuracy_score(y_val, pipe.predict(X_val))
        self.log("val/linear_probe_bacc", acc, prog_bar=True)

        knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
        knn.fit(X_tr, y_tr)
        knn_acc = balanced_accuracy_score(y_val, knn.predict(X_val))
        self.log("val/knn3_bacc", knn_acc)

    def configure_optimizers(self):
        no_decay = [
            "bias",
            "LayerNorm.weight",
            "LayerNorm.bias",
            "embedding",
            "token",
            "time_encoder",
        ]

        backbone_decay, backbone_no_decay, head_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "reconstruction_head" in name:
                head_params.append(param)
            elif any(nd in name for nd in no_decay):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_decay,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {"params": backbone_no_decay, "lr": self.lr, "weight_decay": 0.0},
                {"params": head_params, "lr": self.lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        scheduler = LambdaLR(
            optimizer,
            lambda step: _ar_lr_lambda(
                step, self.warmup_steps, self.total_steps, self.eta_min_factor
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        self.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )

    @staticmethod
    def denormalize_flux(normalized_flux, mean, std):
        """Denormalize flux: x_original = normalized * std + mean

        Use precomputed mean/std from training data to invert normalization at generation time.
        """
        return normalized_flux * std + mean

    def gradfilter_ema(self, m, grads=None, alpha=0.95, lamb=2.0):
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

    def on_after_backward(self):
        self.gradients_ = self.gradfilter_ema(m=self.model, grads=self.gradients_)
