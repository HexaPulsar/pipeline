from typing import Dict, Optional
import torch.nn as nn
import torch
import torchmetrics
from collections import OrderedDict
import numpy as np
import pytorch_lightning as pl

import torchmetrics.classification
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY

import matplotlib.pyplot as plt

import glob
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

TAXONOMY = ELASTICC_TAXONOMY


class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model,
        classifier,
        loss,
        experiment_type: str,
        lc_load_ckpt=None,
        tab_load_ckpt=None,
        freeze_lc=False,
        freeze_tab=False,
        weight_str_parse_lc=None,
        weight_str_parse_tab=None,
        **kwargs,
    ):
        super().__init__()
        self.gradients_ = None

        self.model = model
        self.classifier = classifier
        self.init_model()
        self.loss = loss
        self.learning_rate = kwargs["learning_rate"]
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.eta_min_factor = kwargs.get("eta_min_factor", 1e-2)

        parse_exp_type = experiment_type.split("_")
        self.modalities = []
        self.modalities += ["LC"] if "LC" in parse_exp_type else []
        self.modalities += (
            ["TAB"] if "MD" in parse_exp_type or "FEAT" in parse_exp_type else []
        )
        self.modalities += (
            ["MIX"]
            if ("MD" in parse_exp_type or "FEAT" in parse_exp_type)
            and ("LC" in parse_exp_type)
            else []
        )

        self.init_metrics()

        if lc_load_ckpt is not None:
            try:
                self._load_checkpoint(
                    lc_load_ckpt,
                    self.model,
                    weight_str_parse_lc,
                    exclude_key="loss",
                )
            except Exception as e:
                print(e, "path", lc_load_ckpt)
        if freeze_lc:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        if tab_load_ckpt is not None:
            self._load_checkpoint(
                tab_load_ckpt,
                self.model.transformer_tab,
                weight_str_parse_tab,
                exclude_key="tab",
            )

        if freeze_tab:
            for name, param in self.model.transformer_tab.named_parameters():
                param.requires_grad = False

    def _load_checkpoint(self, ckpt_path, model_component, weight_parse_rule, exclude_key="loss"):
        """Load checkpoint weights into a model component with key remapping."""
        _ckpt = glob.glob(ckpt_path + "*.ckpt")[-1]
        checkpoint_ = torch.load(_ckpt)
        weights = OrderedDict()

        for key in checkpoint_["state_dict"].keys():
            if exclude_key in key:
                continue
            weights[key.replace(weight_parse_rule[0], weight_parse_rule[1])] = checkpoint_["state_dict"][key]

        model_component.load_state_dict(weights, strict=True)
        print(f"loaded checkpoint {_ckpt}")

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

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

    def training_step(self, batch_data, batch_idx):

        labels = batch_data.pop("labels")
        preds = self.classifier(self.model(**batch_data))
        loss = 0

        if "LC" in preds.keys():
            partial_loss = self.loss(
                preds["LC"], labels.long()
            )  # + self.triplet_loss(embs, labels.long())
            loss += partial_loss
            self.LC_train_metrics(preds["LC"], labels.long())

            self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
            self.log(
                f"loss_train/lc",
                partial_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        if "TAB" in preds.keys():

            self.TAB_train_metrics(preds["TAB"], labels.long())
            self.log_dict(self.TAB_train_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds["TAB"], labels.long())
            loss += partial_loss
            self.log(
                f"loss_train/tab",
                partial_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if "MIX" in preds.keys():
            self.MIX_train_metrics(preds["MIX"], labels.long())
            self.log_dict(self.MIX_train_metrics, on_step=True, on_epoch=True)
            partial_loss = self.loss(preds["MIX"], labels.long())
            loss += partial_loss
            self.log(
                f"loss_train/mix",
                partial_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        # if (self.global_step + 1) % 100 == 0:  # every 100 steps
        #   for name, param in self.named_parameters():
        #      self.logger.experiment.add_histogram(name, param, self.global_step)
        # loss = loss / len(self.modalities)
        self.log("loss_train/total", loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("logit_scale", self.classifier.temp.item(),on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.epoch_labels = None
        return super().on_validation_epoch_start()

    def _compute_hierarchical_metrics(self, predictions, labels, modality):
        """Compute hierarchical F1 metrics for transient/stochastic/periodic groups."""
        df = pd.DataFrame({
            "true": labels.clone().long().detach().cpu().numpy(),
            "pred": np.argmax(predictions.clone().detach().cpu().numpy(), axis=-1),
        })

        metrics_dict = {}
        hierarchies = ["transient", "stochastic", "periodic"]

        for hierarchy in hierarchies:
            group = getattr(TAXONOMY, hierarchy)
            group_df = df.query(f"true in {list(group.group.values())}")
            report = classification_report(
                group_df["true"],
                group_df["pred"],
                target_names=list(group.group.keys()),
                labels=list(group.group.values()),
                digits=4,
                output_dict=True,
                zero_division=0,
            )
            metrics_dict[hierarchy] = report["macro avg"]["f1-score"]

        mean_f1 = sum(metrics_dict.values()) / len(metrics_dict)

        # Log metrics
        for hierarchy, f1_score in metrics_dict.items():
            self.log(
                f"validation/{hierarchy}",
                f1_score,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            f"validation/hier_mean_f1",
            mean_f1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch_data, batch_idx):
        labels = batch_data.pop("labels")
        embs = self.model(**batch_data)
        preds = self.classifier(embs)
        loss = 0

        if "LC" in preds.keys():
            partial_loss = self.loss(
                preds["LC"], labels.long()
            )  # + self.triplet_loss(embs, labels.long())
            loss += partial_loss
            self.LC_valid_metrics(preds["LC"], labels.long())
            self.log_dict(
                self.LC_valid_metrics, on_step=False, on_epoch=True, prog_bar=True
            )
            self.validation_cm(preds["LC"], labels.long())
            self.log(
                f"loss_validation/lc",
                partial_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            self._compute_hierarchical_metrics(preds["LC"], labels, "LC")

        if "TAB" in preds.keys():
            self.TAB_valid_metrics(preds["TAB"], labels.long())
            self.log_dict(self.TAB_valid_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds["TAB"], labels.long())
            loss += partial_loss
            self.validation_cm(preds["TAB"], labels.long())
            self.log(
                f"loss_validation/tab",
                partial_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            # self.log('validation/f1_score_harmonic',2 / (self.TAB_valid_metrics['f1_macro'].compute()**-1 + self.f1_hier_macro_val(hier_map_preds, hier_map_labels)**-1), on_step=False, on_epoch=True, sync_dist=True)
        if "MIX" in preds.keys():
            self.MIX_valid_metrics(preds["MIX"], labels.long())
            self.log_dict(
                self.MIX_valid_metrics, on_step=False, on_epoch=True, prog_bar=True
            )
            partial_loss = self.loss(preds["MIX"], labels.long())
            loss += partial_loss
            self.epoch_labels = (
                torch.concat([self.epoch_labels, labels.detach()])
                if self.epoch_labels is not None
                else labels.detach()
            )
            self.validation_cm(preds["MIX"], labels.long())
            self.log(
                f"loss_validation/mix",
                partial_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation_pr_diff",
                (
                    self.MIX_valid_metrics["precision"].compute()
                    - self.MIX_valid_metrics["recall"].compute()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self._compute_hierarchical_metrics(preds["MIX"], labels, "MIX")

        # loss = loss / len(self.modalities)
        self.log(
            f"loss_validation/total", loss, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    def on_validation_epoch_end(self):
        pass

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

        return {"optimizer": optimizer}

    def map_label_tensor(self, labels):
        mapping_dict = {
            0: 1,
            1: 1,
            3: 1,
            5: 1,
            8: 1,
            2: 2,
            6: 2,
            7: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            4: 0,
            9: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
        }
        mapping_tensor = torch.tensor(
            [mapping_dict.get(int(label), -1) for label in labels], device=labels.device
        )
        return mapping_tensor

    def init_metrics(self):
        thr = 0.5
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.classification.Accuracy(
                    task="multiclass",
                    num_classes=self.classifier.num_classes,
                    threshold=thr,
                ),
                "f1_macro": torchmetrics.classification.F1Score(
                    task="multiclass",
                    num_classes=self.classifier.num_classes,
                    average="macro",
                    threshold=thr,
                ),
                "f1_weighted": torchmetrics.classification.F1Score(
                    task="multiclass",
                    num_classes=self.classifier.num_classes,
                    average="weighted",
                    threshold=thr,
                ),
                "recall": torchmetrics.classification.Recall(
                    task="multiclass",
                    num_classes=self.classifier.num_classes,
                    average="macro",
                    threshold=thr,
                ),
                "precision": torchmetrics.classification.Precision(
                    task="multiclass",
                    num_classes=self.classifier.num_classes,
                    average="macro",
                    threshold=thr,
                ),
            }
        )
        self.validation_cm = torchmetrics.classification.ConfusionMatrix(
            task="multiclass",
            num_classes=self.classifier.num_classes,
            normalize="true",
            threshold=thr,
        )
        # self.f1_hier_macro_val =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro", threshold=thr)
        # self.f1_hier_macro_train =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro", threshold=thr)
        # self.f1_8 =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro")
        # self.prcurve = torchmetrics.classification.MulticlassPrecisionRecallCurve(num_classes=self.classifier.num_classes,average = 'macro')
        if "LC" in self.modalities:
            self.LC_train_metrics = metrics.clone(prefix="training/LC/")
            self.LC_valid_metrics = metrics.clone(prefix="validation/LC/")
        if "TAB" in self.modalities:
            self.TAB_train_metrics = metrics.clone(prefix="training/TAB/")
            self.TAB_valid_metrics = metrics.clone(prefix="validation/TAB/")
        if "MIX" in self.modalities:
            self.MIX_train_metrics = metrics.clone(prefix="training/MIX/")
            self.MIX_valid_metrics = metrics.clone(prefix="validation/MIX/")

    def get_confusion_matrix(
        self,
        preds,
        target,
        taxonomy,
        dataset_type: str,
        plot_title: str,
        order_classes: list[str],
    ):

        fs = 11
        y_true = [taxonomy.values_as_keys()[i] for i in np.array(target).astype(int)]
        y_pred = [taxonomy.values_as_keys()[i] for i in np.array(preds).astype(int)]

        cm = confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=order_classes, normalize="true"
        )
        np.set_printoptions(precision=4, suppress=True)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots(figsize=(11, 11))
        decimals = 2
        ax.imshow(
            np.around(cm, decimals=decimals), interpolation="nearest", cmap=cmap
        )
        new_color = cmap(1.0)

        # Añadiendo manualmente las anotaciones
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f"{np.around(cm[i, j], decimals=decimals)}"
                color = (
                    "white" if cm[i, j] > 0.5 else new_color
                )
                ax.text(
                    j, i, text, ha="center", va="center", color=color, fontsize=fs
                )

        # Ajustes finales y mostrar la gráfica
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xticks(np.arange(len(order_classes)))
        ax.set_yticks(np.arange(len(order_classes)))
        ax.set_xticklabels(order_classes)
        ax.set_yticklabels(order_classes)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        f1_ = classification_report(
            y_true,
            y_pred,
            target_names=list(taxonomy().keys()),
            digits=4,
            output_dict=True,
            zero_division=0,
        )["macro avg"]["f1-score"]
        ax.set_title(
            f"{plot_title}: {dataset_type} | macro f1: {np.round(f1_,4)}",
            fontsize=16,
            pad=13,
        )
        ax.set_xlabel("Predicted label", fontsize=16, labelpad=13)  # Label del eje x
        ax.set_ylabel("True label", fontsize=16, labelpad=13)  # Label del eje y

        # ax.xaxis.label.set_size(16)
        # ax.yaxis.label.set_size(16)
        # ax.xaxis.labelpad = 13
        # ax.yaxis.labelpad = 13
        return fig
