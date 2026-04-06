import os
from typing import Dict, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics
from collections import OrderedDict
import numpy as np
import pytorch_lightning as pl

# from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR
import torchmetrics.classification
from tqdm import tqdm
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY

import matplotlib.pyplot as plt
import io
import seaborn as sns

# from lion_pytorch import Lion


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import glob
from copy import deepcopy
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
        report_lc=False,
        report_tab=False,
        report_mix=False,
        weight_str_parse_lc=None,
        weight_str_parse_tab=None,
        **kwargs,
    ):
        super().__init__()
        self.gradients_ = None

        self.model = model
        self.classifier = classifier
        self.init_model()
        self.warmup = 0
        self.loss = loss
        self.learning_rate = kwargs["learning_rate"]

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

        self.init_metrics(report_lc, report_tab, report_mix)

        if lc_load_ckpt is not None:
            try:
                _ckpt = glob.glob(lc_load_ckpt + "*.ckpt")[-1]
                print(_ckpt)
            except Exception as e:
                print(e, "path", lc_load_ckpt)
            checkpoint_ = torch.load(_ckpt)
            weights = OrderedDict()

            for key in checkpoint_["state_dict"].keys():
                #
                #  print(key)
                # nput()
                if "loss" in key:
                    continue

                elif "model" in key:
                    # elif 'model' in key:
                    weights[
                        key.replace(
                            f"{weight_str_parse_lc[0]}", f"{weight_str_parse_lc[1]}"
                        )
                    ] = checkpoint_["state_dict"][key]
                    # weights[key.replace('transformer_lc.', "")] = checkpoint_["state_dict"][key]

            # self.model.load_state_dict(weights, strict=True)
            # print(weights.keys())
            #  print(self.model)
            self.model.load_state_dict(weights, strict=True)
            # print(f"loaded LC checkpoint {_ckpt}".format(_ckpt))
        if freeze_lc:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        if tab_load_ckpt is not None:

            _ckpt = glob.glob(lc_load_ckpt + "*.ckpt")[-1]
            checkpoint_ = torch.load(_ckpt)
            weights = OrderedDict()
            for key in checkpoint_["state_dict"].keys():
                if "loss" in key:
                    continue
                elif "tab" in key:

                    weights[
                        key.replace(
                            f"{weight_str_parse_tab[0]}", f"{weight_str_parse_tab[1]}"
                        )
                    ] = checkpoint_["state_dict"][key]
            self.model.transformer_tab.load_state_dict(weights, strict=True)
            print(f"loaded TAB checkpoint {_ckpt}".format(_ckpt))

        if freeze_tab:
            for name, param in self.model.transformer_tab.named_parameters():
                param.requires_grad = False
                # print(f"loaded TAB checkpoint {_ckpt}".format(_ckpt))

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)
                # if 'alpha_' in name:
                #        nn.init.uniform_(p,0,1)

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

    def infer_at_time(self, batch_data, time):
        labels = batch_data.pop("labels")
        batch_data["time"] = batch_data * (batch_data["time"].max() < time)
        batch_data["data"] = batch_data * (batch_data < time)
        embs = self.model(**batch_data)
        preds = self.classifier(embs)

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

            df = pd.DataFrame(
                {
                    "true": labels.clone().long().detach().cpu().numpy(),
                    "pred": np.argmax(
                        preds["LC"].clone().detach().cpu().numpy(), axis=-1
                    ),
                }
            )

            transient_dict = TAXONOMY.transient.group
            transients = df.query("true in {}".format(list(transient_dict.values())))

            ##
            stochastic_dict = TAXONOMY.stochastic.group
            stochastics = df.query("true in {}".format(list(stochastic_dict.values())))

            ####
            periodic_dict = TAXONOMY.periodic.group
            periodics = df.query("true in {}".format(list(periodic_dict.values())))

            mean_f1 = (
                sum(
                    [
                        classification_report(
                            transients["true"],
                            transients["pred"],
                            target_names=list(TAXONOMY.transient.group.keys()),
                            labels=list(TAXONOMY.transient.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                        classification_report(
                            stochastics["true"],
                            stochastics["pred"],
                            target_names=list(TAXONOMY.stochastic.group.keys()),
                            labels=list(TAXONOMY.stochastic.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                        classification_report(
                            periodics["true"],
                            periodics["pred"],
                            target_names=list(TAXONOMY.periodic.group.keys()),
                            labels=list(TAXONOMY.periodic.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                    ]
                )
                / 3
            )

            self.log(
                f"validation/hier_mean_f1",
                mean_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                f"validation/transient",
                classification_report(
                    transients["true"],
                    transients["pred"],
                    target_names=list(TAXONOMY.transient.group.keys()),
                    labels=list(TAXONOMY.transient.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation/stochastic",
                classification_report(
                    stochastics["true"],
                    stochastics["pred"],
                    target_names=list(TAXONOMY.stochastic.group.keys()),
                    labels=list(TAXONOMY.stochastic.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation/periodic",
                classification_report(
                    periodics["true"],
                    periodics["pred"],
                    target_names=list(TAXONOMY.periodic.group.keys()),
                    labels=list(TAXONOMY.periodic.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

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
            df = pd.DataFrame(
                {
                    "true": labels.clone().long().detach().cpu().numpy(),
                    "pred": np.argmax(
                        preds["MIX"].clone().detach().cpu().numpy(), axis=-1
                    ),
                }
            )

            transient_dict = TAXONOMY.transient.group
            transients = df.query("true in {}".format(list(transient_dict.values())))

            ##
            stochastic_dict = TAXONOMY.stochastic.group
            stochastics = df.query("true in {}".format(list(stochastic_dict.values())))

            ####
            periodic_dict = TAXONOMY.periodic.group
            periodics = df.query("true in {}".format(list(periodic_dict.values())))

            mean_f1 = (
                sum(
                    [
                        classification_report(
                            transients["true"],
                            transients["pred"],
                            target_names=list(TAXONOMY.transient.group.keys()),
                            labels=list(TAXONOMY.transient.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                        classification_report(
                            stochastics["true"],
                            stochastics["pred"],
                            target_names=list(TAXONOMY.stochastic.group.keys()),
                            labels=list(TAXONOMY.stochastic.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                        classification_report(
                            periodics["true"],
                            periodics["pred"],
                            target_names=list(TAXONOMY.periodic.group.keys()),
                            labels=list(TAXONOMY.periodic.group.values()),
                            digits=4,
                            output_dict=True,
                        )["macro avg"]["f1-score"],
                    ]
                )
                / 3
            )

            self.log(
                f"validation/hier_mean_f1",
                mean_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                f"validation/transient",
                classification_report(
                    transients["true"],
                    transients["pred"],
                    target_names=list(TAXONOMY.transient.group.keys()),
                    labels=list(TAXONOMY.transient.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation/stochastic",
                classification_report(
                    stochastics["true"],
                    stochastics["pred"],
                    target_names=list(TAXONOMY.stochastic.group.keys()),
                    labels=list(TAXONOMY.stochastic.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation/periodic",
                classification_report(
                    periodics["true"],
                    periodics["pred"],
                    target_names=list(TAXONOMY.periodic.group.keys()),
                    labels=list(TAXONOMY.periodic.group.values()),
                    digits=4,
                    output_dict=True,
                )["macro avg"]["f1-score"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # loss = loss / len(self.modalities)
        self.log(
            f"loss_validation/total", loss, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    def on_validation_epoch_end(self):
        return
        tax = ZTF_TAXONOMY()
        cm = self.validation_cm.compute().cpu().numpy().astype(float)
        fig = plt.figure(figsize=(12, 10))

        sns.heatmap(
            np.round(cm, decimals=2),
            annot=True,
            cmap=plt.cm.Blues,
            ax=fig.add_subplot(111),
        )
        plt.xticks(
            ticks=range(0, self.classifier.num_classes), rotation=45, labels=tax.keys()
        )
        plt.yticks(
            ticks=range(0, self.classifier.num_classes), rotation=45, labels=tax.keys()
        )

        if len(self.modalities) == 3:
            plt.title(
                f"F1-Score: {self.MIX_valid_metrics['f1_macro'].compute().item()}"
            )
        else:
            if "LC" in self.modalities:
                plt.title(
                    f"F1-Score: {self.LC_valid_metrics['f1_macro'].compute().item()}"
                )
            elif "TAB" in self.modalities:
                plt.title(
                    f"F1-Score: {self.TAB_valid_metrics['f1_macro'].compute().item()}"
                )

        plt.tight_layout()

        # Convert the Matplotlib figure to a tensor
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, pad_inches=0.05)  # png
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = T.ToTensor()(
            image
        )  # Convert PIL image to torch tensor (C, H, W)
        self.logger.experiment.add_image(
            "validation cm", image_tensor, self.global_step
        )
        plt.close(fig)  # Close the figure to free memory
        self.validation_cm.reset()
        return super().on_validation_epoch_end()

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

    def init_metrics(self, report_lc, report_tab, report_mix):
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
            self.LC_train_metrics = metrics.clone(prefix=f'{"training/LC/"}')
            self.LC_valid_metrics = metrics.clone(prefix=f'{"validation/LC/"}')
        if "TAB" in self.modalities:
            self.TAB_train_metrics = metrics.clone(prefix=f'{ "training/TAB/"}')
            self.TAB_valid_metrics = metrics.clone(prefix=f'{"validation/TAB/"}')
        if "MIX" in self.modalities:
            self.MIX_train_metrics = metrics.clone(prefix=f'{"training/MIX/"}')
            self.MIX_valid_metrics = metrics.clone(prefix=f'{"validation/MIX/"}')

    def report(self, report_lc, report_tab, report_mix):
        pass

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
        fig, ax = plt.subplots(figsize=(11, 11))  # , dpi=110)
        decimals = 2
        im = ax.imshow(
            np.around(cm, decimals=decimals), interpolation="nearest", cmap=cmap
        )
        # color map
        new_color = cmap(1.0)

        # Añadiendo manualmente las anotaciones con la media y desviación estándar
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    # print(cm[i, j])
                    text = f"{np.around(cm[i, j], decimals=decimals)}"
                    color = (
                        "white" if cm[i, j] > 0.5 else new_color
                    )  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(
                        j, i, text, ha="center", va="center", color=color, fontsize=fs
                    )
                else:
                    text = f"{np.around(cm[i, j], decimals=decimals)}"
                    color = (
                        "white" if cm[i, j] > 0.5 else new_color
                    )  # Blanco para la diagonal, tono de azul para otras celdas
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
