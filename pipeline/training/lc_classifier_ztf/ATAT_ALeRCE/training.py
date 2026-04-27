import warnings
import logging
import math

import colorlog
import hydra
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torchvision.transforms import RandomChoice, RandomApply, Compose
from hydra.utils import instantiate

from src.augmentations import LightCurveTransform as LC
from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule
from src.layers.transformer.ATAT import LightCurveTransformer, TabularTransformer, Combinator
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier
from src.utils.CustomParser import ATATConfig

# Only filter specific deprecation warnings, not all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hierarchy mappings: class -> parent hierarchy
HIER_CLASS_MAPPING = {
    0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
    2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
    4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
}

# Hierarchy classes -> valid class indices
HIER_VALID_CLASSES = {
    0: [4, 9, 16, 17, 18, 19, 20, 21],
    1: [0, 1, 3, 5, 8],
    2: [2, 6, 7, 10, 11, 12, 13, 14, 15]
}

HIER_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0

# SSL training constants
LC_WINDOW_MIN = 12
LC_WINDOW_MAX = 204
LC_WINDOW_STEP = 6
LC_WINDOW_SELECT_PROB = 0.95
LC_TIME_FACTOR_RANGE = (0.95, 1.05)
LC_DATA_FACTOR_RANGE = (0.95, 1.05)
NUM_FACTOR_SAMPLES = 100
RANDOM_SUBSAMPLE_WINDOW = 200
GAUSSIAN_FILTER_STDS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1]

MASK_TIME_NUM_SEGMENTS = 2
MASK_TIME_PROB = 0.1
MASK_DATA_NUM_SEGMENTS = 2
MASK_DATA_PROB = 0.1

class HierLoss(nn.Module):
    """Hierarchical loss combining hierarchy and fine-grained class predictions."""

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def modify_logits(self, hier_preds, class_preds):
        """Mask class logits by valid classes for predicted hierarchy.

        Args:
            hier_preds: Logits for hierarchy prediction (batch, 3)
            class_preds: Logits for class prediction (batch, num_classes)

        Returns:
            Modified class probabilities with invalid classes masked to 0
        """
        hier_preds_soft = nn.functional.softmax(hier_preds, dim=-1)
        hier_classes = torch.argmax(hier_preds_soft, dim=-1).int()
        class_probs = nn.functional.softmax(class_preds, dim=-1)

        # Create mask for valid classes per batch element
        batch_size = class_preds.shape[0]
        mask = torch.zeros_like(class_probs)

        for hier_class in HIER_VALID_CLASSES:
            valid_indices = HIER_VALID_CLASSES[hier_class]
            hier_mask = hier_classes == hier_class
            mask[hier_mask, valid_indices] = 1

        return class_probs * mask

    def map_label_tensor(self, labels):
        """Map fine-grained labels to hierarchy labels."""
        mapping_tensor = torch.tensor(
            [HIER_CLASS_MAPPING.get(int(label), -1) for label in labels],
            device=labels.device,
            dtype=torch.long
        )
        return mapping_tensor

    def forward(self, preds, labels):
        """Compute hierarchical loss.

        Args:
            preds: Tuple of (hierarchy_logits, class_logits)
            labels: Fine-grained class labels

        Returns:
            Combined hierarchy and class loss
        """
        hier_loss = self.loss(preds[0], self.map_label_tensor(labels.long()))
        modified_logits = self.modify_logits(preds[0], preds[1])
        class_loss = self.loss(modified_logits, labels)

        combined_loss = (HIER_LOSS_WEIGHT * hier_loss + CLASS_LOSS_WEIGHT * class_loss) / 2
        return combined_loss

def build_transforms(cfg):
    """Build augmentation transform pipeline for SSL training."""
    transforms = []
    augmentation_prob = cfg.online_transforms.p_

    # Window selection transforms
    if cfg.online_transforms.use_window_select:
        windows = [
            LC.WindowSelect(cfg.lc.num_bands, window_size=w, apply_to_classes=None)
            for w in range(LC_WINDOW_MIN, LC_WINDOW_MAX, LC_WINDOW_STEP)
        ]
        windows.extend([
            LC.MAXWindowSelect(cfg.lc.num_bands, window_size=w, apply_to_classes=None)
            for w in range(LC_WINDOW_MIN, LC_WINDOW_MAX, LC_WINDOW_STEP)
        ])
        transforms.append(RandomApply([RandomChoice(windows)], p=LC_WINDOW_SELECT_PROB))

    # Simple time/data factor transforms
    if cfg.online_transforms.use_simple_time_factor:
        time_factors = list(np.linspace(LC_TIME_FACTOR_RANGE[0], LC_TIME_FACTOR_RANGE[1], NUM_FACTOR_SAMPLES))
        transforms.append(RandomApply([LC.TimeFactor(factor=time_factors, apply_to_classes=None)], p=augmentation_prob))

    if cfg.online_transforms.use_simple_data_factor:
        data_factors = list(np.linspace(LC_DATA_FACTOR_RANGE[0], LC_DATA_FACTOR_RANGE[1], NUM_FACTOR_SAMPLES))
        transforms.append(RandomApply([LC.Factor(factor=data_factors, apply_to_classes=None)], p=augmentation_prob))

    # Band permutation
    if cfg.online_transforms.use_band_permute:
        transforms.append(RandomApply([LC.BandPermute(cfg.lc.num_bands, apply_to_classes=None)], p=augmentation_prob))

    # Gaussian filters (always applied)
    transforms.append(RandomApply(
        [LC.GaussianFilter(cfg.lc.num_bands, filter_std=GAUSSIAN_FILTER_STDS, apply_to_classes=None)],
        p=augmentation_prob
    ))
    transforms.append(RandomApply(
        [LC.GaussianTimeFilter(cfg.lc.num_bands, filter_std=GAUSSIAN_FILTER_STDS, apply_to_classes=None)],
        p=augmentation_prob
    ))

    # Gaussian noise on flux
    if getattr(cfg.online_transforms, 'use_gauss_noise', False):
        transforms.append(RandomApply([LC.GaussianNoise(cfg.lc.num_bands)], p=augmentation_prob))

    # Gaussian noise on time
    if getattr(cfg.online_transforms, 'use_time_gauss_noise', False):
        transforms.append(RandomApply(
            [LC.TimeGaussianNoise(cfg.lc.num_bands, apply_to_classes=None)],
            p=augmentation_prob
        ))

    # Random subsample to fixed window
    if getattr(cfg.online_transforms, 'use_random_subsample', False):
        transforms.append(RandomApply(
            [LC.RandomSubsample(cfg.lc.num_bands, window_size=RANDOM_SUBSAMPLE_WINDOW)],
            p=augmentation_prob
        ))

    # Masking transforms (always applied)
   # transforms.extend([
   #     LC.RandomMaskTimeVector(MASK_TIME_NUM_SEGMENTS, MASK_TIME_PROB, None),
   ##     LC.RandomMaskDataVector(MASK_DATA_NUM_SEGMENTS, MASK_DATA_PROB, None)
    #])

    return transforms



def setup_lc_model(cfg):
    """Setup light curve only model."""
    transformer = LightCurveTransformer(**cfg.lc)
    classifier = MultimodalClassifier(
        experiment_type='LC',
        lc_input_size=cfg.lc.embedding_size,
        tab_input_size=cfg.tab.embedding_size,
        use_mix=False,
        use_lc=True,
        use_tab=False,
        combine_logits=False,
        num_classes=cfg.num_classes
    )
    loss = nn.CrossEntropyLoss()

    pl_model = ClassifierModule(
        model=transformer,
        classifier=classifier,
        loss=loss,
        freeze_lc=False,
        freeze_tab=False,
        report_lc=False,
        report_mix=False,
        lc_load_ckpt=cfg.lc.checkpoint,
        tab_load_ckpt=cfg.tab.checkpoint,
        lc_freeze=cfg.lc.freeze_weights,
        tab_freeze=cfg.tab.freeze_weights,
        weight_str_parse_lc=('model.', ''),
        **cfg
    )
    return pl_model


def setup_tab_model(cfg):
    """Setup tabular (metadata) only model."""
    transformer = TabularTransformer(**cfg.tab)
    classifier = MultimodalClassifier(
        tab_input_size=cfg.tab.embedding_size,
        use_tab=True,
        num_classes=cfg.num_classes
    )
    loss = nn.CrossEntropyLoss()

    pl_model = ClassifierModule(
        model=transformer,
        classifier=classifier,
        loss=loss,
        freeze_lc=False,
        freeze_tab=False,
        report_lc=False,
        report_mix=False,
        lc_load_ckpt=cfg.lc.checkpoint,
        tab_load_ckpt=cfg.tab.checkpoint,
        lc_freeze=cfg.lc.freeze_weights,
        tab_freeze=cfg.tab.freeze_weights,
        weight_str_parse_lc=('model.', ''),
        **cfg
    )
    return pl_model


def setup_multimodal_model(cfg, experiment_type):
    """Setup multimodal model combining light curve and tabular data."""
    transformer_lc = LightCurveTransformer(**cfg.lc)
    transformer_tab = TabularTransformer(**cfg.tab)
    model = Combinator(transformer_lc, transformer_tab)

    classifier = MultimodalClassifier(
        experiment_type=experiment_type,
        lc_input_size=cfg.lc.embedding_size,
        tab_input_size=cfg.tab.embedding_size,
        use_mix=True,
        use_lc=False,
        use_tab=False,
        combine_logits=False,
        num_classes=cfg.num_classes
    )
    loss = nn.CrossEntropyLoss()

    pl_model = ClassifierModule(
        model=model,
        classifier=classifier,
        loss=loss,
        freeze_lc=False,
        freeze_tab=False,
        report_lc=True,
        report_mix=False,
        lc_load_ckpt=cfg.lc.checkpoint,
        weight_str_parse_lc=('model.', ''),
        weight_str_parse_tab=('model.', ''),
        **cfg
    )
    return pl_model


def log_lc_config(cfg):
    """Log light curve model configuration."""
    log_message = (
        f"Model Configuration:\n"
        f"{'='*30}\n"
        f"• Use_conv          : {'✓' if cfg.lc.use_conv else '✗'}\n"
        f"• Use_stats         : {'✓' if cfg.lc.use_stats else '✗'}\n"
        f"• Use_acceleration  : {'✓' if cfg.lc.use_acceleration else '✗'}\n"
        f"• Use_velocity      : {'✓' if cfg.lc.use_velocity else '✗'}\n"
        f"• Use_metadata      : {'✓' if cfg.lc.use_metadata else '✗'}\n"
        f"• Use_features      : {'✓' if cfg.lc.use_features else '✗'}\n"
        f"• Use_anomaly_gate  : {'✓' if cfg.lc.use_anomaly_gate else '✗'}\n"
        f"{'='*30}\n"
        f"• Sequence l2 norm  : {'✓' if cfg.lc.use_sequence_norm else '✗'}\n"
        f"• Depth             : {cfg.lc.num_encoders}\n"
        f"• Heads             : {cfg.lc.num_heads}\n"
        f"• Dropout           : {cfg.lc.dropout}\n"
        f"{'='*30}\n"
        f"• Timefilm gelu     : {'✓' if cfg.lc.use_timefilm_gelu else '✗'}\n"
        f"• Timefilm norm     : {'✓' if cfg.lc.use_timefilm_norm else '✗'}\n"
        f"• Output exp()      : {'✓' if cfg.lc.use_exp else '✗'}\n"
        f"{'='*30}"
    )
    print(log_message)


@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name='supervised_training')
def main(cfg: ATATConfig):

    cfg = instantiate(cfg).ATATConfig

    # Setup logging
    logging.root.handlers = []
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cfg.log_filename, encoding="utf-8"),
            handler,
        ],
    )

    cfg.datamodule.dataset.experiment_type = cfg.experiment_type

    # Setup transforms
    transforms = build_transforms(cfg)
    cfg.datamodule.dataset.train_transforms = transforms

    # Setup data module
    pl_datal = LitData(**cfg.datamodule)

    # Setup model based on experiment type
    if cfg.experiment_type == 'LC':
        log_lc_config(cfg)
        pl_model = setup_lc_model(cfg)
    elif cfg.experiment_type in ('MD', 'MD_FEAT'):
        pl_model = setup_tab_model(cfg)
    elif cfg.experiment_type in ('LC_MD', 'LC_MD_FEAT'):
        pl_model = setup_multimodal_model(cfg, cfg.experiment_type)
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment_type}")

    # Setup trainer
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger=list(cfg.loggers.values()),
        **cfg.trainer
    )

    # Train
    trainer.fit(pl_model, pl_datal)

if __name__ == "__main__":
    main()