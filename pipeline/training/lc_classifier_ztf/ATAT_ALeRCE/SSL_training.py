import warnings
import logging

import colorlog
import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer
from torchvision.transforms import RandomChoice, RandomApply, Compose, RandomOrder
from hydra.utils import instantiate

from src.data.modules.LitPretrain import LitPretrain
from src.data.modules.LitPretrainMM import LitPretrainMM
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule
from src.augmentations import LightCurveTransform as LC
from src.layers.transformer.ATAT import LightCurveTransformer, TabularTransformer, Combinator
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from src.utils.CustomParser import ATATConfig

# Only filter specific deprecation warnings, not all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# SSL training constants
LC_WINDOW_MIN = 12
LC_WINDOW_MAX = 204
LC_WINDOW_STEP = 6
LC_WINDOW_SELECT_PROB = 0.95
LC_TIME_FACTOR_RANGE = (0.95, 1.05)
LC_DATA_FACTOR_RANGE = (0.95, 1.05)
NUM_FACTOR_SAMPLES = 100
GAUSSIAN_FILTER_STDS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1]

MASK_TIME_NUM_SEGMENTS = 2
MASK_TIME_PROB = 0.1
MASK_DATA_NUM_SEGMENTS = 2
MASK_DATA_PROB = 0.1

def setup_logging(cfg):
    """Configure logging with file and console handlers."""
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
    return logging.getLogger(__name__)


def build_ssl_transforms(cfg):
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

    # Masking transforms (always applied)
    transforms.extend([
        LC.RandomMaskTimeVector(MASK_TIME_NUM_SEGMENTS, MASK_TIME_PROB, None),
        LC.RandomMaskDataVector(MASK_DATA_NUM_SEGMENTS, MASK_DATA_PROB, None)
    ])

    return transforms


def setup_ssl_lc_model(cfg):
    """Setup SSL model for light curve only."""
    transformer = LightCurveTransformer(**cfg.lc)
    projector = VICRegProjector(
        VICReg(
            cfg.vicreg.inv_coeff,
            cfg.vicreg.var_coeff,
            cfg.vicreg.cov_coeff,
            cfg.datamodule.batch_size
        ),
        cfg.vicreg.shape_projector_1,
        cfg.vicreg.shape_projector_2
    )
    pl_model = PretrainModule(
        model=transformer,
        loss=projector,
        lr=cfg.learning_rate,
        eval_regressor=False,
        **cfg
    )
    return pl_model


def setup_ssl_tab_model(cfg):
    """Setup SSL model for tabular (metadata) only."""
    transformer = TabularTransformer(**cfg.tab)
    projector = VICRegProjector(
        VICReg(
            cfg.vicreg.inv_coeff,
            cfg.vicreg.var_coeff,
            cfg.vicreg.cov_coeff,
            cfg.datamodule.batch_size
        ),
        cfg.vicreg.shape_projector_1,
        cfg.vicreg.shape_projector_2
    )
    pl_model = PretrainModule(
        model=transformer,
        loss=projector,
        lr=cfg.learning_rate
    )
    return pl_model


def setup_ssl_multimodal_model(cfg, experiment_type):
    """Setup SSL model for multimodal (LC + tabular) training.

    Args:
        cfg: Configuration object
        experiment_type: Either 'LC_MD' or 'LC_MD_FEAT'

    Returns:
        Tuple of (pl_model, data_module) for multimodal training
    """
    projector = VICRegProjector(
        VICReg(
            cfg.vicreg.inv_coeff,
            cfg.vicreg.var_coeff,
            cfg.vicreg.cov_coeff,
            cfg.datamodule.batch_size
        ),
        cfg.vicreg.shape_projector_1,
        cfg.vicreg.shape_projector_2
    )
    transformer_lc = LightCurveTransformer(**cfg.lc)
    transformer_tab = TabularTransformer(**cfg.tab)

    if experiment_type == 'LC_MD':
        model = Combinator(transformer_lc, transformer_tab)
        pl_model = PretrainMMModule(
            model_lc=transformer_lc,
            model_tab=transformer_tab,
            loss=projector,
            lr=cfg.learning_rate
        )
        pl_datal = LitPretrainMM(**cfg.datamodule)
    else:  # LC_MD_FEAT
        model = Combinator(transformer_lc, transformer_tab, how='sum')
        pl_model = PretrainModule(
            model=model,
            loss=projector,
            lr=cfg.learning_rate
        )
        pl_datal = LitPretrain(**cfg.datamodule)

    return pl_model, pl_datal


@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name='ssl_training')
def main(cfg: ATATConfig):
    cfg = instantiate(cfg).ATATConfig

    # Setup logging
    logger = setup_logging(cfg)
    logger.info(f"Starting SSL training with experiment type: {cfg.experiment_type}")

    cfg.datamodule.dataset.experiment_type = cfg.experiment_type

    # Setup transforms
    transforms = build_ssl_transforms(cfg)
    cfg.datamodule.dataset.transforms_1 = transforms
    cfg.datamodule.dataset.transforms_2 = transforms

    # Setup model and data module based on experiment type
    if cfg.experiment_type == 'LC':
        logger.info("Setting up LC-only SSL model")
        pl_model = setup_ssl_lc_model(cfg)
        pl_datal = LitPretrain(**cfg.datamodule)
    elif cfg.experiment_type in ('MD', 'FEAT'):
        logger.info(f"Setting up {cfg.experiment_type}-only SSL model")
        pl_model = setup_ssl_tab_model(cfg)
        pl_datal = LitPretrain(**cfg.datamodule)
    elif cfg.experiment_type in ('LC_MD', 'LC_MD_FEAT'):
        logger.info(f"Setting up multimodal SSL model: {cfg.experiment_type}")
        pl_model, pl_datal = setup_ssl_multimodal_model(cfg, cfg.experiment_type)
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment_type}")

    # Setup trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger=list(cfg.loggers.values()),
        **cfg.trainer
    )

    # Compile and train
    logger.info("Compiling model with torch.compile")
    pl_model = torch.compile(pl_model)

    logger.info("Starting training")
    trainer.fit(pl_model, pl_datal)

if __name__ == "__main__":

    main()
