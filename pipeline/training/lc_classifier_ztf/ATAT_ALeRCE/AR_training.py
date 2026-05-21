import warnings
import logging

try:
    import colorlog
    has_colorlog = True
except ImportError:
    has_colorlog = False

import hydra
import torch
from pytorch_lightning import Trainer
from hydra.utils import instantiate

from src.data.modules.LitPretrainAR import LitPretrainAR
from src.models.PretrainARModule import PretrainARModule
from src.layers.transformer.ATAT import LightCurveTransformer
from src.utils.CustomParser import ATATConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_logging(cfg):
    """Configure logging with file and console handlers."""
    logging.root.handlers = []
    if has_colorlog:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
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


@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name='supervised_training')
def main(cfg: ATATConfig):
    cfg = instantiate(cfg).ATATConfig

    # Setup logging
    logger = setup_logging(cfg)
    logger.info(f"Starting AR pretraining with experiment type: {cfg.experiment_type}")

    cfg.datamodule.dataset.experiment_type = cfg.experiment_type

    # Setup model
    logger.info("Setting up autoregressive LightCurveTransformer")
    transformer = LightCurveTransformer(**cfg.lc)

    pl_model = PretrainARModule(
        model=transformer,
        embedding_size=cfg.lc.embedding_size,
        num_bands=cfg.lc.num_bands,
        lr=cfg.learning_rate,
        warmup_steps=cfg.pretrain.warmup_steps,
        total_steps=cfg.pretrain.total_steps,
        eta_min_factor=cfg.pretrain.eta_min_factor,
        weight_decay=1e-3,
        eval_probe=cfg.datamodule.eval_probe,
        context_size=getattr(cfg, 'context_size', 1),
        normalize_flux=getattr(cfg, 'normalize_flux', False),
    )

    # Setup data module
    logger.info("Initializing data module")
    pl_datal = LitPretrainAR(**cfg.datamodule)

    # Setup trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger=list(cfg.loggers.values()),
        **cfg.trainer
    )

    # Train (no torch.compile for AR due to dynamic causal mask)
    logger.info("Starting training")
    trainer.fit(pl_model, pl_datal)


if __name__ == "__main__":
    main()
