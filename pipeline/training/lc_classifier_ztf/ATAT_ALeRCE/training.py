import warnings
import logging
import colorlog
import pickle
import yaml
import glob
import os

warnings.filterwarnings("ignore")


from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import AdvancedProfiler
from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.classifiers.TokenClassifier import TokenClassifier
from src.utils.CustomParser import CustomParser
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig, ATATDatasetArgs
from  hydra.utils import instantiate



@hydra.main(version_base=None, config_path="./src/configs/", config_name= 'supervised_training')
def main(cfg:ATATConfig):
    #print(cfg)
    
    cfg = instantiate(cfg).ATATConfig
    
    logger = logging.getLogger()
    logging.root.handlers = []
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s"
        )
    )
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cfg.log_filename, encoding="utf-8"),
            handler,
        ],
    )
    assert cfg.experiment_type == 'LC' , cfg.experiment_type
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    pl_datal = LitData(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        print(transformer)
        classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(transformer,classifier, loss,**cfg) 
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )

    trainer.fit(pl_model, pl_datal)

    #pref_ = global_config.CHECKPOINT_PREFIX

if __name__ == "__main__":
    main()