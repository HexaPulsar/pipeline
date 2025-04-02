import warnings
import logging
import colorlog 

warnings.filterwarnings("ignore")


from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule
 
from pytorch_lightning import Trainer

from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.classifiers.TokenClassifier import TokenClassifier

import torch.nn as nn
import hydra

from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from src.losses.FocalLoss import FocalLoss


@hydra.main(version_base=None, config_path="./src/configs/ELASTICC/", config_name= 'supervised_training')
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

        classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        #loss = FocalLoss(1,task_type='multi-class', num_classes=cfg.num_classes) 
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(transformer,classifier, loss, 
                                    cfg.checkpoint,
                                    freeze_transformer=False,**cfg) 
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )

    trainer.fit(pl_model, pl_datal)

    #pref_ = global_config.CHECKPOINT_PREFIX

if __name__ == "__main__":
    main()