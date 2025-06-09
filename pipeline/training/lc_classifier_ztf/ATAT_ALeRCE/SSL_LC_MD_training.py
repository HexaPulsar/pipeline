import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule

from src.augmentations import LightCurveTransform as LC
from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.transformer.tabular import TabularTransformer
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate

from torchvision.transforms import RandomChoice, RandomApply,RandomAdjustSharpness,RandomSolarize

import numpy as np

@hydra.main(version_base=None, config_path="./src/configs/ZTF", config_name= 'SSL_LC_MD')
def main(cfg:ATATConfig):
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
    WINDOW_1 = -1
    transforms = [  #LC.Factor(list(np.linspace(0.90,1.1,100))),
                   #LC.Factor([1,-1]),
                    #LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = list(np.linspace(0.90,1.1,100)) ),
                    #LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = [1,-1] ),
                    #LC.TimeShift(),
                    #LC.GaussianNoise(num_bands=cfg.lc.num_bands,window = WINDOW_1, std = 10),

                    RandomApply([LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.1)],p = 0.5),
                    #RandomApply([LC.MaskFirstN(2,mask_first= [-1,0,1,2])],p = 0.5),
                    #RandomApply([LC.Exptime(2,window = -1)],p = 0.5),
                    #RandomApply([LC.ZScoreUndersample(min_samples=50, thr = None, inject_gauss_noise=False)],p = 0.5),
                    #RandomApply([LC.CutFirstN(2,mask_first= [-1,0,1,2])],p = 0.5),
                    ]
    transforms +=[LC.TimeNormalization()]
    cfg.datamodule.dataset.transforms_1 = [LC.TimeNormalization()]
    cfg.datamodule.dataset.transforms_2 = []
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    pl_datal = LitPretrain(**cfg.datamodule)
    if cfg.experiment_type == 'LC_MD':
        projector = VICRegProjector(VICReg(25,25,1),'128-512-512')
        pl_model = PretrainMMModule(model_lc= LightCurveTransformer(**cfg.lc),
                                  model_tab = TabularTransformer(**cfg.tab),
                                  loss=projector,
                                  lr = cfg.learning_rate)

    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )
    # Trainer model pl routine # trsainer fit models
    trainer.fit(pl_model, pl_datal)

if __name__ == "__main__":
    
    main()
