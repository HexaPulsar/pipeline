import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule
from src.models.PretrainModuleFusion import PretrainModuleFusion
from src.layers.transformer.Combinator import Combinator
from src.augmentations import LightCurveTransform as LC
#from src.augmentations import TabularTransformations as TAB
from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.transformer.tabular import TabularTransformer
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from torchvision.transforms import RandomChoice, RandomApply, Compose

import numpy as np

@hydra.main(version_base=None, config_path="./src/configs/ZTF", config_name= 'ssl_training')
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
    #assert cfg.experiment_type == 'LC' , cfg.experiment_type
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    
    transforms = [  
                #RandomChoice([LC.WindowSelect(2, window_size=w_) for w_ in list(range(10, 110, 10))]),
                #RandomChoice([LC.MaskWindow(2, window_size= w_) for w_ in list(range(10, 110, 10))]),
                RandomChoice([LC.MaskWindow(2, window_size= w_) for w_ in list(range(10, 210, 10))]),
                LC.BandPermute(2),
                #LC.CutBand(cfg.lc.num_bands),
                LC.TimeShift(),
                LC.ZeroOutTime(),
                #LC.TimeFactor(2, 10,[-1, 0.5,0.9,1.01, 1.5]),
                #LC.Factor([-1,1,0.5,1.5]),
                LC.CutFirstN(2,mask_first=[-1,0,1,2,4,8,15,32,64,128]),
                ]
    p_ = 0.5
    list_of_transforms = [RandomApply([t], p = p_) for t in transforms]
    cfg.datamodule.dataset.transforms_1 = list_of_transforms
    cfg.datamodule.dataset.transforms_2 = [] #list_of_transforms
    pl_datal = LitPretrain(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        projector = VICRegProjector(VICReg(cfg.vicreg.inv_coeff,
                                           cfg.vicreg.var_coeff,
                                           cfg.vicreg.cov_coeff),
                                           cfg.vicreg.shape_projector_1,
                                           cfg.vicreg.shape_projector_2)
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate)

    if cfg.experiment_type == 'MD' or cfg.experiment_type == 'FEAT':
        transformer = TabularTransformer(**cfg.tab)
        projector = VICRegProjector(VICReg(cfg.vicreg.inv_coeff,
                                           cfg.vicreg.var_coeff,
                                           cfg.vicreg.cov_coeff),
                                           cfg.vicreg.shape_projector_1,
                                           cfg.vicreg.shape_projector_2)
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate)

    if cfg.experiment_type == 'LC_MD':
        projector =  VICRegProjector(VICReg(cfg.vicreg.inv_coeff,
                                           cfg.vicreg.var_coeff,
                                           cfg.vicreg.cov_coeff),
                                           cfg.vicreg.shape_projector_1,
                                           cfg.vicreg.shape_projector_2)
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        pl_model = PretrainModuleFusion(model,
                                  loss=projector,
                                  lr = cfg.learning_rate)
    if cfg.experiment_type == 'LC_MD_FEAT':
        projector =  VICRegProjector(VICReg(cfg.vicreg.inv_coeff,
                                           cfg.vicreg.var_coeff,
                                           cfg.vicreg.cov_coeff),
                                           cfg.vicreg.shape_projector_1,
                                           cfg.vicreg.shape_projector_2)
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        pl_model = PretrainModuleFusion(model,
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
