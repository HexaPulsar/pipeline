import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.data.modules.LitPretrainMM import LitPretrainMM
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule
from src.models.PretrainModuleFusion import PretrainModuleFusion
from src.augmentations import LightCurveTransform as LC
#from src.augmentations import TabularTransformations as TAB
from src.layers.transformer.ATAT import LightCurveTransformer, TabularTransformer, Combinator
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from pytorch_lightning import Trainer

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from torchvision.transforms import RandomChoice, RandomApply, Compose, RandomOrder

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
   
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    p_ =0.5
    apply_to = None
    transforms = [  
            RandomApply([LC.GaussianNoise(2)], p = p_),

            RandomApply([LC.GaussFactor(2, scale = 1e-4, apply_to_classes=None)], p = p_),
            RandomApply([LC.GaussTimeFactor(2, scale = 1e-4, apply_to_classes=None)], p = p_),

            RandomApply([LC.Factor( factor = list(np.linspace(0.95,1.05, 100)), apply_to_classes=None)], p = p_),
            RandomApply([LC.TimeFactor( factor = list(np.linspace(0.95,1.05, 100)), apply_to_classes=None)], p = p_),
            
            RandomApply([LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = [-1,1e-3,1e-2,0.1,0.2], apply_to_classes=None)], p = p_),
            RandomApply([LC.TimeGaussianFilter(num_bands=cfg.lc.num_bands,filter_std = [-1,1e-3,1e-2,0.1,0.2], apply_to_classes=None)], p = p_),

            
            RandomApply([RandomChoice([LC.WindowSelect(2, window_size=w_, apply_to_classes=None) for w_ in list(range(25, 225, 25))])],p =1), 

            RandomApply([LC.Roll(2,max_roll = 200,  apply_to_classes=None)], p = 1),

            RandomApply([LC.BandPermute(2, apply_to_classes=None)], p = p_),
            #RandomApply([RandomChoice([LC.SobelFilterMask(keep = 'above',threshold=thr) for thr in [0.01,0.05, 0.1, 0.15,0.2]])],p = p_),
            RandomApply([RandomChoice([LC.SobelFilterMask(keep = 'below',threshold=thr) for thr in [0.01,0.05, 0.1, 0.15,0.2,0.5]])],p = 1),

            RandomApply([ LC.CutBand(cfg.lc.num_bands)], p = 1e-5),  
            #RandomApply([LC.TimeGaussianNoise(2)], p = p_),

            ]*1
    list_of_transforms = transforms
    cfg.datamodule.dataset.transforms_1 =list_of_transforms
    cfg.datamodule.dataset.transforms_2 = list_of_transforms
    pl_datal = LitPretrain(**cfg.datamodule)
    
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        projector = VICRegProjector(VICReg(cfg.vicreg.inv_coeff,
                                           cfg.vicreg.var_coeff,
                                           cfg.vicreg.cov_coeff, 
                                           cfg.datamodule.batch_size),
                                           cfg.vicreg.shape_projector_1,
                                           cfg.vicreg.shape_projector_2)
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate, eval_regressor=False)

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


        pl_datal = LitPretrainMM(**cfg.datamodule)
        pl_model = PretrainMMModule(model_lc=transformer_lc,
                                    model_tab=transformer_md,
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
        model  = Combinator(transformer_lc,transformer_md, how = 'sum')
        pl_model = PretrainModule(model,
                                  loss=projector,
                                  lr = cfg.learning_rate)
        
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        ) 
    trainer.fit(pl_model, pl_datal)

if __name__ == "__main__":
    
    main()
