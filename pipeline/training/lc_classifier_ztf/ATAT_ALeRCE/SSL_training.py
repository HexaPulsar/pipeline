import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.data.modules.LitPretrainMM import LitPretrainMM
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule
from src.augmentations import LightCurveTransform as LC
from src.augmentations import TabularTransformations as TAB
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

@hydra.main(version_base=None, config_path="./src/configs/ELASTICC", config_name= 'ssl_training')
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

    p_ =cfg.online_transforms.p_

    transforms = []

    apply_  = None

    windows =[LC.WindowSelect(cfg.lc.num_bands, window_size=w_, apply_to_classes=None) for w_ in list(range(30, 204, 6))]
    windows.extend([ LC.MAXWindowSelect(cfg.lc.num_bands, window_size=w_, apply_to_classes=None) for w_ in list(range(30, 204, 6))])
    transforms.extend([RandomApply([RandomChoice(windows
                                )],p =1)
                                ])  if cfg.online_transforms.use_window_select else None

    #transforms.extend([ RandomApply([LC.GaussTimeFactor(cfg.lc.num_bands, scale = 1e-4, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_time_gauss_factor else None
    #transforms.extend([ RandomApply([LC.GaussFactor(cfg.lc.num_bands, scale = 1e-4, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_gauss_factor else None

    transforms.extend([ RandomApply([LC.TimeFactor( factor = list(np.linspace(0.99,1.01, 100)), apply_to_classes=apply_)], p = p_),]) if cfg.online_transforms.use_simple_time_factor else None
    transforms.extend([ RandomApply([LC.Factor( factor = list(np.linspace(0.99,1.01, 100)), apply_to_classes=apply_)], p = p_),]) if cfg.online_transforms.use_simple_data_factor else None

    transforms.extend([ RandomApply([LC.BandPermute(cfg.lc.num_bands, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_band_permute else None

    #transforms.extend([ RandomApply([LC.GaussianFilter(cfg.lc.num_bands,filter_std = [1e-5,1e-4,1e-3,1e-2,1e-1,-1], apply_to_classes=apply_)], p = p_)])
    #transforms.extend([ RandomApply([LC.GaussianTimeFilter(cfg.lc.num_bands,filter_std = [1e-5,1e-4,1e-3,1e-2,1e-1,-1], apply_to_classes=apply_)], p = p_)])
#########

#########

   # transforms.extend([LC.RandomMaskTimeVector(2,0.5,None)])
   # transforms.extend([LC.RandomMaskDataVector(2,0.5,None)])

   # transforms.extend([TAB.TABGaussianNoise(0,1e-2)])
    #transforms.extend([TAB.RandomMask(0.01)])
    #transforms.extend([TAB.Factor(factor = list(np.linspace(0.95,1.0, 100)))])

    list_of_transforms = transforms
    cfg.datamodule.dataset.transforms_1 =list_of_transforms
    cfg.datamodule.dataset.transforms_2 = list_of_transforms #[] #list_of_transforms #[] #[] # list_of_transforms
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
                                           cfg.vicreg.cov_coeff,
                                           cfg.datamodule.batch_size),
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
