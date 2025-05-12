import warnings
import logging
import colorlog

#warnings.filterwarnings("ignore")

from src.layers.transformer.Combinator import Combinator
from src.data.modules.LitPretrain import LitPretrain
from src.models.PretrainModule import PretrainModule
from src.models.PretrainMMModule import PretrainMMModule

from src.augmentations import LightCurveTransform as LC
from src.augmentations import TabularTransformations as TAB
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
    WINDOW_1 = -1
    transforms = [  
                    LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = list(np.linspace(0.9999,1.001,100)) ),
                    LC.TimeShift(),
                    LC.GaussianNoise(num_bands=cfg.lc.num_bands, std = 1e-5),
                                   
                    RandomChoice([ LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.05),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.01),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 1e-5),
                                   LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.1),
                                     ]),
                   
                    LC.CutFirstN(2,mask_first= [-1,0,1,2,4,8,15,32,64]),
                    LC.ZScoreUndersample(min_samples=150, thr = None, inject_gauss_noise=False),

                    LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = list(np.linspace(0.9999,1.001,100)) ),

                    LC.TimeShift(),
                    LC.GaussianNoise(num_bands=cfg.lc.num_bands, std = 1e-5),
       
                    RandomChoice([ LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.05),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 1e-5),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.01),
                                   LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.1),
                                     ]),
                   
                    LC.CutFirstN(2,mask_first= [-1,0,1,2,4,8,16,32,64]),
                    LC.ZScoreUndersample(min_samples=150, thr = None, inject_gauss_noise=False),

                    #RandomChoice([#LC.RandomSobelFilterMask('magnitude', keep='above',threshold_range = (0.1,0.2)),
                                  #LC.OnlyMaskPadding(),
                                  #LC.MaskFirstN(2,mask_first= [-1,0,1,2]),
                                #])
                    ]
    #transforms = [
                  #TAB.Scale(),
                  #TAB.Jitter(),
                  #TAB.Shift()
    #              ]
    p_ = 1
    transforms = [RandomApply([t], p = p_) for t in transforms]
    transforms +=[LC.TimeNormalization(), TAB.DealWithInfs(), TAB.DealWithNaNs()]
    cfg.datamodule.dataset.transforms_1 = transforms
    cfg.datamodule.dataset.transforms_2 = [LC.TimeNormalization(), TAB.DealWithInfs(), TAB.DealWithNaNs()] #transforms
    pl_datal = LitPretrain(**cfg.datamodule)

    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        projector = VICRegProjector(VICReg(25,25,1),'128-512-512')
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate)
    if cfg.experiment_type == 'MD':
        transformer = TabularTransformer(**cfg.tab)
        projector = VICRegProjector(VICReg(1,49,1),'128-128-128')
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate)

    if cfg.experiment_type == 'LC_MD':
        projector = VICRegProjector(VICReg(25,25,1),'256-2048-2048')
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        pl_model = PretrainModule(model=model,loss=projector,lr = cfg.learning_rate)

    #elif args.general['experiment_type'] == 'md':
    #    pl_model = LitPreTrainVICREGTAB(**args.all_args)
    #elif args.general['experiment_type'] == 'lc_md':
    ##    pl_datal = LitPretrain(
    #   output_augmented_batch = True,transform_1=transforms_1, transform_2=transforms_2, **args.general
    #   )
    #   pl_model = CLIPLCMD(**args.all_args)
    
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )
    # Trainer model pl routine # trsainer fit models
    trainer.fit(pl_model, pl_datal)

if __name__ == "__main__":
    
    main()
