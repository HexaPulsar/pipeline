import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.models.PretrainModule import PretrainModule

from src.augmentations import LightCurveTransform as LC
from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate

from torchvision.transforms import RandomChoice

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
    assert cfg.experiment_type == 'LC' , cfg.experiment_type
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    WINDOW_1 = -1
    transforms = [  RandomChoice([LC.MaskFirstN(mask_first= [0,1,2]),
                                LC.SobelFilterMask('above', threshold = 0.1),
                                LC.SobelFilterMask('above', threshold = 0.3),
                               # LC.SobelFilterMask('above', threshold = 0.5),
                                LC.SobelFilterMask('below', threshold = 0.1),
                               # LC.SobelFilterMask('below', threshold = 0.01),
                               # LC.SobelFilterMask('below', threshold = 0.001),


                                ]),
                                
                    #RandomChoice([LC.GaussianNoise(num_bands=cfg.lc.num_bands,window = WINDOW_1),
                    #            LC.ShiftData(num_bands=cfg.lc.num_bands,window=WINDOW_1),]),
                    #RandomChoice([
                    #                LC.TimeFactor(cfg.lc.num_bands, window=WINDOW_1,factor = list(np.linspace(0.95,1.05,100))),
                    #                LC.TimePoissonNoise(num_bands=cfg.lc.num_bands, rate =1.5 ,window = WINDOW_1),
                    #                LC.Exptime(num_bands=cfg.lc.num_bands,window = WINDOW_1), ]),
                    ]
    cfg.datamodule.dataset.transforms_1 = transforms
    cfg.datamodule.dataset.transforms_2 = []
    pl_datal = LitPretrain(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        projector = VICRegProjector(VICReg(5,55,50),'128-256-256')
        pl_model = PretrainModule(model=transformer,loss=projector,lr = cfg.learning_rate)

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
