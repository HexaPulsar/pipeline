import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.data.modules.LitPretrain import LitPretrain
from src.models.PretrainModule import PretrainModule
from src.data.handlers.SSLDataset import SSLDataset 
from src.augmentations import LightCurveTransform as LC
from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.utils.projector import VICRegProjector
from src.layers.ProjectorBaseModel import ProjectorBaseModel
from src.losses.VICReg import VICReg

from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning import Trainer
from torchvision.transforms import Compose, RandomApply, RandomChoice


import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate



import numpy as np

@hydra.main(version_base=None, config_path="./src/configs/", config_name= 'ssl_training')
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
    print(cfg)
    WINDOW_1 = -1
    WINDOW_2 = 6
    
    transforms = [  LC.MaskFirstN(mask_first= [0,1,2]),
                    LC.MaskWindow(num_bands = 2,window =5),
                    LC.MaskWindow(num_bands = 2,window =10),
                    LC.GaussianNoise(num_bands=2,window = WINDOW_1),
                    LC.TimeFactor(2,-1,list(np.linspace(0.5,1.5,100))),
                    LC.TimePoissonNoise(num_bands=2, rate =1.5 ,window = WINDOW_1),
                    LC.Exptime(num_bands=2,window = WINDOW_1), 
                    LC.ShiftData(2,WINDOW_1),
                    ]
    cfg.datamodule.dataset.transforms_1 = transforms
    cfg.datamodule.dataset.transforms_2 = []
    pl_datal = LitPretrain(**cfg.datamodule)
    
    if cfg.experiment_type == 'LC':

        transformer = LightCurveTransformer(**cfg.lc)
        
        projector = VICRegProjector(VICReg(25,25,1),'128-256-256')
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
