import warnings
import logging
import colorlog 

warnings.filterwarnings("ignore")

from src.augmentations import LightCurveTransform as LC

from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule
 
from pytorch_lightning import Trainer

from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.transformer.tabular import TabularTransformer
from src.layers.transformer.Combinator import Combinator
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier

from src.layers.classifiers.TokenClassifier import TokenClassifier

import torch.nn as nn
import hydra
import numpy as np
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from src.losses.FocalLoss import FocalLoss
from torchvision.transforms import RandomChoice, RandomApply


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np

@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name= 'supervised_training')
#@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name= 'LC_MD')
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
    WINDOW_1 = -1

    transforms = [  
                    LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = list(np.linspace(0.9999,1.001,100)) ),
                    LC.TimeShift(),
                    LC.GaussianNoise(num_bands=cfg.lc.num_bands, std = 1e-5),
                                   
                    RandomChoice([ LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.05),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.01),
                                   LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.1),
                                     ]),
                   
                    LC.CutFirstN(2,mask_first= [-1,0,1,2,4,8,16,32,64,128]),
                    LC.ZScoreUndersample(min_samples=150, thr = None, inject_gauss_noise=False),

                    LC.TimeFactor(num_bands=cfg.lc.num_bands, window = -1,factor = list(np.linspace(0.9999,1.001,100)) ),

                    LC.TimeShift(),
                    LC.GaussianNoise(num_bands=cfg.lc.num_bands, std = 1e-5),
       
                    RandomChoice([ LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.05),
                                    LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.01),
                                   LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = 0.1), 
                                     ]),
                   
                    LC.CutFirstN(2,mask_first= [-1,0,1,2,4,8,16,32,64,128]),
                    LC.ZScoreUndersample(min_samples=150, thr = None, inject_gauss_noise=False),

                    #RandomChoice([#LC.RandomSobelFilterMask('magnitude', keep='above',threshold_range = (0.1,0.2)),
                                  #LC.OnlyMaskPadding(),
                                  #LC.MaskFirstN(2,mask_first= [-1,0,1,2]),
                                #])
                    ]
    cfg.datamodule.dataset.transforms = [RandomApply([t],p = 0.1) for t in transforms]
    pl_datal = LitData(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        #classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        classifier = MultimodalClassifier(lc_input_size=cfg.lc.embedding_size,
                                          tab_input_size=None, 
                                          num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        print(cfg.checkpoint) 
        pl_model = ClassifierModule(transformer,classifier, loss, 
                                    lc_load_ckpt=cfg.checkpoint,
                                    freeze_transformer=False,**cfg) 
    if cfg.experiment_type == 'MD':
        transformer = TabularTransformer(**cfg.tab)
        #classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        classifier = MultimodalClassifier(tab_input_size=cfg.lc.embedding_size,
                                          num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
       # print(cfg.checkpoint) 
        pl_model = ClassifierModule(transformer,classifier, loss, 
                                    load_ckpt=cfg.checkpoint,
                                    freeze_transformer=False,weight_str_parse='model.',**cfg) 
    if cfg.experiment_type == 'LC_MD':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        classifier = MultimodalClassifier(cfg.lc.embedding_size, cfg.tab.embedding_size, num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model,classifier, loss, 
                                   # lc_load_ckpt= '/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/LC/FULL_new/',
                                    #tab_load_ckpt='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/MD/3ENC_v0/',
                                    freeze_transformer=False,**cfg) 
    if cfg.experiment_type == 'LC_MD_FEAT':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_tab = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_tab)
        classifier = MultimodalClassifier(cfg.lc.embedding_size, cfg.tab.embedding_size, num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model,classifier, loss, 
                                    load_ckpt=cfg.checkpoint,
                                    freeze_transformer=False,**cfg) 
    #cfg.callbacks.update({'f1log':MacroF1PerClassLogger()})
    trainer = Trainer(
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )

    trainer.fit(pl_model, pl_datal)

    #pref_ = global_config.CHECKPOINT_PREFIX

if __name__ == "__main__":
    main()