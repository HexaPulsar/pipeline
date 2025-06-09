import warnings
import logging
import colorlog 

warnings.filterwarnings("ignore")

from src.augmentations import LightCurveTransform as LC
from src.augmentations import TabularTransformations as TAB

from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule
from src.models.ClassifierModuleHier import ClassifierModuleHier
 
from pytorch_lightning import Trainer

from src.layers.transformer.lightcurve import LightCurveTransformer
from src.layers.transformer.tabular import TabularTransformer
from src.layers.transformer.Combinator import Combinator
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier

from src.layers.classifiers.TokenClassifier import TokenClassifier
from src.losses.FocalLoss import FocalLoss
import torch.nn as nn
import hydra
import numpy as np
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from src.losses.FocalLoss import FocalLoss
from torchvision.transforms import RandomChoice, RandomApply, Compose
from src.models.ClassifierModule import HierLoss

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np

class HierLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hier_loss = nn.CrossEntropyLoss()
        self.class_loss = nn.CrossEntropyLoss()
    def modify_logits(self,hier_preds, class_preds):
        mapping_dict  = {0:[4,9,16,17,18,19,20,21],
                         1:[0,1,3,5,8],
                         2:[2,6,7,10,11,12,13,14,15]}
        hier_classes = torch.argmax(hier_preds, dim = -1).int()
        for i in range((class_preds).shape[0]):   

            correct_logits =  mapping_dict[int(hier_classes[i].item())]
            classes = set(range(hier_preds.shape[-1]))
            class_preds[i,list(classes- set(correct_logits))] = 0
           # print(class_preds[i,:])
           # input()
        return class_preds
    def map_label_tensor(self,labels):
        mapping_dict = {
            0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
            2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
            4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
        }
        mapping_tensor = torch.tensor([mapping_dict.get(int(label), -1) for label in labels], device = labels.device)
        return mapping_tensor
    
    def forward(self, preds ,labels):
        
        hier_loss = self.hier_loss(preds[0],  self.map_label_tensor(labels.long()))
        modified_logits = self.modify_logits(preds[0],preds[1]) 
        class_loss = self.class_loss(modified_logits,  labels.long())
        hier_weight = 0.9
        class_weight = 0.1
        loss =(hier_weight* hier_loss + class_weight *class_loss)
        return loss
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
   # assert cfg.experiment_type == 'LC' , cfg.experiment_type
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type
    
    transforms = [  
                #RandomChoice([LC.WindowSelect(2, window_size=w_) for w_ in list(range(10, 110, 10))]),
                #RandomChoice([LC.WindowSelect(2, window_size=6)]),
                RandomChoice([LC.MaskWindow(2, window_size= w_) for w_ in list(range(10, 210, 10))]),
                RandomChoice([LC.BandPermute(2) ]),
                RandomChoice([LC.GaussianNoise(2,window=-1,std=1)]),
                #RandomChoice([LC.Factor([0.5,1,1.5])]),
                LC.GaussianFilter(num_bands=cfg.lc.num_bands,filter_std = [1e-5,1e-4,1e-3,1e-2, 0.1]),
                #LC.TimeFactor(2, 10, [0.999,1.001]), 
                LC.TimeGaussianNoise(2),
                LC.Roll(2),
                #LC.CutBand(cfg.lc.num_bands),  
               # LC.CutFirstN(2,mask_first=[1,2,4,8,15,32,64,128]),
                ]
    p_ = 1
    list_of_transforms = [RandomApply([t], p = p_) for t in transforms]
    cfg.datamodule.dataset.transforms = list_of_transforms

    pl_datal = LitData(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        transformer = LightCurveTransformer(**cfg.lc)
        #classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        classifier = MultimodalClassifier(lc_input_size=cfg.lc.embedding_size,
                                          tab_input_size=None, 
                                          use_lc = True,
                                          num_classes= cfg.num_classes)
        #loss =nn.CrossEntropyLoss() # HierLoss()#
        #loss = FocalLoss(5,alpha = [100], task_type='multi-class', num_classes=cfg.num_classes)
        weights = np.array([0.0006, 0.0005, 0.0006, 0.0007, 0.0006, 0.001 , 0.0005, 0.0007,
       0.0012, 0.0011, 0.0005, 0.0005, 0.0005, 0.0005, 0.0013, 0.001 ,
       0.0036, 0.0093, 0.0192, 0.0159, 0.0065, 0.0667])
        print(cfg.checkpoint) 
        loss = FocalLoss(gamma = 5, alpha = torch.tensor(weights), task_type='multi-class', num_classes=cfg.num_classes)
        #loss = HierLoss()
        pl_model = ClassifierModule(model = transformer,
                                    classifier= classifier,
                                     loss =  loss, 
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =False,
                                     report_mix = False, 
                                    lc_load_ckpt= cfg.lc.checkpoint,
                                    tab_load_ckpt=cfg.tab.checkpoint,
                                    lc_freeze= cfg.lc.freeze_weights,
                                    tab_freeze = cfg.tab.freeze_weights,
                                    #weight_str_parse_lc='model.transformer_lc.',
                                    #weight_str_parse_tab='model.transformer_tab.',
                                    **cfg)
        
        

    
    if cfg.experiment_type == 'MD' or cfg.experiment_type == 'MD_FEAT':
        transformer = TabularTransformer(**cfg.tab)
        #classifier = TokenClassifier(num_classes=cfg.num_classes,embedding_size=cfg.lc.embedding_size)
        classifier = MultimodalClassifier(tab_input_size=cfg.tab.embedding_size,
                                          use_tab=True,
                                          num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
       # print(cfg.checkpoint) 
        pl_model = ClassifierModule(model = transformer,
                                    classifier= classifier,
                                     loss =  loss, 
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =False,
                                     report_mix = False, 
                                    lc_load_ckpt= cfg.lc.checkpoint,
                                    tab_load_ckpt=cfg.tab.checkpoint,
                                    lc_freeze= cfg.lc.freeze_weights,
                                    tab_freeze = cfg.tab.freeze_weights,
                                    #weight_str_parse_lc='model.transformer_lc.',
                                    #weight_str_parse_tab='model.transformer_tab.',
                                    **cfg)
    if cfg.experiment_type == 'LC_MD':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        classifier = MultimodalClassifier(experiment_type=cfg.experiment_type,
                                          lc_input_size=cfg.lc.embedding_size, 
                                                tab_input_size=cfg.tab.embedding_size, 
                                                use_mix = True,
                                                num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model = model,
                                    classifier= classifier,
                                     loss =  loss, 
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =False,
                                     report_mix = False, 
                                    #weight_str_parse_lc='model.transformer_lc.',
                                    #weight_str_parse_tab='model.transformer_tab.',
                                    **cfg)
    if cfg.experiment_type == 'LC_MD_FEAT':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        classifier = MultimodalClassifier(experiment_type=cfg.experiment_type,
                                          lc_input_size=cfg.lc.embedding_size, 
                                                tab_input_size=cfg.tab.embedding_size, 
                                                use_mix = True,
                                                num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model = model,
                                    classifier= classifier,
                                     loss =  loss, 
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =False,
                                     report_mix = False, 
                                    #lc_load_ckpt= cfg.lc.checkpoint,
                                    #tab_load_ckpt=cfg.tab.checkpoint,
                                    #lc_freeze= cfg.lc.freeze_weights,
                                    #tab_freeze = cfg.tab.freeze_weights,
                                    #weight_str_parse_lc='model.transformer_lc.',
                                    #weight_str_parse_tab='model.transformer_tab.',
                                    **cfg) 
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