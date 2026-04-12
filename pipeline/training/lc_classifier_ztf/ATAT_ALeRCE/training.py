import warnings
import logging
import colorlog

warnings.filterwarnings("ignore")

from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY
from src.augmentations import LightCurveTransform as LC

from src.data.modules.LitData import LitData
from src.models.ClassifierModule import ClassifierModule

from pytorch_lightning import Trainer

from src.layers.transformer.ATAT import LightCurveTransformer, TabularTransformer, Combinator
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier

import torch.nn as nn
import hydra
import numpy as np
from src.utils.CustomParser import ATATConfig
from  hydra.utils import instantiate
from torchvision.transforms import RandomChoice, RandomApply, Compose


import torch
import numpy as np

class HierLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def modify_logits(self, hier_preds, class_preds):
        mapping_dict  = {
            0: [4, 9, 16, 17, 18, 19, 20, 21],
            1: [0, 1, 3, 5, 8],
            2: [2, 6, 7, 10, 11, 12, 13, 14, 15]
        }

        hier_preds = nn.functional.softmax(hier_preds, dim=-1)
        hier_classes = torch.argmax(hier_preds, dim=-1).int()
        class_probs = nn.functional.softmax(class_preds, dim=-1)



        modified_logits = []
        for i in range(class_preds.shape[0]):
            valid_indices = mapping_dict[int(hier_classes[i].item())]
            mask = torch.zeros_like(class_probs[i])
            mask[valid_indices] = 1
            masked_logits = class_probs[i] * mask  # No in-place operation here
            modified_logits.append(masked_logits)
        modified_logits = torch.stack(modified_logits, dim=0)
        return modified_logits


    def map_label_tensor(self,labels):
        mapping_dict = {
            0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
            2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
            4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
        }
        mapping_tensor = torch.tensor([mapping_dict.get(int(label), -1) for label in labels], device = labels.device)
        return mapping_tensor

    def forward(self, preds ,labels):

        hier_loss = self.loss(preds[0],  self.map_label_tensor(labels.long()))
        modified_logits = self.modify_logits(preds[0],preds[1])
        class_loss = self.loss(modified_logits,  labels)
        hier_weight =1
        class_weight =1
        loss =(hier_weight* hier_loss + class_weight *class_loss)/2
        return loss

@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name= 'supervised_training')
#@hydra.main(version_base=None, config_path="./src/configs/ZTF/", config_name= 'LC_MD')
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
   # assert cfg.experiment_type == 'LC' , cfg.experiment_type
    cfg.datamodule.dataset.experiment_type = cfg.experiment_type

    transients = set(ZTF_TAXONOMY.transient().values())
    stochastic = set(ZTF_TAXONOMY.stochastic().values())
    periodic = set(ZTF_TAXONOMY.periodic().values())

    p_ =cfg.online_transforms.p_

    transforms = []
    apply_ = None
    windows =[LC.WindowSelect(cfg.lc.num_bands, window_size=w_, apply_to_classes=None) for w_ in list(range(6, 204, 6))]
    windows.extend([ LC.MAXWindowSelect(cfg.lc.num_bands, window_size=w_, apply_to_classes=None) for w_ in list(range(6, 204, 6))])
    transforms.extend([RandomApply([RandomChoice(windows
                                )],p =1)
                                ])  if cfg.online_transforms.use_window_select else None

   # transforms.extend([ RandomApply([LC.GaussTimeFactor(cfg.lc.num_bands, scale = 1e-4, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_time_gauss_factor else None
   # transforms.extend([ RandomApply([LC.GaussFactor(cfg.lc.num_bands, scale = 1e-4, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_gauss_factor else None

    transforms.extend([ RandomApply([LC.TimeFactor( factor = list(np.linspace(0.99,1.01, 100)), apply_to_classes=apply_)], p = p_),]) if cfg.online_transforms.use_simple_time_factor else None
    transforms.extend([ RandomApply([LC.Factor( factor = list(np.linspace(0.99,1.01, 100)), apply_to_classes=apply_)], p = p_),]) if cfg.online_transforms.use_simple_data_factor else None

    transforms.extend([ RandomApply([LC.BandPermute(cfg.lc.num_bands, apply_to_classes=apply_)], p = p_)]) if cfg.online_transforms.use_band_permute else None


    transforms.extend([ RandomApply([LC.GaussianFilter(cfg.lc.num_bands,filter_std = [1e-5,1e-4,1e-3,1e-2,1e-1,-1], apply_to_classes=apply_)], p = p_)])
    transforms.extend([ RandomApply([LC.GaussianTimeFilter(cfg.lc.num_bands,filter_std = [1e-5,1e-4,1e-3,1e-2,1e-1,-1], apply_to_classes=apply_)], p = p_)])

    cfg.datamodule.dataset.train_transforms = transforms
    #print(transforms)
   # cfg.datamodule.dataset.val_transforms = transforms

    pl_datal = LitData(**cfg.datamodule)
    if cfg.experiment_type == 'LC':
        log_message = (
        f"Model Configuration:\n"
        f"{'='*30}\n"
        f"• Use_conv          : {'✓' if cfg.lc.use_conv else '✗'}\n"
        f"• Use_stats         : {'✓' if cfg.lc.use_stats else '✗'}\n"
        f"• Use_acceleration  : {'✓' if cfg.lc.use_acceleration else '✗'}\n"
        f"• Use_velocity      : {'✓' if cfg.lc.use_velocity else '✗'}\n"
        f"• Use_medatadata    : {'✓' if cfg.lc.use_metadata else '✗'}\n"
        f"• Use_features      : {'✓' if cfg.lc.use_features else '✗'}\n"
        f"{'='*30}\n"
        f"• Sequence l2 norm  : {'✓' if cfg.lc.use_sequence_norm else '✗'}\n"
        f"• Depth            : {cfg.lc.num_encoders}\n"
        f"• Heads            : {cfg.lc.num_heads}\n"
        f"• Dropout           : {cfg.lc.dropout}\n"
        f"{'='*30}\n"
        f"• Timefilm gelu     : {'✓' if cfg.lc.use_timefilm_gelu else '✗'}\n"
        f"• Timefilm norm     : {'✓' if cfg.lc.use_timefilm_norm else '✗'}\n"
        f"• Output exp()      : {'✓' if cfg.lc.use_exp else '✗'}\n"
        f"{'='*30}"
        )
        print(log_message)
        transformer = LightCurveTransformer(**cfg.lc)
        classifier = MultimodalClassifier(experiment_type=cfg.experiment_type,
                                          lc_input_size=cfg.lc.embedding_size,
                                                tab_input_size=cfg.tab.embedding_size,
                                                use_mix = False,
                                                use_lc = True,
                                                use_tab=False,
                                                combine_logits=False,
                                                num_classes= cfg.num_classes)
        loss =nn.CrossEntropyLoss()

        pl_model = ClassifierModule(model = transformer,
                                    classifier= classifier,
                                     loss =  loss,
                                     freeze_lc=True,
                                     freeze_tab=False,
                                     report_lc =False,
                                     report_mix = False,
                                    lc_load_ckpt= cfg.lc.checkpoint,
                                    tab_load_ckpt=cfg.tab.checkpoint,
                                    lc_freeze= cfg.lc.freeze_weights,
                                    tab_freeze = cfg.tab.freeze_weights,
                                    weight_str_parse_lc= ('model.',''),
                                    #weight_str_parse_tab='model.transformer_tab.',
                                    **cfg)

    if cfg.experiment_type == 'MD' or cfg.experiment_type == 'MD_FEAT':
        transformer = TabularTransformer(**cfg.tab)
        print(cfg.callbacks.model_checkpoint.monitor)
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
                                     weight_str_parse_lc= ('model.',''),
                                    #weight_str_parse_tab=  ('model.',''),
                                    **cfg)
    if cfg.experiment_type == 'LC_MD':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        classifier = MultimodalClassifier(experiment_type=cfg.experiment_type,
                                          lc_input_size=cfg.lc.embedding_size,
                                                tab_input_size=cfg.tab.embedding_size,
                                                use_mix = True,
                                                use_lc = False,
                                                use_tab=False,
                                                combine_logits=False,
                                                num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model = model,
                                    classifier= classifier,
                                     loss =  loss,
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =True,
                                     report_mix = False,
                                    lc_load_ckpt= cfg.lc.checkpoint,
                                    #tab_load_ckpt=cfg.tab.checkpoint,

                                     weight_str_parse_lc= ('model.',''),
                                    weight_str_parse_tab=('model.',''),
                                    **cfg)

    if cfg.experiment_type == 'LC_MD_FEAT':
        transformer_lc = LightCurveTransformer(**cfg.lc)
        transformer_md = TabularTransformer(**cfg.tab)
        model  = Combinator(transformer_lc,transformer_md)
        classifier = MultimodalClassifier(experiment_type=cfg.experiment_type,
                                          lc_input_size=cfg.lc.embedding_size,
                                                tab_input_size=cfg.tab.embedding_size,
                                                use_mix = True,
                                                use_lc = False,
                                                use_tab=False,
                                                combine_logits=False,
                                                num_classes= cfg.num_classes)
        loss = nn.CrossEntropyLoss()
        pl_model = ClassifierModule(model = model,
                                    classifier= classifier,
                                     loss =  loss,
                                     freeze_lc=False,
                                     freeze_tab=False,
                                     report_lc =True,
                                     report_mix = False,
                                    lc_load_ckpt= cfg.lc.checkpoint,
                                    #tab_load_ckpt=cfg.tab.checkpoint,

                                     weight_str_parse_lc= ('model.',''),
                                    weight_str_parse_tab=('model.',''),
                                    **cfg)
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
       # profiler="simple",
        callbacks=list(cfg.callbacks.values()),
        logger= list(cfg.loggers.values()),
        **cfg.trainer
        )

    trainer.fit(pl_model, pl_datal)
    #trainer.test(pl_model,pl_datal)

    #pref_ = global_config.CHECKPOINT_PREFIX

if __name__ == "__main__":
    main()