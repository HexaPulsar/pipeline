import warnings
import logging
import colorlog
import pickle
import yaml
import glob
import os 
warnings.filterwarnings("ignore")

from custom_parser import CustomParser

import torch
from collections import OrderedDict
from src.data.modules.LitPretrain import LitPretrain 
from src.models.lightcurve.selfsupervised.pretrain import LitPreTrainVICREGLC
from src.models.tabular.selfsupervised.pretrain import LitPreTrainVICREG
from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.SSLDataset import SSLDataset
from lightning.fabric.loggers import TensorBoardLogger,CSVLogger

from torchvision.transforms import Compose, RandomApply, RandomChoice
from src.augmentations import LightCurveTransform as LC


from CustomTrainer import MyCustomTrainer, CustomEarlyStopping, CustomCheckpoint 

LOG_FILENAME = "atatRefactory.log"
ABS_PATH = os.path.abspath(".") # Absolute path for at src level package

class DirectoryHandler:
    def __init__(self,
                 ABS_PATH,
                 args,
                 ):  
         
         
        self.general = args['general']
        self.path = self.handler_dirs()
        with open(os.path.join(self.path,"args.yaml"), "w") as file:
            yaml.dump(args, file, sort_keys=False)
    # create folder if not exist
    def handler_dirs(self):
        my_new_path = [ABS_PATH,'results',"ZTF_ff",]
        if self.general["use_lightcurves"]:
            my_new_path.append("LC")
        if self.general["use_metadata"]:
            my_new_path.append("MD")
        if self.general["use_features"]:
            my_new_path.append("FEAT")
        if self.general["online_opt_tt"]:
            exp_path += "MTA"
        if self.general["use_augmented_dataset"]:
            exp_path += "AUG"
        s = '/'.join(my_new_path) + '/' + self.general['experiment_name'] + '/'
        exp_path = os.path.join(s) 
        # child path
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        return exp_path

    def handler_ckpt_path(self,path):
        out_path = glob.glob(path + "*.ckpt")[0]
        return out_path

if __name__ == "__main__":
    # logger
    logger = logging.getLogger()
    logging.root.handlers = []


    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
    )
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILENAME, encoding="utf-8"), handler,],
    )
    args = CustomParser(model_config_yaml_path='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/h5file/dict_info.yaml')
     
    # experiment type (modalities) setup
    directories = DirectoryHandler(ABS_PATH,args.all_args)
    pl_model = LitPreTrainVICREGLC(**args.all_args)
    from lightning.pytorch.Trainer import Trainer
    
    trainer = Trainer()
    
    trainer = MyCustomTrainer(
        callbacks=[CustomEarlyStopping, CustomCheckpoint()],
        loggers=[TensorBoardLogger(root_dir=directories.path, name="tensorboard", version="."),
                   CSVLogger(root_dir=directories.path, name=".", version=".")],
        
        accelerator="gpu",
        devices='auto',

        max_epochs=args.general["num_epochs"])
    train_dataloader = get_dataloader(
                                        batch_size=args.general['batch_size'],
                                        dataset_used=SSLDataset(
                                            data_root=args.general['data_root'], 
                                            set_type="train", 
                                            
                                            **args.all_args
                                        ),
                                        set_type="train" 
    )
    val_dataloader = get_dataloader(
                                    batch_size=args.general['batch_size'],
                                    dataset_used=SSLDataset(
                                        data_root=args.general['data_root'], 
                                        set_type="validation", 
                                        transform_1= Compose([#LC.MaskFirstN([-1,0,1,2]),
                                                            LC.WindowMask(2,window_size=20),
                                                            RandomChoice([LC.GaussianNoise(num_bands=2,mean = 0,std = 1e-2),
                                                            LC.TimeGaussianNoise(num_bands=2,mean = 0, std = 10),
                                                            LC.TimeFactor([i/10 for i in range(8,13)])])
                                                    ]),
                                        transform_2= Compose([#LC.MaskFirstN([-1,0,1,2]),
                                                            LC.WindowMask(2,window_size=20),
                                                            RandomChoice([LC.GaussianNoise(num_bands=2,mean = 0,std = 1e-2),
                                                            LC.TimeGaussianNoise(num_bands=2,mean = 0, std = 10),
                                                            LC.TimeFactor([i/10 for i in range(8,13)])])
                                                    ]),
                                        **args.all_args
                                    ),
                                    set_type="validation" 
    )
    
    trainer.fit(model = pl_model,
                train_loader= train_dataloader,
                val_loader = val_dataloader,
                ckpt_path=directories.path)
