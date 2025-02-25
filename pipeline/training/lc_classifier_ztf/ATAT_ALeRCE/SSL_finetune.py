import warnings
import logging
import colorlog
import pickle
import yaml
import glob
import os 
warnings.filterwarnings("ignore")

from custom_parser import parse_model_args, handler_parser

from src.data.modules.LitPretrain import LitPretrain
from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.src.models.lightcurve.contrastive.LitPreTrainVICREGLC import LitPreTrainVICREG

from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.src.models.multimodal.contrastive.LitcATAT import LitcATAT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer


LOG_FILENAME = "atatRefactory.log"

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

import torch
from collections import OrderedDict

 


def get_path_results(exp_path, args):
    components = []
    if args["general"]["use_lightcurves"]:
        components.append("LC")
    if args["general"]["use_metadata"]:
        components.append("MD")
    if args["general"]["use_features"]:
        components.append("FEAT")

    if components:
        exp_path += "/" + "_".join(components)
    exp_path += "/{}".format(args["general"]["experiment_name"])

    # print('exp_path: '.format(exp_path))
    return exp_path


# create folder if not exist
def handler_dirs(path, args):
    dict_name = {
        "ztf_ff": "/ZTF_ff",
        "elasticc_1": "/ELASTICC_1",
        "elasticc_2": "/ELASTICC_2",
    }

    # create child folders based on configs
    exp_path = path
    exp_path += dict_name[args["general"]["name_dataset"]]
    exp_path = get_path_results(exp_path, args)

    if args["general"]["online_opt_tt"]:
        exp_path += "/MTA"

    if args["general"]["use_augmented_dataset"]:
        exp_path += "/AUG"

    # child path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    return exp_path


def save_yaml(args, path):
    with open(f"{path}/args.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)


def load_yaml(path):
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args


def handler_ckpt_path(path):
    out_path = glob.glob(path + "*.ckpt")[0]
    return out_path


# Absolute path for at src level package
ABS_PATH = os.path.abspath(".")

if __name__ == "__main__":
    args = vars(parse_model_args(arg_dict=None))

    config_training = load_yaml(path="configs/training.yaml")
    if args["experiment_type_general"] not in config_training.keys():
        raise "You put an experiment name that is not found in the ./configs/training.yaml file"

    args.update(config_training[args["experiment_type_general"]])

    args = handler_parser(args)
    path = handler_dirs(ABS_PATH + f"/results/", args)
    save_yaml(args, path)

    # general updates
    args_general = args["general"]

    ############################  DATALOADERS  ############################
    pl_datal = LitPretrain(**args_general)

    ############################  CALLBACKS  ############################
    all_callbacks = []
    all_callbacks += [
        ModelCheckpoint(
            monitor="alignment/loss_validation/total_loss",  # "F1Score_MLPloss/val"
            dirpath=path,
            save_top_k=1,
            mode="min",  # )]
            every_n_train_steps=1,
            filename="my_best_checkpoint-{step}",
        )
    ]

    all_callbacks += [
        EarlyStopping(
            monitor="alignment/loss_validation/total_loss",
            min_delta=0.00,
            patience=args_general["patience"],
            verbose=False,
            mode="min",
        )
    ]

    all_callbacks += [LearningRateMonitor(logging_interval="step")]

    all_loggers = []
    all_loggers += [TensorBoardLogger(save_dir=path, name="tensorboard", version=".")]
    all_loggers += [CSVLogger(save_dir=path, name=".", version=".")]

    # load from checkpoint if there is one

    ############################  MODEL  ############################
    pl_model = LitcATAT(lc_ckpt='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/LC/pretrain/'
                        ,ft_ckpt='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/MD/pretrain/'
                        ,**args)
 

    if args_general["change_clf"]:
        pl_model.atat.change_clf(args_general["num_classes"])

    ############################  TRAINING  ############################

    trainer = Trainer(
        callbacks=all_callbacks,
        logger=all_loggers,
        val_check_interval=0.5,
        log_every_n_steps=100,
        accelerator="gpu",
        #devices=[],
        min_epochs=1,
        max_epochs=args_general["num_epochs"],
        #check_val_every_n_epoch=5,
        #accumulate_grad_batches = 5,
        num_sanity_val_steps=0,
    )

    # Trainer model pl routine # trsainer fit models
    trainer.fit(pl_model, pl_datal)
