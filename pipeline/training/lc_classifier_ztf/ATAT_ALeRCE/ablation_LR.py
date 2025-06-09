# %%
from ReportPretraining import ReportPretraining, ReportZeroshot, InitBaseModel, InitDataLoader
from src.layers.transformer.lightcurve import LightCurveTransformer
import umap.plot
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY,ZTF_TAXONOMY
import glob
from torch import device, load
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

# %%
PATH = '/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/LC/v16/'
DEVICE = 'cuda:2'

# %%
import yaml
from  hydra.utils import instantiate
path_args = glob.glob(f"{PATH}/.hydra/*config*")[0]
with open(path_args, "r") as file:
    args = yaml.safe_load(file)
args =  instantiate(args)

# %%
from torchmetrics.classification import MulticlassF1Score

# %%
def predict(model,dataloader):
    target = None
    preds_out = None
    model.eval().to(device=DEVICE)
    for b1 in tqdm(dataloader):
        b1 = {key: value.to(device=DEVICE) for key, value in b1.items()}
        t = b1["labels"]
        
        lc_emb = model(**b1)  # [:, 0, :]
        preds_out = (
            np.concatenate([preds_out, lc_emb.detach().cpu().numpy()])
            if preds_out is not None
            else lc_emb.cpu().detach().numpy()
        )
        target = (
            np.concatenate([target, t.cpu().detach().numpy()])
            if target is not None
            else t.detach().cpu().numpy()
        )
    model.to(device="cpu")
    return preds_out, target

# %%
def load_ckpt(model, ckpt = None):
    if ckpt is None:
        print('no ckpt provided!')
        return model
    od_atat = OrderedDict()
    checkpoint_clip = load(
                    ckpt, map_location=DEVICE
                )
    for key in checkpoint_clip["state_dict"].keys():
        if 'model' not in key:
            continue
        od_atat[key.replace("model.", "")] = checkpoint_clip[
            "state_dict"
        ][key]
    model.load_state_dict(od_atat, strict=True)
    return model

# %%
dl = InitDataLoader(**args.ATATConfig.datamodule)
dl.set_sampler(True)
list_of_ckpts =  glob.glob(f"{PATH}*pretrain_ckpt*")


# %%

for i,ckpt_ in enumerate(list_of_ckpts):
    if i % 5 == 0:
        model = load_ckpt(LightCurveTransformer(**args.ATATConfig.lc), ckpt_)

        X_train, y_train = predict(model,dl.train)
        X_test, y_test = predict(model,dl.validation)


        std_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # z = (x - mean) / std
            ('model', LogisticRegression(random_state=0, max_iter = 1000))
        ])
        std_pipeline.fit(X_train,y_train)
        
        test_y_pred = std_pipeline.predict(X_test)

        #classification = classification_report(y_test,test_y_pred, target_names=list(ZTF_TAXONOMY().keys()),digits = 4)
        #print(classification)
        dict_ =   classification_report(y_test,test_y_pred, target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=True)
        with open('{}'.format('{}'.format('/'.join(ckpt_.split('/')[:-1])+'/'+ckpt_.replace('ckpt','pkl').split('/')[-1])), 'wb') as f:
            pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
list_of_pkls =  glob.glob(f"{PATH}*.pkl*")[1:5]

for pkl_ in list_of_pkls:
    print(pkl_)
    with open(pkl_, 'rb') as f:
        test = pickle.load(f)
        print(test['macro avg']['f1-score'])

