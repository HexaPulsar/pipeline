import numpy as np 
from tqdm import tqdm 
import glob
import yaml
from  hydra.utils import instantiate
from torch import device, load
from collections import OrderedDict 


class InitBackbone:
    def __init__(
        self,
        path_to_config_yaml, 
        model,
        arg_key,
        device = 'cpu', ):
        self.device = device
        self.path_to_config_yaml = path_to_config_yaml
        self.args = self._load_yaml_args(path_to_config_yaml).ATATConfig
        self.backbone = self.init_model(model, arg_key)

    @staticmethod
    def _load_yaml_args(path_to_config_yaml):
        
        path_args = glob.glob(f"{path_to_config_yaml}/.hydra/*config*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        args =  instantiate(args)
        return args
    
    def init_model(self,model, arg_key:str):
        model = model(**self.args[arg_key]) 
        return model   
       
    def create_ordered_dict(self,
                            checkpoint_name: str = 'pretrain_ckpt',
                            remove_if_in_key_list:list = ['projection'], 
                            rename_keys: tuple = ('model.',''), print_keys = False):
            checkpoint_path_clip = glob.glob(f"{self.path_to_config_yaml}*{checkpoint_name}*")
            assert isinstance(rename_keys,tuple)
            print('Using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                #print(key)
                if print_keys:
                    print('old key name:',key)
                if any([remove_if_in_key in key for remove_if_in_key in remove_if_in_key_list ]):
                    continue
                od_atat[key.replace(f"{rename_keys[0]}", f"{rename_keys[1]}")] = checkpoint_clip[
                    "state_dict"
                ][key]
            return od_atat
    
    def load_backbone_weights(self,weights: dict, strict = True):
        self.backbone.load_state_dict(weights, strict=strict)
        print("     - Loaded backbone weights")

    def predict(self,dataloader, device = None, return_count_len  = False):
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
        count_len = None
       
        self.backbone.eval().to(device=self.device)
        
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            emb = self.backbone(**b1)
            flux = np.count_nonzero(b1['data'].clone().detach().cpu().numpy(), axis =1)
            count_len = (
                np.concatenate([count_len, flux])
                if count_len is not None
                else flux
            )   
            preds_out = (
                np.concatenate([preds_out, emb.detach().cpu().numpy()])
                if preds_out is not None
                else emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.backbone.to(device="cpu")
        return (preds_out, target, count_len) if return_count_len else (preds_out, target)

