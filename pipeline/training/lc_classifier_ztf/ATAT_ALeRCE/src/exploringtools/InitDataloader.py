from copy import deepcopy
from src.data.modules.LitData import LitData

class InitDataLoader:
    def __init__(self,update_dataset_path,datamodule_args,update_batch_size =128):
        dataset ={
            key: value 
            for key, value in datamodule_args.dataset.items() 
            if key not in ['transforms_1','transforms_2']
        }
        dataset['feature_key'] = 'extracted_feat_2048'
        datamodule_args_copy = deepcopy(datamodule_args)
        datamodule_args_copy['dataset'] = dataset
        datamodule_args_copy['dataset']['data_root'] = update_dataset_path
        datamodule_args_copy['batch_size'] = update_batch_size
        self.datamodule = datamodule_args_copy
        self.pl_datal = LitData(**self.datamodule)
         
        self.train_dataset = self.pl_datal.train_dataloader()
        self.validation_dataset = self.pl_datal.val_dataloader()
        self.test_dataset = self.pl_datal.test_dataloader()

    def set_sampler(self,use_sampler:bool = False):
        if use_sampler:
            self.datamodule['train_use_sampler']  = True
            self.pl_datal =  LitData(**self.datamodule)
            print(self.pl_datal)
            self.train = self.pl_datal.train_dataloader()
        else:
            self.datamodule['train_use_sampler']  = False
            self.pl_datal =  LitData(**self.datamodule)
            self.train = self.pl_datal.train_dataloader()

