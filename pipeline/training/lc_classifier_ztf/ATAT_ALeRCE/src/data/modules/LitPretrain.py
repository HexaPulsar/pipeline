import pytorch_lightning as pl

from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.SSLDataset import SSLDataset
import glob
from torch.utils.data import ConcatDataset
import logging
class LitPretrain(pl.LightningDataModule):
    def __init__(self,
     data_root: str = "path/to/dir", batch_size: int = 128, **kwargs):
        super().__init__()

        self.data_root = glob.glob('/home/mdelafuente/sixplusdets/*/*/*.h5', recursive = True) # data_root 
        assert isinstance(self.data_root,list)
        self.batch_size =  batch_size
        self.kwargs = kwargs

    
    def train_dataloader(self):
       # print(self.data_root)
        
       
        if isinstance(self.data_root,str):
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=SSLDataset(
                    data_root=self.data_root, set_type="train", **self.kwargs
                ),
                set_type="train" 
            )
        else:
            list_of_datasets = []
            for directory in self.data_root:
                dataset_init = SSLDataset(data_root=directory, set_type="train", **self.kwargs)
                list_of_datasets.append(dataset_init)
            concatenated_datasets = ConcatDataset(list_of_datasets)
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=concatenated_datasets,
                set_type="train" 
            )
            
            

    def val_dataloader(self):
        if isinstance(self.data_root,str):
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=SSLDataset(
                    data_root=self.data_root, set_type="validation", **self.kwargs
                ),
                set_type="validation" 
            )
        else:
            list_of_datasets = []
            for directory in self.data_root:
                dataset_init = SSLDataset(data_root=directory, set_type="validation", **self.kwargs)
                list_of_datasets.append(dataset_init)
            concatenated_datasets = ConcatDataset(list_of_datasets)
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=concatenated_datasets,
                set_type="validation" 
            )

    def test_dataloader(self):
         if isinstance(self.data_root,str):
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=SSLDataset(
                    data_root=self.data_root, set_type="test", **self.kwargs
                ),
                set_type="test" 
            )
         else:
            list_of_datasets = []
            for directory in self.data_root:
                dataset_init = SSLDataset(data_root=directory, set_type="test", **self.kwargs)
                list_of_datasets.append(dataset_init)
            concatenated_datasets = ConcatDataset(list_of_datasets)
            logging.log(concatenated_datasets.__len__)
            return get_dataloader(
                batch_size=self.batch_size,
                dataset_used=concatenated_datasets,
                set_type="test" 
            )