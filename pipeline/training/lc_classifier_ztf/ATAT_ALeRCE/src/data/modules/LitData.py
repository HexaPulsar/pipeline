import pytorch_lightning as pl

from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.CustomDataset import ATATDataset


class LitData(pl.LightningDataModule):
    def __init__(self, data_root: str = "path/to/dir", batch_size: int = 32, **kwargs):
        super().__init__()

        self.data_root = data_root
        self.batch_size = 256
        self.kwargs = kwargs
        print("KWARGS")
        print(self.kwargs)
        
    def train_dataloader(self):
        return get_dataloader(
                    batch_size=self.batch_size,
                  dataset_used=ATATDataset(
                data_root=self.data_root, set_type="train", **self.kwargs
            ),
            set_type="train",
            **self.kwargs
        )

    def val_dataloader(self):
        return get_dataloader(
            batch_size=self.batch_size,
            dataset_used=ATATDataset(
                data_root=self.data_root, set_type="validation", **self.kwargs
            ),
            set_type="validation",
             **self.kwargs
        )

    def test_dataloader(self):
        return get_dataloader(
            batch_size=self.batch_size,
            dataset_used=ATATDataset(
                data_root=self.data_root, set_type="test", **self.kwargs
            ),
            set_type="test",
             **self.kwargs
        )
