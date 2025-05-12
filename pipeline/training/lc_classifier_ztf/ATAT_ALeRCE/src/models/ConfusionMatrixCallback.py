
import os
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any, List


class EmbeddingSaver(Callback):
    def __init__(
        self,

    ):
        self.validation_embeddings = []

    def on_validation_start(self, trainer, pl_module):
        # Reset collected data at the beginning of validation
        self.collected_embeddings = []
        
    def on_validation_batch_end(self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch,
        batch_idx: int,
        dataloader_idx: int = 0
        ):
        pass
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if we don't need to save this epoch
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_every_n_epochs != 0:
            return
        
        # Skip if no embeddings were collected
        if not self.collected_embeddings:
            return
        
        # Concatenate all collected embeddings
        all_embeddings = torch.cat(self.collected_embeddings, dim=0)
        
        # Process collected metadata
        metadata = {}
        for key, values in self.collected_metadata.items():
            if values:  # Check if we collected any values
                if isinstance(values[0], torch.Tensor):
                    metadata[key] = torch.cat(values, dim=0).cpu()
                else:
                    # Handle non-tensor metadata
                    metadata[key] = values
        
        # Save as PyTorch file
        save_path = os.path.join(
            self.output_dir, 
            f"{self.filename_prefix}_epoch_{current_epoch}.pt"
        )
        
        save_dict = {
            "embeddings": all_embeddings,
            "epoch": current_epoch,
            **metadata
        }
        
        torch.save(save_dict, save_path)
        
        # Optionally save as numpy file
        if self.save_as_numpy:
            np_save_path = os.path.join(
                self.output_dir, 
                f"{self.filename_prefix}_epoch_{current_epoch}.npy"
            )
            np.save(np_save_path, all_embeddings.numpy())
        
        print(f"Saved {len(all_embeddings)} embeddings to {save_path}")