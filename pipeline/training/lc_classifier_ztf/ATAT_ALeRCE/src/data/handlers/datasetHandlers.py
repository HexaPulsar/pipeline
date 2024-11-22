from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch



def get_dataloader(
    dataset_used,
    batch_size=128, #TODO corregir parseo de batch_size 
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    set_type="train",
    use_sampler = False,
    **kwargs
):
    
    """extra args for Dataloader object"""

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    """ if is data train data must be sampled balanced or not  equally_sample = True (balanced batch)"""
 
    if use_sampler:
        print("USING SAMPLER")
        if set_type == "train":
            # balancedd samples
            class_sample_count = np.array(
                [
                    len(np.where(dataset_used.labels == t)[0])
                    for t in np.unique(dataset_used.labels)
                ]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in dataset_used.labels])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(
                samples_weight.type("torch.DoubleTensor"), len(samples_weight)
            )

            loader = DataLoader(dataset_used, sampler=sampler, **loader_kwargs)
        else:
                print("NOT USING SAMPLER")
                
                loader = DataLoader(dataset_used, sampler=None, shuffle=True, **loader_kwargs)
    else:
        print("NOT USING SAMPLER")

        loader = DataLoader(dataset_used, sampler=None, shuffle=True, **loader_kwargs)

    return loader
