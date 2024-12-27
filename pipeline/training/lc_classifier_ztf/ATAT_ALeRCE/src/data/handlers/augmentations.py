import torch
import numpy as np
import random


class ThreeTimeMask:
    "Callable implementation of threetimemask function to integrate with the torchvision.transforms"

    def __init__(self,use_features,use_lightcurves,extracted_feat = None):
        self.use_features = use_features
        self.use_lightcurves = use_lightcurves
        self.extracted_feat = extracted_feat
        self.time_eval_list = [16,128,2048]

    def __call__(self, sample):
        time_eval = np.random.choice(self.time_eval_list)  
        if self.use_lightcurves:
            mask, time = sample["mask"], sample["time"]
            mask_time = (time <= time_eval).bool()
            sample["mask"] = (mask * mask_time).bool()
        
        if self.use_features:
            sample["extracted_feat"] = self.extracted_feat[time_eval][sample['idx']]
        return sample


class SCAugmentation:
    "Callable implementation of scagumentation function to integrate with the torchvision.transforms"
    def __init__(self,per_init_time, list_time_to_eval,use_features,use_lightcurves, extracted_feat_dict = None):
        self.per_init_time = per_init_time
        self.list_time_to_eval = list_time_to_eval 
        self.use_features = use_features
        self.use_lightcurves = use_lightcurves
        self.extracted_feat_dict = extracted_feat_dict

    def __call__(self, sample):
        """sample is a dictionary obj"""
        mask, time_alert = sample["mask"], sample["time"]
        random_value = random.uniform(0, 1)
        max_time = (time_alert * mask).max()
        init_time = self.per_init_time * max_time
        eval_time = init_time + (max_time - init_time) * random_value
        if self.use_lightcurves:
            
            """ random value to asing new light curve """

            
            mask_time = (time_alert <= eval_time).bool()
            sample["mask"] = (mask * mask_time).bool()
        """ if lc features are using in training tabular feeat is updated to specific time (near to)"""

        if self.use_features:
            # update features with features at eval_time value
            sample.update(
                {"extracted_feat": self.extracted_feat_dict[self.list_time_to_eval[
                        (eval_time.numpy() <= self.list_time_to_eval).argmax()
                    ]][sample['idx'], :]}
            )
        """ multiplication of mask, where are both enabled is the final mask """
        
        return sample