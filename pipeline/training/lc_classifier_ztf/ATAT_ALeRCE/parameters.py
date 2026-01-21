import torchvision.models as models
import torch
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import glob
from collections import OrderedDict
from ReportPretraining import ReportClassification
from src.layers.transformer.ATAT import LightCurveTransformer,TabularTransformer

from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY
import matplotlib.pyplot as plt
import numpy as np

N = 3
EMB =   128
SUBEMB = 128
input = {'data':torch.rand(256,200,2), 'time': torch.rand(256,200,2),'mask': torch.rand(256,200,2).bool()}
model = LightCurveTransformer(input_size= 1,
        embedding_size= EMB,
        embedding_size_sub= SUBEMB,
        num_heads= 4,
        num_encoders= N,
        Tmax= 3000.0,
        num_harmonics= 4,
        pe_type= 'tm',
        num_bands= 2,
        dropout = 0.00,
        checkpoint = None,
        freeze_weights = True,
        use_velocity = True,
        use_acceleration = True,
        use_stats = False,
        use_metadata =False,
        use_features = False,
        use_conv = True,
        use_timefilm_gelu=True,

        use_timefilm_norm=True,
        use_sequence_norm=True,
        use_exp=True,
        metadata_num_features = 6,
        features_num_features = 181,)  # Example

from calflops import calculate_flops

flops, macs, params = calculate_flops(model=model,
                                      kwargs=input,
                                      output_as_string=True,
                                      output_precision=4)


import torchvision.models as models
input = {'tabular_feat': torch.rand(256,187).bool()}
model = TabularTransformer(
        embedding_size= EMB,
        embedding_size_sub= SUBEMB,
        num_heads= 4,
        num_encoders= N,
        length_size = 187,
        dropout = 0.00,
        checkpoint = None,
        freeze_weights = False,)  # Example

from calflops import calculate_flops

flops, macs, params = calculate_flops(model=model,
                                      kwargs=input,
                                      output_as_string=True,
                                      output_precision=4)