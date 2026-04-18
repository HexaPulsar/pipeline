from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy


class WindowApply:
    @staticmethod
    def apply_to_window(tensor_in, tensor_out, window_size, max_sample_n):
        if max_sample_n == 0:
            return tensor_in
        seq_window = torch.randint(0, max_sample_n, (2,)).sort()[0]
        start, end = seq_window[0].item(), seq_window[1].item()
        tensor_out[start:end] = tensor_in[start:end]
        return tensor_out