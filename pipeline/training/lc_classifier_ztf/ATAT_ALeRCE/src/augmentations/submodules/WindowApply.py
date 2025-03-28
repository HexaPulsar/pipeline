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
        seq_window = np.random.randint(0, max_sample_n, size=(2,))
        seq_window.sort()

        start, end = int(seq_window[0]), int(seq_window[1])  # Explicitly convert to Python integers
        #print(start,end,abs(start-end), max_sample_n)
        if abs(start-end) > max_sample_n:
            return tensor_in
        tensor_out[start:end] = tensor_in[start:end]
        return tensor_out