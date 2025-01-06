import torch
class Jitter: 
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]
        mask = x !=0
        # Create a mask for non-zero values
        
        self.max_jitter = torch.rand(1).to(device = x.device).item()

        # Generate random jitter for each channel independently within the range [-max_jitter, max_jitter]
        jitter = (torch.rand_like(x, device=x.device) * 2 - 1) * self.max_jitter  # Shape: [bs, seqlen, channels]

        # Apply jitter only to the non-zero values
        x_with_jitter = x + jitter*mask
        x_with_jitter = torch.clip(x_with_jitter, 0,1.1)

        sample['tabular_feat'] = x_with_jitter
        return sample

class Factor:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['tabular_feat']
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(0.98, 1.02)
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['tabular_feat'] = data * factor
        #sample['time'] = time * scale_factors

        return sample

class Shift:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['tabular_feat']
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(-0.01, 0.01)
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['tabular_feat'] = data + factor
        #sample['time'] = time * scale_factors

        return sample
    
class RandomShift:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['tabular_feat']
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(-0.01, 0.01)
        mask = torch.rand_like(data, device=data.device) >=0.5
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['tabular_feat'] = data + factor*mask
        #sample['time'] = time * scale_factors

        return sample

class Scale:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['tabular_feat'] 
        # Generate a random scale factor for each point in sample['tabular_feat']
        scale_factors = torch.empty(sample['tabular_feat'].shape,device = data.device).uniform_(self.min_scale, self.max_scale)

        # Scale each point of sample['data'] with its corresponding scale factor
        sample['tabular_feat'] = data * scale_factors
        #sample['time'] = time * scale_factors

        return sample

class GaussianNoise: 
    def __init__(self):
        self.mean = 0
        self.std = 0
        
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]
        mask = (x != 0)
        self.std =  torch.FloatTensor(1).uniform_(0, 1.5).to(device = x.device).item()
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(self.mean, self.std, size=x.shape).to(device = x.device)
         
        # Apply noise only to the non-
        # zero values
        x_with_noise = x + noise*mask
        x_with_noise = torch.clip(x_with_noise, 0,1.1)
        sample['tabular_feat'] = x_with_noise
        return sample
    

class RandomMask:
    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        mask = ( torch.rand_like(sample['tabular_feat'])>=0.5).bool()
        
        sample['mask'] = mask
        return sample
