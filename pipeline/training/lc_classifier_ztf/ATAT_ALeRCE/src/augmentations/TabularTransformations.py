import torch
class Jitter: 
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]

        # Create a mask for non-zero values
        
        self.max_jitter = torch.rand(1).to(device = x.device).item()

        # Generate random jitter for each channel independently within the range [-max_jitter, max_jitter]
        jitter = (torch.rand_like(x, device=x.device) * 2 - 1) * self.max_jitter  # Shape: [bs, seqlen, channels]

        # Apply jitter only to the non-zero values
        x_with_jitter = (x + jitter).masked_fill_(x==0,0)
        x_with_jitter = torch.clip(x_with_jitter, 0,1.1)

        sample['tabular_feat'] = x_with_jitter
        return sample
    
class GaussianNoise: 
    def __init__(self):
        self.mean = 0
        self.std = 0
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]
    
        self.std =  torch.FloatTensor(1).uniform_(0, 0.01).to(device = x.device).item()
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(self.mean, self.std, size=x.shape).to(device = x.device)
         
        # Apply noise only to the non-
        # zero values
        x_with_noise = (x + noise).masked_fill_(x == 0, 0)
        #x_with_noise = torch.clip(x_with_noise, 0,1.1)
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
