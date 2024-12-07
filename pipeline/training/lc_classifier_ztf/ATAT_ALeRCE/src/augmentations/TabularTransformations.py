import torch
class Jitter: 
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]

        # Create a mask for non-zero values
        mask = (x != 0)  # Mask with the same shape as x
        self.max_jitter = torch.rand(1).to(device = x.device).item()

        # Generate random jitter for each channel independently within the range [-max_jitter, max_jitter]
        jitter = (torch.rand_like(x, device=x.device) * 2 - 1) * self.max_jitter  # Shape: [bs, seqlen, channels]

        # Apply jitter only to the non-zero values
        x_with_jitter = x + jitter * mask
        x_with_jitter = torch.clip(x_with_jitter, 0,1.1)

        sample['tabular_feat'] = x_with_jitter
        return sample
    
class GaussianNoise: 
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]
        mask = (x != 0)  # Mask for non-zero values
        self.mean = 0 #torch.rand(1).to(device = x.device).item()
        self.std =  torch.FloatTensor(1).uniform_(0, 0.01).to(device = x.device).item()
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(self.mean, self.std, size=x.shape).to(device = x.device)
         
        # Apply noise only to the non-
        # zero values
        x_with_noise = x + noise * mask
        x_with_noise = torch.clip(x_with_noise, 0,1.1)
        sample['tabular_feat'] = x_with_noise
        return sample