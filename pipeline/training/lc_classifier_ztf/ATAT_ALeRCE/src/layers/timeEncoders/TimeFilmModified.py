import torch
import torch.nn as nn 
     
def roll_tensor(v):
    v = torch.roll(v, shifts = (1,), dims = 0)
    v[0,:] = 0  
    return v


def d_dt(x,t):
    dt = t - roll_tensor(t) 
    dx = x - roll_tensor(x)
    return dx,dt,dx/dt.masked_fill_(dt == 0, 1)


class EarlyFusionEncoder(nn.Module):
    def __init__(self, n_harmonics=1, 
                 embedding_size=64, 
                 Tmax=1000.0, 
                 input_size=1, 
                 dropout = 0.01,
                 bias = False,
                 use_velocity:bool = True,
                 use_acceleration: bool = True,
                 use_stats: bool = True,
                 use_metadata:bool = False,
                 use_coordinates: bool = False, 
                 use_timespan: bool =  False):
        super().__init__() 
        
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_stats = use_stats
        self.use_metadata = use_metadata
        print(use_velocity,
              use_acceleration,
              use_stats,
              use_metadata)
        self.dropout = nn.Dropout(dropout)
        self.timefilm_coeffs = TimeFilmCoeffs(n_harmonics,
                                         embedding_size,
                                         Tmax,
                                         dropout)
        self.linear_x = nn.Sequential(nn.Linear(in_features = input_size,
                                                    out_features = embedding_size,
                                                    bias = bias),
                                        nn.Dropout(dropout)
                                        )
        if use_velocity:
            self.velocity = Velocity(input_size,
                                     embedding_size=embedding_size,
                                     dropout = dropout)
        if use_acceleration:
            self.acceleration = Acceleration(input_size = input_size,
                                             embedding_size=embedding_size,
                                             dropout = dropout)
        if use_stats:
            self.stats = Stats(embedding_size)

        if use_metadata: 
            self.metadata  = Metadata(num_features=6,
                                      embedding_size = embedding_size)
            
    def forward(self,
                    x, 
                    t, 
                    metadata,
                    coordinates,
                    allwise, 
                    timespan):  
        x_out = self.linear_x(x)
        x_out.add_(self.velocity(x,t)) if self.use_velocity else None
        x_out.add_(self.acceleration(x,t)) if self.use_acceleration else None
        x_out.add_(self.stats(x)) if self.use_stats else None
        x_out.add_( self.metadata(x,metadata)) if self.use_metadata else None

        alpha, beta = self.timefilm_coeffs(t)
        out = x_out* alpha  +  beta
        return self.dropout(out)

class Velocity(nn.Module):
    def __init__(self,
                input_size, 
                embedding_size,
                dropout):
        super().__init__()
        self.linear_velocity = nn.Sequential(nn.Linear(in_features = input_size,
                                                   out_features = embedding_size,
                                                   bias = False),
                                        nn.Dropout(1e-5),
                                        )
    
    def forward(self,x,t):
        dx,dt,dxdt = d_dt(x,t) 
        return self.linear_velocity(dxdt)

class Acceleration(nn.Module):
    def __init__(self,
                input_size, 
                embedding_size,
                dropout):
        super().__init__()
        self.linear_acceleration = nn.Sequential(nn.Linear(in_features = input_size,
                                                    out_features = embedding_size,
                                                    bias = False),
                                                    nn.Dropout(1e-5),
                                                    )
    def forward(self, x,t):
        dx,_,_ = d_dt(x,t) 
        _,_,dxdtdt  = d_dt(dx,t)
        return self.linear_acceleration(dxdtdt)


class AlphaCoeffs(nn.Module):
    def __init__(self, n_harmonics, embedding_size, dropout):
        super().__init__()
        self.register_parameter('alpha_sincos',nn.Parameter(torch.randn(2*n_harmonics, embedding_size)))
        self.alpha_norm = nn.RMSNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,embedding):
        alpha = self.dropout(torch.matmul(embedding, self.alpha_sincos))
        alpha = self.alpha_norm(alpha)
        alpha = nn.functional.gelu(alpha)
        return alpha
    
class BetaCoeffs(nn.Module):
    def __init__(self, n_harmonics, embedding_size, dropout):
        super().__init__()
        self.register_parameter('beta_sincos',nn.Parameter(torch.randn(2*n_harmonics, embedding_size)))
        self.beta_norm = nn.RMSNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,embedding):
        beta = self.dropout(torch.matmul(embedding, self.beta_sincos))
        beta = self.beta_norm(beta)
        beta = nn.functional.gelu(beta)
        return beta 

class ZeroIgnoredStats(nn.Module):
    def __init__(self, dim, unbiased=True, keepdim=False):
        super().__init__()
        self.dim = dim
        self.unbiased = unbiased
        self.keepdim = keepdim

    def forward(self, x):
        # Mask for non-zero elements
        mask = x != 0
        mask_float = mask.type_as(x)

        # Count non-zero elements along dim
        count_nonzero = torch.sum(mask_float, dim=self.dim, keepdim=True)
        count_nonzero = torch.clamp(count_nonzero, min=1)  # avoid div by zero

        # Compute masked mean
        masked_sum = torch.sum(x * mask_float, dim=self.dim, keepdim=True)
        mean = masked_sum / count_nonzero

        # Compute squared differences, masked
        squared_diff = ((x - mean) ** 2) * mask_float
        sum_squared_diff = torch.sum(squared_diff, dim=self.dim, keepdim=True)

        # Unbiased or population variance denominator
        if self.unbiased:
            denom = count_nonzero - 1
            denom = torch.clamp(denom, min=1)
        else:
            denom = count_nonzero

        variance = sum_squared_diff / denom
        std = torch.sqrt(variance)

        if not self.keepdim:
            mean = mean.squeeze(self.dim)
            std = std.squeeze(self.dim)

        return [mean, std]


class Stats(nn.Module):
    def __init__(self, 
                 embedding_size,
                 min_: bool =True, 
                 max_: bool =True, 
                 mean: bool =True, 
                 std: bool = True,
                 range_:bool = True,
                 negative_count: bool = True,
                 positive_count:bool = True, 
                 q1 = False,
                 q2 = False,
                 q3 = False):
        super().__init__()
        stats = [min_,
                max_,
                mean,
                std,
                range_,
                positive_count,
                negative_count,
                q1,
                q2,
                q3]
        self.project_stats = nn.Linear(in_features=sum(stats),out_features=embedding_size, bias = False)
        self.zero_ignore = ZeroIgnoredStats(dim = 1,keepdim= False)

    def forward(self,x):
        x_ = x.clone()
        masked = x_.masked_fill(x_ ==0, float('-inf'))
        max_ = torch.argmax(masked,dim = 1)
        masked = x_.masked_fill(x_ ==0, float('inf'))
        min_ = torch.argmin(masked,dim = 1)
        negative_count = (x < 0).sum(dim = -2)
        pos_count = (x > 0).sum(dim = -2)
        stats = [max_,
                min_,
                negative_count,
                pos_count,
                max_ - min_,
                #torch.quantile(x, 0.25, dim = 1),
                #torch.quantile(x, 0.50, dim = 1),
                #torch.quantile(x, 0.75, dim = 1),
                    ]
        stats.extend(self.zero_ignore(x))
        stats = torch.concat(stats, dim = -1)
        stats = stats.unsqueeze(-2).repeat(1,x.size(1),1)
        mask = x != 0
        stats = stats * mask
        return self.project_stats(stats) 

class SymmetryEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: Tensor of shape (batch_size, seq_len), expected to be padded with zeros.

        Returns:
            Tensor of shape (batch_size,) containing symmetry scores in [0, 1].
        """
        if batch.ndim != 2:
            raise ValueError("Input must be a 2D tensor (batch_size, seq_len)")

        scores = []
        for row in batch:
            nonzero = (row != 0).nonzero(as_tuple=True)[0]
            if nonzero.numel() == 0:
                scores.append(torch.tensor(1.0, device=batch.device))  # All zeros
                continue
            start, end = nonzero[0].item(), nonzero[-1].item() + 1
            core = row[start:end]
            reversed_core = torch.flip(core, dims=[0])
            mse = torch.mean((core - reversed_core) ** 2)
            var = torch.var(core, unbiased=False)
            score = 1 - (mse / var) if var.item() != 0 else torch.tensor(1.0, device=batch.device)
            scores.append(score)

        return torch.stack(scores).unsqueeze(-1)  # shape (batch_size,1) ready for projection

class Coordinates(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def forward(self, tabular_feat):
        pass

class AllWise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def forward(self, tabular_feat):
        pass

class Metadata(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.linear_metadata = nn.Sequential(nn.Linear(num_features, embedding_size, bias = False),
                                             nn.Dropout(1e-5),
                                             nn.LayerNorm(embedding_size),
                                             nn.GELU(),
                                             nn.Linear(embedding_size,embedding_size, bias= True))
    def forward(self,x,metadata):
        assert metadata is not None, "Metadata not provided."
        metadata = metadata.unsqueeze(-2).repeat(1,x.size(1),1)
        mask = x != 0
        metadata = metadata * mask
        metadata = self.linear_metadata(metadata)
        return metadata



class TimeFilmCoeffs(nn.Module):
    def __init__(self, n_harmonics=1, 
                 embedding_size=64, 
                 Tmax=1000.0, 
                 dropout = 0.01):
            super().__init__() 
            self.alpha_coeffs = AlphaCoeffs(n_harmonics,
                                            embedding_size, 
                                            dropout)
            self.beta_coeffs = BetaCoeffs(n_harmonics,
                                          embedding_size,
                                          dropout)
            self.n_harmonics = n_harmonics
            self.register_buffer("ar", torch.tensor( 2 * torch.pi ) * (torch.arange(1,n_harmonics+1).unsqueeze(0).unsqueeze(0)))
            self.register_buffer("Tmax", torch.tensor(Tmax, dtype = float))
            

    @staticmethod
    def get_sin_cos(t):
        sin = torch.sin(t) 
        cos = torch.cos(t)
        return torch.exp(sin).masked_fill_(sin == 0, 0) , torch.exp(cos).masked_fill_(cos == 1, 0)         
    
    def forward(self,t):  
        t = self.ar*t.repeat(1,1,self.n_harmonics) / self.Tmax
        sin_emb, cos_emb = self.get_sin_cos(t)
        emb = torch.concat([sin_emb, cos_emb], dim = -1)
        return self.alpha_coeffs(emb), self.beta_coeffs(emb)
