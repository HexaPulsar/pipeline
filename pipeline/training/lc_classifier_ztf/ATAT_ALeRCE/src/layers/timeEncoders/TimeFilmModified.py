import torch
from torch import nn

def roll_tensor(v):
    v = torch.roll(v, shifts=(1,), dims=1)
    v[:, 0, :] = 0
    return v


def d_dt(x, t, use_exp=False, tmax=2048):
    dt = t - roll_tensor(t)
    dx = x - roll_tensor(x)
    if use_exp:
        exp_ = torch.exp(torch.sin(dx / dt.masked_fill_(dt == 0, 1)  ))
        return dx, dt, exp_
    return dx, dt, torch.sin(dx / dt.masked_fill_(dt == 0, 1)  )

class EarlyFusionEncoder(nn.Module):
    def __init__(
        self,
        n_harmonics=4,
        embedding_size=64,
        projections_inner_size=64,
        Tmax=1000.0,
        input_size=1,
        dropout=0.001,
        bias=True,
        metadata_num_features=None,
        features_num_features=None,
        use_velocity: bool = False,
        use_acceleration: bool = False,
        use_stats: bool = False,
        use_metadata: bool = False,
        use_tabular_transformer=False,
        use_features=False,
        use_timefilm_norm=False,
        use_timefilm_gelu=False,
        use_sequence_norm=False,
        use_conv=False,
        use_exp=False,
    ):
        super().__init__()
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_stats = use_stats
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.dropout = nn.Dropout(dropout)
        embedding_size = embedding_size * 1
        self.timefilm_coeffs = TimeFilmCoeffs(
            n_harmonics,
            embedding_size,
            Tmax,
            dropout,
            exponential=use_exp,
            norm=use_timefilm_norm,
            gelu=use_timefilm_gelu,
            bias=True,
        )
        inner_size = embedding_size
        self.use_conv = use_conv
        self.dropout = nn.Dropout(dropout)
        if self.use_conv:
            self.conv = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=inner_size, bias=bias, kernel_size=5,padding = 2) )
            self.linear_x = nn.Sequential(
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(in_features=inner_size, out_features=embedding_size, bias=bias),
            )
        else:
            self.linear_x = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=inner_size, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features=inner_size, out_features=embedding_size, bias=bias),
            )
        if use_velocity:
            self.velocity = Velocity(input_size, embedding_size=embedding_size,inner_size= inner_size, bias=bias, dropout=dropout
            )
        if use_acceleration:
            self.acceleration = Acceleration(
                input_size=input_size,
                embedding_size=embedding_size,
                inner_size=inner_size,
                bias=bias,
                dropout=dropout,
            )
        if use_stats:
            self.stats = Stats(embedding_size)


        if use_metadata or use_features:
            if all([use_metadata, use_features]):
                input_size = metadata_num_features + features_num_features
            elif all([use_metadata, not use_features]):
                input_size = metadata_num_features
            else:
                input_size = features_num_features
            self.tabular_data = TabularData(
                input_size=input_size,
                inner_size=embedding_size,
                bias=bias,
                dropout=dropout,
            )


            print("using tabular transformer")
        self.use_tabular_transformer = use_tabular_transformer
        # self.global_rnn = SimpleRNN(128)
        self.use_exp = use_exp

    def forward(self, x, t, metadata, features):

        # if self.use_velocity:
        #    x_out = x_out + self.velocity(x,t)

        vel = self.velocity(x, t, use_exp=self.use_exp) if self.use_velocity else 0
        acc = (
            self.acceleration(x, t, use_exp=self.use_exp)
            if self.use_acceleration
            else 0
        )
        stats = self.stats(x) if self.use_stats else 0
        if all([self.use_metadata, not self.use_features]):
            tabular = self.tabular_data(x,metadata)
        elif all([not self.use_metadata,self.use_features]):
            tabular = self.tabular_data(x,features)
        elif all([self.use_metadata, self.use_features]):
            tabular_data = torch.concat([metadata, features], axis = 1)
            tabular = self.tabular_data(x, tabular_data)
        else:
            tabular = 0

        if self.use_conv:
            x_out = self.conv(x.permute(0,2,1)).permute(0,2,1)
            x_out = self.linear_x(x_out)
        else:
            x_out = self.linear_x(x)
        alpha, beta = self.timefilm_coeffs(t)

        x_out = (
            x_out * alpha + beta + vel + acc + stats + tabular
        )
        return self.dropout(x_out)



class Velocity(nn.Module):
    def __init__(self, input_size, embedding_size, inner_size,bias, dropout):
        super().__init__()
        self.linear_vel = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(
                in_features=inner_size, out_features=embedding_size, bias=bias
            ),

        )

    def forward(self, x, t, use_exp):
        _, _, dxdt = d_dt(x, t, use_exp)
        return self.linear_vel(dxdt)


class Acceleration(nn.Module):
    def __init__(self, input_size, embedding_size,inner_size, bias, dropout):
        super().__init__()
        self.linear_acc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(
                in_features=inner_size, out_features=embedding_size, bias=bias
            ),
        )

    def forward(self, x, t, use_exp):
        dx, dt, _ = d_dt(x, t, use_exp)
        _, _, dxdtdt = d_dt(dx, t, use_exp)
        return self.linear_acc(dxdtdt)


class AlphaCoeffs(nn.Module):
    def __init__(
        self,
        n_harmonics,
        embedding_size,
        dropout,
        norm,
        gelu,
    ):
        super().__init__()
        self.register_parameter(
            "alpha_sincos", nn.Parameter(torch.randn(2 * n_harmonics, embedding_size))
        )
        layers = []
        #changed order of fropout
        layers.extend([nn.Dropout(dropout)]) if dropout is not None else None
        layers.extend([nn.LayerNorm(embedding_size)]) if norm else None
        layers.extend([nn.GELU()]) if gelu else None
        self.normalize = nn.Sequential(*layers)

    def forward(self, embedding):
        return self.normalize(torch.matmul(embedding, self.alpha_sincos))


class BetaCoeffs(nn.Module):
    def __init__(
        self,
        n_harmonics,
        embedding_size,
        dropout,
        norm,
        gelu,
    ):
        super().__init__()
        self.register_parameter(
            "beta_sincos", nn.Parameter(torch.randn(2 * n_harmonics, embedding_size))
        )
        layers = []
        layers.extend([nn.Dropout(dropout)]) if dropout is not None else None
        layers.extend([nn.LayerNorm(embedding_size)]) if norm else None

        layers.extend([nn.GELU()]) if gelu else None

        self.normalize = nn.Sequential(*layers)

    def forward(self, embedding):
        return self.normalize(torch.matmul(embedding, self.beta_sincos))


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



class TimeFilmCoeffs(nn.Module):
    def __init__(
        self, n_harmonics, embedding_size, Tmax, dropout, norm, gelu, bias, exponential
    ):
        super().__init__()
        self.alpha_coeffs = AlphaCoeffs(
            n_harmonics, embedding_size, dropout, norm, gelu
        )
        self.bias = bias
        if self.bias:
            self.beta_coeffs = BetaCoeffs(
                n_harmonics, embedding_size, dropout, norm, gelu
            )
        self.n_harmonics = n_harmonics
        self.register_buffer(
            "ar",
            torch.tensor(2 * torch.pi)
            * (torch.arange(1, n_harmonics + 1).unsqueeze(0).unsqueeze(0)),
        )
        self.register_buffer("Tmax", torch.tensor(Tmax, dtype=float))
        self.exponential = exponential
        self.dropout = nn.Dropout(0.0)

    def get_sin_cos(self, t):
        sin = self.dropout(torch.sin(t))
        cos = self.dropout(torch.cos(t))
        if self.exponential:
            return torch.exp(sin).masked_fill_(sin == 0, 0), torch.exp(
                cos.masked_fill_(cos == 1, 0)
            )

        return sin, cos

    def forward(self, t):
        t = self.ar * t.repeat(1, 1, self.n_harmonics) / self.Tmax

        sin_emb, cos_emb = self.get_sin_cos(t)
        emb = torch.concat([sin_emb, cos_emb], dim=-1)
        return (
            (self.alpha_coeffs(emb), self.beta_coeffs(emb))
            if self.bias
            else (self.alpha_coeffs(emb), 0)
        )

class TabularData(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(inner_size, inner_size, bias=bias),
        )

    def forward(self, x, tabular):
        assert tabular is not None, "tabular data not provided."
        tabular = tabular.unsqueeze(-2).repeat(1, x.size(1), 1)
        mask = x != 0
        tabular = tabular * mask
        tabular = self.linear(tabular)
        return tabular

class Metadata(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        self.linear_metadata = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(inner_size, inner_size, bias=bias),
        )

    def forward(self, x, metadata):
        assert metadata is not None, "Metadata not provided."
        metadata = metadata.unsqueeze(-2).repeat(1, x.size(1), 1)
        mask = x != 0
        metadata = metadata * mask
        metadata = self.linear_metadata(metadata)
        return metadata

class Features(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        self.linear_features = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(inner_size, inner_size, bias=bias),
        )

    def forward(self, x, features):
        assert features is not None, "Metadata not provided."
        features = features.unsqueeze(-2).repeat(1, x.size(1), 1)
        mask = x != 0
        features = features * mask
        features = self.linear_features(features)
        return features

class Stats(nn.Module):
    def __init__(
        self,
        embedding_size,
        min_: bool = True,
        max_: bool = True,
        mean: bool = True,
        std: bool = True,
        range_: bool = True,
        negative_count: bool = True,
        positive_count: bool = True,
        q1=False,
        q2=False,
        q3=False,
        bias=False,
    ):
        super().__init__()
        stats = [
            min_,
            max_,
            mean,
            std,
            range_,
            positive_count,
            negative_count,
            q1,
            q2,
            q3,
        ]
        self.project_stats = nn.Sequential(
            nn.Linear(in_features=sum(stats), out_features=embedding_size, bias=bias),
            nn.Dropout(0.01),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(
                in_features=embedding_size, out_features=embedding_size, bias=bias
            ),
        )
        self.zero_ignore = ZeroIgnoredStats(dim=1, keepdim=False)

    def forward(self, x):
        x_ = x.clone()
        masked = x_.masked_fill(x_ == 0, float("-inf"))
        max_ = torch.argmax(masked, dim=1)
        masked = x_.masked_fill(x_ == 0, float("inf"))
        min_ = torch.argmin(masked, dim=1)
        negative_count = (x < 0).sum(dim=-2)
        pos_count = (x > 0).sum(dim=-2)
        stats = [
            max_,
            min_,
            negative_count,
            pos_count,
            max_ - min_,
        ]
        stats.extend(self.zero_ignore(x))
        stats = torch.concat(stats, dim=-1)
        stats = stats.unsqueeze(-2).repeat(1, x.size(1), 1)
        mask = x != 0
        stats = stats * mask
        return self.project_stats(stats)
