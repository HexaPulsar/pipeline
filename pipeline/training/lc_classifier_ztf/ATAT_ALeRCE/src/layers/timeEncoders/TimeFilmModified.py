import torch
from torch import nn


def roll_tensor(v):
    v = torch.roll(v, shifts=(1,), dims=1)
    v[:, 0, :] = 0
    return v


def d_dt(x, t, use_exp=False, tmax=2048):
    x_rolled = roll_tensor(x)
    t_rolled = roll_tensor(t)
    dt = t - t_rolled
    dx = x - x_rolled
    dt_safe = dt.masked_fill(dt == 0, 1)
    ratio = dx / dt_safe
    if use_exp:
        return dx, dt, torch.exp(torch.sin(ratio))
    return dx, dt, torch.sin(ratio)


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
        use_anomaly_gate=False,
    ):
        super().__init__()
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_stats = use_stats
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_anomaly_gate = use_anomaly_gate
        self.dropout = nn.Dropout(dropout)
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
        inner_size = projections_inner_size
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=inner_size,
                    bias=bias,
                    kernel_size=5,
                    padding=2,
                )
            )
            self.linear_x = nn.Sequential(
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(
                    in_features=inner_size, out_features=embedding_size, bias=bias
                ),
            )
        else:
            self.linear_x = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=inner_size, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=inner_size, out_features=embedding_size, bias=bias
                ),
            )
        if use_velocity:
            self.velocity = Velocity(
                input_size,
                embedding_size=embedding_size,
                inner_size=inner_size,
                bias=bias,
                dropout=dropout,
            )
            self.norm_vel = nn.LayerNorm(embedding_size)
        if use_acceleration:
            self.acceleration = Acceleration(
                input_size=input_size,
                embedding_size=embedding_size,
                inner_size=inner_size,
                bias=bias,
                dropout=dropout,
            )
            self.norm_acc = nn.LayerNorm(embedding_size)
        if use_stats:
            self.stats = Stats(embedding_size)
            self.norm_stats = nn.LayerNorm(embedding_size)

        if use_anomaly_gate:
            self.anomaly_gate = AnomalyGate(
                input_size=input_size,
                embedding_size=embedding_size,
                inner_size=inner_size,
                bias=bias,
                dropout=dropout,
            )

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
            self.norm_tabular = nn.LayerNorm(embedding_size)
        self.use_tabular_transformer = use_tabular_transformer
        self.use_exp = use_exp

    def forward(self, x, t, metadata, features):
        if self.use_conv:
            x_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            x_out = self.linear_x(x_out)
        else:
            x_out = self.linear_x(x)

        alpha, beta = self.timefilm_coeffs(t)
        x_out = x_out * alpha + beta

        if self.use_velocity or self.use_acceleration or self.use_anomaly_gate:
            dx, dt, vel_ratio = d_dt(x, t, use_exp=self.use_exp)
            if self.use_velocity:
                x_out = x_out + self.norm_vel(self.velocity(dx, dt, vel_ratio))
            if self.use_acceleration:
                dx_acc, _, acc_ratio = d_dt(dx, t, use_exp=self.use_exp)
                x_out = x_out + self.norm_acc(self.acceleration(dx_acc, dt, acc_ratio))
            if self.use_anomaly_gate:
                x_out = x_out * self.anomaly_gate(dx, vel_ratio, acc_ratio)

        if self.use_stats:
            x_out = x_out + self.norm_stats(self.stats(x))

        if self.use_metadata or self.use_features:
            if all([self.use_metadata, not self.use_features]):
                tabular = self.tabular_data(x, metadata)
            elif all([not self.use_metadata, self.use_features]):
                tabular = self.tabular_data(x, features)
            else:
                tabular_data = torch.concat([metadata, features], axis=1)
                tabular = self.tabular_data(x, tabular_data)
            x_out = x_out + self.norm_tabular(tabular)

        return self.dropout(x_out)


class Velocity(nn.Module):
    def __init__(self, input_size, embedding_size, inner_size, bias, dropout):
        super().__init__()
        self.linear_vel = nn.Sequential(
            nn.Linear(in_features=3 * input_size, out_features=inner_size, bias=bias),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=inner_size, out_features=embedding_size, bias=bias),
        )

    def forward(self, dx, dt, vel_ratio):
        return self.linear_vel(torch.cat([dx, dt, vel_ratio], dim=-1))


class Acceleration(nn.Module):
    def __init__(self, input_size, embedding_size, inner_size, bias, dropout):
        super().__init__()
        self.linear_acc = nn.Sequential(
            nn.Linear(in_features=3 * input_size, out_features=inner_size, bias=bias),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=inner_size, out_features=embedding_size, bias=bias),
        )

    def forward(self, dx_acc, dt, acc_ratio):
        return self.linear_acc(torch.cat([dx_acc, dt, acc_ratio], dim=-1))


class AnomalyGate(nn.Module):
    def __init__(self, input_size, embedding_size, inner_size, bias, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=inner_size, bias=bias),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=inner_size, out_features=embedding_size, bias=bias),
        )

    def forward(self, dx, vel_ratio, acc_ratio):
        surprise = torch.cat([vel_ratio.abs(), acc_ratio.abs()], dim=-1)
        return torch.sigmoid(self.mlp(surprise))


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
        layers.extend([nn.Dropout(dropout)]) if dropout is not None else None
        layers.extend([nn.LayerNorm(embedding_size)]) if norm else None
        layers.extend([nn.GELU()]) if gelu else None
        self.normalize = nn.Sequential(*layers)

    def forward(self, embedding):
        out = torch.einsum('...i,ij->...j', embedding, self.alpha_sincos)
        return self.normalize(out)


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
        out = torch.einsum('...i,ij->...j', embedding, self.beta_sincos)
        return self.normalize(out)


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
            "ar_norm",
            (2 * torch.pi * torch.arange(1, n_harmonics + 1) / Tmax)
            .unsqueeze(0).unsqueeze(0),
        )
        self.exponential = exponential

    def get_sin_cos(self, t):
        sin = torch.sin(t)
        cos = torch.cos(t)
        if self.exponential:
            return torch.exp(sin).masked_fill_(sin == 0, 0), torch.exp(
                cos.masked_fill_(cos == 1, 0)
            )

        return sin, cos

    def forward(self, t):
        t = self.ar_norm * t.expand(-1, -1, self.n_harmonics)

        sin_emb, cos_emb = self.get_sin_cos(t)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return (
            (self.alpha_coeffs(emb), self.beta_coeffs(emb))
            if self.bias
            else (self.alpha_coeffs(emb), 0)
        )


class TabularData(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        expansion_size = inner_size * 4
        self.linear = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_size, expansion_size, bias=bias),
            nn.LayerNorm(expansion_size),
            nn.GELU(),
            nn.Linear(expansion_size, inner_size, bias=bias),
        )

    def forward(self, x, tabular):
        assert tabular is not None, "tabular data not provided."
        tabular = tabular.unsqueeze(1).expand(-1, x.size(1), -1)
        mask = x != 0
        tabular = tabular * mask
        tabular = self.linear(tabular)
        return tabular


class Metadata(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        expansion_size = inner_size * 4
        self.linear_metadata = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_size, inner_size, bias=bias),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(inner_size, inner_size, bias=bias),
        )

    def forward(self, x, metadata):
        assert metadata is not None, "Metadata not provided."
        metadata = metadata.unsqueeze(1).expand(-1, x.size(1), -1)
        mask = x != 0
        metadata = metadata * mask
        metadata = self.linear_metadata(metadata)
        return metadata


class Features(nn.Module):
    def __init__(self, input_size, inner_size, bias, dropout):
        super().__init__()
        expansion_size = inner_size * 4
        self.linear_features = nn.Sequential(
            nn.Linear(input_size, inner_size, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_size, inner_size, bias=bias),
            nn.LayerNorm(inner_size),
            nn.GELU(),
            nn.Linear(inner_size, inner_size, bias=bias),
        )

    def forward(self, x, features):
        assert features is not None, "Features not provided."
        features = features.unsqueeze(1).expand(-1, x.size(1), -1)
        mask = x != 0
        features = features * mask
        features = self.linear_features(features)
        return features


class Stats(nn.Module):
    # 9 features: max, min, range, mean, std, negative_frac, positive_frac, skewness, kurtosis
    _N_STATS = 9

    def __init__(self, embedding_size, bias=False):
        super().__init__()
        expansion_size = embedding_size * 4
        self.project_stats = nn.Sequential(
            nn.Linear(in_features=self._N_STATS, out_features=expansion_size, bias=bias),
            nn.LayerNorm(expansion_size),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(in_features=expansion_size, out_features=embedding_size, bias=bias),
            nn.LayerNorm(embedding_size),
        )
        self.residual_proj = nn.Linear(in_features=self._N_STATS, out_features=embedding_size, bias=bias)
        self.zero_ignore = ZeroIgnoredStats(dim=1, keepdim=True)

    def forward(self, x):
        mask = (x != 0).float()
        obs_count = mask.sum(dim=1, keepdim=True).clamp(min=1)

        masked_max = x.masked_fill(x == 0, float("-inf"))
        max_ = torch.max(masked_max, dim=1, keepdim=True)[0]

        masked_min = x.masked_fill(x == 0, float("inf"))
        min_ = torch.min(masked_min, dim=1, keepdim=True)[0]

        negative_frac = (x < 0).float().sum(dim=1, keepdim=True) / obs_count
        positive_frac = (x > 0).float().sum(dim=1, keepdim=True) / obs_count

        mean, std = self.zero_ignore(x)  # both [B, 1, input_size]
        std_safe = std.clamp(min=1e-6)

        centered = (x - mean) * mask
        skewness = (centered ** 3 * mask).sum(dim=1, keepdim=True) / obs_count / std_safe ** 3
        kurtosis = (centered ** 4 * mask).sum(dim=1, keepdim=True) / obs_count / std_safe ** 4

        stats = torch.cat([
            max_,
            min_,
            max_ - min_,
            mean,
            std,
            negative_frac,
            positive_frac,
            skewness,
            kurtosis,
        ], dim=-1)  # [B, 1, 9]

        stats = stats.expand(-1, x.size(1), -1) * mask
        return self.project_stats(stats) + self.residual_proj(stats)
