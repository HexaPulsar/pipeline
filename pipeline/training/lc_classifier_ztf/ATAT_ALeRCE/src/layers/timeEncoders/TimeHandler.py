import torch
import torch.nn as nn
from .TimeFilmModified import EarlyFusionEncoder


class TimeHandler(nn.Module):
    def __init__(
        self,
        num_bands=2,
        input_size=1,
        embedding_size=64,
        projections_inner_size=32,
        Tmax=1500.0,
        metadata_num_features=6,
        features_num_features=0,
        use_acceleration: bool = False,
        use_velocity: bool = False,
        use_stats: bool = False,
        use_metadata: bool = False,
        use_features: bool = False,
        use_timefilm_gelu: bool = False,
        use_timefilm_norm: bool = False,
        use_exp: bool = False,
        use_conv : bool = False,
        use_tabular_transformer: bool = False,
        use_anomaly_gate: bool = False,
    ):
        super(TimeHandler, self).__init__()

        self.embedding_size = embedding_size
        self.time_encoders = nn.ModuleList(
            [
                EarlyFusionEncoder(
                    embedding_size=embedding_size,
                    projections_inner_size=projections_inner_size,
                    input_size=input_size,
                    Tmax=Tmax,
                    metadata_num_features=metadata_num_features,
                    features_num_features=features_num_features,
                    use_acceleration=use_acceleration,
                    use_velocity=use_velocity,
                    use_metadata=use_metadata,
                    use_stats=use_stats,
                    use_features=use_features,
                    use_timefilm_norm=use_timefilm_norm,
                    use_timefilm_gelu=use_timefilm_gelu,
                    use_exp=use_exp,
                    use_conv =use_conv,
                    use_tabular_transformer=use_tabular_transformer,
                    use_anomaly_gate=use_anomaly_gate,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, x, t, mask, metadata=None, features=None):
        x_mod = [
            self.time_encoders[i](x[..., i:i+1], t[..., i:i+1], metadata=metadata, features=features)
            for i in range(x.shape[-1])
        ]
        return (
            torch.cat(x_mod, dim=1),
            torch.cat([mask[..., i:i+1] for i in range(mask.shape[-1])], dim=1),
            torch.cat([t[..., i:i+1] for i in range(t.shape[-1])], dim=1),
        )
