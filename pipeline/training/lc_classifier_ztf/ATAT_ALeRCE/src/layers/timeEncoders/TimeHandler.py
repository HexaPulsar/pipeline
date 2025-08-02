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
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, x, t, mask, metadata=None, features=None):

        x_mod = []
        t_mod = []
        m_mod = []

        for i in range(x.shape[-1]):
            slices_x = [slice(None)] * (x.dim() - 1) + [slice(i, i + 1)]
            slices_t = [slice(None)] * (t.dim() - 1) + [slice(i, i + 1)]
            slices_m = [slice(None)] * (mask.dim() - 1) + [slice(i, i + 1)]

            if x.dim() != t.dim():
                x_band = self.time_encoders[i](
                    x[slices_x], t[slices_t], metadata=metadata, features=features
                )
            else:
                x_band = self.time_encoders[i](
                    x[slices_x], t[slices_t], metadata=metadata, features=features
                )

            t_band = t[slices_t]
            m_band = mask[slices_m]

            x_mod.append(x_band)
            t_mod.append(t_band)
            m_mod.append(m_band)

        x_mod = torch.cat(x_mod, axis=1)
        m_mod = torch.cat(m_mod, axis=1)
        t_mod = torch.cat(t_mod, axis=1)
        # x_mod.reshape(x_mod.size(0),x_mod.size(1)//subsequence_length,subsequence_length, embedding_size)
        return (x_mod, m_mod, t_mod)
