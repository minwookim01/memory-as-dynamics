"""Predictive-memory module for RPM.

PredictiveMemory realizes the predictive-memory branch of Reliability-Guided
Predictive Memory (RPM). Given a short temporal tracklet of observation-driven memory
features, it rolls the state forward with state-space (Mamba) dynamics to
produce (i) a predictive latent ``z_tilde`` used for reliability evaluation
and (ii) a decoder-aligned dense prompt used by the SAM2 mask decoder.

The same module is used for both training and inference:

* Training only learns the prompt: ``forward(tracklet)`` returns the predictive
  latent and the predictive logit, which are supervised against the next-frame
  features / mask. No reliability gating is applied.
* Inference passes the current-frame feature so the reliability consistency
  check (``d_feature``) can suppress unreliable predictions
  (``RPM_prompt_mode=True``).
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block as Mamba2Block

from sam2.modeling.sam2_utils import MLP, LayerNorm2d


def sinusoidal_time_embedding(T, dim, device):
    """Sinusoidal temporal positional embedding.

    Args:
        T: number of frames in the tracklet.
        dim: embedding dimension (256).
    Returns:
        Tensor of shape ``[T, dim]``.
    """
    t = torch.arange(T, device=device).float()                      # [T]
    div = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * (-math.log(10000.0) / dim)
    )                                                               # [dim/2]

    pe = torch.zeros(T, dim, device=device)                         # [T, dim]
    pe[:, 0::2] = torch.sin(t[:, None] * div)
    pe[:, 1::2] = torch.cos(t[:, None] * div)
    return pe


class PredictiveMemory(nn.Module):
    """Predictive-memory module for RPM.

    Args:
        RPM_resolution: input crop size used to build the tracklet (224 or 384).
            Determines the latent grid (224 -> 14x14, 384 -> 24x24).
        RPM_prompt_mode: if True, apply the reliability consistency check at
            inference time and drop the predictive prompt when prediction and
            observation disagree. Training leaves this False.
        RPM_feature: ``d_feature`` threshold for predictive activation; the
            prompt is suppressed when ``d_feature > RPM_feature``.
        dim: feature dimension.
        depth: number of stacked Mamba2 blocks.
    """

    def __init__(
        self,
        RPM_resolution: int = 224,
        RPM_prompt_mode: bool = False,
        RPM_feature: float = 0.0,
        dim: int = 256,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        depth: int = 24,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.RPM_resolution = RPM_resolution
        self.RPM_prompt_mode = RPM_prompt_mode
        self.RPM_feature = RPM_feature

        self.norm_mamba = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)

        # State-space latent dynamics over the tracklet memory tokens.
        mamba_factory = partial(Mamba2, d_state=128, d_conv=4, expand=2)
        self.mamba_layers = nn.ModuleList([
            Mamba2Block(
                dim=dim,
                mixer_cls=mamba_factory,
                mlp_cls=nn.Identity,
                norm_cls=norm_layer,
                fused_add_norm=True,
                residual_in_fp32=True,
            )
            for _ in range(depth)
        ])

        # Predictive-prompt generator: normalization + two-layer MLP.
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dim,
            num_layers=2,
            activation=act_layer,
        )

        # Spatial projection / upsampling to a decoder-aligned dense prompt.
        # 224 -> latent 14x14 needs two upsamples (14 -> 28 -> 56);
        # 384 -> latent 24x24 needs one (24 -> 48), refined to 64x64 in forward.
        if self.RPM_resolution == 224:
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
                LayerNorm2d(dim // 2),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
                LayerNorm2d(dim // 4),
                nn.GELU(),
            )
        elif self.RPM_resolution == 384:
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
                LayerNorm2d(dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
                LayerNorm2d(dim // 4),
                nn.GELU(),
            )
        else:
            raise ValueError(
                f"Unsupported RPM_resolution={self.RPM_resolution}. Use 224 or 384."
            )

        self.logit = nn.Conv2d(dim // 4, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(
        self,
        tracklet: torch.Tensor,
        pred_iou=None,
        pred_obj=None,
        current_feature=None,
        pred_mask=None,
    ):
        """Generate the predictive latent and dense prompt from a tracklet.

        Args:
            tracklet: observation-driven memory features ``[T, dim, H, W]``.
            current_feature: current-frame feature ``[1, dim, H, W]`` used for the
                reliability consistency check (inference only).
            pred_iou, pred_obj, pred_mask: decoder-derived signals carried from
                the previous frame (reserved for reliability logging / extensions).
        Returns:
            ``(latent, logit)`` where ``latent`` is the predictive latent
            ``z_tilde`` (``[1, dim, H, W]``) and ``logit`` is the dense-prompt
            logit (``[1, 1, 64, 64]``), or ``None`` if the prompt is suppressed.
        """
        T, C, H, W = tracklet.shape
        device = tracklet.device

        # Add temporal positional embedding to preserve frame order.
        t_emb = sinusoidal_time_embedding(T, self.dim, device)      # [T, dim]
        t_emb = t_emb[:, :, None, None]                             # [T, dim, 1, 1]
        x = tracklet + 0.1 * t_emb

        # Flatten tracklet to a single token sequence: [1, T * H * W, dim].
        x = x.flatten(2).transpose(1, 2)                            # [T, H*W, dim]
        x = x.reshape(1, -1, x.shape[-1])                           # [1, T*H*W, dim]

        # State-space dynamics over the tracklet.
        hidden_states = x
        residual = None
        for layer in self.mamba_layers:
            hidden_states, residual = layer(hidden_states, residual)
        x_mamba = self.norm_mamba(hidden_states + residual)         # [1, L, dim]

        # Softmax-weighted temporal pooling -> single predictive latent.
        B = x_mamba.shape[0]
        x_mamba = x_mamba.view(B, T, H * W, self.dim)
        w = torch.softmax(x_mamba.mean(-1), dim=1)
        mamba_q = (x_mamba * w.unsqueeze(-1)).sum(dim=1)            # [B, H*W, dim]

        mamba_q = self.mlp(self.norm_mlp(mamba_q))
        latent = mamba_q.transpose(1, 2).reshape(B, self.dim, H, W)  # [B, dim, H, W]

        # Spatial projection / upsampling to the decoder-aligned dense prompt.
        x = self.decoder(latent)
        x = F.interpolate(x, size=(64, 64), mode="bilinear")
        logit = self.logit(x).float()                               # [B, 1, 64, 64]

        # Reliability-guided predictive activation (inference only).
        if self.RPM_prompt_mode and current_feature is not None:
            phi_current = current_feature.mean(dim=(2, 3))          # [1, dim]
            phi_predict = latent.mean(dim=(2, 3))                   # [1, dim]
            d_feature = 1.0 - F.cosine_similarity(phi_predict, phi_current, dim=1)
            if d_feature > self.RPM_feature:
                logit = None  # prediction disagrees with observation -> suppress

        return latent, logit
