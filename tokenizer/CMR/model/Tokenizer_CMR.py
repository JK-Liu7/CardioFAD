import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from LeanVAE.autoencoder import *
from einops import rearrange
from monai.networks.nets import UNet
from timm.models.vision_transformer import VisionTransformer


def sample_gumbel(shape, device=None, dtype=None, eps: float = 1e-20):
    """Sample standard Gumbel(0,1) noise."""
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_st_topk(logits: torch.Tensor,
                           K: int,
                           tau: float = 1.0):
    """
    Gumbel-Softmax straight-through top-k.

    Args:
        logits: [B, T] unnormalized scores for each time step.
        K: number of keyframes to select.
        tau: temperature for Gumbel-Softmax (lower -> sharper).

    Returns:
        mask_st: [B, T]  -- straight-through k-hot mask (forward hard, backward soft)
        indices: [B, K]  -- selected keyframe indices (for logging / inference)
        y_soft: [B, T]   -- underlying soft weights (gradient carrier)
    """
    B, T = logits.shape
    device = logits.device
    dtype = logits.dtype


    g = sample_gumbel((B, T), device=device, dtype=dtype)

    y_soft = F.softmax((logits + g) / tau, dim=-1)


    topk = torch.topk(y_soft, K, dim=-1)
    indices = topk.indices

    y_hard = torch.zeros_like(y_soft)

    y_hard.scatter_(1, indices, 1.0)


    mask_st = y_hard + y_soft - y_soft.detach()
    return mask_st, indices, y_soft


class KeyframeSelector(nn.Module):
    """
    Keyframe selector in latent space.

    Input:
      z_time  [B, T, D]  - ECG/pseudo-ECG tokens (MUST have gradients)
      z_traj  [B, T, D]  - visual tokens (OPTIONAL; will be stop-gradient if used)

    Output:
      logits  [B, T]
      q       [B, T]
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_visual: bool = False,
    ):
        super().__init__()
        self.use_visual = use_visual
        in_dim = latent_dim * (2 if use_visual else 1)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, z_time: torch.Tensor, z_traj: torch.Tensor = None):
        """
        z_time: [B, T, D]
        z_traj: [B, T, D] (optional)
        """
        if self.use_visual:
            assert z_traj is not None, "use_visual=True requires z_traj"

            z_traj = z_traj.detach()
            x_in = torch.cat([z_traj, z_time], dim=-1)
        else:
            x_in = z_time

        x = self.input_proj(x_in)
        x = self.encoder(x)
        logits = self.score_head(x).squeeze(-1)
        q = torch.softmax(logits, dim=-1)
        return logits, q


class Keyframe2Trajectory(nn.Module):
    """
    Keyframe selector that runs in the cardiac latent space.
    Input:
      z_traj   [B, T, D]  - cardiac MRI tokens
      idx_k   [B, K]  - indices of keyframes
    Output:
      loss_rec
    """
    def __init__(
        self,
        latent_dim,
        depth,
        channels,
        strides,
        num_frames,
        num_keyframes
    ):
        super().__init__()
        self.keyframe_rec = UNet(spatial_dims=3,
                                 in_channels=int(latent_dim * depth + 1),
                                 out_channels=int(latent_dim * depth),
                                 channels=channels,
                                 strides=strides)
        self.num_frames = num_frames
        self.num_keyframes = num_keyframes

    def forward(self, z_traj, logits):
        """
        full latent: z_traj [B * T, C, D, H, W]
        keyframe indices: idx_k [B, K]
        """
        z_traj = rearrange(z_traj, "(b t) c d h w -> b c t d h w", t=self.num_frames)
        B, C, T, D, H, W = z_traj.shape

        z_traj_ = rearrange(z_traj, "b c t d h w -> b (c d) t h w")

        mask_st, indices, y_soft = gumbel_softmax_st_topk(logits, K=self.num_keyframes, tau=1.0)

        mask_3d = mask_st.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        mask_3d = mask_3d.expand(-1, 1, -1, H, W)

        z_traj_ = z_traj_ * mask_3d
        z_in = torch.cat([z_traj_, mask_3d], dim=1)

        z_rec = self.keyframe_rec(z_in)
        z_rec = z_rec.view(B, C, T, D, H, W)

        mask_nonkey = 1.0 - mask_st
        mask_nonkey = mask_nonkey.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        diff2 = (z_rec - z_traj) ** 2
        mask_full = mask_nonkey.expand_as(diff2)

        num_voxels = mask_full.sum()
        loss_rec = (diff2 * mask_full).sum() / (num_voxels + 1e-8)

        return loss_rec


class Tokenizer_CMR(nn.Module):
    def __init__(
            self,
            args,

    ):
        super().__init__()

        self.args = args

        self.MRI_tok = LeanVAE(args)
        self.MRI_proj = nn.Sequential(
            nn.Linear(self.args.mri_latent_dim, self.args.proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.proj_dim * 2, self.args.proj_dim),
        )

        self.ECG_encoder = VisionTransformer(img_size=self.args.ECG_size,
                                        patch_size=self.args.ECG_enc_patch,
                                        in_chans=1,
                                        embed_dim=args.ECG_enc_dim,
                                        depth=args.ECG_enc_depth,
                                        num_heads=8)
        self.ECG_proj = nn.Sequential(
            nn.Linear(self.args.ECG_enc_dim, self.args.proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.proj_dim * 2, self.args.proj_dim),
        )

        self.keyframe_selector = KeyframeSelector(self.args.proj_dim,
                                                  self.args.selector_hidden,
                                                  self.args.selector_layers,
                                                  self.args.selector_heads,
                                                  use_visual=False)

        self.latent_depth = (self.args.volume_depth - 1) // 4 + 1
        self.keyframe2trajectory = Keyframe2Trajectory(self.args.mri_latent_dim, self.latent_depth, self.args.unet_channels, self.args.unet_strides,
                                                       self.args.num_frames, self.args.num_keyframes)

    def coverage_regularization_bins(self,
                                     q: torch.Tensor,
                                     num_bins: int = 4) -> torch.Tensor:

        B, T = q.shape
        p = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        bin_size = T // num_bins
        masses = []
        start = 0
        for b in range(num_bins):
            end = start + bin_size
            if b == num_bins - 1:
                end = T
            mass_b = p[:, start:end].sum(dim=1)
            masses.append(mass_b)
            start = end
        masses = torch.stack(masses, dim=1)

        p_bin = masses / (masses.sum(dim=1, keepdim=True) + 1e-8)

        log_p_bin = torch.log(p_bin + 1e-8)
        log_u = math.log(1.0 / num_bins)
        kl_div = (p_bin * (log_p_bin - log_u)).sum(dim=1)

        cov_loss = kl_div.mean()
        return cov_loss


    def get_keyframe_indices(self, q: torch.Tensor) -> torch.Tensor:
        B, T = q.shape
        K = min(self.args.num_keyframes, T)
        _, indices_unsorted = torch.topk(q, k=K, dim=-1, largest=True, sorted=False)
        keyframe_idx, _ = torch.sort(indices_unsorted, dim=-1)
        return keyframe_idx


    def build_alignment_matrix(self, q: torch.Tensor,) -> torch.Tensor:
        B, T = q.shape
        device = q.device

        phi = torch.linspace(0.0, 1.0, T, device=device)
        phase_k = phi.view(1, T, 1)
        phase_t = phi.view(1, 1, T)

        d_time = torch.abs(phase_k - phase_t)
        d_time = torch.minimum(d_time, 1.0 - d_time)
        d_time = d_time.expand(B, -1, -1)

        q_k = q.unsqueeze(2)
        q_t = q.unsqueeze(1)
        imp = 0.5 * (q_k + q_t)

        imp_mean = imp.mean(dim=(1, 2), keepdim=True)
        imp_norm = imp / (imp_mean + 1e-8)

        d2 = d_time.pow(2) / (imp_norm + 1e-8)
        A_unnorm = torch.exp(-d2 / (2.0 * (self.args.sigma_align ** 2) + 1e-8))
        A = A_unnorm / (A_unnorm.sum(dim=-1, keepdim=True) + 1e-8)
        return A


    def local_contrastive_loss(
        self,
        z_traj: torch.Tensor,
        z_time: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = z_traj.shape

        z_traj = F.normalize(z_traj, dim=-1)
        z_time = F.normalize(z_time, dim=-1)
        S = torch.einsum("bkd, btd -> bkt", z_traj, z_time) / math.sqrt(D)

        A = self.build_alignment_matrix(q)

        logits_e2v = S / self.args.tau
        log_probs_e2v = logits_e2v.log_softmax(dim=-1)

        w_t = (1 + self.args.lambda_q * q).unsqueeze(1)

        loss_e2v = -(A * w_t * log_probs_e2v).sum(dim=-1).mean()

        S_rev = S.transpose(1, 2).contiguous()
        logits_v2e = S_rev / self.args.tau
        log_probs_v2e = logits_v2e.log_softmax(dim=-1)

        A_rev = A.transpose(1, 2).contiguous()
        w_t_row = w_t.transpose(1, 2).contiguous()
        loss_v2e = -(A_rev * w_t_row * log_probs_v2e).sum(dim=-1).mean()
        loss_cl = 0.5 * (loss_e2v + loss_v2e)

        return loss_cl


    def forward(self, x_mri: torch.Tensor, x_ecg: torch.Tensor, stage: str = "full"):
        _, mri_rec, z_traj, _, _, posterior = self.MRI_tok(x_mri)
        loss_kl = posterior.kl().mean()

        if stage == "mri_pretrain":
            return {
                "loss_kl": loss_kl,
                "rec_mri": mri_rec,
                "z_mri": z_traj,
            }

        z_traj_s = rearrange(z_traj, "(b t) c d h w -> b t c h w d", t=self.args.num_frames)
        z_traj_ = z_traj.mean(dim=(2, 3, 4))
        z_traj_ = rearrange(z_traj_, "(b t) c -> b t c", t=self.args.num_frames)

        z_time = self.ECG_encoder.forward_features(x_ecg)
        z_time = z_time[:, 1:, :]

        z_traj_proj = self.MRI_proj(z_traj_)
        z_time_proj = self.ECG_proj(z_time)

        logits, q = self.keyframe_selector(z_time_proj, z_traj_proj)
        idx_k = self.get_keyframe_indices(q)

        loss_conv = self.coverage_regularization_bins(q, self.args.num_bins)

        loss_k2t = self.keyframe2trajectory(z_traj, logits)

        loss_cl = self.local_contrastive_loss(z_traj_proj, z_time_proj, q)

        return {
            'loss_kl': loss_kl,
            'loss_conv': loss_conv,
            'loss_k2t': loss_k2t,
            'loss_cl': loss_cl,
            'z_mri': z_traj_s,
            'z_ecg': z_time,
            'rec_mri': mri_rec,
            'keyframe_q': q,
            'keyframe_idx': idx_k
        }


