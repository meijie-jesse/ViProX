"""
RAFT optical-flow utilities for experiment 7.3 (explicit-flow baseline).

Inference follows the official princeton-vl/RAFT demo pipeline:
  InputPadder -> RAFT(iters=20, test_mode) -> unpad

Supports:
  - backend='torchvision' (default, no extra deps)
  - backend='official' (uses ./RAFT/core + downloaded .pth weights)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_OFFICIAL_SMALL_CKPT = os.path.join(PROJECT_ROOT, 'RAFT', 'models', 'raft-small.pth')
DEFAULT_OFFICIAL_FULL_CKPT = os.path.join(PROJECT_ROOT, 'RAFT', 'models', 'raft-things.pth')


class InputPadder:
    """Pads images so H and W are divisible by 8 (from princeton-vl/RAFT)."""

    def __init__(self, dims, mode: str = 'sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def freeze_batchnorm(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def rgb01_to_raft_tensor(
    image_rgb01: torch.Tensor,
    backend: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Convert RGB in [0, 1] to backend-specific RAFT input."""
    image = image_rgb01.clamp(0.0, 1.0)
    if target_size is not None:
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
    if backend == 'torchvision':
        return 2.0 * image - 1.0
    if backend == 'official':
        return image * 255.0
    raise ValueError(f'Unknown RAFT backend: {backend}')


def imagenet_to_rgb01(image_norm: torch.Tensor, mean, std) -> torch.Tensor:
    return (image_norm * std + mean).clamp(0.0, 1.0)


def _resolve_official_checkpoint(checkpoint_path: Optional[str], small: bool) -> str:
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        return checkpoint_path
    default_path = DEFAULT_OFFICIAL_SMALL_CKPT if small else DEFAULT_OFFICIAL_FULL_CKPT
    if os.path.isfile(default_path):
        return default_path
    raise FileNotFoundError(
        'Official RAFT checkpoint not found. Run:\n'
        '  cd RAFT && ./download_models.sh\n'
        f'Expected one of: {checkpoint_path}, {default_path}'
    )


def _load_official_raft(checkpoint_path: str, small: bool, device: torch.device) -> nn.Module:
    raft_core = os.path.join(PROJECT_ROOT, 'RAFT', 'core')
    if raft_core not in sys.path:
        sys.path.insert(0, raft_core)
    from raft import RAFT

    class RaftArgs:
        small = small
        mixed_precision = False
        alternate_corr = False
        dropout = 0

    model = RAFT(RaftArgs())
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    freeze_batchnorm(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_torchvision_raft(small: bool, device: torch.device) -> nn.Module:
    if small:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    else:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    model.to(device)
    model.eval()
    freeze_batchnorm(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


class RaftFlowExtractor(nn.Module):
    """
    Frozen RAFT flow estimator aligned with the official demo settings.

    Accepts ImageNet-normalized ACT images or raw RGB in [0, 1] (precompute path).
    """

    def __init__(
        self,
        flow_image_size: Tuple[int, int] = (240, 320),
        num_iters: int = 20,
        backend: str = 'torchvision',
        checkpoint_path: Optional[str] = None,
        small: bool = True,
        input_format: str = 'imagenet',
    ):
        super().__init__()
        self.flow_image_size = flow_image_size
        self.num_iters = num_iters
        self.backend = backend
        self.small = small
        self.input_format = input_format

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if backend == 'official':
            ckpt = _resolve_official_checkpoint(checkpoint_path, small=small)
            self.raft = _load_official_raft(ckpt, small=small, device=device)
            print(f'Loaded official RAFT ({os.path.basename(ckpt)}), iters={num_iters}')
        elif backend == 'torchvision':
            self.raft = _load_torchvision_raft(small=small, device=device)
            print(f'Loaded torchvision RAFT-{"small" if small else "large"}, iters={num_iters}')
        else:
            raise ValueError(f'Unknown backend: {backend}')

        self.register_buffer('_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prepare_pair(
        self,
        image_prev: torch.Tensor,
        image_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputPadder]:
        if self.input_format == 'imagenet':
            prev_rgb = imagenet_to_rgb01(image_prev, self._mean, self._std)
            curr_rgb = imagenet_to_rgb01(image_curr, self._mean, self._std)
        elif self.input_format == 'rgb01':
            prev_rgb = image_prev
            curr_rgb = image_curr
        else:
            raise ValueError(f'Unknown input_format: {self.input_format}')

        prev = rgb01_to_raft_tensor(prev_rgb, self.backend, self.flow_image_size)
        curr = rgb01_to_raft_tensor(curr_rgb, self.backend, self.flow_image_size)
        padder = InputPadder(prev.shape, mode='sintel')
        prev, curr = padder.pad(prev, curr)
        return prev, curr, padder

    @torch.no_grad()
    def forward(
        self,
        image_prev: torch.Tensor,
        image_curr: torch.Tensor,
        flow_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_batch(image_prev, image_curr, flow_init=flow_init)

    @torch.no_grad()
    def forward_batch(
        self,
        image_prev: torch.Tensor,
        image_curr: torch.Tensor,
        flow_init: Optional[torch.Tensor] = None,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Batched RAFT inference. Inputs: [B, 3, H, W]."""
        if image_prev.ndim == 3:
            image_prev = image_prev.unsqueeze(0)
            image_curr = image_curr.unsqueeze(0)
        if image_prev.shape[0] != image_curr.shape[0]:
            raise ValueError('prev/curr batch size mismatch')

        self.raft.eval()
        freeze_batchnorm(self.raft)

        prev, curr, padder = self._prepare_pair(image_prev, image_curr)
        device_type = 'cuda' if prev.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=use_amp and device_type == 'cuda'):
            if self.backend == 'official':
                if flow_init is not None:
                    _, flow_up = self.raft(
                        prev, curr, iters=self.num_iters, flow_init=flow_init, test_mode=True
                    )
                else:
                    _, flow_up = self.raft(prev, curr, iters=self.num_iters, test_mode=True)
            else:
                flow_up = self.raft(prev, curr, num_flow_updates=self.num_iters)[-1]

        return padder.unpad(flow_up.float())

    @torch.no_grad()
    def forward_with_init(
        self,
        image_prev: torch.Tensor,
        image_curr: torch.Tensor,
        flow_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (flow, low_res_flow_for_warm_start)."""
        self.raft.eval()
        freeze_batchnorm(self.raft)

        prev, curr, padder = self._prepare_pair(image_prev, image_curr)
        next_init = None

        if self.backend == 'official':
            flow_low, flow_up = self.raft(
                prev, curr, iters=self.num_iters, flow_init=flow_init, test_mode=True
            )
            next_init = flow_low
            return padder.unpad(flow_up), next_init

        flow_preds = self.raft(prev, curr, num_flow_updates=self.num_iters)
        return padder.unpad(flow_preds[-1]), None


class FlowEncoder(nn.Module):
    """Encode multi-camera optical flow (+ compact motion stats) into hidden_dim."""

    def __init__(self, num_cams: int, out_dim: int = 256):
        super().__init__()
        self.num_cams = num_cams
        in_ch = 2 * num_cams
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # Per-camera (mean_u, mean_v, mean_speed) -> 3 * num_cams
        self.stat_proj = nn.Linear(3 * num_cams, 64)
        self.out_proj = nn.Sequential(
            nn.Linear(128 + 64, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _flow_stats(flow: torch.Tensor, num_cams: int) -> torch.Tensor:
        stats = []
        for cam_id in range(num_cams):
            cam_flow = flow[:, cam_id * 2:(cam_id + 1) * 2]
            u = cam_flow[:, 0]
            v = cam_flow[:, 1]
            speed = torch.sqrt(u * u + v * v + 1e-6)
            stats.extend([u.mean(dim=(1, 2)), v.mean(dim=(1, 2)), speed.mean(dim=(1, 2))])
        return torch.stack(stats, dim=1)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv(flow)
        stat_feat = self.stat_proj(self._flow_stats(flow, self.num_cams))
        return self.out_proj(torch.cat([conv_feat, stat_feat], dim=1))
