"""SHiP: add Gaussian noise to selected tokens of DiT blocks during the first denoising steps.

This module holds the model-agnostic parts of SHiP:
  - the config (which blocks / site / tokens, noise scale, and step window),
  - denoising-step tracking,
  - deterministic noise sampling tied to the image seed,
  - the perturbation itself and generic hook builders.

Model-specific parts (which tokens are target / reference / text, and which modules to hook)
live in models/<model>/ship_hooks.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

MaskFn = Callable[[int, torch.device], Optional[torch.Tensor]]

VALID_TARGET_BLOCKS = ("dual", "single", "both")
VALID_SITES = ("attn_out", "mlp_out", "block_input", "block_output")
VALID_TOKENS = ("target", "ref", "text")
VALID_NOISE_MODES = ("block_relative_std", "absolute")


@dataclass
class ShipConfig:
    enabled: bool = True
    target_blocks: str = "both"             # dual | single | both
    site: str = "attn_out"                  # attn_out | mlp_out | block_input | block_output
    selected_token: str = "target"          # target | ref | text
    noise_mode: str = "block_relative_std"  # block_relative_std | absolute
    noise_std: float = 3.0                  # lambda: relative to the block output std, or absolute
    early_steps: int = 3                    # k: number of perturbed denoising steps
    start_step: int = 0
    num_steps: int = 28


def build_ship_config(cfg: Dict[str, Any]) -> ShipConfig:
    ship_raw = cfg["SHIP"]
    noise_raw = ship_raw.get("NOISE", {})
    ship_cfg = ShipConfig(
        enabled=bool(ship_raw.get("ENABLED", True)),
        target_blocks=str(ship_raw.get("TARGET_BLOCKS", "both")).lower(),
        site=str(ship_raw.get("SITE", "attn_out")).lower(),
        selected_token=str(ship_raw.get("SELECTED_TOKEN", "target")).lower(),
        noise_mode=str(noise_raw.get("MODE", "block_relative_std")),
        noise_std=float(noise_raw.get("STD", 3.0)),
        early_steps=int(ship_raw.get("EARLY_STEPS", 3)),
        start_step=int(ship_raw.get("START_STEP", 0)),
        num_steps=int(cfg.get("NUM_STEPS", 28)),
    )
    for name, value, valid in (
        ("TARGET_BLOCKS", ship_cfg.target_blocks, VALID_TARGET_BLOCKS),
        ("SITE", ship_cfg.site, VALID_SITES),
        ("SELECTED_TOKEN", ship_cfg.selected_token, VALID_TOKENS),
        ("NOISE.MODE", ship_cfg.noise_mode, VALID_NOISE_MODES),
    ):
        if value not in valid:
            raise ValueError(f"SHIP.{name} must be one of {list(valid)}, got {value!r}")
    return ship_cfg


# =============================================================================
# Deterministic noise seeds
# =============================================================================

def _mix_u64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF


def _seed_for_noise(base_seed: int, step_idx: int, kind: int) -> int:
    x = base_seed & 0xFFFFFFFFFFFFFFFF
    x ^= (kind * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (step_idx * 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF
    x = _mix_u64(x)
    return int(x & 0x7FFFFFFFFFFFFFFF)


# =============================================================================
# State
# =============================================================================

class Ship:
    """Per-process SHiP state. Model-specific token layouts are stored by subclasses."""

    def __init__(self):
        self.cfg = ShipConfig(enabled=False)
        self.sample_seed: Optional[int] = None
        self.step_key: Optional[float] = None
        self.step_idx: int = -1
        self.early_steps: int = 0
        self.start_step: int = 0
        self.text_len: int = 0
        self._noise_cache: Dict[tuple, torch.Tensor] = {}
        self._hook_handles: List[Any] = []

    def configure(self, cfg: ShipConfig):
        self.cfg = cfg
        num_steps = max(int(cfg.num_steps), 1)
        early_steps = max(0, int(cfg.early_steps))
        self.early_steps = min(early_steps, num_steps) if early_steps > 0 else 0
        self.start_step = max(0, int(cfg.start_step))
        self.reset()

    def reset(self):
        self.step_key = None
        self.step_idx = -1
        self.text_len = 0
        self._noise_cache.clear()

    def start_sample(self, seed: int):
        """Call before generating each image; the noise is tied to the image seed."""
        self.reset()
        self.sample_seed = int(seed)

    def on_step(self, timestep: Any):
        """Called on every transformer forward; a new timestep value starts a new step."""
        if timestep is None:
            return
        try:
            step_key = (
                float(timestep.flatten()[0].detach().cpu().item())
                if isinstance(timestep, torch.Tensor)
                else float(timestep)
            )
        except Exception:
            return
        if self.step_key != step_key:
            self.step_key = step_key
            self.step_idx += 1
            self._noise_cache.clear()

    def in_early_window(self) -> bool:
        if not self.cfg.enabled or self.early_steps <= 0:
            return False
        return self.start_step <= self.step_idx < (self.start_step + int(self.early_steps))

    def _sample_noise(self, like: torch.Tensor, std: float, kind: int) -> torch.Tensor:
        shape = tuple(like.shape)
        device = like.device
        dtype = like.dtype

        seed = int(self.sample_seed) if self.sample_seed is not None else -1
        step_key = int(self.step_idx)
        cache_key = (seed, step_key, int(kind), device, dtype, shape)
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]

        if float(std) == 0.0:
            out = torch.zeros(shape, device=device, dtype=dtype)
            self._noise_cache[cache_key] = out
            return out

        want_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        if seed >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(_seed_for_noise(seed, step_key, kind=kind))
            sample = torch.randn(shape, device=device, dtype=want_dtype, generator=generator)
        else:
            sample = torch.randn(shape, device=device, dtype=want_dtype)

        # zero mean over the feature dimension
        if sample.shape[-1] > 1:
            sample = sample - sample.mean(dim=-1, keepdim=True)

        sample = sample * float(std)
        out = sample.to(dtype=dtype) if want_dtype != dtype else sample
        self._noise_cache[cache_key] = out
        return out

    def perturb(self, output: torch.Tensor, mask_1d: Optional[torch.Tensor], kind: int) -> torch.Tensor:
        """Add noise to the tokens selected by mask_1d (over dim 1) of a (B, N, D) tensor."""
        if mask_1d is None or mask_1d.numel() != output.shape[1] or not mask_1d.any():
            return output

        selected_clean = output[:, mask_1d, :]
        if selected_clean.numel() == 0:
            return output

        cfg = self.cfg
        clean_std = float(selected_clean.detach().float().std(unbiased=False).item())
        std = float(cfg.noise_std) * clean_std if cfg.noise_mode == "block_relative_std" else float(cfg.noise_std)

        noisy = output + self._sample_noise(like=output, std=std, kind=kind)
        mask_f = mask_1d[None, :, None].to(dtype=output.dtype, device=output.device)
        return output + mask_f * (noisy - output)

    def add_hook(self, handle: Any):
        self._hook_handles.append(handle)

    def clear_hooks(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()


# =============================================================================
# Generic hook builders
# =============================================================================

def get_tensor_arg(args: Tuple[Any, ...], kwargs: Dict[str, Any], name: str, pos: int) -> Any:
    x = kwargs.get(name, None)
    if x is None and len(args) > pos:
        x = args[pos]
    return x


def replace_tensor_arg(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: str,
    pos: int,
    value: torch.Tensor,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if name in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs[name] = value
        return args, new_kwargs
    if len(args) > pos:
        new_args = list(args)
        new_args[pos] = value
        return tuple(new_args), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs[name] = value
    return args, new_kwargs


def tensor_output_hook(ship: Ship, mask_fn: MaskFn, kind: int):
    """Forward hook for modules that return a single (B, N, D) tensor."""
    def hook(module, inputs, output):
        if not ship.in_early_window():
            return output
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            return output
        return ship.perturb(output, mask_fn(int(output.shape[1]), output.device), kind)

    return hook


def tuple_output_hook(ship: Ship, mask_fn: MaskFn, kind: int, index: int):
    """Forward hook for modules that return a tuple of tensors; perturbs output[index]."""
    def hook(module, inputs, output):
        if not ship.in_early_window():
            return output
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        x = output[index]
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            return output
        out_list = list(output)
        out_list[index] = ship.perturb(x, mask_fn(int(x.shape[1]), x.device), kind)
        return tuple(out_list) if isinstance(output, tuple) else out_list

    return hook


def input_pre_hook(ship: Ship, mask_fn: MaskFn, kind: int, name: str, pos: int):
    """Forward pre-hook (with_kwargs=True) that perturbs the block input `name`."""
    def hook(module, args, kwargs):
        if not ship.in_early_window():
            return None
        x = get_tensor_arg(args, kwargs, name, pos)
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            return None
        mask = mask_fn(int(x.shape[1]), x.device)
        if mask is None:
            return None
        return replace_tensor_arg(args, kwargs, name, pos, ship.perturb(x, mask, kind))

    return hook
