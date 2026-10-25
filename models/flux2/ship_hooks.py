"""SHiP hooks for FLUX.2.

Token layout: the image stream is [target tokens | reference tokens], where target tokens have
img_ids[:, 0] == 0 and reference images have img_ids[:, 0] = 10, 20, ...
Single-stream blocks see [text tokens | image tokens].

Single-stream blocks fuse attention and MLP: attn.to_out([attention | MLP]). As in FLUX.1,
attn_out perturbs the attention part and mlp_out the MLP part, both before the output projection.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ship import Ship, ShipConfig, input_pre_hook, tensor_output_hook, tuple_output_hook


class Flux2Ship(Ship):
    def __init__(self):
        super().__init__()
        self.img_target_mask: Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        self.img_target_mask = None

    def _has_reference(self) -> bool:
        mask = self.img_target_mask
        return mask is not None and bool(mask.any()) and not bool(mask.all())

    def _image_mask(self, n_img: int, offset: int, total: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over `total` tokens whose image tokens start at `offset`."""
        mask = self.img_target_mask
        if mask is None or mask.numel() != n_img or not self._has_reference():
            return None
        if self.cfg.selected_token == "target":
            img_mask = mask.to(device=device)
        elif self.cfg.selected_token == "ref":
            img_mask = ~mask.to(device=device)
        else:
            return None
        full_mask = torch.zeros(total, dtype=torch.bool, device=device)
        full_mask[offset : offset + n_img] = img_mask
        return full_mask

    def dual_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over one stream of a double-stream block (text or image)."""
        if self.cfg.selected_token == "text":
            if self.text_len <= 0 or int(seq_len) != int(self.text_len) or not self._has_reference():
                return None
            return torch.ones(seq_len, dtype=torch.bool, device=device)
        return self._image_mask(int(seq_len), 0, int(seq_len), device)

    def single_mask(self, total_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over the joint [text | image] sequence of single-stream blocks."""
        text_len = max(0, min(int(self.text_len), int(total_tokens)))
        if self.cfg.selected_token == "text":
            if text_len <= 0 or not self._has_reference():
                return None
            mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
            mask[:text_len] = True
            return mask
        n_img = max(int(total_tokens) - text_len, 0)
        return self._image_mask(n_img, text_len, int(total_tokens), device)


SHIP = Flux2Ship()


def _wrap_transformer_forward(transformer: nn.Module):
    """Track the denoising step, text length and target-token mask on every transformer call."""
    if hasattr(transformer, "_ship_transformer_orig_forward"):
        return
    transformer._ship_transformer_orig_forward = transformer.forward

    def forward(*args, **kwargs):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        timestep = kwargs.get("timestep", None)
        img_ids = kwargs.get("img_ids", None)
        if encoder_hidden_states is None and len(args) >= 2 and isinstance(args[1], torch.Tensor):
            encoder_hidden_states = args[1]
        if timestep is None and len(args) >= 3 and isinstance(args[2], (torch.Tensor, int, float)):
            timestep = args[2]
        if img_ids is None and len(args) >= 4 and isinstance(args[3], torch.Tensor):
            img_ids = args[3]

        SHIP.on_step(timestep)
        SHIP.text_len = int(encoder_hidden_states.shape[1]) if isinstance(encoder_hidden_states, torch.Tensor) else 0

        SHIP.img_target_mask = None
        if isinstance(img_ids, torch.Tensor):
            ids = img_ids[0] if img_ids.ndim == 3 else img_ids
            if ids.ndim == 2 and ids.shape[-1] >= 1:
                SHIP.img_target_mask = ids[:, 0] == 0

        try:
            return transformer._ship_transformer_orig_forward(*args, **kwargs)
        finally:
            SHIP.img_target_mask = None
            SHIP.text_len = 0

    transformer.forward = forward


def _dual_attn_output_hook(kind: int):
    """attn returns a tensor or a tuple of streams; perturb the first stream with a valid mask."""
    def hook(module, inputs, output):
        if not SHIP.in_early_window():
            return output
        if isinstance(output, torch.Tensor):
            return tensor_output_hook(SHIP, SHIP.dual_mask, kind)(module, inputs, output)
        if not isinstance(output, (tuple, list)):
            return output
        for j, x in enumerate(output):
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                continue
            mask = SHIP.dual_mask(int(x.shape[1]), x.device)
            if mask is not None:
                out_list = list(output)
                out_list[j] = SHIP.perturb(x, mask, kind)
                return tuple(out_list) if isinstance(output, tuple) else out_list
        return output

    return hook


def _single_attn_part_pre_hook(kind: int, attn_dim: int):
    """attn.to_out input is [attention | MLP] along features; perturb the attention part only."""
    def hook(module, args):
        if not SHIP.in_early_window():
            return None
        x = args[0] if args else None
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            return None
        mask = SHIP.single_mask(int(x.shape[1]), x.device)
        if mask is None:
            return None
        attn_part = SHIP.perturb(x[..., :attn_dim], mask, kind)
        return (torch.cat([attn_part, x[..., attn_dim:]], dim=-1),) + tuple(args[1:])

    return hook


def enable_ship_hooks(pipe_or_transformer: Any, cfg: ShipConfig):
    SHIP.clear_hooks()
    SHIP.configure(cfg)

    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    _wrap_transformer_forward(transformer)

    site = cfg.site
    text = cfg.selected_token == "text"

    if cfg.target_blocks in ("dual", "both"):
        for i, block in enumerate(transformer.transformer_blocks):
            if site == "attn_out":
                SHIP.add_hook(block.attn.register_forward_hook(_dual_attn_output_hook(kind=3000 + i)))
            elif site == "mlp_out":
                module = block.ff_context if text else block.ff
                SHIP.add_hook(module.register_forward_hook(tensor_output_hook(SHIP, SHIP.dual_mask, kind=1000 + i)))
            elif site == "block_input":
                name, pos = ("encoder_hidden_states", 1) if text else ("hidden_states", 0)
                hook = input_pre_hook(SHIP, SHIP.dual_mask, 8000 + i, name, pos)
                SHIP.add_hook(block.register_forward_pre_hook(hook, with_kwargs=True))
            elif site == "block_output":
                # block returns (text, image)
                hook = tuple_output_hook(SHIP, SHIP.dual_mask, kind=10000 + i, index=0 if text else 1)
                SHIP.add_hook(block.register_forward_hook(hook))

    if cfg.target_blocks in ("single", "both"):
        for i, block in enumerate(transformer.single_transformer_blocks):
            if site == "attn_out":
                hook = _single_attn_part_pre_hook(kind=2000 + i, attn_dim=int(block.attn.inner_dim))
                SHIP.add_hook(block.attn.to_out.register_forward_pre_hook(hook))
            elif site == "mlp_out":
                hook = tensor_output_hook(SHIP, SHIP.single_mask, kind=7000 + i)
                SHIP.add_hook(block.attn.mlp_act_fn.register_forward_hook(hook))
            elif site == "block_input":
                hook = input_pre_hook(SHIP, SHIP.single_mask, 9000 + i, "hidden_states", 0)
                SHIP.add_hook(block.register_forward_pre_hook(hook, with_kwargs=True))
            elif site == "block_output":
                SHIP.add_hook(block.register_forward_hook(tensor_output_hook(SHIP, SHIP.single_mask, kind=11000 + i)))
