"""SHiP hooks for Qwen-Image-Edit (Qwen-Image-Edit-2511, FireRed-Image-Edit-1.0).

All blocks are double-stream. Token layout: the image stream is [target tokens | reference tokens],
with lengths taken from the img_shapes argument of every transformer call.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ship import Ship, ShipConfig, input_pre_hook, tensor_output_hook, tuple_output_hook


def _shape_len(shape: Any) -> int:
    if isinstance(shape, (list, tuple)) and len(shape) >= 3:
        return int(shape[0]) * int(shape[1]) * int(shape[2])
    return 0


class QwenShip(Ship):
    def __init__(self):
        super().__init__()
        self.target_len: Optional[int] = None
        self.ref_len: int = 0

    def reset(self):
        super().reset()
        self.target_len = None
        self.ref_len = 0

    def set_layout(self, encoder_hidden_states: Any, img_shapes: Any):
        """img_shapes[0] = [(frames, h, w) of the target, (frames, h, w) of each reference, ...]."""
        if isinstance(encoder_hidden_states, torch.Tensor) and encoder_hidden_states.ndim == 3:
            self.text_len = int(encoder_hidden_states.shape[1])
        else:
            self.text_len = 0

        self.target_len, self.ref_len = None, 0
        if isinstance(img_shapes, list) and img_shapes and isinstance(img_shapes[0], (list, tuple)):
            lengths = [_shape_len(s) for s in img_shapes[0] if isinstance(s, (list, tuple)) and len(s) >= 3]
            if lengths:
                self.target_len, self.ref_len = lengths[0], sum(lengths[1:])

    def image_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.target_len is None or self.target_len <= 0 or self.ref_len <= 0:
            return None
        target_len = min(int(self.target_len), int(seq_len))
        ref_end = min(target_len + int(self.ref_len), int(seq_len))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.cfg.selected_token == "target":
            mask[:target_len] = True
        elif self.cfg.selected_token == "ref":
            if ref_end <= target_len:
                return None
            mask[target_len:ref_end] = True
        else:
            return None
        return mask

    def text_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.cfg.selected_token != "text" or self.text_len <= 0 or self.ref_len <= 0:
            return None
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[: min(int(self.text_len), int(seq_len))] = True
        return mask


SHIP = QwenShip()


def _wrap_transformer_forward(transformer: nn.Module):
    """Track the denoising step and token layout on every transformer call."""
    if hasattr(transformer, "_ship_transformer_orig_forward"):
        return
    transformer._ship_transformer_orig_forward = transformer.forward

    def forward(*args, **kwargs):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        timestep = kwargs.get("timestep", None)
        img_shapes = kwargs.get("img_shapes", None)
        if encoder_hidden_states is None and len(args) >= 2 and isinstance(args[1], torch.Tensor):
            encoder_hidden_states = args[1]
        if timestep is None and len(args) >= 4 and isinstance(args[3], (torch.Tensor, int, float)):
            timestep = args[3]
        if img_shapes is None and len(args) >= 5:
            img_shapes = args[4]

        SHIP.on_step(timestep)
        SHIP.set_layout(encoder_hidden_states, img_shapes)
        return transformer._ship_transformer_orig_forward(*args, **kwargs)

    transformer.forward = forward


def enable_ship_hooks(pipe_or_transformer: Any, cfg: ShipConfig):
    SHIP.clear_hooks()
    SHIP.configure(cfg)

    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    _wrap_transformer_forward(transformer)

    site = cfg.site
    text = cfg.selected_token == "text"
    mask_fn = SHIP.text_mask if text else SHIP.image_mask

    for i, block in enumerate(transformer.transformer_blocks):
        if site == "attn_out":
            # attn returns (image, text)
            SHIP.add_hook(block.attn.register_forward_hook(tuple_output_hook(SHIP, mask_fn, kind=1000 + i, index=1 if text else 0)))
        elif site == "mlp_out":
            module = block.txt_mlp if text else block.img_mlp
            SHIP.add_hook(module.register_forward_hook(tensor_output_hook(SHIP, mask_fn, kind=2000 + i)))
        elif site == "block_input":
            name, pos = ("encoder_hidden_states", 1) if text else ("hidden_states", 0)
            SHIP.add_hook(block.register_forward_pre_hook(input_pre_hook(SHIP, mask_fn, 3000 + i, name, pos), with_kwargs=True))
        elif site == "block_output":
            # block returns (text, image)
            SHIP.add_hook(block.register_forward_hook(tuple_output_hook(SHIP, mask_fn, kind=4000 + i, index=0 if text else 1)))
