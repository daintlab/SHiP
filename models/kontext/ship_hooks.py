"""SHiP hooks for FLUX.1-Kontext.

Token layout: the image stream is [target tokens | reference tokens], where target tokens have
img_ids[:, 0] == 0. Single-stream blocks see [text tokens | image tokens].

Single-stream block interface depends on the diffusers version:
  - before 0.35: block(hidden_states=[text | image]) -> [text | image]
  - 0.35 and later: block(hidden_states=image, encoder_hidden_states=text) -> (text, image)
For block_input / block_output on single blocks, the later interface is handled by joining
[text | image], perturbing it exactly as in the earlier interface, and splitting it again.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ship import (
    Ship,
    ShipConfig,
    get_tensor_arg,
    input_pre_hook,
    replace_tensor_arg,
    tensor_output_hook,
    tuple_output_hook,
)


class KontextShip(Ship):
    def __init__(self):
        super().__init__()
        self.img_target_mask: Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        self.img_target_mask = None

    def _has_reference(self) -> bool:
        mask = self.img_target_mask
        return mask is not None and bool(mask.any()) and not bool(mask.all())

    def image_mask(self, n_img: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over image-stream tokens."""
        mask = self.img_target_mask
        if mask is None or mask.numel() != n_img or not self._has_reference():
            return None
        target_mask = mask.to(device=device)
        apply_mask = torch.zeros(n_img, dtype=torch.bool, device=device)
        if self.cfg.selected_token == "target":
            apply_mask |= target_mask
        if self.cfg.selected_token == "ref":
            apply_mask |= ~target_mask
        return apply_mask

    def text_mask(self, n_txt: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over text-stream tokens."""
        if self.cfg.selected_token != "text" or n_txt <= 0 or not self._has_reference():
            return None
        return torch.ones(n_txt, dtype=torch.bool, device=device)

    def single_mask(self, total_seq: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask over the joint [text | image] sequence of single-stream blocks."""
        txt_len = int(self.text_len)
        if txt_len <= 0 or txt_len >= total_seq:
            return None
        img_mask = self.image_mask(total_seq - txt_len, device)
        if img_mask is None:
            return None
        full_mask = torch.zeros(total_seq, dtype=torch.bool, device=device)
        if self.cfg.selected_token == "text":
            full_mask[:txt_len] = True
        full_mask[txt_len:] = img_mask
        return full_mask


SHIP = KontextShip()


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
        if timestep is None and len(args) >= 4 and isinstance(args[3], (torch.Tensor, int, float)):
            timestep = args[3]
        if img_ids is None and len(args) >= 5 and isinstance(args[4], torch.Tensor):
            img_ids = args[4]

        SHIP.on_step(timestep)

        if isinstance(encoder_hidden_states, torch.Tensor) and encoder_hidden_states.ndim == 3:
            SHIP.text_len = int(encoder_hidden_states.shape[1])
        else:
            SHIP.text_len = 0

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


def _takes_text_separately(block: nn.Module) -> bool:
    return "encoder_hidden_states" in inspect.signature(block.forward).parameters


def _perturb_joint(txt: torch.Tensor, img: torch.Tensor, kind: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Perturb [text | image] as one sequence and split it back into (text, image)."""
    x = torch.cat([txt, img], dim=1)
    mask = SHIP.single_mask(int(x.shape[1]), x.device)
    if mask is None:
        return None
    x = SHIP.perturb(x, mask, kind)
    t = txt.shape[1]
    return x[:, :t], x[:, t:]


def _split_single_input_pre_hook(kind: int):
    def hook(module, args, kwargs):
        if not SHIP.in_early_window():
            return None
        img = get_tensor_arg(args, kwargs, "hidden_states", 0)
        txt = get_tensor_arg(args, kwargs, "encoder_hidden_states", 1)
        if not isinstance(img, torch.Tensor) or not isinstance(txt, torch.Tensor) or img.ndim != 3 or txt.ndim != 3:
            return None
        joint = _perturb_joint(txt, img, kind)
        if joint is None:
            return None
        args, kwargs = replace_tensor_arg(args, kwargs, "encoder_hidden_states", 1, joint[0])
        return replace_tensor_arg(args, kwargs, "hidden_states", 0, joint[1])

    return hook


def _split_single_output_hook(kind: int):
    def hook(module, inputs, output):
        if not SHIP.in_early_window():
            return output
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            return output
        txt, img = output
        if not isinstance(txt, torch.Tensor) or not isinstance(img, torch.Tensor) or txt.ndim != 3 or img.ndim != 3:
            return output
        joint = _perturb_joint(txt, img, kind)
        if joint is None:
            return output
        return tuple(joint) if isinstance(output, tuple) else list(joint)

    return hook


def enable_ship_hooks(pipe_or_transformer: Any, cfg: ShipConfig):
    SHIP.clear_hooks()
    SHIP.configure(cfg)

    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    _wrap_transformer_forward(transformer)

    site = cfg.site
    text = cfg.selected_token == "text"
    dual_mask = SHIP.text_mask if text else SHIP.image_mask

    if cfg.target_blocks in ("dual", "both"):
        for i, block in enumerate(transformer.transformer_blocks):
            if site == "attn_out":
                # attn returns (image, text)
                hook = tuple_output_hook(SHIP, dual_mask, kind=1000 + i, index=1 if text else 0)
                SHIP.add_hook(block.attn.register_forward_hook(hook))
            elif site == "mlp_out":
                module = block.ff_context if text else block.ff
                SHIP.add_hook(module.register_forward_hook(tensor_output_hook(SHIP, dual_mask, kind=3000 + i)))
            elif site == "block_input":
                if text:
                    hook = input_pre_hook(SHIP, dual_mask, 5000 + i, "encoder_hidden_states", 1)
                else:
                    hook = input_pre_hook(SHIP, dual_mask, 5100 + i, "hidden_states", 0)
                SHIP.add_hook(block.register_forward_pre_hook(hook, with_kwargs=True))
            elif site == "block_output":
                # block returns (text, image)
                hook = tuple_output_hook(SHIP, dual_mask, kind=(7000 if text else 7100) + i, index=0 if text else 1)
                SHIP.add_hook(block.register_forward_hook(hook))

    if cfg.target_blocks in ("single", "both"):
        for i, block in enumerate(transformer.single_transformer_blocks):
            if site == "attn_out":
                SHIP.add_hook(block.attn.register_forward_hook(tensor_output_hook(SHIP, SHIP.single_mask, kind=2000 + i)))
            elif site == "mlp_out":
                SHIP.add_hook(block.act_mlp.register_forward_hook(tensor_output_hook(SHIP, SHIP.single_mask, kind=4000 + i)))
            elif site == "block_input":
                if _takes_text_separately(block):
                    hook = _split_single_input_pre_hook(6000 + i)
                else:
                    hook = input_pre_hook(SHIP, SHIP.single_mask, 6000 + i, "hidden_states", 0)
                SHIP.add_hook(block.register_forward_pre_hook(hook, with_kwargs=True))
            elif site == "block_output":
                if _takes_text_separately(block):
                    hook = _split_single_output_hook(8000 + i)
                else:
                    hook = tensor_output_hook(SHIP, SHIP.single_mask, kind=8000 + i)
                SHIP.add_hook(block.register_forward_hook(hook))
