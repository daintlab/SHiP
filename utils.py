from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional, Tuple

import torch
import yaml
from PIL import Image


def dist_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(out_root: str, cfg: dict[str, Any], filename: str = "config.json") -> str:
    ensure_dir(out_root)
    out_path = os.path.join(out_root, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return out_path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    for key in (name, name.lower(), name.upper()):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(f"Missing config section: {name}")


def load_config(config_path: str, section: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    common = data.get("common", data.get("COMMON", {})) or {}
    if not isinstance(common, dict):
        raise ValueError("Config section 'common' must be a mapping.")
    return _deep_merge(common, _section(data, section))


def save_image(img: Image.Image, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    img.save(out_path)


def sanitize_name(text: str, max_len: int = 120) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z가-힣_.\-]+", "_", text)
    return text[:max_len]


def list_subjects(src_root: str, whitelist: Optional[List[str]] = None) -> List[str]:
    subjects: List[str] = []
    for name in sorted(os.listdir(src_root)):
        path = os.path.join(src_root, name)
        if os.path.isdir(path):
            subjects.append(name)
    if whitelist:
        allow = set(whitelist)
        subjects = [subject for subject in subjects if subject in allow]
    return subjects


def load_prompts_for_subject(prompt_root: str, subject: str) -> List[Tuple[str, str]]:
    """One prompt per line of {prompt_root}/{subject}.txt; subjects without this file are skipped."""
    subject_txt = os.path.join(prompt_root, f"{subject}.txt")
    if not os.path.isfile(subject_txt):
        return []
    prompts: List[Tuple[str, str]] = []
    with open(subject_txt, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append((f"{subject}_{i:03d}", line))
    return prompts


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32
