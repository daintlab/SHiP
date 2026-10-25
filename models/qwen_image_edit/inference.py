from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ship import build_ship_config
from ship_hooks import SHIP, enable_ship_hooks
from utils import (
    dist_info as _dist_info,
    ensure_dir as _ensure_dir,
    get_dtype as _get_dtype,
    list_subjects as _list_subjects,
    load_config as _load_config,
    load_prompts_for_subject as _load_prompts_for_subject,
    sanitize_name as _sanitize_name,
    save_config as _save_config,
    save_image as _save_image,
)


DEFAULT_CONFIG_PATH = str(ROOT / "config.yml")
CFG: dict[str, Any] = {}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--check_remaining", action="store_true")
    return parser.parse_args()


# =============================================================================
# Build pipe
# =============================================================================

def build_pipe():
    import diffusers

    pipeline_cls = getattr(diffusers, str(CFG.get("PIPELINE_CLASS", "QwenImageEditPlusPipeline")))
    kwargs: Dict[str, Any] = {
        "torch_dtype": _get_dtype(CFG.get("DTYPE", "bf16")),
        "token": os.environ.get("HF_TOKEN"),
    }
    if CFG.get("DEVICE_MAP") is not None:
        kwargs["device_map"] = CFG["DEVICE_MAP"]
    if CFG.get("MAX_MEMORY") is not None:
        kwargs["max_memory"] = CFG["MAX_MEMORY"]

    pipe = pipeline_cls.from_pretrained(CFG["MODEL_ID"], **kwargs)
    if "device_map" not in kwargs:
        pipe = pipe.to(CFG["DEVICE"])
    return pipe


# =============================================================================
# Run one
# =============================================================================

def run_one(pipe, prompt: str, ref_image: Image.Image, seed: int) -> Image.Image:
    generator = torch.Generator(device=CFG["GENERATOR_DEVICE"]).manual_seed(int(seed))

    if SHIP.cfg.enabled:
        SHIP.start_sample(int(seed))

    kwargs: Dict[str, Any] = dict(
        image=ref_image,
        prompt=prompt,
        negative_prompt=CFG.get("NEGATIVE_PROMPT", " "),
        true_cfg_scale=CFG["TRUE_CFG_SCALE"],
        num_inference_steps=CFG["NUM_STEPS"],
        generator=generator,
    )
    # null in the config = pipeline default (WIDTH/HEIGHT: derived from the reference image)
    for key, name in (
        ("WIDTH", "width"),
        ("HEIGHT", "height"),
        ("GUIDANCE_SCALE", "guidance_scale"),
        ("MAX_SEQUENCE_LENGTH", "max_sequence_length"),
    ):
        if CFG.get(key) is not None:
            kwargs[name] = CFG[key]

    out = pipe(**kwargs)
    return out.images[0]


# =============================================================================
# Main
# =============================================================================

Task = Tuple[str, str, str, int]


def _task_output_path(subject: str, prompt_name: str, seed: int) -> str:
    return os.path.join(CFG["OUT_ROOT"], subject, _sanitize_name(prompt_name), f"seed_{int(seed)}.png")


def _collect_tasks(verbose: bool = True) -> Tuple[List[Task], List[Task]]:
    subjects = _list_subjects(CFG["SRC_IMAGE_ROOT"], CFG["SUBJECT_WHITELIST"])
    all_tasks: List[Task] = []
    remaining_tasks: List[Task] = []

    for subject in subjects:
        prompts = _load_prompts_for_subject(CFG["PROMPT_ROOT"], subject)
        if not prompts:
            if verbose:
                print(f"[SKIP] no prompts for subject={subject}", flush=True)
            continue

        total_prompts = len(prompts)
        max_prompts = CFG["EXPECT_NUM_PROMPTS"]
        if max_prompts is not None:
            max_prompts = int(max_prompts)
            if total_prompts < max_prompts and verbose:
                print(
                    f"[WARN] subject={subject} available_prompts={total_prompts} "
                    f"requested={max_prompts} -> using all available prompts",
                    flush=True,
                )
            prompts = prompts[:max_prompts]

        if verbose:
            print(f"[INFO] subject={subject} using_prompts={len(prompts)} (available={total_prompts})", flush=True)

        for prompt_name, prompt_text in prompts:
            for seed in CFG["SEEDS"]:
                task = (subject, prompt_name, prompt_text, int(seed))
                all_tasks.append(task)
                if not os.path.isfile(_task_output_path(subject, prompt_name, int(seed))):
                    remaining_tasks.append(task)

    return all_tasks, remaining_tasks


def main(section: str = "qwen_image_edit"):
    """section: config.yml section (models/firered/inference.py passes "firered")."""
    global CFG
    args = _parse_args()
    CFG = _load_config(args.config, section)

    if args.check_remaining:
        all_tasks, remaining_tasks = _collect_tasks(verbose=False)
        print(f"[CHECK] total_tasks={len(all_tasks)} | remaining_tasks={len(remaining_tasks)}", flush=True)
        for subject, prompt_name, _prompt_text, seed in remaining_tasks[:20]:
            print(f"[MISSING] subject={subject} prompt={_sanitize_name(prompt_name)} seed={seed}", flush=True)
        if len(remaining_tasks) > 20:
            print(f"[MISSING] ... {len(remaining_tasks) - 20} more", flush=True)
        raise SystemExit(1 if remaining_tasks else 0)

    local_rank, env_rank, env_world = _dist_info()
    if args.rank is None and os.environ.get("RANK") is not None:
        args.rank = env_rank
    if args.world_size is None and os.environ.get("WORLD_SIZE") is not None:
        args.world_size = env_world
    if args.rank is None or args.world_size is None:
        args.rank, args.world_size = 0, 1

    torch.cuda.set_device(local_rank)
    CFG["DEVICE"] = f"cuda:{local_rank}"
    CFG["GENERATOR_DEVICE"] = f"cuda:{local_rank}"

    torch.set_grad_enabled(False)
    ship_cfg = build_ship_config(CFG)

    if args.rank == 0:
        cfg_path = _save_config(CFG["OUT_ROOT"], CFG)
        print(f"[CFG] saved to {cfg_path}", flush=True)

    all_tasks, remaining_tasks = _collect_tasks(verbose=True)
    my_tasks = remaining_tasks[args.rank :: args.world_size]
    print(
        f"[RANK {args.rank}] total_tasks={len(all_tasks)} | "
        f"remaining_tasks={len(remaining_tasks)} | my_tasks={len(my_tasks)}",
        flush=True,
    )
    if not my_tasks:
        return

    pipe = build_pipe()
    enable_ship_hooks(pipe, ship_cfg)

    failed_tasks = 0
    for subject, prompt_name, prompt_text, seed in my_tasks:
        ref_path = os.path.join(CFG["SRC_IMAGE_ROOT"], subject, CFG["REF_IMAGE_NAME"])
        if not os.path.isfile(ref_path):
            print(f"[SKIP] missing ref image: {ref_path}", flush=True)
            failed_tasks += 1
            continue

        from diffusers.utils import load_image

        ref_image = load_image(ref_path).convert("RGB")
        prompt_name = _sanitize_name(prompt_name)
        prompt_dir = os.path.join(CFG["OUT_ROOT"], subject, prompt_name)
        _ensure_dir(prompt_dir)

        prompt_txt_path = os.path.join(prompt_dir, "prompt.txt")
        if not os.path.isfile(prompt_txt_path):
            with open(prompt_txt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text + "\n")

        out_path = os.path.join(prompt_dir, f"seed_{seed}.png")
        if os.path.isfile(out_path):
            continue

        print(f"[TASK] rank={args.rank} subject={subject} prompt={prompt_name} seed={seed}", flush=True)
        try:
            image = run_one(pipe, prompt_text, ref_image, int(seed))
            _save_image(image, out_path)
            print("[SAVE]", out_path, flush=True)
        except Exception as e:
            failed_tasks += 1
            print(f"[ERROR] {subject=} {prompt_name=} {seed=} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if failed_tasks:
        print(f"[RANK {args.rank}] failed_tasks={failed_tasks}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
