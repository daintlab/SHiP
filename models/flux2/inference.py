from __future__ import annotations

import argparse
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ship import build_ship_config
from ship_hooks import SHIP, enable_ship_hooks
from utils import (
    ensure_dir as _ensure_dir,
    get_dtype as _get_dtype,
    list_subjects as _list_subjects,
    load_config as _load_config,
    load_prompts_for_subject as _load_prompts_for_subject,
    sanitize_name as _sanitize_name,
    save_image as _save_image,
    save_config as _save_config,
)


DEFAULT_CONFIG_PATH = str(ROOT / "config.yml")
CFG: dict[str, Any] = _load_config(DEFAULT_CONFIG_PATH, "flux2")

# =============================================================================
# Build pipe
# =============================================================================

def build_pipe() -> Flux2Pipeline:
    from diffusers import Flux2Pipeline

    pipe = Flux2Pipeline.from_pretrained(
        CFG["MODEL_ID"],
        torch_dtype=_get_dtype(CFG["DTYPE"]),
        token=os.environ.get("HF_TOKEN"),
        device_map=CFG["DEVICE_MAP"],
        max_memory=CFG["MAX_MEMORY"],
    )
    print("[hf_device_map]", pipe.hf_device_map, flush=True)
    return pipe


# =============================================================================
# Run one
# =============================================================================

def run_one(pipe: Flux2Pipeline, prompt: str, ref_image: Optional[Image.Image], seed: int) -> Image.Image:
    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        width=CFG["WIDTH"],
        height=CFG["HEIGHT"],
        num_inference_steps=CFG["NUM_STEPS"],
        guidance_scale=CFG["GUIDANCE_SCALE"],
        generator=torch.Generator(device="cuda:0").manual_seed(int(seed)),
        max_sequence_length=CFG["MAX_SEQUENCE_LENGTH"],
    )

    has_reference = bool(CFG["USE_REFERENCE_IMAGE"] and ref_image is not None)
    if has_reference:
        kwargs["image"] = ref_image

    if SHIP.cfg.enabled:
        SHIP.start_sample(int(seed))

    out = pipe(**kwargs)
    return out.images[0]


# =============================================================================
# Main
# =============================================================================

Task = Tuple[str, str, str, int]


def _task_output_path(subject: str, prompt_name: str, seed: int) -> str:
    return os.path.join(
        CFG["OUT_ROOT"],
        subject,
        _sanitize_name(prompt_name),
        f"seed_{int(seed)}.png",
    )


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
            print(
                f"[INFO] subject={subject} using_prompts={len(prompts)} "
                f"(available={total_prompts})",
                flush=True,
            )

        for prompt_name, prompt_text in prompts:
            for seed in CFG["SEEDS"]:
                task = (subject, prompt_name, prompt_text, int(seed))
                all_tasks.append(task)

                if not os.path.isfile(_task_output_path(subject, prompt_name, int(seed))):
                    remaining_tasks.append(task)

    return all_tasks, remaining_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--check_remaining", action="store_true")
    args = parser.parse_args()

    global CFG
    CFG = _load_config(args.config, "flux2")

    if args.check_remaining:
        all_tasks, remaining_tasks = _collect_tasks(verbose=False)
        print(
            f"[CHECK] total_tasks={len(all_tasks)} | remaining_tasks={len(remaining_tasks)}",
            flush=True,
        )
        for subject, prompt_name, _prompt_text, seed in remaining_tasks[:20]:
            print(
                f"[MISSING] subject={subject} prompt={_sanitize_name(prompt_name)} seed={seed}",
                flush=True,
            )
        if len(remaining_tasks) > 20:
            print(f"[MISSING] ... {len(remaining_tasks) - 20} more", flush=True)
        raise SystemExit(1 if remaining_tasks else 0)

    if args.rank is None:
        env_rank = os.environ.get("RANK")
        if env_rank is not None:
            args.rank = int(env_rank)
    if args.world_size is None:
        env_world_size = os.environ.get("WORLD_SIZE")
        if env_world_size is not None:
            args.world_size = int(env_world_size)

    if args.rank is None or args.world_size is None:
        parser.error(
            "--rank and --world_size are required unless --check_remaining is used "
            "(or launch with torchrun, which sets RANK and WORLD_SIZE)"
        )

    torch.set_grad_enabled(False)

    ship_cfg = build_ship_config(CFG)

    if args.rank == 0:
        cfg_path = _save_config(CFG["OUT_ROOT"], CFG)
        print(f"[CFG] saved to {cfg_path}", flush=True)

    all_tasks, remaining_tasks = _collect_tasks(verbose=True)

    my_tasks = remaining_tasks[args.rank::args.world_size]

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
        subject_dir = os.path.join(CFG["SRC_IMAGE_ROOT"], subject)
        ref_path = os.path.join(subject_dir, CFG["REF_IMAGE_NAME"])

        ref_image: Optional[Image.Image] = None
        if CFG["USE_REFERENCE_IMAGE"]:
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

        print(
            f"[TASK] rank={args.rank} subject={subject} prompt={prompt_name} seed={seed}",
            flush=True,
        )

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
