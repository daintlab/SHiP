from __future__ import annotations

import argparse
import os
import traceback
from typing import Any, List

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import sys
from pathlib import Path

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

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEFAULT_CONFIG_PATH = str(ROOT / "config.yml")
CFG: dict[str, Any] = _load_config(DEFAULT_CONFIG_PATH, "kontext")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    return parser.parse_args()
# =============================================================================
# Main
# =============================================================================

def main():
    global CFG
    args = _parse_args()
    CFG = _load_config(args.config, "kontext")
    torch.set_grad_enabled(False)

    local_rank, rank, world = _dist_info()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dtype = _get_dtype(CFG.get("DTYPE", "bf16"))

    if rank == 0:
        print("[INIT] loading FLUX Kontext pipeline...", flush=True)

    pipe = FluxKontextPipeline.from_pretrained(
        CFG["MODEL_ID"],
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    ship_cfg = build_ship_config(CFG)

    enable_ship_hooks(pipe, ship_cfg)

    subjects = _list_subjects(CFG["SRC_IMAGE_ROOT"], CFG["SUBJECT_WHITELIST"])
    if not subjects:
        raise RuntimeError(f"No subjects found in {CFG['SRC_IMAGE_ROOT']}")

    seeds: List[int] = list(CFG["SEEDS"])
    out_root = CFG["OUT_ROOT"]
    _ensure_dir(out_root)
    if rank == 0:
        _save_config(out_root, CFG)

    job_idx = 0

    for subject in subjects:
        ref_path = os.path.join(CFG["SRC_IMAGE_ROOT"], subject, CFG["REF_IMAGE_NAME"])
        if not os.path.exists(ref_path):
            if rank == 0:
                print(f"[SKIP] missing ref image: {ref_path}", flush=True)
            continue

        ref_img = load_image(ref_path).convert("RGB")

        prompts = _load_prompts_for_subject(CFG["PROMPT_ROOT"], subject)

        if not prompts:
            if rank == 0:
                print(f"[SKIP] no prompts for subject={subject}", flush=True)
            continue

        total_prompts = len(prompts)

        prompt_indices = CFG.get("PROMPT_INDICES", None)
        if prompt_indices is not None:
            valid_indices = [i for i in prompt_indices if 0 <= i < len(prompts)]

            if rank == 0:
                print(
                    f"[INFO] subject={subject} selected_prompt_indices={valid_indices} "
                    f"(requested={prompt_indices}, available={len(prompts)})",
                    flush=True,
                )

            prompts = [prompts[i] for i in valid_indices]

        max_prompts = CFG["EXPECT_NUM_PROMPTS"]

        if max_prompts is not None:
            max_prompts = int(max_prompts)

            if len(prompts) < max_prompts:
                if rank == 0:
                    print(
                        f"[WARN] subject={subject} available_prompts={len(prompts)} "
                        f"requested={max_prompts} -> using all available prompts",
                        flush=True,
                    )

            prompts = prompts[:max_prompts]

        if rank == 0:
            print(
                f"[INFO] subject={subject} using_prompts={len(prompts)} "
                f"(available={total_prompts})",
                flush=True,
            )

        for (pname, prompt) in prompts:
            pname_safe = _sanitize_name(pname)
            prompt_dir = os.path.join(out_root, subject, pname_safe)
            _ensure_dir(prompt_dir)

            for seed in seeds:
                if (job_idx % world) != rank:
                    job_idx += 1
                    continue
                job_idx += 1

                out_path = os.path.join(prompt_dir, f"seed_{seed}.png")
                if os.path.exists(out_path):
                    continue

                try:
                    gen = torch.Generator(device=device).manual_seed(int(seed))

                    if SHIP.cfg.enabled:
                        SHIP.start_sample(int(seed))

                    out = pipe(
                        image=ref_img,
                        prompt=prompt,
                        guidance_scale=CFG["GUIDANCE_SCALE"],
                        num_inference_steps=CFG["NUM_STEPS"],
                        width=CFG["WIDTH"],
                        height=CFG["HEIGHT"],
                        generator=gen,
                    )

                    img = out.images[0]
                    _save_image(img, out_path)

                    if rank == 0:
                        print(f"[OK] subject={subject} prompt={pname_safe} seed={seed} -> {out_path}", flush=True)

                except Exception as e:
                    if rank == 0:
                        print(
                            f"[ERR] subject={subject} prompt={pname_safe} seed={seed} "
                            f"| {type(e).__name__}: {e}",
                            flush=True,
                        )
                        traceback.print_exc()

    if rank == 0:
        print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
