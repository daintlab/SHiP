"""Compute CLIP-I, DINO, CLIP-T, Vendi and LPIPS for generated images (OUT_ROOT of any model).

Per prompt, all seeds are scored together: CLIP-I / DINO / CLIP-T are averaged over images,
Vendi (CLIP features) and pairwise LPIPS measure diversity across seeds. Prompt scores are
averaged per subject, and subject scores are averaged into the final numbers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from transformers import BitImageProcessor, CLIPModel, CLIPProcessor, Dinov2Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import list_subjects, load_config, load_prompts_for_subject, sanitize_name

METRICS = ["clip_i", "dino", "clip_t", "vendi", "lpips"]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
LPIPS_BATCH = 64


def _mean(values: List[float]) -> float:
    return round(float(np.mean(values)), 4)


def calculate_vendi_score(embeddings: torch.Tensor) -> float:
    n = embeddings.shape[0]
    if n <= 1:
        return 1.0
    sim_matrix = (embeddings @ embeddings.T) / n
    eigvals = torch.linalg.eigvalsh(sim_matrix)
    eigvals = torch.clamp(eigvals, min=1e-10)
    return float(torch.exp(-torch.sum(eigvals * torch.log(eigvals))).item())


class Evaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-small", torch_dtype=torch.float32).to(device).eval()
        self.dino_proc = BitImageProcessor.from_pretrained("facebook/dinov2-small")
        self.lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().float()
        self.lpips_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @staticmethod
    def _normalize(feat) -> torch.Tensor:
        # transformers>=5 returns a model output instead of a tensor from get_*_features
        if not isinstance(feat, torch.Tensor):
            if getattr(feat, "pooler_output", None) is not None:
                feat = feat.pooler_output
            elif getattr(feat, "image_embeds", None) is not None:
                feat = feat.image_embeds
            else:
                feat = feat[0]
        feat = feat.float()
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def clip_text(self, text: str) -> torch.Tensor:
        inputs = self.clip_proc(text=[text], padding=True, return_tensors="pt").to(self.device)
        return self._normalize(self.clip.get_text_features(**inputs))

    @torch.no_grad()
    def clip_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        return self._normalize(self.clip.get_image_features(**inputs))

    @torch.no_grad()
    def dino_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.dino_proc(images=images, return_tensors="pt").to(self.device)
        return self._normalize(self.dino(**inputs).last_hidden_state[:, 0, :])

    @torch.no_grad()
    def pairwise_lpips(self, images: List[Image.Image]) -> float:
        if len(images) < 2:
            return 0.0
        x = torch.stack([self.lpips_transform(img) for img in images]).to(self.device)
        pairs = list(itertools.combinations(range(len(images)), 2))
        distances = []
        for k in range(0, len(pairs), LPIPS_BATCH):
            i, j = zip(*pairs[k : k + LPIPS_BATCH])
            distances.append(self.lpips_model(x[list(i)], x[list(j)]).flatten())
        return float(torch.cat(distances).mean().item())

    def score_prompt(self, ref: Image.Image, prompt: str, images: List[Image.Image]) -> Dict[str, float]:
        clip_feats = self.clip_image(images)
        dino_feats = self.dino_image(images)
        return {
            "clip_i": round(float((clip_feats @ self.clip_image([ref]).T).mean().item() * 100.0), 4),
            "dino": round(float((dino_feats @ self.dino_image([ref]).T).mean().item() * 100.0), 4),
            "clip_t": round(float((clip_feats @ self.clip_text(prompt).T).mean().item() * 100.0), 4),
            "vendi": round(calculate_vendi_score(clip_feats), 4),
            "lpips": round(self.pairwise_lpips(images), 4),
        }


def _load_images(prompt_dir: str) -> List[Image.Image]:
    files = sorted(f for f in os.listdir(prompt_dir) if f.lower().endswith(IMAGE_EXT))
    return [Image.open(os.path.join(prompt_dir, f)).convert("RGB") for f in files]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yml"), help="Path to YAML config file (eval section)")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config, "eval", base=None)
    src_root, prompt_root, gen_root = cfg["SRC_IMAGE_ROOT"], cfg["PROMPT_ROOT"], cfg["OUT_ROOT"]
    out_path = os.path.join(gen_root, "metrics.json")

    torch.set_grad_enabled(False)
    evaluator = Evaluator(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    results: Dict[str, dict] = {}
    for subject in list_subjects(src_root):
        prompts = load_prompts_for_subject(prompt_root, subject)
        if not prompts or not os.path.isdir(os.path.join(gen_root, subject)):
            print(f"[SKIP] subject={subject} (no prompts or no generated images)", flush=True)
            continue

        ref = Image.open(os.path.join(src_root, subject, cfg["REF_IMAGE_NAME"])).convert("RGB")
        prompt_scores: Dict[str, Dict[str, float]] = {}
        for pname, prompt in tqdm(prompts, desc=subject):
            prompt_dir = os.path.join(gen_root, subject, sanitize_name(pname))
            images = _load_images(prompt_dir) if os.path.isdir(prompt_dir) else []
            if not images:
                print(f"[SKIP] no images in {prompt_dir}", flush=True)
                continue
            prompt_scores[pname] = {"prompt": prompt, "num_images": len(images), **evaluator.score_prompt(ref, prompt, images)}

        if prompt_scores:
            overall = {m: _mean([s[m] for s in prompt_scores.values()]) for m in METRICS}
            results[subject] = {"overall": overall, "prompts": prompt_scores}
            print(f"[OK] subject={subject} {overall}", flush=True)

    if not results:
        raise RuntimeError(f"No generated images found under {gen_root}")

    overall = {m: _mean([r["overall"][m] for r in results.values()]) for m in METRICS}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "num_subjects": len(results), "subjects": results}, f, indent=2, ensure_ascii=False)

    print(f"[DONE] overall={overall} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
