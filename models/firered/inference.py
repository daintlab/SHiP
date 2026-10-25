"""FireRed-Image-Edit-1.0 has the Qwen-Image-Edit architecture and pipeline, so it reuses
models/qwen_image_edit/ with the `firered` section of config.yml."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

QWEN_DIR = Path(__file__).resolve().parents[1] / "qwen_image_edit"
sys.path.insert(0, str(QWEN_DIR))

spec = importlib.util.spec_from_file_location("qwen_image_edit_inference", QWEN_DIR / "inference.py")
qwen_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen_inference)


if __name__ == "__main__":
    qwen_inference.main(section="firered")
