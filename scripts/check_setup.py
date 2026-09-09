#!/usr/bin/env python3
"""Validate the portable GrainTTS checkout without starting a training run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/LJSpeech/preprocess.yaml"
EXPECTED_HIFIGAN_SHA256 = (
    "3fac378c5918fb2c102733f21eeaa8e9a4ca6cda24dbfddc55bbb947c78d562f"
)


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing required path: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


with CONFIG.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

preprocessed = require(ROOT / config["path"]["preprocessed_path"])
for filename in ("train.txt", "val.txt", "speakers.json", "stats.json"):
    require(preprocessed / filename)
for dirname in ("mel", "pitch", "energy", "duration"):
    require(preprocessed / dirname)

with (preprocessed / "train.txt").open("r", encoding="utf-8") as handle:
    first = handle.readline().rstrip("\n").split("|")
if len(first) != 4:
    raise SystemExit("train.txt does not have the expected four-column format")

basename, speaker = first[0], first[1]
mel = require(preprocessed / "mel" / f"{speaker}-mel-{basename}.npy")
mel_array = np.load(mel, mmap_mode="r")
if mel_array.ndim != 2 or mel_array.shape[1] != 80:
    raise SystemExit(f"unexpected Mel shape: {mel_array.shape}")

with (preprocessed / "stats.json").open("r", encoding="utf-8") as handle:
    stats = json.load(handle)
if "pitch" not in stats or "energy" not in stats:
    raise SystemExit("stats.json is missing pitch or energy statistics")

hifigan = require(ROOT / "common/hifigan/LJ_V2/generator_v2")
actual_hash = sha256(hifigan)
if actual_hash != EXPECTED_HIFIGAN_SHA256:
    raise SystemExit(f"unexpected HiFi-GAN checkpoint SHA-256: {actual_hash}")

print("GrainTTS setup is ready.")
print(f"  Python/PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Dataset:        {(ROOT / 'data/LJSpeech-1.1').resolve()}")
print(f"  First Mel:      {mel_array.shape}")
print(f"  HiFi-GAN hash:  {actual_hash}")

