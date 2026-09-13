#!/usr/bin/env python3
"""Validate the portable GrainTTS checkout without starting a training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/LJSpeech/preprocess.yaml"


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing required path: {path}")
    return path

with CONFIG.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

preprocessed = require(ROOT / config["path"]["preprocessed_path"])
for filename in ("train.txt", "val.txt", "speakers.json", "stats.json"):
    require(preprocessed / filename)
for dirname in ("mel", "pitch", "energy", "duration"):
    require(preprocessed / dirname)

first_line = ""
for split in ("train.txt", "val.txt"):
    with (preprocessed / split).open("r", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
    if first_line:
        break
first = first_line.split("|")
if len(first) != 4:
    raise SystemExit("metadata splits do not have the expected four-column format")

basename, speaker = first[0], first[1]
mel = require(preprocessed / "mel" / f"{speaker}-mel-{basename}.npy")
mel_array = np.load(mel, mmap_mode="r")
if mel_array.ndim != 2 or mel_array.shape[1] != 80:
    raise SystemExit(f"unexpected Mel shape: {mel_array.shape}")

with (preprocessed / "stats.json").open("r", encoding="utf-8") as handle:
    stats = json.load(handle)
if "pitch" not in stats or "energy" not in stats:
    raise SystemExit("stats.json is missing pitch or energy statistics")

require(ROOT / "hifigan/LJ_V2/generator_v2")

print("GrainTTS setup is ready.")
print(f"  Python/PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Dataset:        {(ROOT / 'data/LJSpeech-1.1').resolve()}")
print(f"  First Mel:      {mel_array.shape}")
print("  HiFi-GAN:       hifigan/LJ_V2/generator_v2")
