"""Prepare LJSpeech features for GrainTTS training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from preprocessing.ljspeech import (
    LJSpeechPreprocessor,
    install_textgrids,
    prepare_raw_ljspeech,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess-config",
        default="configs/LJSpeech/preprocess.yaml",
        help="Path to the GrainTTS preprocessing configuration",
    )
    parser.add_argument(
        "--textgrid-dir",
        help="Downloaded TextGrid directory; omit if TextGrids are already installed",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N utterances (for setup testing only)",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    config_path = Path(args.preprocess_config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not args.skip_raw:
        prepare_raw_ljspeech(
            config, overwrite=args.overwrite_raw, limit=args.limit
        )
    output = Path(config["path"]["preprocessed_path"])
    if args.textgrid_dir:
        count = install_textgrids(args.textgrid_dir, output)
        print(f"Installed {count} TextGrid files.")
    expected = output / "TextGrid" / "LJSpeech"
    if not any(expected.glob("*.TextGrid")):
        raise FileNotFoundError(
            f"No TextGrids found under {expected}. Pass --textgrid-dir after downloading them."
        )

    device = resolve_device(args.device)
    print(f"Preprocessing on {device} with split seed {args.seed}.")
    LJSpeechPreprocessor(config, device=device, seed=args.seed).build(limit=args.limit)


if __name__ == "__main__":
    main()
