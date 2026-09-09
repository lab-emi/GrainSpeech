"""
Per-operator parameter breakdown for EfficientSpeech (phoneme2mel only).
Lists every module that directly owns at least one learnable parameter,
including activations with weights (DyT, etc.).

Usage:
    python count_params.py --networks-file layers/networks.py
"""

import torch
import argparse
import importlib
import importlib.util
import os
import sys

from utils.tools import get_args

DUMMY_PREPROCESS_CONFIG = {
    "path": {"preprocessed_path": "preprocessed_data/LJSpeech"},
    "preprocessing": {"audio": {"sampling_rate": 22050}},
}

def own_params(module):
    """Parameters directly owned by this module (excludes sub-modules)."""
    return sum(p.numel() for p in module._parameters.values() if p is not None)


def collect_rows(root):
    rows = []
    for name, mod in root.named_modules():
        n = own_params(mod)
        if n == 0:
            continue
        type_name = type(mod).__name__
        display   = name if name else "(root)"
        rows.append((display, type_name, n))
    return rows


def print_table(title, rows, section_total, grand_total):
    name_w  = max(len(r[0]) for r in rows) + 2
    type_w  = max(len(r[1]) for r in rows) + 2
    param_w = max(len(f"{grand_total:,}") + 2, 10)

    header = (f"{'Module path':<{name_w}} {'Type':<{type_w}} "
              f"{'Params':>{param_w}}   {'% Section':>9}   {'% Total':>8}")
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  {title}  ({section_total:,} params,  {section_total/grand_total*100:.1f}% of total)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    prev_section = None
    for path, type_name, n in rows:
        section = path.split(".")[0]
        if prev_section is not None and section != prev_section:
            print()
        prev_section = section

        pct_sec   = n / section_total * 100
        pct_total = n / grand_total   * 100
        print(f"{path:<{name_w}} {type_name:<{type_w}} {n:>{param_w},}   "
              f"{pct_sec:>8.2f}%   {pct_total:>7.2f}%")

    print(sep)
    print(f"{'SECTION TOTAL':<{name_w}} {'':<{type_w}} {section_total:>{param_w},}   "
          f"{'100.00%':>9}   {section_total/grand_total*100:>7.2f}%")
    print(sep)


def _resolve_networks_file(path_arg):
    if os.path.isabs(path_arg):
        return path_arg
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path_arg))


def _load_networks_module(networks_file):
    spec = importlib.util.spec_from_file_location("layers.networks", networks_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load networks module from: {networks_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["layers.networks"] = module

    layers_pkg = importlib.import_module("layers")
    layers_pkg.PhonemeEncoder = module.PhonemeEncoder
    layers_pkg.MelDecoder = module.MelDecoder
    layers_pkg.Phoneme2Mel = module.Phoneme2Mel


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--networks-file",
        default="layers/networks.py",
        help="Path to a networks file, relative to graintts/ or absolute.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    networks_file = _resolve_networks_file(known.networks_file)
    if not os.path.exists(networks_file):
        raise FileNotFoundError(f"networks file not found: {networks_file}")

    _load_networks_module(networks_file)
    from model import EfficientSpeech

    args = get_args()

    # ------------------------------------------------------------------ #
    # 1. EfficientSpeech (phoneme2mel only, trainable)
    # ------------------------------------------------------------------ #
    print(f"\nUsing networks file: {networks_file}")
    print(f"Building EfficientSpeech  embed_dim={args.embed_dim} ...")

    es_model = EfficientSpeech(
        preprocess_config=DUMMY_PREPROCESS_CONFIG,
        embed_dim=args.embed_dim,
        depth=args.depth,
        n_blocks=args.n_blocks,
        block_depth=args.block_depth,
        reduction=args.reduction,
        head=args.head,
        kernel_size=args.kernel_size,
        decoder_kernel_size=args.decoder_kernel_size,
        expansion=args.expansion,
        hifigan_checkpoint=args.hifigan_checkpoint,
        verbose=False,
    )
    es_root  = es_model.phoneme2mel
    es_total = sum(p.numel() for p in es_root.parameters())
    es_rows  = collect_rows(es_root)

    bar = "=" * 72
    print(f"\n{bar}\nEfficientSpeech  phoneme2mel  (PyTorch repr)\n{bar}\n")
    print(es_root)
    print()

    # ------------------------------------------------------------------ #
    # 2. Print table
    # ------------------------------------------------------------------ #
    grand_total = es_total

    print_table("EfficientSpeech  [trainable]", es_rows, es_total, grand_total)

    # Summary
    w = 60
    print(f"\n{'='*w}")
    print(f"  TOTAL")
    print(f"{'='*w}")
    print(f"  EfficientSpeech (trainable) : {es_total:>10,}  (100.0%)")
    print(f"{'='*w}\n")


if __name__ == "__main__":
    main()
