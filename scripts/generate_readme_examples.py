"""Run the released model and export the README's speech and Mel examples.

Run from the repository root after the Quick Start installation:
    python scripts/generate_readme_examples.py --device cpu
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import torch
import yaml

from infer import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_STATS, DEFAULT_VOCODER,
    resolve_device, text_to_arpabet,
)
from model_l1_ssim_gvar import GrainSpeech
from text import text_to_sequence


EXAMPLES = (
    ("compact-speech", "Small models can give every word a voice."),
    ("morning-light", "The morning light falls softly on the quiet garden."),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_mel(mel, text, duration, output):
    """Plot the actual acoustic-model prediction, without interpolation."""
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#101f31", "axes.labelcolor": "#101f31",
        "xtick.color": "#5b6875", "ytick.color": "#5b6875",
        "axes.edgecolor": "#d9e2e8",
    }):
        fig, ax = plt.subplots(figsize=(11, 3.4), layout="constrained")
        plot = ax.imshow(
            mel.T, origin="lower", aspect="auto", interpolation="nearest",
            extent=(0, duration, -0.5, mel.shape[1] - 0.5),
            cmap="magma", vmin=-11.5, vmax=2.0,
        )
        ax.set(xlabel="Time (s)", ylabel="Mel bin")
        ax.set_yticks([0, 20, 40, 60, 79])
        ax.set_title(f'“{text}”', loc="left", fontsize=14, pad=14)
        fig.colorbar(plot, ax=ax, label="Log Mel amplitude", pad=0.02)
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/examples")
    args = parser.parse_args()
    device = resolve_device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    config = yaml.safe_load((ROOT / DEFAULT_CONFIG).read_text())
    config["path"]["preprocessed_path"] = str((ROOT / DEFAULT_STATS).parent)
    model = GrainSpeech.load_from_checkpoint(
        ROOT / DEFAULT_CHECKPOINT, map_location=device,
        preprocess_config=config,
        hifigan_checkpoint=str(ROOT / DEFAULT_VOCODER), infer_device=device,
    ).to(device).eval()
    torch.manual_seed(0)
    sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    records = []

    for slug, sentence in EXAMPLES:
        phones = text_to_arpabet(sentence)
        arpabet = "{" + " ".join(phones) + "}"
        phoneme = torch.tensor(
            [text_to_sequence(arpabet, cleaners)], dtype=torch.long, device=device,
        )
        with torch.inference_mode():
            audio, lengths, prediction = model({
                "phoneme": phoneme,
                "phoneme_mask": torch.zeros_like(phoneme, dtype=torch.bool),
            })
        frames = int(lengths[0])
        mel = prediction[0, :frames].float().cpu().numpy()
        waveform = audio[0, :frames * hop_length].float().cpu().numpy()
        if frames <= 0 or not np.isfinite(mel).all() or not np.isfinite(waveform).all():
            raise RuntimeError(f"Invalid model output for {slug}")
        duration = len(waveform) / sample_rate
        np.save(output / f"{slug}.npy", mel)
        pcm = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
        wavfile.write(output / f"{slug}.wav", sample_rate, pcm)
        plot_mel(mel, sentence, duration, output / f"{slug}.png")
        records.append({
            "id": slug, "text": sentence, "arpabet": arpabet,
            "mel_shape": list(mel.shape), "duration_seconds": duration,
            "mel_range": [float(mel.min()), float(mel.max())],
            "files": {ext: {"path": f"{slug}.{ext}", "sha256": sha256(output / f"{slug}.{ext}")}
                      for ext in ("npy", "wav", "png")},
        })
        print(f"{slug}: {frames} frames, {duration:.3f} s, {sample_rate} Hz", flush=True)

    metadata = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "checkpoint_sha256": sha256(ROOT / DEFAULT_CHECKPOINT),
        "vocoder_sha256": sha256(ROOT / DEFAULT_VOCODER),
        "config_sha256": sha256(ROOT / DEFAULT_CONFIG),
        "stats_sha256": sha256(ROOT / DEFAULT_STATS),
        "device": device, "seed": 0, "torch_threads": 1,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "lightning", "numpy", "scipy", "matplotlib", "g2p-en", "nltk")},
        "sampling_rate": sample_rate, "hop_length": hop_length,
        "mel_source": "Direct GrainSpeech acoustic-model prediction before HiFi-GAN",
        "mel_layout": "time_frames, mel_bins; natural-log amplitude",
        "plot": {"cmap": "magma", "vmin": -11.5, "vmax": 2.0, "interpolation": "nearest"},
        "examples": records,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
