"""Synthesize speech with the released GrainTTS checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from g2p_en import G2p
from scipy.io import wavfile

from model_l1_ssim_gvar import EfficientSpeech
from text import text_to_sequence
from text.symbols import symbols


DEFAULT_CHECKPOINT = "checkpoints/graintts_l1_ssim_gvar.ckpt"
DEFAULT_CONFIG = "configs/LJSpeech/preprocess.yaml"
DEFAULT_STATS = "configs/LJSpeech/stats.json"
DEFAULT_VOCODER = "hifigan/LJ_V2/generator_v2"
PAUSE_MARKS = {",", ";", ":", ".", "!", "?"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="English text to synthesize")
    source.add_argument(
        "--phonemes",
        help="Space-separated ARPAbet phonemes, optionally enclosed in braces",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preprocess-config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stats",
        default=DEFAULT_STATS,
        help="Training-set pitch and energy statistics",
    )
    parser.add_argument("--hifigan-checkpoint", default=DEFAULT_VOCODER)
    parser.add_argument("--output", default="outputs/graintts.wav")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: CUDA when available)",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return requested


def text_to_arpabet(text):
    supported = {symbol[1:] for symbol in symbols if symbol.startswith("@")}
    phones = []
    for token in G2p()(text):
        if token in supported:
            phones.append(token)
        elif token in PAUSE_MARKS and phones and phones[-1] != "sp":
            phones.append("sp")

    while phones and phones[-1] == "sp":
        phones.pop()
    if not phones:
        raise ValueError("The text frontend produced no supported ARPAbet phonemes")
    return phones


def parse_phonemes(value):
    phones = value.strip().removeprefix("{").removesuffix("}").split()
    supported = {symbol[1:] for symbol in symbols if symbol.startswith("@")}
    unsupported = sorted(set(phones) - supported)
    if unsupported:
        raise ValueError(f"Unsupported phonemes: {', '.join(unsupported)}")
    if not phones:
        raise ValueError("No phonemes were provided")
    return phones


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.preprocess_config)
    stats_path = Path(args.stats)
    vocoder_checkpoint = Path(args.hifigan_checkpoint)
    output = Path(args.output)

    for path, label in (
        (checkpoint, "GrainTTS checkpoint"),
        (config_path, "preprocessing configuration"),
        (stats_path, "LJSpeech statistics"),
        (vocoder_checkpoint, "HiFi-GAN checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    with config_path.open("r", encoding="utf-8") as stream:
        preprocess_config = yaml.safe_load(stream)
    preprocess_config["path"]["preprocessed_path"] = str(stats_path.parent)

    phones = text_to_arpabet(args.text) if args.text else parse_phonemes(args.phonemes)
    arpabet = "{" + " ".join(phones) + "}"
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sequence = text_to_sequence(arpabet, cleaners)
    phoneme = torch.tensor([sequence], dtype=torch.long, device=device)
    model_input = {
        "phoneme": phoneme,
        "phoneme_mask": torch.zeros_like(phoneme, dtype=torch.bool),
    }

    model = EfficientSpeech.load_from_checkpoint(
        checkpoint,
        map_location=device,
        preprocess_config=preprocess_config,
        hifigan_checkpoint=str(vocoder_checkpoint),
        infer_device=device,
    ).to(device)
    model.eval()

    with torch.inference_mode():
        audio, mel_lengths, _ = model(model_input)

    audio_config = preprocess_config["preprocessing"]["audio"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sample_count = int(mel_lengths[0]) * hop_length
    waveform = audio[0, :sample_count].float().cpu().numpy()
    waveform = np.clip(waveform * audio_config["max_wav_value"], -32768, 32767)
    waveform = waveform.astype(np.int16)

    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, audio_config["sampling_rate"], waveform)
    duration = len(waveform) / audio_config["sampling_rate"]
    print(f"Phonemes: {arpabet}")
    print(f"Wrote {output} ({duration:.3f} s, {audio_config['sampling_rate']} Hz)")


if __name__ == "__main__":
    main()
