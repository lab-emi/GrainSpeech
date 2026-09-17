"""LJSpeech feature preparation adapted from EfficientSpeech/FastSpeech 2.

The source preprocessing implementation is distributed under Apache-2.0.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import pyworld
import tgt
import torch
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from preprocessing.audio import TacotronSTFT, get_mel_from_wav
from text import _clean_text


SPEAKER = "LJSpeech"
SILENCE_PHONES = {"sil", "sp", "spn"}


def prepare_raw_ljspeech(
    config: dict, overwrite: bool = False, limit: int | None = None
) -> None:
    """Create the normalized wav/lab layout expected by the feature extractor."""
    corpus_dir = Path(config["path"]["corpus_path"])
    raw_dir = Path(config["path"]["raw_path"]) / SPEAKER
    metadata = corpus_dir / "metadata.csv"
    wav_dir = corpus_dir / "wavs"
    if not metadata.is_file() or not wav_dir.is_dir():
        raise FileNotFoundError(
            f"Expected metadata.csv and wavs/ under LJSpeech root: {corpus_dir}"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    with metadata.open(encoding="utf-8") as handle:
        rows = list(handle)
    if limit is not None:
        rows = rows[:limit]
    for row in tqdm(rows, desc="Preparing wav/lab files"):
        fields = row.rstrip("\n").split("|")
        if len(fields) != 3:
            raise ValueError(f"Malformed LJSpeech metadata row: {row[:80]!r}")
        basename, _, normalized_text = fields
        destination_wav = raw_dir / f"{basename}.wav"
        destination_lab = raw_dir / f"{basename}.lab"
        if overwrite or not destination_wav.exists():
            waveform, _ = librosa.load(
                corpus_dir / "wavs" / f"{basename}.wav", sr=sample_rate
            )
            peak = float(np.max(np.abs(waveform)))
            if peak == 0:
                raise ValueError(f"Silent waveform: {basename}.wav")
            waveform = waveform / peak * max_wav_value
            wavfile.write(destination_wav, sample_rate, waveform.astype(np.int16))
        if overwrite or not destination_lab.exists():
            destination_lab.write_text(
                _clean_text(normalized_text, cleaners), encoding="utf-8"
            )


def install_textgrids(source: str | Path, preprocessed_dir: str | Path) -> int:
    """Copy downloaded TextGrids into TextGrid/LJSpeech."""
    source = Path(source).expanduser().resolve()
    candidates = [source / SPEAKER, source]
    source_dir = next(
        (candidate for candidate in candidates if any(candidate.glob("*.TextGrid"))),
        None,
    )
    if source_dir is None:
        raise FileNotFoundError(
            f"No .TextGrid files found in {source} or {source / SPEAKER}"
        )
    destination = Path(preprocessed_dir) / "TextGrid" / SPEAKER
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for textgrid in tqdm(sorted(source_dir.glob("*.TextGrid")), desc="Installing TextGrids"):
        shutil.copy2(textgrid, destination / textgrid.name)
        count += 1
    return count


class LJSpeechPreprocessor:
    def __init__(self, config: dict, device: torch.device, seed: int = 1234):
        self.config = config
        self.in_dir = Path(config["path"]["raw_path"])
        self.out_dir = Path(config["path"]["preprocessed_path"])
        self.val_size = config["preprocessing"]["val_size"]
        self.sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.seed = seed

        pitch = config["preprocessing"]["pitch"]
        energy = config["preprocessing"]["energy"]
        self.pitch_phoneme_averaging = pitch["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = energy["feature"] == "phoneme_level"
        self.pitch_normalization = pitch["normalization"]
        self.energy_normalization = energy["normalization"]

        stft = config["preprocessing"]["stft"]
        mel = config["preprocessing"]["mel"]
        self.stft = TacotronSTFT(
            stft["filter_length"],
            stft["hop_length"],
            stft["win_length"],
            mel["n_mel_channels"],
            self.sample_rate,
            mel["mel_fmin"],
            mel["mel_fmax"],
        ).to(device)

    def build(self, limit: int | None = None) -> list[str]:
        for directory in ("mel", "pitch", "energy", "duration"):
            (self.out_dir / directory).mkdir(parents=True, exist_ok=True)

        wav_files = sorted((self.in_dir / SPEAKER).glob("*.wav"))
        if limit is not None:
            wav_files = wav_files[:limit]
        if not wav_files:
            raise FileNotFoundError(f"No prepared wav files found under {self.in_dir / SPEAKER}")

        metadata: list[str] = []
        basenames: list[str] = []
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        frame_count = 0
        for wav_path in tqdm(wav_files, desc="Extracting GrainSpeech features"):
            result = self.process_utterance(wav_path.stem)
            if result is None:
                continue
            row, pitch, energy, frames = result
            metadata.append(row)
            basenames.append(wav_path.stem)
            if pitch.size:
                pitch_scaler.partial_fit(pitch.reshape(-1, 1))
            if energy.size:
                energy_scaler.partial_fit(energy.reshape(-1, 1))
            frame_count += frames

        if not metadata:
            raise RuntimeError("No utterances were processed; check the TextGrid directory")

        pitch_mean, pitch_std = self._scaler_values(
            pitch_scaler, self.pitch_normalization
        )
        energy_mean, energy_std = self._scaler_values(
            energy_scaler, self.energy_normalization
        )
        pitch_min, pitch_max = self._normalize_files(
            "pitch", basenames, pitch_mean, pitch_std
        )
        energy_min, energy_max = self._normalize_files(
            "energy", basenames, energy_mean, energy_std
        )

        (self.out_dir / "speakers.json").write_text(
            json.dumps({SPEAKER: 0}), encoding="utf-8"
        )
        stats = {
            "pitch": [pitch_min, pitch_max, pitch_mean, pitch_std],
            "energy": [energy_min, energy_max, energy_mean, energy_std],
        }
        (self.out_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

        random.Random(self.seed).shuffle(metadata)
        validation_size = min(self.val_size, len(metadata))
        self._write_lines("val.txt", metadata[:validation_size])
        self._write_lines("train.txt", metadata[validation_size:])
        hours = frame_count * self.hop_length / self.sample_rate / 3600
        print(f"Processed {len(metadata)} utterances ({hours:.2f} hours).")
        return metadata

    def process_utterance(
        self, basename: str
    ) -> tuple[str, np.ndarray, np.ndarray, int] | None:
        textgrid_path = self.out_dir / "TextGrid" / SPEAKER / f"{basename}.TextGrid"
        if not textgrid_path.is_file():
            return None
        textgrid = tgt.io.read_textgrid(textgrid_path)
        phones, durations, start, end = self._get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        if start >= end:
            return None

        wav_path = self.in_dir / SPEAKER / f"{basename}.wav"
        waveform, _ = librosa.load(wav_path, sr=self.sample_rate)
        waveform = waveform[
            int(self.sample_rate * start) : int(self.sample_rate * end)
        ].astype(np.float32)
        raw_text = (self.in_dir / SPEAKER / f"{basename}.lab").read_text(
            encoding="utf-8"
        ).strip()

        pitch, times = pyworld.dio(
            waveform.astype(np.float64),
            self.sample_rate,
            frame_period=self.hop_length / self.sample_rate * 1000,
        )
        pitch = pyworld.stonemask(
            waveform.astype(np.float64), pitch, times, self.sample_rate
        )[: sum(durations)]
        if np.count_nonzero(pitch) <= 1:
            return None

        mel, energy = get_mel_from_wav(waveform, self.stft)
        frame_total = sum(durations)
        if min(mel.shape[1], len(energy), len(pitch)) < frame_total:
            raise RuntimeError(
                f"Alignment for {basename} requires {frame_total} frames, but the "
                "audio features are shorter"
            )
        mel, energy, pitch = mel[:, :frame_total], energy[:frame_total], pitch[:frame_total]
        durations = np.asarray(durations, dtype=np.int64)

        if self.pitch_phoneme_averaging:
            nonzero = np.flatnonzero(pitch)
            interpolate = interp1d(
                nonzero,
                pitch[nonzero],
                fill_value=(pitch[nonzero[0]], pitch[nonzero[-1]]),
                bounds_error=False,
            )
            pitch = self._phoneme_average(interpolate(np.arange(len(pitch))), durations)
        if self.energy_phoneme_averaging:
            energy = self._phoneme_average(energy, durations)

        np.save(self.out_dir / "duration" / f"{SPEAKER}-duration-{basename}.npy", durations)
        np.save(self.out_dir / "pitch" / f"{SPEAKER}-pitch-{basename}.npy", pitch)
        np.save(self.out_dir / "energy" / f"{SPEAKER}-energy-{basename}.npy", energy)
        np.save(self.out_dir / "mel" / f"{SPEAKER}-mel-{basename}.npy", mel.T)
        row = "|".join([basename, SPEAKER, "{" + " ".join(phones) + "}", raw_text])
        return row, self._remove_outlier(pitch), self._remove_outlier(energy), frame_total

    def _get_alignment(self, tier) -> tuple[list[str], list[int], float, float]:
        phones: list[str] = []
        durations: list[int] = []
        start = end = 0.0
        end_index = 0
        for interval in tier._objects:
            phone = interval.text
            if not phones and phone in SILENCE_PHONES:
                continue
            if not phones:
                start = interval.start_time
            phones.append(phone)
            durations.append(0)  # converted below after retaining the interval
            durations[-1] = int(
                round(interval.end_time * self.sample_rate / self.hop_length)
                - round(interval.start_time * self.sample_rate / self.hop_length)
            )
            if phone not in SILENCE_PHONES:
                end = interval.end_time
                end_index = len(phones)
        return phones[:end_index], durations[:end_index], start, end

    @staticmethod
    def _phoneme_average(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
        output = np.zeros(len(durations), dtype=np.float64)
        position = 0
        for index, duration in enumerate(durations):
            if duration > 0:
                output[index] = np.mean(values[position : position + duration])
            position += duration
        return output

    @staticmethod
    def _remove_outlier(values: np.ndarray) -> np.ndarray:
        q25, q75 = np.percentile(values, [25, 75])
        lower, upper = q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25)
        return values[(values > lower) & (values < upper)]

    @staticmethod
    def _scaler_values(scaler: StandardScaler, normalize: bool) -> tuple[float, float]:
        if not normalize:
            return 0.0, 1.0
        return float(scaler.mean_[0]), float(scaler.scale_[0])

    def _normalize_files(
        self, name: str, basenames: list[str], mean: float, std: float
    ) -> tuple[float, float]:
        minimum, maximum = np.inf, -np.inf
        for basename in basenames:
            path = self.out_dir / name / f"{SPEAKER}-{name}-{basename}.npy"
            values = (np.load(path) - mean) / std
            np.save(path, values)
            minimum = min(minimum, float(np.min(values)))
            maximum = max(maximum, float(np.max(values)))
        return minimum, maximum

    def _write_lines(self, filename: str, rows: list[str]) -> None:
        text = "".join(f"{row}\n" for row in rows)
        (self.out_dir / filename).write_text(text, encoding="utf-8")
