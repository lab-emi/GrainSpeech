"""Mel-spectrogram extraction adapted from EfficientSpeech/FastSpeech 2.

The original implementation is distributed under the Apache License 2.0.
This version removes the hard-coded CUDA dependency so preprocessing can run
on either CPU or GPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window


def dynamic_range_compression(x: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-5) * coefficient)


class STFT(torch.nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length

        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = filter_length // 2 + 1
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff]), np.imag(fourier_basis[:cutoff])]
        )
        forward_basis = torch.tensor(fourier_basis[:, None, :], dtype=torch.float32)

        fft_window = get_window("hann", win_length, fftbins=True)
        left = (filter_length - win_length) // 2
        right = filter_length - win_length - left
        fft_window = np.pad(fft_window, (left, right))
        forward_basis *= torch.tensor(fft_window, dtype=torch.float32)
        self.register_buffer("forward_basis", forward_basis)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.view(input_data.size(0), 1, input_data.size(1))
        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.filter_length // 2, self.filter_length // 2, 0, 0),
            mode="reflect",
        ).squeeze(1)
        transformed = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
        )
        cutoff = self.filter_length // 2 + 1
        real = transformed[:, :cutoff]
        imaginary = transformed[:, cutoff:]
        return torch.sqrt(real.square() + imaginary.square())


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        sampling_rate: int,
        mel_fmin: float,
        mel_fmax: float,
    ):
        super().__init__()
        self.stft = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        self.register_buffer("mel_basis", torch.tensor(mel_basis, dtype=torch.float32))

    def mel_spectrogram(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.min(waveform) < -1 or torch.max(waveform) > 1:
            raise ValueError("waveform must be normalized to [-1, 1]")
        magnitudes = self.stft.transform(waveform)
        mel = dynamic_range_compression(torch.matmul(self.mel_basis, magnitudes))
        energy = torch.linalg.vector_norm(magnitudes, dim=1)
        return mel, energy


@torch.inference_mode()
def get_mel_from_wav(
    waveform: np.ndarray, stft: TacotronSTFT
) -> tuple[np.ndarray, np.ndarray]:
    device = stft.mel_basis.device
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device)[None]
    mel, energy = stft.mel_spectrogram(waveform_tensor)
    return (
        mel.squeeze(0).cpu().numpy().astype(np.float32),
        energy.squeeze(0).cpu().numpy().astype(np.float32),
    )
