# GrainTTS

**Authors:** Zitao Liang, Chang Gao\*  
\* Corresponding author.

GrainTTS is a compact text-to-speech model that addresses two quality bottlenecks:
encoder context allocation and Mel-spectrogram oversmoothing. A receptive-field
study finds no consistent benefit from self-attention beyond 15 phonemes. Based
on this result, GrainTTS uses a fixed-receptive-field convolutional encoder with
`W=15`, reducing pitch, energy, and duration prediction errors by 36.0%, 17.3%,
and 3.4%, respectively. A local gradient-variance (GVar) loss improves
high-frequency retention in the generated Mel spectrograms.

The acoustic model contains 264.8K parameters. It reports Mel-generation mRTFs
of 2215.0 on an AMD Ryzen 9 9950X3D CPU and 17.9 on an STM32H747XI MCU with
1 MB SRAM. Under a unified UTMOS evaluation, GrainTTS scores 4.088—comparable
to MixerTTS and higher than FastSpeech 2—while using only 1.3% and 0.8% of their
parameters, respectively.

This repository contains the GrainTTS model, loss variants, training code, and
reproducible configurations. Datasets, generated audio, virtual environments,
and unrelated experiment artifacts are intentionally excluded. The released
acoustic-model checkpoint is included for direct inference.

## Demo

Audio samples are available at:
[**GrainTTS Audio Demo**](https://lab-emi.github.io/GrainTTS-Audio-Demo/)

## Repository contents

- `graintts`: GrainTTS architecture and loss variants.
- `common`: the HiFi-GAN v2 vocoder required by the model.
- `configs`: portable LJSpeech preprocessing configuration.
- `hparams`: exact experiment hyperparameters.
- `scripts`: dataset mapping and setup validation.

## Quick start

The tested environment is Python 3.11.15, PyTorch 2.12.0 (CUDA 13.0), and
Lightning 2.6.5. A complete package snapshot is stored in
`environment/reference-pip-freeze.txt`.

```bash
git clone <repository-url> GrainTTS
cd GrainTTS

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

./scripts/configure_data.sh /absolute/path/to/LJSpeech-1.1
python scripts/check_setup.py
```

`configure_data.sh` does not copy the dataset. It creates local symbolic links
to the supplied LJSpeech root and verifies the preprocessed training files.

## Inference

The released GrainTTS acoustic-model checkpoint is included at
`checkpoints/graintts_l1_ssim_gvar.ckpt`. It is an inference-only checkpoint:
training progress, optimizer state, scheduler state, and callbacks are omitted.

Its expected SHA-256 is:

```text
9361086fc8539d1ef6cdfbfd759608af42c258be3977c306515681b984ad3fb6
```

Synthesize English text on CPU (use `--device cuda` for GPU inference):

```bash
python graintts/infer.py \
  --checkpoint checkpoints/graintts_l1_ssim_gvar.ckpt \
  --text "Grain T T S is a compact text to speech model." \
  --device cpu \
  --output outputs/graintts.wav
```

The inference command uses `g2p-en` to convert ordinary English text to the
ARPAbet symbols used during training. For exact control, replace `--text` with
`--phonemes "G R EY1 N T IY1 T IY1 EH1 S"`.

## Training

Train the primary GrainTTS configuration:

```bash
python graintts/train_l1_ssim_gvar.py \
  --run-name graintts-l1-ssim-gvar \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --hifigan-checkpoint common/hifigan/LJ_V2/generator_v2 \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 128 --num_workers 4 --max_epochs 5000 \
  --lr 0.001 --weight-decay 0.00001 --infer-device cuda
```

The three training entry points are:

```text
graintts/train.py                  # L1
graintts/train_l1_ssim.py          # L1 + SSIM
graintts/train_l1_ssim_gvar.py     # L1 + SSIM + GVar (GrainTTS)
```

The primary training script writes checkpoints and TensorBoard logs under
`lightning_logs/<run-name>/`; validation audio is written under `val_outputs/`.
These outputs are ignored by Git.

## Included vocoder

The HiFi-GAN v2 checkpoint required by the model is included in this repository.
Its expected SHA-256 is:

```text
3fac378c5918fb2c102733f21eeaa8e9a4ca6cda24dbfddc55bbb947c78d562f
```

## Data layout

The supplied dataset root must contain:

```text
LJSpeech-1.1/
└── preprocessed_data/LJSpeech/
    ├── train.txt
    ├── val.txt
    ├── speakers.json
    ├── stats.json
    ├── mel/
    ├── pitch/
    ├── energy/
    └── duration/
```

No dataset content is tracked by Git.
