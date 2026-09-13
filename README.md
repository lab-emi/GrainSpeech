# GrainTTS

**Authors:** Zitao Liang, Chang Gao\*  
\* Corresponding author.

This is the official repository for the paper **“GrainTTS: Less Context, More
Detail for Compact Speech Synthesis.”** GrainTTS introduces two changes for
compact text-to-speech synthesis:

1. A **fixed-receptive-field convolutional encoder** that uses focused phoneme
   context for acoustic prediction.
2. An **anti-oversmoothing Mel loss** that combines L1, SSIM, and local
   gradient-variance (GVar) supervision.

![GrainTTS architecture](assets/graintts_architecture.png)

[GrainTTS Audio Demo](#graintts-audio-demo) ·
[GrainTTS Online Playground](#graintts-online-playground) ·
[GrainTTS Quick Start](#graintts-quick-start) ·
[GrainTTS Training](#graintts-training)

## GrainTTS Audio Demo

Listen to GrainTTS samples generated from the LJSpeech dataset:
[**GrainTTS Audio Demo**](https://lab-emi.github.io/GrainTTS-Audio-Demo/).

## GrainTTS Online Playground

[**Launch the GrainTTS Online Playground**](TODO) — coming soon. The playground
will provide real-time text-to-speech synthesis directly from a web page.

## GrainTTS Quick Start

The reference environment uses Python 3.11, PyTorch 2.12, and Lightning 2.6.
The complete package snapshot is available in
`environment/reference-pip-freeze.txt`.

```bash
git clone https://github.com/lab-emi/GrainTTS.git
cd GrainTTS

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger cmudict
```

The last command installs the language resources used by `g2p-en` to convert
ordinary English text into ARPAbet phonemes. It only needs to be run once per
environment. Inference with `--phonemes` does not use this conversion step.

Synthesize English text with the released checkpoint:

```bash
python graintts/infer.py \
  --checkpoint checkpoints/graintts_l1_ssim_gvar.ckpt \
  --text "Grain T T S is a compact text to speech model." \
  --device cpu \
  --output outputs/graintts.wav
```

Use `--device cuda` for GPU inference. To control the pronunciation directly,
replace `--text` with a space-separated ARPAbet sequence, for example:

```bash
python graintts/infer.py \
  --checkpoint checkpoints/graintts_l1_ssim_gvar.ckpt \
  --phonemes "G R EY1 N T IY1 T IY1 EH1 S" \
  --device cpu \
  --output outputs/graintts.wav
```

The released inference checkpoint contains the trained GrainTTS acoustic model
and HiFi-GAN vocoder weights. The small `configs/LJSpeech/stats.json` file lets
inference run without downloading the training dataset. The standalone HiFi-GAN
file under `hifigan/LJ_V2/` is also kept because the current model
constructor uses it while loading the checkpoint and when starting a new
training run.

## GrainTTS Training

Want to modify GrainTTS or train your own variant? Complete the Quick Start
installation first, then prepare LJSpeech as follows.

### 1. Download LJSpeech

Download [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) and extract it
directly under the repository's `data/` directory. The final location must be:

```text
GrainTTS/
└── data/
    └── LJSpeech-1.1/
        ├── metadata.csv
        └── wavs/
```

In other words, `metadata.csv` must be available at
`data/LJSpeech-1.1/metadata.csv` when commands are run from the repository root.

### 2. Download the phoneme alignments

Download the precomputed
[LJSpeech TextGrids](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4).
Place the downloaded `.TextGrid` files at:

```text
GrainTTS/data/LJSpeech-1.1/TextGrid/LJSpeech/
```

The `.TextGrid` files should be directly inside the final `LJSpeech/` folder,
not inside an additional nested directory.

A TextGrid records the time interval occupied by each phoneme in an utterance.
GrainTTS uses these phoneme-to-audio alignments to obtain duration targets and
to align pitch, energy, and Mel-spectrogram features with the phoneme sequence.

We thank the [EfficientSpeech](https://github.com/roatienza/efficientspeech)
authors for their LJSpeech processing workflow. The preprocessing code in this
repository is adapted from EfficientSpeech, which follows the FastSpeech 2 data
pipeline.

### 3. Preprocess LJSpeech

Run the preprocessing command from the repository root:

```bash
python graintts/preprocess.py \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --textgrid-dir data/LJSpeech-1.1/TextGrid \
  --device auto
```

`--device auto` uses CUDA when it is available and otherwise runs on CPU.

This command cleans the transcripts, prepares normalized waveforms, installs
the TextGrids, and generates Mel spectrograms, pitch, energy, durations,
metadata splits, and normalization statistics. It writes everything under the
local `data/LJSpeech-1.1/` directory rather than into Git:

```text
data/LJSpeech-1.1/
├── metadata.csv
├── wavs/
├── TextGrid/LJSpeech/
├── raw_data/LJSpeech/LJSpeech/
└── preprocessed_data/LJSpeech/
    ├── TextGrid/LJSpeech/
    ├── mel/
    ├── pitch/
    ├── energy/
    ├── duration/
    ├── train.txt
    ├── val.txt
    ├── speakers.json
    └── stats.json
```

Full preprocessing creates normalized waveform copies and NumPy feature files,
so make sure the `data/` directory has enough free disk space.

For a quick setup test, add `--limit 1`; do not use this option for full
training. Once preprocessing finishes, validate the data layout:

```bash
python scripts/check_setup.py
```

### 4. Train GrainTTS

Train the paper model with L1, SSIM, and GVar supervision:

```bash
python graintts/train_l1_ssim_gvar.py \
  --run-name graintts-l1-ssim-gvar \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --hifigan-checkpoint hifigan/LJ_V2/generator_v2 \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 128 --num_workers 4 --max_epochs 5000 \
  --lr 0.001 --weight-decay 0.00001 --infer-device cuda
```

The available training objectives are:

```text
graintts/train.py                  # L1
graintts/train_l1_ssim.py          # L1 + SSIM
graintts/train_l1_ssim_gvar.py     # L1 + SSIM + GVar (GrainTTS)
```

Training checkpoints and TensorBoard logs are written under
`lightning_logs/<run-name>/`. Resume a full Lightning checkpoint by adding:

```bash
--checkpoint lightning_logs/<run-name>/checkpoints/last.ckpt
```

The released checkpoint in `checkpoints/` is inference-only and cannot resume
training. Add `--compile` to enable `torch.compile` for training.

No LJSpeech audio, TextGrid, or generated dataset feature is tracked by Git.
