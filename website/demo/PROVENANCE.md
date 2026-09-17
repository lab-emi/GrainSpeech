# Demo recording provenance

Imported from [lab-emi/GrainTTS-Audio-Demo](https://github.com/lab-emi/GrainTTS-Audio-Demo)
at commit [`ffb3140acb3f3ebdb4ce96059bcf4f2bcaebbef4`](https://github.com/lab-emi/GrainTTS-Audio-Demo/tree/ffb3140acb3f3ebdb4ce96059bcf4f2bcaebbef4).

The 65 WAV files are unchanged: five shared LJSpeech utterances for each of
12 acoustic-model variants and the original-recording reference. All are mono
PCM-16 at 22,050 Hz. SHA-256 hashes and the original evaluation metadata remain
in `manifest.json`. Only file paths were prefixed with `audio/` to reflect the
new directory. Redundant base64 JavaScript audio copies were not imported.

The site uses these clearer display labels while preserving original model IDs:

| Original ID | Website label |
| --- | --- |
| `es117_l1` | GrainSpeech (L1) |
| `es117_l1_ssim_gvar` | GrainSpeech (L1 + SSIM + GVar) |
| `es0_l1_ssim_gvar` | EfficientSpeech-Tiny (L1 + SSIM + GVar) |
| `ground-truth` | Original recording |

Transcripts, sample order, parameter counts and original manifest metrics are
preserved. The 128-utterance aggregate results shown on the page are taken from
Table 1 of the [preprint](https://arxiv.org/html/2609.18856v1#S3), rather than
computed from these five listening examples. Synthesized audio uses a shared
HiFi-GAN v2 vocoder; reference recordings bypass the vocoder.
