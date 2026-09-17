#!/usr/bin/env python3
"""Build the GitHub Pages paper site using only the Python standard library."""

from array import array
from html import escape
import json
from pathlib import Path
import shutil
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website"
OUTPUT = ROOT / "_site"
MAIN_MODEL = "es117_l1_ssim_gvar"
LABELS = {
    "es117_l1": "GrainSpeech (L1)",
    MAIN_MODEL: "GrainSpeech",
    "es0_l1_ssim_gvar": "EfficientSpeech-Tiny (L1 + SSIM + GVar)",
    "ground-truth": "Original recording",
}


def audio_card(model, sample, *, highlight=False):
    clip = next(f for f in model["files"] if f["sample_id"] == sample["sample_id"])
    label = LABELS.get(model["model_id"], model["display_name"])
    if model["model_id"] == MAIN_MODEL:
        detail = "L1 + SSIM + GVar · 0.265M parameters"
    elif model["parameters"] is None:
        detail = "LJSpeech · human reference"
    else:
        detail = f'{model["parameters"] / 1_000_000:.3f}M acoustic parameters'
    path = escape("demo/" + clip["path"], quote=True)
    duration = f'{clip["duration_seconds"]:.2f} s'
    audio_label = escape(f'{label}, sample {sample["display_order"]}: {sample["text"]}', quote=True)
    tag = '<span class="tag">This work</span>' if highlight else ""
    return f'''<article class="audio-card{' highlight' if highlight else ''}" data-model="{model['model_id']}">
      <div class="audio-card-header"><div><h4>{escape(label)}</h4><small>{escape(detail)} · {duration}</small></div>{tag}</div>
      <audio controls preload="none" aria-label="{audio_label}" src="{path}"><a href="{path}">Download WAV</a></audio>
      <a class="download-audio" href="{path}" download>Download WAV</a>
    </article>'''


def demo_html(manifest):
    models = {model["model_id"]: model for model in manifest["models"]}
    buttons = ['<span>Choose a sentence</span>']
    panels = []
    for index, sample in enumerate(manifest["samples"]):
        number = sample["display_order"]
        panel_id = f"sample-{number}"
        buttons.append(
            f'<button type="button" data-sample="{panel_id}" aria-controls="{panel_id}" '
            f'aria-pressed="{str(index == 0).lower()}">Sample <span>{number:02}</span></button>'
        )
        primary = audio_card(models[MAIN_MODEL], sample, highlight=True)
        reference = audio_card(models["ground-truth"], sample)
        groups = []
        for group, title in (
            ("ours", "GrainSpeech ablation"),
            ("es-series", "EfficientSpeech family"),
            ("other-models", "Other acoustic models"),
        ):
            cards = "\n".join(
                audio_card(model, sample)
                for model in manifest["models"]
                if model["group"] == group and model["model_id"] != MAIN_MODEL
            )
            groups.append(f'<div class="model-group"><h3>{title}</h3><div class="model-grid">{cards}</div></div>')
        hidden = " hidden" if index else ""
        panels.append(f'''<section id="{panel_id}" class="sample-panel" aria-labelledby="sentence-{number}"{hidden}>
          <div class="transcript-label"><span id="sentence-{number}">SENTENCE {number:02}</span><span>{escape(sample['sample_id'])}</span></div>
          <p class="transcript">“{escape(sample['text'])}”</p>
          <div class="featured-comparison">{primary}{reference}</div>
          <details class="comparisons" open><summary>Compare all models <span>12 acoustic variants · one shared sentence</span></summary>{''.join(groups)}</details>
        </section>''')
    return "\n".join(buttons), "\n".join(panels)


def waveform_svg(path):
    """Peak envelope from the actual featured PCM audio, not decorative noise."""
    with wave.open(str(path), "rb") as recording:
        if recording.getsampwidth() != 2 or recording.getnchannels() != 1:
            raise ValueError("The featured waveform expects mono PCM-16 audio")
        samples = array("h", recording.readframes(recording.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    count = 96
    peaks = [max(abs(v) for v in samples[i * len(samples) // count:(i + 1) * len(samples) // count]) for i in range(count)]
    maximum = max(peaks) or 1
    bars = []
    for i, peak in enumerate(peaks):
        height = max(2, peak / maximum * 54)
        bars.append(f'<rect x="{i * 4}" y="{(58-height)/2:.2f}" width="2" height="{height:.2f}" rx="1"/>')
    return '<svg viewBox="0 0 384 58" preserveAspectRatio="none" fill="currentColor">' + "".join(bars) + "</svg>"


def main():
    manifest = json.loads((SOURCE / "demo/manifest.json").read_text())
    buttons, panels = demo_html(manifest)
    featured = SOURCE / "demo/audio/audio_es117_l1_ssim_gvar_04_LJ037-0157.wav"
    html = (SOURCE / "index.html").read_text()
    for marker, content in {
        "<!-- SAMPLE_BUTTONS -->": buttons,
        "<!-- DEMO_SAMPLES -->": panels,
        "<!-- WAVEFORM -->": waveform_svg(featured),
    }.items():
        if html.count(marker) != 1:
            raise ValueError(f"Expected exactly one template marker: {marker}")
        html = html.replace(marker, content)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(SOURCE, OUTPUT, ignore=shutil.ignore_patterns("README.md"))
    for relative in ("grainspeech-banner.png", "grainspeech_architecture.png"):
        destination = OUTPUT / "assets" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "assets" / relative, destination)
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    (OUTPUT / ".nojekyll").touch()
    print(f"Built {OUTPUT}: {len(manifest['samples'])} sentences, {len(manifest['models'])} model/reference tracks each.")


if __name__ == "__main__":
    main()
