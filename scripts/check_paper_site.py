#!/usr/bin/env python3
"""Verify the published HTML, asset paths and unmodified demo recordings."""

from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import wave

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.urls = []
        self.players = []
        self.models = []
        self.panels = []
        self.buttons = []
        self.h1_count = 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.urls.extend(attrs[key] for key in ("src", "href") if key in attrs)
        if tag == "h1":
            self.h1_count += 1
        if tag == "audio":
            assert "controls" in attrs and attrs.get("preload") == "none"
            assert attrs.get("aria-label"), "Audio controls must have a model and sentence label"
            self.players.append(attrs["src"])
        if "data-model" in attrs:
            self.models.append(attrs["data-model"])
        if "sample-panel" in attrs.get("class", "").split():
            self.panels.append(attrs)
        if "data-sample" in attrs:
            self.buttons.append(attrs)


def main():
    page = Page()
    html = (SITE / "index.html").read_text()
    page.feed(html)
    assert page.h1_count == 1
    assert len(page.ids) == len(set(page.ids)), "Duplicate HTML IDs"
    assert not re.search(r"<!-- (?:WAVEFORM|SAMPLE_BUTTONS|DEMO_SAMPLES) -->", html)
    assert "https://arxiv.org/abs/2609.18856" in page.urls
    assert "https://github.com/lab-emi/GrainSpeech" in page.urls
    assert {"assets/emi-logo.svg", "assets/tu-delft-logo.svg"} <= set(page.urls)
    for url in page.urls:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        assert not parts.path.startswith("/"), f"Project Pages needs a relative asset path: {url}"
        if parts.path:
            assert (SITE / unquote(parts.path)).is_file(), f"Missing asset: {url}"
        elif parts.fragment:
            assert parts.fragment in page.ids, f"Broken anchor: {url}"
    for url in re.findall(r"url\(['\"]?([^)'\"]+)", (SITE / "styles.css").read_text()):
        assert (SITE / url).is_file(), f"Missing CSS asset: {url}"
    assert len(page.panels) == len(page.buttons) == 5
    assert sum("hidden" not in p for p in page.panels) == 1
    assert sum(b.get("aria-pressed") == "true" for b in page.buttons) == 1
    assert {b["aria-controls"] for b in page.buttons} == {p["id"] for p in page.panels}

    manifest = json.loads((SITE / "demo/manifest.json").read_text())
    assert len(manifest["models"]) == 13 and len(manifest["samples"]) == 5
    expected_players = ["assets/examples/compact-speech.wav"]
    sample_ids = {sample["sample_id"] for sample in manifest["samples"]}
    for model in manifest["models"]:
        assert {clip["sample_id"] for clip in model["files"]} == sample_ids
        for clip in model["files"]:
            relative = "demo/" + clip["path"]
            path = SITE / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"], f"Changed recording: {path}"
            with wave.open(str(path), "rb") as recording:
                assert recording.getframerate() == clip["sample_rate"] == 22050
                assert recording.getnchannels() == clip["channels"] == 1
                assert recording.getsampwidth() == clip["sample_width_bytes"] == 2
                assert recording.getnframes() == clip["frames"]
            expected_players.append(relative)
    assert Counter(page.players) == Counter(expected_players), "Every recording must be playable exactly once"
    assert Counter(page.models) == Counter({m["model_id"]: 5 for m in manifest["models"]})
    print("Verified 66 audio players, all 65 original recording hashes and WAV headers, five sample panels, local links, fonts and logos.")


if __name__ == "__main__":
    main()
