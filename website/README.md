# GrainSpeech paper website

The single-page paper website and listening demo are maintained in the same
repository as the acoustic model: <https://lab-emi.github.io/GrainSpeech/>.
GitHub Actions builds and publishes it on changes to `main`. In repository
**Settings → Pages**, the source must be **GitHub Actions**.

## Local preview

From the repository root, with Python 3.10+ (no third-party packages required):

```bash
python3 scripts/build_paper_site.py
python3 scripts/check_paper_site.py
python3 -m http.server 4322 --directory _site
```

Open <http://localhost:4322>. Edit `website/index.html`, `styles.css` and `site.js`,
then rebuild. The build inserts the sentence panels from `demo/manifest.json`,
draws a waveform from the featured audio, and copies the relevant shared assets
from the root `assets/` directory into `_site/`. Do not edit `_site/` directly.
Relative asset paths support both local preview and the GitHub Pages project URL.

## Audio and artwork

- All 65 original comparison recordings were imported byte-for-byte from
  `lab-emi/GrainTTS-Audio-Demo`. See [audio provenance](demo/PROVENANCE.md).
- The 1.28-second opening clip is the GrainSpeech (L1 + SSIM + GVar) rendition of
  LJSpeech sample `LJ037-0157`: “Taken from Oswald.” It reuses the unchanged
  Sample 04 comparison recording, with its provenance in `demo/manifest.json`.
- Players load audio on demand. Choosing a sentence pauses hidden players;
  starting a recording pauses any other recording. The complete comparison
  works without JavaScript, with all five sentences shown.
- Model comparisons are expanded by default. Visitors can collapse them, and
  their choice is preserved when switching sentences.
- The EMI logo is the original SVG from the
  [EMI website repository](https://github.com/lab-emi/website-emi/blob/main/public/images/emi-logo.svg).
- The TU Delft logo is the original SVG served by the
  [official TU Delft website](https://www.tudelft.nl/_assets/2f383d4a929ad3d42eff11e81cdd4068/img/logo.svg),
  retrieved on 2026-09-17. Institutional logos retain their owners' rights.
- Manrope and Newsreader match the lab website. Their OFL licenses are kept in
  `fonts/`. The shared banner's provenance is in `assets/branding/README.md` at
  the repository root.

The static website does not run text-to-speech inference in the visitor's
browser. Use the software's Quick Start to synthesize new sentences.
