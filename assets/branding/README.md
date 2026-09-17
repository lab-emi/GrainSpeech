# GrainSpeech artwork

The README concept banner uses the white, navy (`#101f31`), Delft blue
(`#007dad`), and cyan (`#37b9e8`) visual language of
[EMI Lab at TU Delft](https://www.tudemi.com/). The new GrainSpeech emblem
combines a grain-shaped speech bubble, a waveform, and circuit traces.

`emi-logo.svg` is the existing lab header logo copied unchanged from
[`lab-emi/website-emi`](https://github.com/lab-emi/website-emi/blob/main/public/images/emi-logo.svg).
`emi-logo.png` is its white-background reference rendering. EMI branding retains
its respective ownership.

The banner was created with the built-in image generation tool, using the EMI
logo and both generated spectrogram plots as references. The exact prompt is in
[`banner-prompt.txt`](banner-prompt.txt). It is concept artwork: use the original
plots and raw arrays in [`../examples/`](../examples/) for inspecting model
predictions, since image generation can change fine details in the inserts.

The examples themselves are not image-generated. They come from running
[`generate_readme_examples.py`](../../scripts/generate_readme_examples.py) with
the released checkpoint. Their manifest records file hashes and generation
settings.
