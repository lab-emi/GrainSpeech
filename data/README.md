# Dataset mount point

Run `../scripts/configure_data.sh /absolute/path/to/LJSpeech-1.1` from anywhere
inside the cloned repository. The dataset root must contain
`preprocessed_data/LJSpeech`. The script creates local symbolic links without
copying any dataset content. Dataset files are ignored by Git and must never be
committed.
