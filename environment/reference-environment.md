# Verified training environment

- Python: 3.11.15
- PyTorch: 2.12.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Lightning: 2.6.5
- NumPy: 1.26.4
- SciPy: 1.14.1

`reference-pip-freeze.txt` records the complete package snapshot from the working
training environment. Platform-specific CUDA packages are recorded for audit;
install them only when the target CUDA driver and package index support them.
