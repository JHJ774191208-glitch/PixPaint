# JiT prior implementation

This directory contains the JiT-related code used by PixPaint:

- `model_jit.py`: JiT network definitions.
- `denoiser.py`: JiT denoiser wrapper used for native generation and training.
- `engine_jit.py` and `main_jit.py`: original JiT training and evaluation loop.
- `prepare_ref.py`: reference-statistic preparation utility.
- `util/`: JiT data-cropping, learning-rate, distributed-training, and model
  utilities.
- `environment.yaml`: JiT's original environment specification.

The files in this directory are kept separate from the public PixPaint
evaluation utilities. They do **not** include PixPaint's mask-guided sampling,
CMI, FPPR, Pixel-Predict Resampling, model checkpoints, datasets, or generated
results.

## Attribution and license

JiT is based on the public implementation at
https://github.com/LTH14/JiT. The upstream MIT license is retained verbatim in
[`LICENSE`](LICENSE). Several JiT source files also cite SiT and Lightning-DiT;
their in-file attribution notices are preserved.
