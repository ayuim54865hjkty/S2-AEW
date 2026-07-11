# S2-AEW

Official core implementation for **Spatial-Spectral Adversarial Explanation Watermarking (S2-AEW)**, a lossless black-box ownership verification framework for hyperspectral image (HSI) classification models.

The released code focuses on the method core:

- spatial-spectral explanation carrier modeling;
- prediction-preserving adversarial explanation sample generation;
- branch-wise and joint ownership verification;
- representative HSI classifiers used in the paper.

Large datasets, model checkpoints, and raw experiment logs are intentionally excluded.

## Repository Structure

```text
s2aew/
  explanation.py     # spatial and spectral LIME-style explanation carriers
  generation.py      # S2-AEW adversarial explanation sample generation
  verification.py    # WSR and binomial evidence-score verification
  targets.py         # deterministic target-bit generation from a private key
  models.py          # representative HSI classifiers
examples/
  toy_example.py     # minimal runnable example without external HSI data
requirements.txt
```

## Installation

```bash
git clone https://github.com/ayuim54865hjkty/S2-AEW.git
cd S2-AEW
pip install -r requirements.txt
```

## Minimal Example

The toy example uses a randomly initialized classifier and random input only to check the pipeline.

```bash
python examples/toy_example.py
```

For real experiments, load a trained HSI classifier and a correctly shaped HSI patch, then call:

```python
from s2aew.generation import generate_s2aew_sample
from s2aew.targets import make_spatial_spectral_targets
from s2aew.verification import verify_s2aew

targets = make_spatial_spectral_targets(
    key="private-key",
    spatial_bits=121,
    spectral_bits=103,
    device=device,
)

watermarked_x = generate_s2aew_sample(
    model=model,
    image=x,
    target_spatial=targets.spatial,
    target_spectral=targets.spectral,
    image_size=11,
    num_channels=103,
    steps=200,
)

report = verify_s2aew(
    model=model,
    image=watermarked_x,
    target_spatial=targets.spatial,
    target_spectral=targets.spectral,
    image_size=11,
    num_channels=103,
)
print(report)
```

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{s2aew,
  title   = {Spatial-Spectral Adversarial Explanation Watermarking for Hyperspectral Image Classification Models},
  author  = {Anonymous},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

## License

This repository is released for academic research use. Please check the license file before redistribution.
