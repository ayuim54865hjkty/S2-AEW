# S2-AEW

Official implementation for **Spatial-Spectral Adversarial Explanation Watermarking (S2-AEW)**, a lossless black-box ownership verification framework for hyperspectral image (HSI) classification models.

S2-AEW protects a trained HSI classifier by encoding ownership evidence into its spatial-spectral explanation behavior. The model parameters and task labels are not modified. Verification is performed by recovering spatial and spectral explanation signs from a suspicious model and testing whether they match the owner's private target bits.

## What This Repository Provides

- **S2 explanation carrier modeling**: deterministic spatial-block and spectral-band perturbation carriers.
- **S2-AEW sample generation**: prediction-preserving adversarial explanation optimization.
- **S2 statistical verification**: branch-wise and joint WSR, binomial p-value, and evidence-score reporting.
- **Key-sample support**: confidence/margin-based key candidate selection helpers.
- **Robustness utilities**: lightweight pruning and classifier-head fine-tuning helpers for stress tests.
- **HSI model backbones**: representative Hamida, Chen, and He 3-D CNN classifiers.
- **Reproducible project structure**: YAML configs, CLI smoke tests, unit tests, and method documentation.

Large datasets, model checkpoints, and raw experiment logs are intentionally excluded. The repository is designed to be small enough for review but complete enough to inspect and reuse the method core.

## Repository Structure

```text
s2aew/
  attacks.py         # pruning/fine-tuning robustness helpers
  config.py          # YAML experiment config dataclasses
  data.py            # HSI normalization and patch utilities
  explanation.py     # spatial and spectral LIME-style explanation carriers
  generation.py      # S2-AEW adversarial explanation sample generation
  keys.py            # stable key-sample candidate selection
  models.py          # representative HSI classifiers
  runner.py          # high-level three-stage S2-AEW pipeline
  targets.py         # deterministic target-bit generation from a private key
  verification.py    # WSR and binomial evidence-score verification
examples/
  toy_example.py     # minimal runnable example without external HSI data
configs/
  paviau_hamida.yaml # example experiment configuration
docs/
  method_overview.md
  reproducibility.md
tests/
  test_core.py
requirements.txt
```

## Installation

```bash
git clone https://github.com/ayuim54865hjkty/S2-AEW.git
cd S2-AEW
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

## Quick Checks

The toy example uses a randomly initialized classifier and random input only to check the software pipeline.

```bash
python examples/toy_example.py
```

The CLI reads a YAML configuration and performs the same smoke-test pipeline:

```bash
s2aew-demo --config configs/paviau_hamida.yaml --toy-steps 5
```

Run unit tests:

```bash
make test
```

## Core Usage

For real experiments, load a trained HSI classifier and a correctly shaped HSI patch, then call:

```python
from s2aew import S2AEWPipeline

pipeline = S2AEWPipeline(
    model=model,
    image_size=11,
    num_channels=103,
    key="private-owner-key",
    device=device,
)

watermarked_x = pipeline.generate(x, steps=200, eps=0.3)
report = pipeline.verify(suspicious_model=model, image=watermarked_x)
print(report)
```

## Verification Outputs

The report contains spatial, spectral, and joint branches:

- `wsr`: watermark success rate in percent;
- `matched`: recovered target signs;
- `total`: total evidence units;
- `p_value`: binomial evidence under the random sign-match model;
- `evidence_score`: `-log10(p)`.

## Paper-Level Experiments

The paper experiments use trained HSI classifiers, selected key samples, and dataset-specific settings. This public release keeps the executable method core and lightweight smoke tests in the repository. Full datasets, checkpoints, and generated watermark samples should be downloaded or released separately because they are large binary artifacts.

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
