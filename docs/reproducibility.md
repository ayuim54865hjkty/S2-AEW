# Reproducibility Notes

This repository intentionally separates lightweight method code from heavy experiment artifacts.

## Included

- core spatial and spectral explanation carriers;
- adversarial explanation sample generation;
- ownership verification metrics;
- representative HSI classifier architectures;
- key selection helpers;
- toy and CLI examples.

## Not Included

- raw HSI datasets;
- pretrained checkpoints;
- generated watermark samples;
- full experiment logs.

These files are large and should be stored in dataset/model repositories or released as supplementary artifacts.

## Recommended Workflow

1. Train or load an HSI classifier.
2. Select correctly classified key samples with sufficient prediction margin.
3. Generate deterministic spatial and spectral target bits from a private key.
4. Optimize prediction-preserving S2-AEW samples.
5. Verify a suspicious model using branch-wise and joint WSR/p-value evidence.

The toy example only checks the software pipeline. It does not reproduce paper-level WSR because it uses a random model and random input.
