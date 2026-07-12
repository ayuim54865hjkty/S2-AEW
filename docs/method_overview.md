# Method Overview

S2-AEW protects an HSI classifier without modifying its parameters. The owner keeps a small private key set of HSI patches. Each key patch is optimized so that the model's explanation signs, rather than its predicted label, carry the watermark.

## 1. Spatial-Spectral Explanation Carrier

For an input patch with shape `C x H x W`, S2-AEW builds two explanation carriers.

- The spatial carrier masks local spatial cells and estimates `H x W` spatial-layout contributions.
- The spectral carrier masks individual HSI bands and estimates `C` spectral-band contributions.

The sign of each contribution becomes one verifiable bit. A single key patch therefore gives `HW + C` ownership evidence units.

## 2. Prediction-Preserving Watermark Sample Generation

Given private target signs, S2-AEW optimizes the input patch under an `L_inf` bound. The objective aligns spatial and spectral explanation signs with the secret bits while penalizing changes in the original prediction distribution.

The protected model remains fixed throughout this process. This is why the watermark is lossless with respect to model parameters.

## 3. Statistical Ownership Verification

Verification replays the same explanation protocol on a suspicious model and compares recovered signs with the private target signs. The implementation reports:

- spatial WSR;
- spectral WSR;
- joint WSR;
- binomial p-value under the random sign-match model;
- `-log10(p)` evidence score.

The p-value is used as a compact evidence score, not as a claim that every explanation bit is physically independent.
