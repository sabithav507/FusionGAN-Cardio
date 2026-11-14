# Verification Record — FusionGAN-Cardio Source Package

Validation date: 13 November 2026. This is a **software verification** report, not a biomedical experiment, and no clinical outcome is claimed.

## Environment used for verification

- Python 3.13, PyTorch 2.10 CPU, torchvision 0.25 CPU.
- Independent smoke fixture generator produces synthetic geometric gray images (NO clinical images).
- Full research configuration targets 256x256; small CPU smoke configuration uses 32x32.

## Executed checks

| Verification | Result |
|---|---|
| `pytest -q` — 10 tests on architecture, batch gradients, spectral PatchGAN shapes, class/view reference isolation, patient leakage, baseline shapes, diffusion sampling, Wilcoxon/Holm math | PASS (10/10) |
| `fusiongan smoke --workdir ...` — auxiliary anatomy encoder, SimCLR evaluator, CAMUS U-Net, teacher, core GAN, gated balance, classifier, heldout test and anatomy metrics | PASS |
| Proposed generator at 256x256, disease/view-conditioned forward | PASS: `[1,1,256,256]` |
| Conditional 70x70 PatchGAN at 256x256 | PASS: `[1,1,30,30]` |
| cGAN train/export pool/shared critic/quality-gated augmentation (CPU, one epoch) | PASS |
| DCGAN, WGAN-GP, cGAN, `stylegan2ada_lite`, `vitgan_lite`, DDPM, LDM training (CPU fixture, one epoch) | PASS for all seven |
| EchoNet binary classifier -> heldout CAMUS static-EF proxy command, with fixture metadata | PASS |
| Inference profiling, quality-score histograms, class balance and confusion-matrix figures | PASS |
| Python bytecode compilation | PASS |

## Known limits

- **No real TMED-2, EchoNet-Dynamic or CAMUS clinical dataset was used during software validation.** Model quality and manuscript numerical results are unverified.
- Real 256x256 multi-seed experiments, actual NVIDIA GPU mixed-precision performance, and medical pretrained external encoders have not been run in this verification environment.
- Data and original checkpoints must be obtained/licensed separately. Exact manuscript Appendix A1–A3 parameters are unavailable from the supplied document.
- The standalone `*_lite` comparators are functional research architectures, **not** canonical author-code reimplementations.

Use `pytest -q` and `fusiongan smoke --workdir outputs/smoke_validation` on the target workstation before installing real data. See README and REPRODUCIBILITY.md for complete commands and additional requirements.
