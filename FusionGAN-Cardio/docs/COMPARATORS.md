# Comparator implementations and fair quality gating

## Implemented families

| CLI name | Self-contained training | Condition handling | Reproduction fidelity |
|---|---|---|---|
| `dcgan` | Yes | Unconditional; one model per group | Practical DCGAN reference |
| `cgan` | Yes | Disease and view embeddings | Practical conv-cGAN reference |
| `wgangp` | Yes | Unconditional; one model per group | WGAN with gradient penalty, conditional study groups via separate models |
| `stylegan2ada_lite` | Yes | Disease/view conditioning; AdaIN-style modulation and restricted ADA heuristic | **NOT official StyleGAN2-ADA** architecture/training |
| `vitgan_lite` | Yes | Disease/view embedding + Transformer latent-token processing | **NOT official ViTGAN** code |
| `ddpm` | Yes | Conditional noise prediction + DDIM sampling | Compact pixel-space diffusion implementation |
| `ldm` | Yes | VAE compression + conditional latent noise prediction | Compact latent diffusion, **not Stable Diffusion** |
| FusionGAN-Cardio | Yes | Disease/view-conditioned full proposed generator | Manuscript-guided implementation; see reproducibility caveats |

All comparator code genuinely runs and trains end to end; no pseudo-code stand-ins. **Nevertheless, published canonical StyleGAN2-ADA and ViTGAN results cannot be reproduced by these lightweight surrogates.** To claim the precise paper's state-of-the-art baseline numbers, install and use the official implementations and their published training settings, with the common candidate-gating pipeline applied to their actual generated outputs. Do not rename these `*_lite` methods to original baseline names in an SCI manuscript.

## Training loop and fairness

1. Use identical training-only image preprocessing and fixed real partitions for every method.
2. Generate raw ungated pools with the same class-view quotas and sampling seeds.
3. Train **one** ResNet-18 realism critic to distinguish genuine training images from generated candidates; each generator must contribute equally to its fake class.
4. Use the **same** real-only teacher, frozen structure encoder, quality thresholds, and deficit rule across all generators.
5. Train a fresh EfficientNet-B0 from the same initial seed for each augmented manifest, with identical optimizer and early-stopping settings.
6. Evaluate only on the untouched real test and aggregate images to original study units for bootstrap.
7. Record per-seed balanced accuracy, Macro-F1, AUROC and confidence intervals; apply exact paired Wilcoxon and Holm corrections.

Unconditional per-group generators with insufficient observed train/validation images cannot be trained faithfully; a documented within-training holdout can support checkpoint selection, but changes the comparison protocol. Quota and quality-gate failures remain visible and must not be silently repaired.

## Candidate generation commands

Use `sample-candidates` before `train-critic`, then `augment` after training the critic. For class-view-specific DCGAN/WGAN-GP use a JSON `checkpoint_map` of the form `{"0:2": "/abs/path/dcgan_0_2.pt", ...}` and specify `--checkpoint-map` on the candidate generation and augmentation commands. Every observed class/view group must have a model checkpoint in that map.

Realism uses `P(real|image)` from the common critic when supplied; for a standalone FusionGAN run it may instead use the sigmoid-normalized global discriminator score. These are **different scoring sources** and must not be mixed within an allegedly matched comparison.
