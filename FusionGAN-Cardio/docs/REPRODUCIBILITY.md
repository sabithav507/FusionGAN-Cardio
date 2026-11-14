# Paper-to-code traceability and limitations

## Implemented mathematical components

| Manuscript reference | Module |
|---|---|
| Eq. (20) and frame extraction Eq. (21) | `data.py`, aspect-preserving grayscale preprocessing; uniform five-frame sampling |
| Eq. (25)–(30) | `models/core.py`: 128-d latent + two embeddings; 4x4 initialization; five residual upsamplings at 256 |
| Eq. (31)–(34) | `models/core.py`: 1x1 feature alignment + sample-wise learned softmax gates |
| Eq. (35)–(38) | `models/core.py`: channel attention + 7x7 spatial attention |
| Eq. (39) | `models/core.py`: final residual upsampling to 1x256x256 sigmoid |
| Eq. (40)–(45) | `models/core.py`: global projection D and receptive field-70 conditional PatchGAN; spectral norm |
| Eq. (46)–(49) | `losses.py`, `train.py`: hinge, generator adversarial, feature matching, class/view head supervision |
| Eq. (50)–(51) | `losses.py`: detached same-condition nearest real train embedding, differentiable generated embedding |
| Eq. (52)–(56) | `losses.py`, `train.py`: mode-seeking inverse-ratio, TV and alternating Adam updates |
| Eq. (57)–(62) | `quality.py`: frozen DenseNet teacher, realism score, feature distance plausibility, weighted gate |
| Eq. (24), (63)–(64) | `quality.py`: 90% group target and accepted-synthetic train-only manifest |
| Eq. (65)–(66) | `train.py`: EfficientNet-B0 with categorical cross entropy and AdamW |
| Eq. (67)–(70) | `metrics.py`, `evaluate.py`: feature Fréchet, KID, PR, CAMUS real/proxy Dice and HD95 |
| Eq. (71)–(74) | `metrics.py`, `statistics.py`: Macro-F1, balanced accuracy, sensitivity/specificity, 95% CIs |
| Algorithms 1–3 | `models/core.py`, `train.py`, `quality.py` |

## Source manuscript details that cannot be established from the supplied file alone

- Appendix Tables A1–A3 (per-seed results, exact full layer-by-layer definitions, detailed hyperparameter specifications) were referenced but not included in the supplied 38-page document. **The code never silently claims they were available.**
- TMED-2 image IDs and exact partition IDs giving the reported 2890/963/958 image counts are absent; available official split statistics differ (e.g. 360/119/120). Exact reconstruction requires original metadata and seed.
- The precise objective by which the frozen ResNet-18 structure encoder was jointly trained on CAMUS and EchoNet was not fully specified. This code uses mask segmentation supervision plus binary EF supervision, explicitly recorded in checkpoint metadata.
- The exact discriminator auxiliary-head training contribution was underspecified; this code adds a separately identifiable real-image class/view cross-entropy term to its hinge loss.
- Feature matching is defined using discriminator layers but sample pairing isn't unique; this implementation uses four batch-averaged feature statistics to avoid forcing arbitrary real/fake pixel correspondence.
- The manuscript states ~16.9M generator parameters, but the available channel-block description produces a different parameter count in this implementation. `count_params()` and model checkpoints provide verifiable figures; do not claim architecture-exact equivalence to missing Appendix A2.
- Full comparison to canonical StyleGAN2-ADA or published ViTGAN requires their original source code; `*_lite` architectural references are not exact replicas.
- Med-FID is valid as a medical-domain feature Fréchet only after medical-domain self-supervised pretraining of the *independent* encoder. Using only the labelled training cohort as fallback differs from the described unlabelled cohort. The evaluator records the actual training source.
- External CAMUS synthetic proxy Dice uses predicted LV structure to align a real expert reference; it cannot be interpreted as independent expert ground truth for synthetic images.

## Fixed main experiment configuration

- Image size: 1x256x256 (input normalized to [0,1]); latent `z` dimension 128; embeddings 32 each.
- Hierarchical feature-taps at 32x32, 64x64 and 128x128; aligned at 64x128x128.
- Multi-scale adversarial branches combined 0.5 global / 0.5 patch.
- GAN learning rates 2e-4, batch size 16, at most 300 epochs. Validation every second epoch, checkpoint minimum Med-FID and early-stopping patience 30 evaluations.
- Generator weights: adversarial 1, feature match 10, class/view 1, structure 5, mode-seeking 1, TV 1e-5.
- Quality gate: `Sc>=0.80` and `Q>=0.75`, with weights `(.40,.30,.30)`; nearest-real k=5 and Gaussian diversity normalization within (class,view).
- Balancing target: ceil(0.90 x max observed real training condition count). All original majority samples remain.
- Downstream AdamW at 1e-4, weight decay 1e-4, batch 32, 50 epochs max, patience 10 on validation Macro-F1.
- Main five seeds: 7, 21, 42, 77, 123; extra paired seeds: 256, 512, 1024, 2048, 4096. Bootstrap 10,000 independent patient/study sampling draws.

## Explicit quality and privacy requirements

- Never use clinical test subjects during generator training, metric-encoder training, gate training, anatomy-encoder fitting, or hyperparameter selection.
- Treat EchoNet static-frame EF and TMED image-level AS labels strictly as proxies.
- Do not tune the quality gate or augmentation ratio on the final test set.
- No scripts download clinical data automatically or redistribute access-controlled data.
- **A passed smoke test proves software wiring and shape correctness, NOT medical fidelity.**
