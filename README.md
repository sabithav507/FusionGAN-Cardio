[![DOI](https://zenodo.org/badge/1457415116.svg)](https://doi.org/10.5281/zenodo.24213570)

# FusionGAN-Cardio

**A Multi-Level Feature Fusion Generative Adversarial Network for Synthetic Echocardiographic Image Generation and Quality-Controlled Augmentation for Cardiovascular Disease Classification**

FusionGAN-Cardio is a modular PyTorch research implementation for disease- and view-conditioned synthesis of grayscale echocardiographic images. It combines multi-level feature fusion, FiLM conditioning, convolutional block attention, anatomy-aware constraints, adversarial learning, synthetic-image quality screening, and deficit-driven augmentation. The package also provides downstream classification, independent generative evaluation, anatomical assessment, baseline comparisons, statistical analysis, visualization, and software tests.

> **Research-use notice:** FusionGAN-Cardio is a research software package, not a medical device. Its image-level aortic stenosis (AS) classification and static-frame ejection-fraction (EF) prediction are proxy tasks. They are not substitutes for Doppler-based AS severity assessment or video-based EF measurement. The software has not been validated for clinical diagnosis or clinical decision-making.

## Project contents

The source package includes:

- PyTorch model implementations and training routines.
- Configuration files for the full research pipeline and a small software smoke test.
- Dataset preparation, manifest validation, and split/leakage audits.
- Supporting anatomy, segmentation-probe, classification-teacher, and generative-evaluation models.
- The proposed FusionGAN-Cardio generator and conditional global and patch discriminators.
- Composite adversarial, feature-matching, class/view, structure, mode-seeking, and total-variation objectives.
- Candidate generation, shared-critic scoring, quality gating, and training-set balancing.
- Downstream classification, generative metrics, anatomical assessment, baselines, statistics, and plotting utilities.
- Unit tests and a non-clinical end-to-end smoke workflow.

The source archive contains software, configuration, tests, and documentation. It does **not** include controlled-access clinical datasets, pretrained checkpoints, or fabricated experimental results.

## Method overview

The proposed generator uses a latent representation together with disease and view conditioning. It applies conditional residual generation, feature alignment and learned gated fusion at three feature scales, FiLM conditioning, and CBAM attention. The discriminator combines a projection-conditioned global branch with a conditional PatchGAN branch.

The training objective combines adversarial learning with feature matching, class/view supervision, a nearest-reference structural constraint, mode-seeking regularization, and total variation regularization.

Synthetic candidates are screened before they are added to the training manifest. The configured quality gate requires both:

- `Sc >= 0.80`
- `Q >= 0.75`, where `Q = 0.4 Sc + 0.3 Sr + 0.3 Sd`

Accepted synthetic images are used for training augmentation only. Original training images are retained, and accepted additions are constrained by the observed disease-view training-group deficit. If candidates do not pass the configured gate, the pipeline records the shortfall rather than weakening the thresholds or inventing accepted samples.

## Supported datasets

The implementation includes adapters for the following datasets. Users must obtain each dataset independently and comply with its access conditions, license, and terms of use.

| Dataset | Intended role in this implementation | Source |
|---|---|---|
| TMED-2 | Echocardiographic still images, study-level AS labels, and view-conditioned synthesis/classification | [TMED-2](https://tmed.cs.tufts.edu/tmed_v2.html) |
| EchoNet-Dynamic | Echocardiography videos, official video partitions, sampled frames, and EF-related proxy supervision | [EchoNet-Dynamic](https://echonet.github.io/dynamic/) |
| CAMUS | End-diastolic/end-systolic images and expert anatomical segmentation masks | [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) |

No dataset is redistributed with this repository. Dataset preparation requires locally obtained source data and metadata. See [`docs/DATASETS.md`](docs/DATASETS.md) for the manifest schema, expected directory structures, label conventions, split requirements, and data-integrity guidance.

### Important dataset and label limitations

- TMED-2 provides study-level AS labels; image-level use of those labels is a proxy and does not constitute Doppler-based severity measurement.
- TMED-2 includes multiple studies from some patients. Patient identifiers or an official patient-disjoint split are needed to verify patient-level separation. Study-level grouping alone is a weaker guarantee.
- EchoNet-Dynamic frames are sampled from videos; the implementation does not synthesize videos or estimate EF from an individual still frame.
- EchoNet-Dynamic EF classes are defined in the implementation as reduced EF below 50% and preserved EF at or above 50%.
- CAMUS mask labels 1, 2, and 3 represent the LV cavity, myocardium, and atrium, respectively.
- Synthetic-image anatomical Dice/HD95 based on nearest-reference alignment is explicitly a proxy, not expert-labelled ground truth for generated images.

## System requirements

- Python 3.10 or later.
- PyTorch 2.1 or later.
- torchvision 0.16 or later, matched to the installed PyTorch build.
- A CPU or compatible CUDA environment.
- Additional disk space for datasets, extracted frames, checkpoints, generated images, and evaluation outputs.

The package dependencies are declared in [`pyproject.toml`](pyproject.toml). CUDA users should install a PyTorch/torchvision build compatible with their CUDA environment by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Installation

Clone or download this repository, then run the following commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,metrics]"
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,metrics]'
```

For a minimal installation without development and optional metric dependencies:

```bash
pip install -e .
```

The optional `metrics` extra installs LPIPS and torchmetrics. Legacy CAMUS `.mhd` files require SimpleITK:

```bash
pip install SimpleITK
```

Newer CAMUS `.nii.gz` files are handled through nibabel, which is included in the declared dependencies. Some optional external metrics require separately obtained pretrained weights. Such weights are not bundled with the source archive.

Verify the PyTorch installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Verify the software installation

The repository provides a non-clinical smoke workflow that uses artificial geometric grayscale images. It is intended to test software wiring, tensor shapes, gradients, and the end-to-end pipeline without requiring clinical datasets.

Run the tests:

```bash
pytest -q
```

Run the smoke workflow:

```bash
fusiongan smoke --workdir outputs/smoke_validation
```

See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

## Data preparation

Obtain the datasets independently from their official sources. Organize the local data and create manifests according to [`docs/DATASETS.md`](docs/DATASETS.md). The full configuration is [`configs/full.yaml`](configs/full.yaml).

### TMED-2

Prepare a metadata CSV containing the required source image and study information, including known view labels and patient identifiers where available. Then run:

```bash
fusiongan --config configs/full.yaml prepare-tmed \
  --metadata data/raw/tmed/image_labels.csv \
  --root data/raw/tmed/images \
  --out data/processed/tmed_manifest.csv
```

### EchoNet-Dynamic

Place `FileList.csv` and the `Videos/` directory under the local EchoNet-Dynamic data root, then run:

```bash
fusiongan --config configs/full.yaml prepare-echonet \
  --root data/raw/EchoNet-Dynamic \
  --images data/processed/echonet_frames \
  --out data/processed/echonet_manifest.csv
```

The adapter preserves official video-level partitions and samples up to five usable frames per video. It does not create synthetic videos.

### CAMUS

Place the CAMUS patient data in the local CAMUS data root, then run:

```bash
fusiongan --config configs/full.yaml prepare-camus \
  --root data/raw/CAMUS \
  --images data/processed/camus_endpoints \
  --out data/processed/camus_manifest.csv
```

Readable official patient split files are used when available; otherwise, the implementation uses a deterministic patient-level split.

### Audit prepared manifests

Run the corresponding audits before training:

```bash
fusiongan audit --manifest data/processed/tmed_manifest.csv --task tmed
fusiongan audit --manifest data/processed/echonet_manifest.csv --task echonet
fusiongan audit --manifest data/processed/camus_manifest.csv --task camus
```

Do not claim verified patient-level isolation when reliable patient identifiers are unavailable. Keep the dataset release, source metadata, manifest, split identifiers, and checksums associated with each experiment.

## Training and evaluation workflow

The following commands describe the principal TMED-2 workflow. They assume that the datasets have been prepared and audited and that the required manifests are available.

### 1. Train supporting models

```bash
fusiongan --config configs/full.yaml train-structure \
  --camus data/processed/camus_manifest.csv \
  --echonet data/processed/echonet_manifest.csv \
  --out outputs/checkpoints/structure.pt

fusiongan --config configs/full.yaml train-probe \
  --camus data/processed/camus_manifest.csv \
  --out outputs/checkpoints/probe.pt

fusiongan --config configs/full.yaml train-teacher \
  --manifest data/processed/tmed_manifest.csv \
  --out outputs/checkpoints/teacher_tmed.pt

fusiongan --config configs/full.yaml train-evaluator \
  --manifest data/processed/tmed_manifest.csv \
  --out outputs/checkpoints/simclr_tmed.pt
```

For an independent SimCLR encoder trained on an unlabelled TMED-2 manifest, provide the optional `--unlabeled-manifest` argument as documented in the project implementation. The unlabelled data must be disjoint from held-out labelled patients/studies.

### 2. Train FusionGAN-Cardio

```bash
fusiongan --config configs/full.yaml train-gan \
  --manifest data/processed/tmed_manifest.csv \
  --task tmed \
  --structure outputs/checkpoints/structure.pt \
  --evaluator outputs/checkpoints/simclr_tmed.pt \
  --out outputs/checkpoints/fusiongan_tmed.pt
```

A separate EchoNet-Dynamic generator and a binary real-only EchoNet teacher are required for the independent EF proxy task. The three-class TMED-2 teacher must not be reused for the two-class EchoNet task.

The training command saves the best validation checkpoint to the requested output path and a resumable `last_<name>.pt` checkpoint alongside it. Resume training using the corresponding `--resume` argument.

### 3. Generate quality-screened training augmentation

```bash
fusiongan --config configs/full.yaml augment \
  --manifest data/processed/tmed_manifest.csv \
  --generator outputs/checkpoints/fusiongan_tmed.pt \
  --teacher outputs/checkpoints/teacher_tmed.pt \
  --structure outputs/checkpoints/structure.pt \
  --out outputs/augmentation_fusion
```

The augmentation output includes a balanced training manifest, an augmentation report, candidate score files, and accepted synthetic PNG images. Only accepted training additions are included in the augmented training manifest. A recorded shortfall means that the configured acceptance criteria were not met for all requested groups.

### 4. Train and evaluate downstream classifiers

Train a real-only reference classifier:

```bash
fusiongan --config configs/full.yaml train-classifier \
  --manifest data/processed/tmed_manifest.csv \
  --out outputs/checkpoints/real_only.pt
```

Train a classifier using the augmented training manifest:

```bash
fusiongan --config configs/full.yaml train-classifier \
  --manifest outputs/augmentation_fusion/balanced_manifest.csv \
  --out outputs/checkpoints/augmented.pt
```

Evaluate on the real held-out test partition:

```bash
fusiongan --config configs/full.yaml eval-classifier \
  --manifest data/processed/tmed_manifest.csv \
  --checkpoint outputs/checkpoints/augmented.pt \
  --out outputs/validation_fusion \
  --split test
```

Held-out classifier evaluation uses real test images; synthetic images are not included in held-out metrics. The implementation reports image-level and per-study aggregated metrics.

### 5. Evaluate generation and anatomy

```bash
fusiongan --config configs/full.yaml eval-generator \
  --manifest data/processed/tmed_manifest.csv \
  --checkpoint outputs/checkpoints/fusiongan_tmed.pt \
  --evaluator outputs/checkpoints/simclr_tmed.pt \
  --out outputs/generative_fusion \
  --split test

fusiongan --config configs/full.yaml eval-anatomy \
  --camus data/processed/camus_manifest.csv \
  --probe outputs/checkpoints/probe.pt \
  --structure outputs/checkpoints/structure.pt \
  --generator outputs/checkpoints/fusiongan_tmed.pt \
  --out outputs/camus_validation
```

The generative evaluation includes medical-feature Fréchet distance, KID, and precision/recall, with optional LPIPS/MS-SSIM and optional independent Inception or RadImageNet-based metrics when their required dependencies and weights are available. Comparisons using medical-feature Fréchet distance must use the same held-out encoder and image subset.

The CAMUS real-image assessment uses expert masks. Synthetic nearest-reference aligned Dice/HD95 is a proxy and must not be reported as expert-labelled synthetic ground truth.

## Baseline comparisons

The implementation includes the following operational baseline options:

- DCGAN
- Conditional GAN (cGAN)
- WGAN-GP
- DDPM
- Latent Diffusion Model (LDM)
- `stylegan2ada_lite`
- `vitgan_lite`

Example cGAN training command:

```bash
fusiongan --config configs/full.yaml train-baseline \
  --name cgan \
  --manifest data/processed/tmed_manifest.csv \
  --evaluator outputs/checkpoints/simclr_tmed.pt \
  --out outputs/checkpoints/cgan.pt
```

For fair quality-gated comparisons, the implementation supports candidate pools and a shared realism critic. Use the same shared critic for all compared generators, including FusionGAN-Cardio, and apply matched evaluation data and settings. Do not compare scores produced by different critics as though they were directly equivalent.

The `stylegan2ada_lite` and `vitgan_lite` implementations are lightweight operational research comparators; they are not claimed to be canonical reproductions of the original StyleGAN2-ADA or ViTGAN implementations. Additional comparator details and limitations are documented in [`docs/COMPARATORS.md`](docs/COMPARATORS.md).

## Statistical analysis and visualization

The package provides paired-seed comparison, study-level bootstrap, and plotting commands. The statistical utilities include mean and standard deviation, Student's t confidence intervals, exact Wilcoxon testing, Holm correction, rank-biserial effect size, Cohen's dz, and study-level bootstrap analysis.

For the paired-seed comparison utility, the input CSV uses the columns `seed`, `method`, and `metric`; the proposed method is labelled `fusiongan`. Example:

```bash
fusiongan compare-seeds \
  --csv outputs/per_seed_macro_f1.csv \
  --out outputs/paired_stats.json
```

Example study-level bootstrap:

```bash
fusiongan bootstrap \
  --a outputs/real_only/test_predictions.csv \
  --b outputs/augmented/test_predictions.csv \
  --metric macro_f1 \
  --reps 10000 \
  --out outputs/bootstrap.json
```

Available visualization commands include training curves, synthetic grids, class balance, confusion matrices, and quality-score plots. Example:

```bash
fusiongan plot-curves \
  --log outputs/checkpoints/fusiongan_tmed.jsonl \
  --out outputs/fig_training.png

fusiongan plot-grid \
  --manifest outputs/augmentation_fusion/balanced_manifest.csv \
  --out outputs/fig_samples.png

fusiongan plot-balance \
  --manifest outputs/augmentation_fusion/balanced_manifest.csv \
  --out outputs/fig_balance.png

fusiongan plot-confusion \
  --predictions outputs/validation_fusion/test_predictions.csv \
  --out outputs/fig_confusion.png

fusiongan plot-quality \
  --augmentation-dir outputs/augmentation_fusion \
  --out outputs/fig_quality.png
```

Statistical significance does not establish prospective clinical utility. Bootstrap resampling in this implementation is performed at the study level rather than the individual-frame level.

## Reproducibility and known limitations

For full implementation caveats, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

## Repository structure

```text
FusionGAN-Cardio/
├── configs/
│   ├── full.yaml
│   └── smoke.yaml
├── docs/
│   ├── COMPARATORS.md
│   ├── DATASETS.md
│   └── REPRODUCIBILITY.md
├── scripts/
│   ├── run_pipeline.sh
│   └── run_seed_study.py
├── src/
│   └── fusiongan_cardio/
│       ├── models/
│       ├── baseline_train.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── losses.py
│       ├── metrics.py
│       ├── quality.py
│       ├── statistics.py
│       ├── train.py
│       └── visualize.py
├── tests/
├── pyproject.toml
├── README.md
└── VALIDATION_REPORT.md
```

## Software verification record

The included [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) records the verification environment and executed checks. The recorded checks include:

- 10 passing unit tests covering architecture, gradients, discriminator output shapes, data-leakage checks, baseline shapes, diffusion sampling, and statistical utilities.
- A non-clinical end-to-end smoke workflow.
- Fixture-based checks of the proposed generator, PatchGAN, baseline training, quality-gated augmentation, classifier evaluation, profiling, and plotting commands.

These results describe software verification in the recorded environment only. They are not clinical study results or evidence of diagnostic performance.

## Data, ethics, and responsible use

Users are responsible for obtaining data lawfully, following dataset-specific terms, protecting patient privacy, and ensuring that any derived outputs are handled according to applicable institutional and legal requirements. Do not upload protected clinical data, patient identifiers, or restricted pretrained weights to a public repository or public archive.

Generated echocardiographic images may contain unrealistic or misleading structures. They must not be used for patient care, diagnosis, treatment decisions, or as a replacement for expert-reviewed clinical imaging.


## Acknowledgements

This implementation is designed to support reproducible research in echocardiographic image synthesis, quality-controlled data augmentation, and cardiovascular image analysis. Dataset creators and maintainers retain ownership of their respective datasets and should be cited according to their terms.


