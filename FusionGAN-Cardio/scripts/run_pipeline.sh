#!/usr/bin/env bash
# Train supporting models, core GAN, quality gate and downstream classifier.
# DATA must already be prepared with prepare-tmed/echonet/camus. No datasets downloaded.
set -euo pipefail
CFG=${CFG:-configs/full.yaml}
TMED=${TMED:-data/processed/tmed_manifest.csv}
ECHO=${ECHO:-data/processed/echonet_manifest.csv}
CAMUS=${CAMUS:-data/processed/camus_manifest.csv}
OUT=${OUT:-outputs/main_run}
mkdir -p "$OUT"

fusiongan --config "$CFG" train-structure --camus "$CAMUS" --echonet "$ECHO" --out "$OUT/structure.pt"
fusiongan --config "$CFG" train-probe --camus "$CAMUS" --out "$OUT/probe.pt"
fusiongan --config "$CFG" train-evaluator --manifest "$TMED" --out "$OUT/evaluator.pt"
fusiongan --config "$CFG" train-teacher --manifest "$TMED" --out "$OUT/teacher.pt"
fusiongan --config "$CFG" train-gan --manifest "$TMED" --structure "$OUT/structure.pt" --evaluator "$OUT/evaluator.pt" --out "$OUT/fusiongan.pt"
fusiongan --config "$CFG" augment --manifest "$TMED" --generator "$OUT/fusiongan.pt" --teacher "$OUT/teacher.pt" --structure "$OUT/structure.pt" --out "$OUT/augmented"
fusiongan --config "$CFG" train-classifier --manifest "$TMED" --out "$OUT/real_only.pt"
fusiongan --config "$CFG" train-classifier --manifest "$OUT/augmented/balanced_manifest.csv" --out "$OUT/augmented_classifier.pt"
fusiongan --config "$CFG" eval-classifier --manifest "$TMED" --checkpoint "$OUT/augmented_classifier.pt" --out "$OUT/classification_test"
fusiongan --config "$CFG" eval-generator --manifest "$TMED" --checkpoint "$OUT/fusiongan.pt" --evaluator "$OUT/evaluator.pt" --out "$OUT/generation_test"
fusiongan --config "$CFG" eval-anatomy --camus "$CAMUS" --probe "$OUT/probe.pt" --structure "$OUT/structure.pt" --generator "$OUT/fusiongan.pt" --out "$OUT/anatomy"
echo "Finished. Results in $OUT"
