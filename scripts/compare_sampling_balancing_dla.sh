#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/compare_sampling_balancing_dla"

# Avoid uv warning when VIRTUAL_ENV points to a different project.
unset VIRTUAL_ENV || true

run_manylatents() {
  local run_dir="$1"
  shift
  uv run python -m manylatents.main \
    "hydra.run.dir=${run_dir}" \
    "$@"
}

rm -rf "${OUT_DIR}"
mkdir -p \
  "${OUT_DIR}/unbalanced" \
  "${OUT_DIR}/random_balanced" \
  "${OUT_DIR}/label_balanced" \
  "${OUT_DIR}/diffusion_condensation_balanced"

# Visible points for this DLA config:
# 5000 + 7*300 = 7100. Label-balanced target is 8*300 = 2400.
RANDOM_BALANCED_N=2400

COMMON_ARGS=(
  "algorithms/latent=phate"
  "data=dla_tree_from_graph_imbalanced_5000_300"
  "callbacks/embedding=minimal"
  "metrics=noop"
  "seed=42"
  "log_level=info"
  "algorithms.latent.gamma=0"
  "algorithms.latent.knn=100"
  "algorithms.latent.t=25"
  "algorithms.latent.n_landmark=2000"
)

run_manylatents \
  "${OUT_DIR}/unbalanced" \
  "${COMMON_ARGS[@]}" \
  "name=dla_5000_300_unbalanced"

run_manylatents \
  "${OUT_DIR}/random_balanced" \
  "${COMMON_ARGS[@]}" \
  "name=dla_5000_300_random_balanced" \
  "+sampling.dataset._target_=manylatents.utils.sampling.RandomSampling" \
  "+sampling.dataset.n_samples=${RANDOM_BALANCED_N}" \
  "+sampling.dataset.seed=42"

run_manylatents \
  "${OUT_DIR}/label_balanced" \
  "${COMMON_ARGS[@]}" \
  "name=dla_5000_300_label_balanced" \
  "+sampling.dataset._target_=manylatents.utils.sampling.BalancedLabelSampling" \
  "+sampling.dataset.stratify_by=metadata" \
  "+sampling.dataset.seed=42"

run_manylatents \
  "${OUT_DIR}/diffusion_condensation_balanced" \
  "${COMMON_ARGS[@]}" \
  "name=dla_5000_300_diffusion_condensation_balanced" \
  "+sampling.dataset._target_=manylatents.utils.sampling.DiffusionCondensationSampling" \
  "+sampling.dataset.target_clusters=8" \
  "+sampling.dataset.landmarks=2000" \
  "+sampling.dataset.knn=100" \
  "+sampling.dataset.decay=40" \
  "+sampling.dataset.seed=42"

uv run python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("outputs/compare_sampling_balancing_dla")
runs = [
    ("unbalanced", "embeddings_dla_5000_300_unbalanced"),
    ("random_balanced", "embeddings_dla_5000_300_random_balanced"),
    ("label_balanced", "embeddings_dla_5000_300_label_balanced"),
    ("diffusion_condensation_balanced", "embeddings_dla_5000_300_diffusion_condensation_balanced"),
]

print("Row counts by run:")
for run_name, prefix in runs:
    run_dir = base / run_name
    csvs = sorted(run_dir.glob(f"{prefix}_*.csv"))
    if not csvs:
        print(f"- {run_name}: no embeddings csv found")
        continue
    df = pd.read_csv(csvs[-1])
    print(f"- {run_name}: {len(df)} rows")
PY

echo "Wrote comparison outputs to ${OUT_DIR}"
