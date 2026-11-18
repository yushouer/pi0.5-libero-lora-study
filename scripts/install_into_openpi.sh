#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/install_into_openpi.sh /path/to/openpi
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="${1:?Please provide openpi directory}"

cd "$OPENPI"

EXPECTED="$(cat "$PROJECT/reproducibility/openpi_commit.txt")"
ACTUAL="$(git rev-parse HEAD)"

if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "ERROR: openpi commit mismatch"
    echo "Expected: $EXPECTED"
    echo "Actual:   $ACTUAL"
    exit 1
fi

# Apply exact training modifications
git apply --check "$PROJECT/patches/openpi_training.patch"
git apply "$PROJECT/patches/openpi_training.patch"

if [ -s "$PROJECT/patches/dependencies.patch" ]; then
    git apply --check "$PROJECT/patches/dependencies.patch"
    git apply "$PROJECT/patches/dependencies.patch"
fi

# Install evaluation scripts
cp "$PROJECT/examples/libero/eval_book.py" \
   examples/libero/eval_book.py

cp "$PROJECT/scripts/eval_book_val_loss.py" \
   scripts/eval_book_val_loss.py

# Reproduce train/validation episode splits
mkdir -p results/book_task09

cp "$PROJECT/results/train_episodes.json" \
   results/book_task09/train_episodes.json

cp "$PROJECT/results/val_episodes.json" \
   results/book_task09/val_episodes.json

# Also support README's short paths
cp "$PROJECT/results/train_episodes.json" results/train_episodes.json
cp "$PROJECT/results/val_episodes.json" results/val_episodes.json

# Restore train-only normalization statistics
mkdir -p assets/pi05_libero_book_lora/physical-intelligence/libero

cp "$PROJECT/assets/norm_stats_book_train.json" \
   assets/pi05_libero_book_lora/physical-intelligence/libero/norm_stats.json

python3 -m py_compile \
    src/openpi/training/config.py \
    src/openpi/training/data_loader.py \
    scripts/eval_book_val_loss.py \
    examples/libero/eval_book.py

echo "Pi0.5 LoRA experiment code installed successfully."
