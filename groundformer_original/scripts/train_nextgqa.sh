#!/usr/bin/env bash
# Train GroundFormer on NExT-GQA and evaluate the best checkpoint on the test split.
#
# Usage:   ./scripts/train_nextgqa.sh [GPU_ID] [SAVE_DIR]
# Example: ./scripts/train_nextgqa.sh 0 checkpoints/nextgqa
set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR=${2:-"$ROOT/checkpoints/nextgqa"}
[[ "$SAVE_DIR" != /* ]] && SAVE_DIR="$ROOT/$SAVE_DIR"
mkdir -p "$SAVE_DIR"

cd "$ROOT/code/GroundFormer"

CUDA_VISIBLE_DEVICES=$GPU python main_dual.py \
  --dataset=nextgqa \
  --test=train \
  --baseline=dual \
  --feat_type=CLIPL \
  --mc=5 \
  --epochs=30 \
  --lr=1e-5 \
  --batch_size=16 \
  --batch_size_val=64 \
  --qmax_words=30 \
  --amax_words=38 \
  --max_feats=32 \
  --embd_dim=768 \
  --ff_dim=768 \
  --n_layers=2 \
  --dropout=0.0 \
  --seed=666 \
  --lan=RoBERTa \
  --vg_loss=1 \
  --sigma=9 \
  --num_thread_reader=4 \
  --save_dir="$SAVE_DIR"

echo "Training done. Best model: $SAVE_DIR/best_model.pth"
