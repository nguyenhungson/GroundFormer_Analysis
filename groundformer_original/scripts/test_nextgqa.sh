#!/usr/bin/env bash
# Evaluate a GroundFormer checkpoint on the NExT-GQA test split.
#
# Usage:   ./scripts/test_nextgqa.sh [GPU_ID] [CHECKPOINT] [SPLIT]
# Example: ./scripts/test_nextgqa.sh 0 checkpoints/nextgqa/best_model.pth test
set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT=${2:-"$ROOT/checkpoints/nextgqa/best_model.pth"}
SPLIT=${3:-test}
[[ "$CKPT" != /* ]] && CKPT="$ROOT/$CKPT"
SAVE_DIR="$(dirname "$CKPT")"

cd "$ROOT/code/GroundFormer"

CUDA_VISIBLE_DEVICES=$GPU python main_dual.py \
  --dataset=nextgqa \
  --test=$SPLIT \
  --baseline=dual \
  --feat_type=CLIPL \
  --mc=5 \
  --batch_size_val=64 \
  --qmax_words=30 \
  --amax_words=38 \
  --max_feats=32 \
  --embd_dim=768 \
  --ff_dim=768 \
  --n_layers=2 \
  --dropout=0.0 \
  --lan=RoBERTa \
  --sigma=9 \
  --num_thread_reader=4 \
  --save_dir="$SAVE_DIR" \
  --pretrain_path="$CKPT"

echo "Evaluation done. Results: $SAVE_DIR/${SPLIT}-res.json, $SAVE_DIR/${SPLIT}_ground_ada.json"
