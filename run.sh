#!/usr/bin/env bash
# Usage:
#   bash run.sh training     — run the training pipeline
#   bash run.sh feature      — run the live feature pipeline
#   bash run.sh backfill     — run the backfill pipeline

set -e

GOMP_DIR=$(gcc --print-file-name=libgomp.so.1 | xargs dirname)
export LD_LIBRARY_PATH="$GOMP_DIR:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/models:$(pwd)/pipelines:$PYTHONPATH"

case "${1:-}" in
  training)  python pipelines/training_pipeline.py ;;
  feature)   python pipelines/feature_pipeline.py ;;
  backfill)  python pipelines/backfill.py ;;
  *)
    echo "Usage: bash run.sh [training|feature|backfill]"
    exit 1
    ;;
esac
