#!/usr/bin/env bash
set -euo pipefail
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
