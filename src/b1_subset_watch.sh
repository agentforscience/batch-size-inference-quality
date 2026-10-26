#!/bin/bash
# For slow (3B) models: once the b4 condition is written, restart the job with B1_SUB=1 so the sequential
# B=1 reference runs on the 340-item subset (every 5th item) instead of all 1,700 items (time budget).
cd "$(dirname "$0")/.."
source .venv/bin/activate
tag=$1; model=$2; mem=$3; gpu=$4; pid=$5
until [[ -f results/gen/$tag/b4.jsonl ]]; do sleep 5; done
kill $pid; sleep 2
echo "$(date +%T) restarting $tag with B1_SUB=1"
CUDA_VISIBLE_DEVICES=$gpu B1_SUB=1 python src/generate.py "$model" "$tag" "$mem" bfloat16 >> logs/gen_$tag.log 2>&1
