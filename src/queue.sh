#!/bin/bash
# Launch jobs one at a time: start a job, wait until it has acquired its GPU slice ("holding" in its log),
# then start the next. This avoids our own jobs racing each other for the same freed memory.
cd "$(dirname "$0")/.."
source .venv/bin/activate
while read -r tag model mem dtype extra; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  if [[ -f results/gen/$tag/b1_rep.jsonl ]]; then echo "skip $tag"; continue; fi
  echo "$(date +%T) launching $tag"
  env $extra nohup python src/generate.py "$model" "$tag" "$mem" "$dtype" > logs/gen_$tag.log 2>&1 &
  pid=$!
  until grep -q "^holding" logs/gen_$tag.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  echo "$(date +%T) $tag acquired (pid $pid)"
  sleep 60   # let the engine finish profiling before the next job measures free memory
done < "$1"
echo "$(date +%T) queue: all launched"
