#!/usr/bin/env bash
# Run one Track 3 public unit through an image (no network, read-only root).
# usage: scripts/run_unit.sh IMAGE UNIT_DIR OUT_DIR
set -euo pipefail
img=$1; u=$(cd "$2" && pwd); o=$3
mkdir -p "$o"; chmod 777 "$o"
docker run --rm --read-only --network=none --tmpfs /tmp:rw,size=256m -v "$u":/input:ro -v "$(cd "$o" && pwd)":/output \
  "$img" simulate --config /input/scenario.json --out /output/trace.parquet >/dev/null
