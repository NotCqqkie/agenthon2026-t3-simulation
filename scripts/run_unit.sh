#!/usr/bin/env bash
# Run one Track 3 public unit through an image (no network, read-only root).
# Units with batch.json + scenarios/ go to simulate-batch, as on the platform.
# usage: scripts/run_unit.sh IMAGE UNIT_DIR OUT_DIR
set -euo pipefail
img=$1; u=$(cd "$2" && pwd); o=$3
mkdir -p "$o"; chmod 777 "$o"
if [ -f "$u/batch.json" ]; then cmd=(simulate-batch --batch-dir /input/scenarios --out-dir /output)
else cmd=(simulate --config /input/scenario.json --out /output/trace.parquet); fi
docker run --rm --read-only --network=none --tmpfs /tmp:rw,size=256m -v "$u":/input:ro -v "$(cd "$o" && pwd)":/output \
  "$img" "${cmd[@]}" >/dev/null
