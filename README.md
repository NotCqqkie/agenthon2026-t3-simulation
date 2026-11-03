# Agenthon 2026 Track 3 (Simulation): 2nd place entry

Source code for the market simulator that placed 2nd in Track 3 (Simulation) of
[Agenthon 2026](https://www.agenthon.net), released under the MIT licence as the winners' reproducibility package.

- Team: VerityLovityFalsityCruelty (Luke Finigan)
- Final entry: CodaBench submission 989526, release M21e, verified Final score 3,249,578.67 events/second
- Final image: `ghcr.io/notcqqkie/agenthon-t3-simulator@sha256:5f4e0536a4f227b4a6e0e087a27a5b97a48577c308054c7cf826c75c6269306c`

## What it does

The task was to re-implement the track's reference agent-based market simulator (ABIDES plus the organizers' adapter) so that every
output is identical to the reference while running as fast as possible. The score is mean simulated events per second
over sealed scenarios. Any mismatch or failure scores 0 for that scenario.

`simulate` is a single static C++17 binary:

- **Bit-exact randomness:** `rng.h` reproduces NumPy's legacy `RandomState` (MT19937 and its distribution
  samplers), so every agent draws the same numbers as in the Python reference.
- **Event queue:** a calendar-wheel priority queue (`queue.h`) with the reference's tie-breaking order.
- **Engine:** order book, matching, the agent types and latency models (`engine.h`, `config.h`), following the
  reference's semantics, including its edge cases.
- **Output:** our own Parquet writer (`parquet.h`) produces the trace files the scorer compares. Small headers handle
  JSON parsing (`json.h`) and SHA-256 (`sha256.h`).
- **Speed:** helper threads for output encoding, profile-guided optimisation, and a static build on an empty
  (`FROM scratch`) image, so container start-up is minimal.

Correctness came first. Before the Final, outputs were value-exact against the reference image on about 6,000 generated
configurations, along the axes the organizers said sealed variants vary on. The Development runs also passed the track's
exactness gate.

## Layout

| Path | What |
|---|---|
| `sim/` | the simulator (`main.cpp` and headers) |
| `train/` | scenarios used only as profile inputs during the PGO build (see [NOTICE.md](NOTICE.md)) |
| `Dockerfile` | two-stage build: instrumented binary, profile run, optimised static binary on `scratch` |
| `scripts/run_unit.sh` | runs one public unit through an image |
| `.github/workflows/reproduce.yml` | builds this source, pulls the Final image, runs both on every public unit and checks every output file is byte-identical |

## Reproduce

Requirements: Docker (linux/amd64).

```bash
docker build -t t3-source .
git clone https://github.com/Agenthon-2026/track3-simulation-public
git -C track3-simulation-public checkout eb8bf07
scripts/run_unit.sh t3-source track3-simulation-public/units/t3-as01-base-mix out/t3-as01-base-mix
```

Inside the container the command is the platform's:

```bash
simulate --config /input/scenario.json --out /output/trace.parquet
```

`simulate-batch --batch-dir DIR --out-dir DIR` runs several scenarios in one process.

The `reproduce` workflow (run it from the Actions tab) checks that this source rebuilds the Final entry's behaviour.
It runs this build and the published Final image on all 71 public units. It requires byte-identical Parquet output and an
identical `events.json`, apart from the run-specific timing and memory fields. Throughput depends on the machine:
the official numbers came from the organizers' timing machine (one run at a time on the same machine and GPU).

## Licence

MIT, see [LICENSE](LICENSE). The simulator is a C++ port of ABIDES (BSD-3-Clause) and the organizers' adapter (MIT);
their notices are in [NOTICE.md](NOTICE.md). Scenario files in `train/` are CC BY 4.0 from the organizers' public
repository.
