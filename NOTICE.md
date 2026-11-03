# Notices

## Scenario files in `train/`

Ten of the eleven files in `train/` are copies of `scenario.json` from public units of
[Agenthon-2026/track3-simulation-public](https://github.com/Agenthon-2026/track3-simulation-public) (commit `eb8bf07`):
t3-as01-base-mix, t3-eq-deterministic-baseline, t3-eq-uniform-wide, t3-gb-mega-throughput, t3-mp02-stp-oldest-baseline,
t3-mp04-mm-heavy-oldest, t3-mr-deep-book-state-size, t3-ra01-fundamental-shock-mid, t3-s001-price-time-priority and
t3-st05-latency-spike-pareto. Each unit's manifest licenses these files under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/) and marks them redistributable. They are unmodified and are used only as
profile-guided-optimization inputs during the build. `train/ztrain-mrdeep32.json` is adapted from the `scenario.json` of
public unit t3-mr-deep-book-state-size (same source and licence). Changes: `horizon_ns` 2,000,000,000 -> 250,000,000,
MarketMaker `count` 2 -> 32, and `scenario_id` set to `ztrain-mrdeep32`. It is also used only for the build profile.

## Reference simulator (ABIDES)

`sim/` is a C++ port, written for the competition, of the Track 3 reference simulator: ABIDES
([jpmorganchase/abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public), pinned commit
`f9cbe51342b7dedd9587e4e069040d68a5c6477f`; kernel, ExchangeAgent, OrderBook, SparseMeanRevertingOracle) together with the
organizers' `baselines/abides_fork` adapter (config, agents, latency model, trace) and their build-time patches in
`baselines/patches/` (exchange self-trade prevention, kernel message ledger, oracle scheduled jump, order-size model) from
[Agenthon-2026/track3-simulation-public](https://github.com/Agenthon-2026/track3-simulation-public). It reproduces their
behaviour so that outputs match bit for bit, and the random-number generator reproduces the output sequences of NumPy's
legacy `RandomState` (MT19937). The Python sources themselves are not included. Because the port follows their logic,
the upstream licence notices are reproduced below.

### ABIDES (BSD-3-Clause)

```
BSD 3-Clause License

Copyright (c) 2021, J.P. Morgan Chase
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Agenthon 2026 Track 3 public repository (MIT)

The organizers' `abides_fork` adapter and patches are published under the MIT licence below; unit data files are CC BY 4.0
per each unit's manifest. See that repository's `LICENSE`, `DATA-LICENSE.md` and `THIRD-PARTY-NOTICES.md`.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Algorithms

- `sim/rng.h` re-implements, without copying code, the output of NumPy's legacy `RandomState`: the MT19937 generator of
  Matsumoto and Nishimura (1998) and NumPy's legacy distribution samplers (NumPy is BSD-3-Clause).
- `sim/config.h` uses Acklam's rational approximation of the inverse normal CDF.
- `sim/sha256.h` implements SHA-256 (FIPS 180-4), with an x86 SHA-NI path that follows the instruction sequence published
  in Intel's SHA extensions reference material.
