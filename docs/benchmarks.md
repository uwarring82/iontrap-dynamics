# Benchmarks

Honest performance baselines for `iontrap-dynamics`, refreshed at each
Phase 2 dispatch. This page doubles as (a) a sizing aid for users
deciding whether the library's performance fits their use case, and
(b) a reference record so future optimisation work can be measured
against a fixed starting point.

**Phase** and **status** per §5 Phase 2: performance. All Phase 2
dispatches have landed: X (`sesolve` dispatch), Y (`solve_ensemble`
parallel sweeps), OO (`csr`-vs-`dense` ops baseline), RR–XX (JAX
backend skeleton → Dynamiqs integrator → LAZY storage → five
time-dependent Hamiltonian builders on JAX), and YY / β.4.5
(cross-backend benchmark at dim ≥ 100 / 5000 steps). This page
summarises every baseline measured so far.

## Headline

The main empirical finding from Dispatches X and Y is that **at the
Hilbert-space scales this library routinely uses (dim ≤ 100, single-
solve cost ≤ tens of ms), QuTiP 5.2's dense-matrix `mesolve` is
already fast.** The two performance dispatches landed cleanly, but
the default settings favour simplicity over micro-optimisation:

- **`sesolve` dispatch** is on by default for pure-ket inputs — not
  because it's faster (it isn't, at these scales), but because the
  Schrödinger equation is the correct dynamics for a pure state.
- **`solve_ensemble` parallelism** defaults to `n_jobs=1` (serial in
  the main process) — joblib's process-spawn + pickle overhead
  dominates at typical jitter-ensemble scales. `n_jobs=-1` with
  `parallel_backend="loky"` pulls ahead once single-solve cost
  exceeds ~15 ms with ≥ 2000 time steps.

Raw performance at small scales is not the bottleneck. Sparse-
matrix tuning (Dispatch OO) found QuTiP 5's CSR default already
optimal at library scales. The JAX-backend performance-
characterisation benchmark (Dispatch YY / β.4.5) returned an
honest null result: **QuTiP 5 is ~2.8× faster than Dynamiqs +
JAX at dim 100 / 5000 steps** across all five time-dependent
builders — the JAX backend's value is positioning, cross-backend
consistency checking, and forward-looking capability (autograd
via a future γ track, GPU dispatch), not wall-clock on CPU at
current library scales. See the "JAX time-dependent Hamiltonians
at scale" section below for the full table and opt-in guidance.

## Canonical smoke-test thresholds (Phase 0.F)

Three scenarios with wall-clock gates, set against the reference
hardware in `WORKPLAN_v0.3.md` §0.F (2023 MacBook Air M2 or
equivalent). Tests live at `tests/benchmarks/test_performance_smoke.py`
and run under the `benchmark` marker.

| Scenario                          | Fock | Steps | Threshold | Measured (ref) |
|-----------------------------------|-----:|------:|----------:|---------------:|
| Single-ion sideband flopping      | 30   | 200   | < 5 s     | ~1.5 s         |
| Two-ion Mølmer–Sørensen gate      | 15   | 500   | < 30 s    | ~0.01 s        |
| Stroboscopic AC-π/2 drive         | 40   | 1000  | < 60 s    | ~0.01 s        |

The two-ion MS gate and stroboscopic AC-π/2 scenarios are orders of
magnitude below their thresholds — the gates were set conservatively
during Phase 0 before the full builder surface had landed, and
QuTiP 5.2 plus the clean builder designs easily beat them. They
nonetheless remain as tripwires against accidental regressions.

Run with:

```bash
pytest tests/benchmarks -v --durations=0
```

## `sesolve` vs `mesolve` (Dispatch X)

`solve(..., solver="auto")` dispatches pure kets to `qutip.sesolve` and
density matrices to `qutip.mesolve`. The measured wall-clock ratio
across five scenarios:

| Scenario          | Hilbert dim | mesolve    | sesolve    | Speedup |
|-------------------|------------:|-----------:|-----------:|--------:|
| carrier fock=4    | 8           | 3.87 ms    | 3.97 ms    | 0.98×   |
| carrier fock=12   | 24          | 4.07 ms    | 4.15 ms    | 0.98×   |
| carrier fock=24   | 48          | 4.34 ms    | 4.22 ms    | 1.03×   |
| RSB fock=8        | 16          | 3.75 ms    | 3.70 ms    | 1.01×   |
| BSB fock=8        | 16          | 3.76 ms    | 3.75 ms    | 1.00×   |
| **Mean**          |             |            |            | **1.00×** |

**Interpretation.** The QuTiP-4-era 2–3× sesolve advantage has
largely been closed in QuTiP 5.2 — `mesolve` on a pure ket is
comparably efficient at these scales because it skips the density-
matrix lifting when it can. The dispatch therefore opts into
`sesolve` for **semantic correctness** (Schrödinger is the right
dynamics for pure states) rather than for measured performance.
Bigger Hilbert spaces (hundreds of dimensions) should see `sesolve`
pull ahead as density-matrix lifting cost scales as N²; that regime
is not routine in the library today.

Run with:

```bash
python tools/run_benchmark_sesolve_speedup.py
```

## `solve_ensemble` parallelism crossover (Dispatch Y)

The ensemble helper wraps `joblib.Parallel` for jitter-style sweeps.
The relevant question is **where the process-spawn + pickle overhead
is paid off**. Measured crossover (ensemble of 20 trials):

| Regime            | Single-solve | Serial (`n_jobs=1`) | loky (`n_jobs=-1`) | Threading (`n_jobs=-1`) |
|-------------------|-------------:|--------------------:|-------------------:|------------------------:|
| small fock=3 n=200   | 2.7 ms    | 88 ms               | 1954 ms            | 72 ms                    |
| medium fock=12 n=500 | 6.2 ms    | 85 ms               | 91 ms              | 157 ms                   |
| large fock=24 n=2000 | 15.5 ms   | 499 ms              | 186 ms (**2.68×**) | 617 ms                   |

**Interpretation.**

- **Small regime** (single-solve < 5 ms): serial wins decisively.
  loky is ~20× slower because process-spawn + pickle cost exceeds
  per-trial solve cost. Threading nudges ahead of serial on this
  one — BLAS calls release the GIL just enough to benefit.
- **Medium regime** (single-solve ~5–10 ms): serial ≈ loky. Noise-
  dominated crossover.
- **Large regime** (single-solve ≥ 15 ms, long step counts): loky
  wins cleanly at 2–3× speedup. Threading hurts at long step counts
  because Python-level stepper overhead accumulates inside the GIL.

`solve_ensemble` therefore defaults to `n_jobs=1`. Flip to
`n_jobs=-1` with `parallel_backend="loky"` when your scenarios sit
in the large regime (MS gates with Fock > 24, full-Lamb-Dicke two-
ion builders, scans with per-trial duration > ~10 μs).

Run with:

```bash
python tools/run_benchmark_ensemble_parallel.py
```

## `csr` vs `dense` operator dtype (Dispatch OO)

QuTiP 5 changed the default `Qobj.data` representation from dense
(QuTiP 4) to compressed-sparse-row. The library inherits that default
without any explicit selection in `hamiltonians.py` / `operators.py`,
so the practical question is **how much of the Phase-2 "sparse-matrix
tuning" plan is already done for free by upstream**. Measured
wall-clock at both dtypes on the `sesolve` fast path:

| Scenario                 | Hilbert dim | CSR     | Dense   | Dense / CSR |
|--------------------------|------------:|--------:|--------:|------------:|
| carrier fock=4           | 8           | 3.03 ms | 3.10 ms | 1.02×       |
| carrier fock=12          | 24          | 3.21 ms | 3.17 ms | 0.99×       |
| carrier fock=24          | 48          | 3.18 ms | 3.25 ms | 1.02×       |
| RSB fock=8               | 16          | 2.84 ms | 2.80 ms | 0.98×       |
| BSB fock=8               | 16          | 2.83 ms | 2.86 ms | 1.01×       |
| RSB fock=60              | 120         | 3.52 ms | 4.80 ms | **1.36×**   |
| BSB fock=60              | 120         | 3.58 ms | 5.34 ms | **1.49×**   |
| two-ion RSB fock=15      | 60          | 3.00 ms | 3.01 ms | 1.00×       |
| **Mean**                 |             |         |         | **1.11×**   |

**Interpretation.**

- **At library-typical Hilbert sizes** (dim ≤ 60, covering all three
  Phase 0.F smoke tests and the entire tutorial bundle), CSR and
  dense tie within ~5 %. The headline `mesolve` cost on QuTiP 5.2 is
  dominated by the SciPy ODE stepper, not by per-step matrix–vector
  product cost, so the dtype choice is lost in the noise.
- **At single-ion Fock ≥ 60** (dim ≥ 120), CSR pulls ahead to
  1.3–1.5×. These regimes appear in hot-ion simulations (Tutorial 8)
  and in the squeezed-state builder family (Tutorial 9) when the
  squeezing parameter pushes the Fock support out.
- **CSR never loses materially.** The smallest ratio recorded is
  0.98× (RSB fock=8), which is within repeat-to-repeat noise.

**What this means for the Phase 2 plan.** The "sparse-matrix tuning"
open item in the original Phase 2 scope is **effectively discharged
by QuTiP 5 defaults** — there is no residual dense-matrix build to
replace with sparse primitives in the library's builder chain. The
library therefore **does not expose a `matrix_format` kwarg** on
`sequences.solve`; Design Principle 5 ("one way to do it at the
public API level") applies, and CSR is the one way because it
never loses. Users who genuinely need a dense representation (e.g.
for debugging with `qutip.Qobj.full()`) can still call
`hamiltonian.to("dense")` manually before handing the operator to
`solve`.

Run with:

```bash
python tools/run_benchmark_sparse_vs_dense.py
```

## JAX time-dependent Hamiltonians at scale (Dispatch YY / β.4.5)

β.4.5 closes the Phase 2 JAX-backend track with a deliberately
**out-of-library-scale** benchmark: all five β.4 time-dependent
Hamiltonian builders run on both backends at Hilbert dim ≥ 100
and trajectory length **5000 steps**. The design-note §7 α.4
staging flagged this as the informative regime — Dispatches X /
Y / OO already established QuTiP 5 is near the floor at
library-typical dim ≤ 60 / few-hundred steps, so any
JAX-vs-QuTiP comparison at small scale is confirmatory.

Measured wall-clock (Dynamiqs 0.3.4 + QuTiP 5.2.3, Apple
silicon, `jax>=0.4`):

| Scenario                       | Hilbert dim | QuTiP [s] | JAX [s] | QuTiP / JAX | Cross-backend Δ |
|--------------------------------|------------:|----------:|--------:|------------:|----------------:|
| `detuned_carrier` (fock=50)    | 100         | 0.040     | 0.093   | 0.43×       | 1.4e-04         |
| `detuned_red_sideband` (fock=50) | 100       | 0.042     | 0.126   | 0.33×       | 4.3e-05         |
| `detuned_blue_sideband` (fock=50) | 100      | 0.043     | 0.126   | 0.34×       | 5.2e-05         |
| `detuned_ms_gate` (two-ion, fock=25) | 100   | 0.046     | 0.160   | 0.29×       | 7.7e-06         |
| `modulated_carrier` (Gaussian, fock=50) | 100 | 0.036    | 0.087   | 0.42×       | 9.3e-05         |
| **Mean**                       |             |           |         | **0.36×**   |                 |

**Interpretation.**

- **Null result on performance.** QuTiP 5 is **~2.8× faster than
  Dynamiqs + JAX** at dim 100 / 5000 steps across all five
  time-dependent builders. The result is consistent across
  structurally different Hamiltonians (carrier, sidebands, MS
  gate, envelope-modulated carrier), so it's an integrator /
  dispatch-overhead characterisation, not a per-builder artefact.
- **Cross-backend numeric equivalence holds.** The worst
  observed disagreement is 1.4e-4 (detuned carrier with σ_z)
  — under the 1e-3 design-target tolerance. Confirms the five
  β.4.1–β.4.4 builders emit mathematically equivalent
  Hamiltonians on both backends; the wall-clock gap is
  integrator-performance, not a translation bug.
- **Why the null.** Dynamiqs + JAX's overhead comes from
  Python-dispatched Tsit5 stepping inside `jax.lax.scan`: JIT
  compile cost (~seconds for first call; not amortised in the
  min-of-3-repeats we report) plus per-step Python dispatch on
  `ModulatedTimeQArray` evaluation. At library scale the QuTiP
  / scipy `solve_ivp` path is lean enough to hold the
  advantage; JAX would need much larger Hilbert spaces (dim
  ≥ 500, density-matrix lift dominated by matmul) or GPU
  dispatch to see a crossover.

**What this means for Phase 2.** The JAX backend's value is
**positioning and capability**, not raw wall-clock — exactly as
the design note's Axis-A framing flagged. The Phase 2
commitment remains justified by: (a) keeping the workplan §1
"Dynamiqs as future backend target" honest; (b) opening the
architectural slot for autograd (γ / α′ design-note options)
and GPU dispatch that QuTiP cannot deliver; (c) giving users a
cross-backend consistency check — if a JAX result disagrees
with QuTiP by more than 1e-3, both backends' convention
interpretations are now on the table, reducing the
convention-version ambiguity that single-backend results carry.

**Opt-in guidance.** At current library scale, `backend="jax"`
is the right choice when:

- You need autograd through solver parameters (future γ scope).
- You plan to run on GPU / TPU (Dynamiqs + JAX on CUDA / Metal).
- You want an independent numerical check on a QuTiP result —
  cross-backend agreement at 1e-4 is strong evidence against
  translation or convention bugs.

At current library scale, `backend="qutip"` (the default) is
the right choice when:

- You want the fastest time-to-result on CPU at dim ≤ ~200.
- You're running ensembles via `solve_ensemble` where JIT
  compile cost cannot be amortised across trials (per-trial
  Hamiltonian means per-trial recompile).
- You have no autograd or GPU requirement.

Run with:

```bash
python tools/run_benchmark_jax_timedep.py
```

## When to opt into each feature

A quick decision guide distilled from the numbers above:

| Concern                              | Default         | Opt in when …                           |
|--------------------------------------|-----------------|-----------------------------------------|
| `solver` (sesolve vs mesolve)        | `"auto"`        | Forcing `"mesolve"` to share code with mixed-state trajectories, or `"sesolve"` to assert a pure-state invariant. |
| `backend` (qutip vs jax)             | `"qutip"`       | Autograd (future γ scope), GPU / TPU dispatch, or independent cross-backend numeric check. Not for raw performance at dim ≤ 200 on CPU — QuTiP is faster there (β.4.5). |
| `solve_ensemble` parallelism         | `n_jobs=1`      | Single-solve cost ≥ 15 ms **and** ensemble size ≥ ~20. Use `parallel_backend="loky"`. |
| `solve_ensemble` threading backend   | n/a             | Short ensembles (small regime, dozens of trials) where BLAS-release is plausible — this is a narrow niche; usually serial is simpler. |
| `storage_mode`                       | `OMITTED`       | Nonlinear observables (concurrence, log-negativity) need full states — use `EAGER`. |

## Reproducing the numbers

All benchmarks are deterministic given seed + hardware. To reproduce
the tables above on your machine:

```bash
# Smoke-test thresholds
pytest tests/benchmarks -v --durations=0

# sesolve vs mesolve
python tools/run_benchmark_sesolve_speedup.py
#   → benchmarks/data/sesolve_speedup/report.json + plot.png

# solve_ensemble parallelism crossover
python tools/run_benchmark_ensemble_parallel.py
#   → benchmarks/data/ensemble_parallel/report.json + plot.png

# csr vs dense operator dtype
python tools/run_benchmark_sparse_vs_dense.py
#   → benchmarks/data/sparse_vs_dense/report.json + plot.png

# JAX time-dependent Hamiltonians at scale (β.4.5)
#   requires the [jax] extras: pip install iontrap-dynamics[jax]
python tools/run_benchmark_jax_timedep.py
#   → benchmarks/data/jax_timedep/report.json + plot.png
```

The JSON reports record the exact environment (Python, NumPy, SciPy,
QuTiP versions + platform) so cross-machine comparisons are
auditable.

## Open Phase 2 items

Per `WORKPLAN_v0.3.md` §5.3 (β.4 as v0.3.x follow-up), the Phase 2
deliverable surface is complete at the `v0.3` milestone. The JAX
backend covers both time-independent (β.1–β.3) and time-dependent
(β.4.1–β.4.4) Hamiltonians with cross-backend numeric equivalence
validated at the 1e-3 design-target tolerance; β.4.5 closed the
performance-characterisation loop with the null-result analysis
above.

Remaining items have been retired or re-scoped:

- **Sparse-matrix tuning** — closed by Dispatch OO (QuTiP 5
  defaults to CSR; the library inherits that default without
  explicit opt-in).
- **JAX backend performance characterisation** — closed by
  Dispatch YY (β.4.5). Honest null result documented above:
  QuTiP 5 is ~2.8× faster than Dynamiqs + JAX at dim 100 /
  5000 steps. Phase 2 value is positioning + capability, not
  wall-clock.

Future follow-ups live outside the `v0.3` milestone:

- **γ track — autograd through solver parameters.** Design-note
  Axis-D. Would reuse β.4's parallel JAX-native builder
  scaffolding at ~60 % coverage. Scoped as a Phase 3+ item.
- **GPU dispatch validation.** CPU-only CI today; users with
  CUDA / Metal JAX builds can run the JAX backend transparently
  but we don't include a GPU-smoke-test in CI.

## Exact-diagonalization envelope (Dispatch AAH)

The Clos 2016 reproduction track (`docs/workplan-clos-2016-integration.md`)
routes through `solve_spectrum`'s dense `scipy.linalg.eigh` path. The
user-facing question it raises — "how large a system can I
exact-diagonalise on my laptop?" — is answered by
`tools/run_benchmark_spectrum_envelope.py`, which builds the non-RWA
Clos 2016 spin–boson Hamiltonian across an `(N, n_c)` grid, runs
`solve_spectrum` in an isolated subprocess per point, and records
wall-clock plus peak RSS.

Measured on this reference hardware (Apple Silicon arm64, macOS 26;
scipy 1.17.1; run recorded at
`benchmarks/data/spectrum_envelope/report.json`):

| N | n_c | dim   | matrix (dense complex128) | `solve_spectrum` wall-clock | peak RSS  |
|---|-----|-------|---------------------------|-----------------------------|-----------|
| 1 | 100 | 202   | 638 KB                    | 0.01 s                      | 161 MB    |
| 2 | 20  | 882   | 11.9 MB                   | 0.4 s                       | 317 MB    |
| 2 | 25  | 1 352 | 27.9 MB                   | 1.3 s                       | 537 MB    |
| 3 | 8   | 1 458 | 32.4 MB                   | 1.6 s                       | 577 MB    |
| 3 | 10  | 2 662 | 108 MB                    | 9.0 s                       | 1.22 GB   |
| 3 | 12  | 4 394 | 295 MB                    | 41.1 s                      | 2.01 GB   |
| 3 | 14  | 6 750 | 695 MB                    | 150 s                       | 3.84 GB   |
| 4 | 5   | 2 592 | 103 MB                    | 8.0 s                       | 1.23 GB   |
| 4 | 6   | 4 802 | 352 MB                    | 52.3 s                      | 2.53 GB   |
| 4 | 7   | 8 192 | 1.00 GB                   | 430 s                       | 3.63 GB   |
| 5 | 3   | 2 048 | 64.0 MB                   | 6.6 s                       | 935 MB    |
| 5 | 4   | 6 250 | 596 MB                    | 118 s                       | 3.14 GB   |

The last three rows (`dim ≥ 6 250`) are deliberate large-dim probes that
push a 16 GB laptop into measurable swap territory; they are gated by
`--include-large` on the benchmark tool because they balloon total
runtime to ~25 min. The largest measured point sits at `dim = 8 192`
with a 1 GB dense matrix, 7 min wall-clock, and 3.6 GB peak RSS — the
top of what is comfortable to sweep across many detunings on
commodity hardware.

**Scaling.** Wall-clock follows the expected $\mathcal{O}(d^{3})$ cost
of symmetric eigendecomposition. Peak RSS tracks the dense-matrix
footprint with a ~4–5× workspace multiplier (lower than the ~6–8×
naively suggested by the small-dim points; scipy's `dsyevr`-based
default is more memory-efficient than the divide-and-conquer
alternative once `d ≳ 4 000`). Both curves collapse onto a single
N-independent trajectory when plotted against Hilbert dimension —
the solver cost depends only on dimension, not on how the
tensor-product factors are arranged.

**Rules of thumb on a 16 GB laptop, calibrated against the measured
data.** Five tiers, with the boundaries pinned by *measured* (not
extrapolated) points where possible:

- `dim ≤ 1 000` — sub-second; trivial for interactive notebook use.
- `dim ≤ 2 700` — 1 s to 10 s; peak RSS under 1.2 GB. Covers the
  AAE / AAF regression suite at fully converged cutoffs (N = 1–3).
- `dim ≤ 5 000` — 10 s to 1 min; peak RSS up to 2.5 GB. Covers
  N = 3 at cutoff = 12 and N = 4 at cutoff = 6.
- `dim ≤ 8 200` — 1 min to 7 min; peak RSS up to 3.6 GB. Covers
  N = 4 at cutoff = 7 (the largest measured point). Single eigh
  is fine; sweeping 16 detunings takes ~2 hr.
- `dim ≳ 15 000` — extrapolated ~30 min and ~7 GB RSS. N = 5 at
  cutoff = 5 (dim = 15 552) sits on the 16 GB boundary; one run
  per detuning point is feasible, sweeps are not.

### Scaling laws and extrapolation to RAM limits

The measured points fit the textbook eigh scaling cleanly. Closed-form
fits (using only points with `dim ≥ 500` for the wall-clock fit, so
the import / assemble baseline does not bend the cubic; the RSS fit
uses every point):

- **Wall-clock**: $t_\text{solve}(d) = \alpha\,d^{3}$ with
  $\alpha \approx 5.6 \times 10^{-10}$ s on the reference hardware.
- **Peak RSS**: $m_\text{peak}(d) = m_{0} + \kappa\,d^{2}$ with
  $m_{0} \approx 365$ MB (Python + numpy + scipy + qutip + dependency
  imports) and $\kappa \approx 69$ B per matrix entry — i.e. the
  per-element workspace is ~4.3× the 16 B `complex128` matrix entry.
  An earlier fit on the small-dim core grid alone reported
  ~6.7× because the divide-and-conquer LAPACK driver dominates RSS
  at small `d` while the more memory-efficient `dsyevr` path scipy
  defaults to wins at `d ≳ 4 000`. The large-dim probes at
  `dim ∈ {6 250, 6 750, 8 192}` pin the asymptotic constant.

`tools/plot_spectrum_envelope_extrapolation.py` rebuilds the figure
below from `report.json`, overlays the fitted laws past the measured
range, and projects them against the canonical wall-clock thresholds
and consumer-RAM tiers.

![exact-diag envelope extrapolation](https://github.com/uwarring82/iontrap-dynamics/blob/main/benchmarks/data/spectrum_envelope/plot_extrapolation.png?raw=true)

**Per-RAM-tier envelope, projected from the RSS fit.** The numbers in
the body of the table are the largest `n_c` you can dense-eigh at
each `(N, RAM)` pair, derived from the Clos 2016 reproduction
parameterisation `dim = 2 · (n_c + 1)^N`. Numbers ≤ converged-cutoff
(N=1 → 7, N=2 → 8, N=3 → 10) mean the converged regression target
is *not* reachable on that RAM tier; numbers above are the slack you
have for cutoff-convergence sweeps.

| RAM tier                  | max dim | wall @ max dim | N=1    | N=2 | N=3 | N=4 | N=5 |
|---------------------------|--------:|---------------:|-------:|----:|----:|----:|----:|
| 8 GB (budget laptop)      |  10 937 | 12 min         |  5 467 |  72 |  16 |   7 |   4 |
| 16 GB (standard laptop)   |  15 646 | 36 min         |  7 822 |  87 |  18 |   8 |   5 |
| 32 GB (workstation)       |  22 253 | 1.7 hr         | 11 125 | 104 |  21 |   9 |   5 |
| 64 GB (high-end WS)       |  31 559 | 4.9 hr         | 15 778 | 124 |  24 |  10 |   5 |
| 128 GB (lab server)       |  44 693 | 14 hr          | 22 345 | 148 |  27 |  11 |   6 |

**Wall-clock is the binding constraint past 32 GB.** Reading off the
extrapolated cubic, the time to a single eigh hits:

| target wall-clock | max dim | comment |
|-------------------|--------:|---------|
| 1 s               |   1 211 | interactive notebook bound |
| 10 s              |   2 610 | tight regression-suite point |
| 1 min             |   4 742 | upper end of AAE / AAF / AAI test budget |
| 10 min            |  10 217 | feasible for one-off runs |
| 1 hr              |  18 565 | overnight-only |

So while a 128 GB server *could* hold the matrix at `dim = 44 693`,
that single eigh would take ~14 hours — and a 16-detuning sweep
(the size of the Clos 2016 figure) would take ~9 days. Past the
16 GB tier the user's effective binding constraint flips from RAM
to wall-clock: the measured `dim = 8 192` point already takes
7 minutes for one eigh and 1.9 hours for a 16-detuning sweep, even
though it sits well inside the 16 GB envelope.

**Cross-checking the "AAG would help" cases against the measured
data.** The new 16 GB cutoff projection (`n_c ≤ 8` at N = 4,
`n_c ≤ 5` at N = 5) is more generous than the original small-dim
fit suggested:

- **N = 4 at fully converged cutoff (~6) fits 8 GB comfortably.**
  Measured: 53 s per eigh, 2.5 GB peak RSS.
- **N = 4 at cutoff = 7** (one step past converged) fits the 16 GB
  envelope cleanly. Measured: 7 min per eigh, 3.6 GB peak RSS.
  Sweep cost (~2 hr per detuning row) is the binding factor.
- **N = 5 at cutoff = 5** sits *exactly* at the 16 GB boundary
  (predicted). Single eigh feasible, sweep impractical.
- **N = 5 at cutoff = 6** (`dim = 33 614`) needs 64 GB-class
  hardware for dense; this is where AAG's interior-window iterative
  path is unambiguously valuable.

**AAG gate status.** On the basis of the *measured* envelope, AAG
earns its keep in two specific cases:

1. **N = 5 reproduction at fully converged cutoff (`n_c ≥ 6`).**
   Dense crosses 64 GB here; iterative shift-invert around the
   initial-state mean energy could plausibly stay under 16 GB by
   exploiting the sparsity-after-tridiagonalisation that dense
   eigh discards.
2. **Detuning-sweep amortisation at `dim ≳ 5 000` on a 16 GB
   laptop.** Single eigh at `dim = 4 802` is 53 s; a 16-detuning
   sweep is 14 min. At `dim = 8 192` the sweep is 1.9 hr. An
   iterative path that re-uses the sparse factorisation across
   detuning windows could cut this materially.

Neither is currently a binding library requirement, so AAG stays
**deferred**. The numbers above give a clean re-activation
trigger: ship AAG when a real user's reproduction work hits one
of the two cases.

Run the measurement and the projection with:

```bash
python tools/run_benchmark_spectrum_envelope.py        # measurement
python tools/plot_spectrum_envelope_extrapolation.py   # fits + projection
```

The benchmark is subprocess-isolated per grid point so peak RSS
reflects a single-run baseline rather than cumulative allocator
state; total runtime on the reference hardware is ~2 min for the
measurement plus < 1 s for the extrapolation pass.
