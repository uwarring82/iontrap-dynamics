# Benchmarks

Honest performance baselines for `iontrap-dynamics`, refreshed at each
Phase 2 dispatch. This page doubles as (a) a sizing aid for users
deciding whether the library's performance fits their use case, and
(b) a reference record so future optimisation work can be measured
against a fixed starting point.

**Phase** and **status** per §5 Phase 2: performance. Dispatches X
(`sesolve` dispatch) and Y (`solve_ensemble` parallel sweeps) have
landed; sparse-matrix tuning and a JAX backend are pending. This page
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

Raw performance at small scales is not the bottleneck; the bigger
wins are elsewhere (sparse-matrix tuning on the builder side, JAX
backend for dense-matrix reduction, larger-system scalability).

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

## When to opt into each feature

A quick decision guide distilled from the numbers above:

| Concern                              | Default         | Opt in when …                           |
|--------------------------------------|-----------------|-----------------------------------------|
| `solver` (sesolve vs mesolve)        | `"auto"`        | Forcing `"mesolve"` to share code with mixed-state trajectories, or `"sesolve"` to assert a pure-state invariant. |
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
```

The JSON reports record the exact environment (Python, NumPy, SciPy,
QuTiP versions + platform) so cross-machine comparisons are
auditable.

## Open Phase 2 items

Per `WORKPLAN_v0.3.md` §5 Phase 2, still to land:

- **Sparse-matrix profiling.** Audit the Hamiltonian builders for
  hidden dense constructions (e.g. full-dim identities tensored into
  embedded operators) and replace with sparse primitives where the
  builder's output is sparse-natured.
- **JAX backend.** Opt-in `solve` variant that runs on JAX arrays
  with JIT-compiled stepping. Biggest potential win for larger
  Hilbert spaces; also opens GPU execution paths. Will include ≥1
  worked example matching the QuTiP reference within cross-platform
  tolerance.

Both are pending dispatches on the `main` branch.
