# Tutorials

Task-oriented walkthroughs of `iontrap-dynamics`. Each tutorial takes
~10 min to read and ~1 s to run. They are Sail material — adaptive
guidance with specific parameter choices, not coastline constraints.

The first tutorial is live; the rest are planned topics roughly in
order of reader dependency. The runnable demo tools under `tools/`
and their committed output bundles under
[`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data)
cover the same ground for users who prefer reading code to prose.

## Available

- [**Tutorial 1 — Carrier Rabi flopping with finite-shot readout**](01_first_rabi_readout.md).
  End-to-end pipeline exercising every architectural layer through
  v0.2: configuration (`IonSystem`, `DriveConfig`, `ModeConfig`),
  `HilbertSpace`, `carrier_hamiltonian`, `sequences.solve`,
  `SpinReadout`, `binomial_summary`. Written as the canonical "Hello
  world" for the library post-v0.2 Convention Freeze.
- [**Tutorial 2 — Red-sideband flopping from Fock ∣1⟩**](02_red_sideband_fock1.md).
  The four-step pattern from Tutorial 1 with the carrier swapped for
  the leading-order red-sideband Hamiltonian and motion initialised
  in `|n = 1⟩`. Introduces the `lamb_dicke_parameter` analytic helper,
  the `number` observable factory, and the `⟨σ_z⟩ + 2⟨n̂⟩ = 1`
  single-phonon-manifold conservation law as a sanity check.
  Parallels `tools/run_benchmark_sideband.py`.
- [**Tutorial 3 — Gaussian π-pulse with `modulated_carrier_hamiltonian`**](03_gaussian_pi_pulse.md).
  First time-dependent Hamiltonian in the series. Swap the static
  carrier for a Gaussian-enveloped drive, normalise the envelope
  amplitude so the pulse area integrates to exactly π, and watch
  the Bloch vector trace a clean y–z meridian. Introduces the
  list-format dispatch through `sequences.solve` and the
  cumulative-integral analytic overlay
  `θ(t) = ∫₀^t Ω · f(t') dt'`. Closes with Blackman / stroboscopic
  / adiabatic-ramp envelope extensions. Parallels
  `tools/run_demo_gaussian_pulse.py`.
- [**Tutorial 4 — Mølmer–Sørensen Bell gate**](04_ms_gate_bell.md).
  First two-ion scenario. Builds a two-²⁵Mg⁺ system sharing an
  axial COM mode, derives the Bell-closing detuning `δ = 2|Ωη|√K`
  and gate time `t_gate = π√K / |Ωη|` from physics inputs via the
  `ms_gate_closing_detuning` / `ms_gate_closing_time` analytic
  helpers, and verifies the four final-state invariants (loop
  closure `⟨n̂⟩ → 0`, equal Bell populations `P(|↓↓⟩) = P(|↑↑⟩) =
  1/2`, odd-parity `P_flip ≡ 0`, ion-exchange symmetry). First
  tutorial to wrap custom `qutip.Qobj` population projectors as
  `Observable` records. Parallels `tools/run_demo_ms_gate.py`.
- [**Tutorial 5 — Custom observables**](05_custom_observables.md).
  Generalises Tutorial 4's `Observable`-record foothold into the
  full construction hook. Four patterns, one per case you'll
  actually hit: multi-subsystem Bell-fidelity projector
  `|Φ⁻⟩⟨Φ⁻|`, two-ion `⟨σ_x σ_x⟩` correlator via
  `HilbertSpace.spin_op_for_ion`, mode Fock-state projector
  `|1⟩⟨1|` via `mode_op_for`, and a non-Hermitian virtual
  `|↓↓⟩⟨↑↑|` as a coherence-phase diagnostic. Closes with
  factory-vs-inline guidance and the `StorageMode.EAGER`
  post-hoc-analysis route (reduced-state partial traces, the
  registered `concurrence_trajectory` evaluator).
- [**Tutorial 6 — Fock truncation diagnosis**](06_fock_truncation.md).
  First diagnostic-layer tutorial. Walks a single scenario
  (thermal initial state `n̄ = 0.5`, static carrier) through all
  four CONVENTIONS §15 statuses — silent OK, Level 1
  `FockConvergenceWarning`, Level 2 `FockQualityWarning`, Level 3
  `ConvergenceError` — by varying `N_Fock` alone. Shows how to
  read `result.warnings` as both a Python warning and a
  structured `ResultWarning` record with diagnostics dict; how to
  tighten ε via the `fock_tolerance` override for
  publication-grade runs; and a diagnosis recipe for a
  `ConvergenceError` from its message.
- [**Tutorial 7 — Hash-verified cache round-trip**](07_cache_round_trip.md).
  End-to-end walk through the persistence layer over the
  Tutorial 2 RSB scenario. Covers the three cache functions —
  `compute_request_hash`, `save_trajectory`, `load_trajectory` —
  the `manifest.json` + `arrays.npz` bundle layout and its
  `StorageMode.OMITTED`-only scope, bit-identical round-trip of
  times + expectations + warnings, and four distinct
  `IntegrityError` failure modes (mismatched hash, missing
  files, tampered manifest, extra/missing npz arrays). Closes
  with three practical-use patterns for the cache (notebook
  skip-recompute, committed reference results, cross-process
  sharing) and the "don't commit 1000-trial sweep bundles"
  caveat.
- [**Tutorial 8 — Full Lamb–Dicke for hot-ion regimes**](08_full_lamb_dicke.md).
  When the `full_lamb_dicke=True` flag on the sideband builders
  matters. Covers the Wineland–Itano closed form
  `Ω_{n,n−1}^full = Ω·|η|·e^(−η²/2)·√((n−1)!/n!)·L_{n−1}^(1)(η²)`,
  the `η²·n ≳ 0.1` rule-of-thumb crossover, and a quantitative
  three-scenario comparison showing 3 % → 16 % → 30 % rate
  shortfall as `n` climbs from 1 to 10 at fixed `η = 0.26`.
  Closes with a when-to-flip decision tree covering
  thermal-start scenarios, MS-gate tuning, and sideband
  cooling cascades.
- [**Tutorial 9 — Squeezed / coherent state preparation**](09_squeezed_coherent_prep.md).
  Move past `qutip.basis` and `qutip.thermal_dm` for the motional
  initial state. Walks through the three named factories —
  `coherent_mode`, `squeezed_vacuum_mode`,
  `squeezed_coherent_mode` — plus `compose_density` for the
  full-space composition. Verifies each factory's ⟨n̂⟩ formula
  and squeezing-quadrature variances against analytic
  predictions, records the `ξ = r·e^(2iφ)` and
  squeeze-then-displace ordering conventions, and closes with
  a red-sideband collapse scenario from `|↓, α = 2⟩` that
  exhibits the canonical Rabi-rate dephasing invisible from a
  pure-Fock start.
- [**Tutorial 10 — Finite-shot statistics**](10_finite_shot_statistics.md).
  Deep dive on the three statistics functions —
  `wilson_interval`, `clopper_pearson_interval`,
  `binomial_summary` — and the `BinomialSummary` dataclass. A
  seven-row anchor table of Wilson vs Clopper–Pearson 95 % CIs
  across canonical `(k, n)` points (showing CP's 10–30 %
  width penalty at low `n` and convergence at `n = 100`); a
  fully-vectorised `binomial_summary` call across a 200-point
  carrier-Rabi trajectory with no Python loop; a four-branch
  Wilson-vs-CP decision tree; and a
  `n_required ≥ z²·p(1−p) / Δ²` shot-budget sizing formula.
  Expands Tutorial 1's single-CI step into the full
  finite-shot reporting surface.

## Planned

Sequenced roughly in order of reader dependency on prior tutorials:

11. **Systematics — jitter ensembles.** Running an inhomogeneous-
    dephasing study via `RabiJitter` + `perturb_carrier_rabi` and
    aggregating an ensemble with `sequences.solve_ensemble`.
    Parallels `tools/run_demo_rabi_jitter.py`.
12. **Two-ion Bell-state entanglement.** The Mølmer–Sørensen
    gate with parity readout and nonlinear entanglement observables
    (concurrence, log-negativity). Parallels
    `tools/run_demo_bell_entanglement.py` + `run_demo_parity_scan.py`.

## Scope and licensing

Tutorials are Sail material — adaptive guidance with specific
parameter choices, not coastline constraints. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).

## Complementary references

- **Install + first run** — [Getting Started](../getting-started.md).
- **Architectural overview** — [Phase 1 Architecture](../phase-1-architecture.md).
- **Binding physics conventions** — [Conventions](../conventions.md) (rendered live from the repo root `CONVENTIONS.md`).
- **Performance baselines** — [Benchmarks](../benchmarks.md).
- **Contributor scope** — [Boundary Decision Tree](../boundary-decision-tree.md).
- **Runnable examples** — the tools under `tools/run_demo_*.py` and the committed bundles under `benchmarks/data/<scenario>/` (each holding `manifest.json` + `arrays.npz` + `demo_report.json` + `plot.png`).
