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

## Planned

Sequenced roughly in order of reader dependency on prior tutorials:

4. **Mølmer–Sørensen Bell gate.** Parallels
   `tools/run_demo_ms_gate.py`. Uses `ms_gate_closing_detuning` and
   `ms_gate_closing_time` analytic helpers to derive gate parameters
   from physics inputs.
5. **Custom observables.** How to wrap a QuTiP `Qobj` as an
   `Observable` record — e.g. two-ion population projectors for the
   MS demo.
6. **Fock truncation diagnosis.** Reading `result.warnings`; the
   `FockConvergenceWarning` / `FockQualityWarning` /
   `ConvergenceError` ladder; using `fock_tolerance` to override ε.
7. **Hash-verified cache round-trip.** `cache.save_trajectory` /
   `cache.load_trajectory`; `compute_request_hash` usage; the
   `[Unreleased]` demo-bundle layout (`manifest.json` + `arrays.npz`
   + `demo_report.json`).
8. **Full Lamb–Dicke for hot-ion regimes.** When the
   `full_lamb_dicke=True` flag matters (Wineland–Itano Laguerre
   structure on `|n⟩ → |n ± 1⟩` rates) and when the leading-order
   sideband Hamiltonian is sufficient.
9. **Squeezed / coherent state-prep.** `coherent_mode`,
   `squeezed_vacuum_mode`, `squeezed_coherent_mode`; composing via
   `states.compose_density`; how the squeeze-then-displace ordering
   maps to the qc.py scenario 5 convention.
10. **Finite-shot statistics.** Wilson vs Clopper–Pearson
    confidence intervals on `BinomialChannel` counts;
    `BinomialSummary` dataclass usage and choice of estimator.
    Parallels `tools/run_demo_wilson_ci.py`; treated briefly in
    Tutorial 1 and expanded here.
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
