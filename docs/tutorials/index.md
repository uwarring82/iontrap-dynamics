# Tutorials

!!! note "Placeholder — Phase 1.E in progress"

    This section is the designated home for task-oriented walkthroughs
    on `iontrap-dynamics`. The content landing here is scoped for Phase
    1.E (Sail material). Until then, most of the public-API surface is
    covered by the [Phase 1 Architecture](../phase-1-architecture.md)
    reference plus the four runnable demo tools under `tools/`, whose
    output bundles live at [`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data).

## What will live here

Planned tutorial topics (roughly in order of reader dependency):

1. **First simulation end-to-end.** Parallels
   `tools/run_demo_carrier.py` — configure an ion, a mode, a drive;
   build a carrier Hamiltonian; call `sequences.solve`; plot ⟨σ_z⟩(t)
   and compare to the analytic cosine.
2. **Red-sideband physics at Fock ∣1⟩.** Parallels
   `tools/run_benchmark_sideband.py`. Introduces the Lamb–Dicke
   parameter helper and the leading-order sideband Hamiltonian.
3. **Pulse shaping with `modulated_carrier_hamiltonian`.** Parallels
   `tools/run_demo_gaussian_pulse.py`. Gaussian envelope with
   calibrated pulse area → clean π-rotation; reader writes their own
   Blackman / stroboscopic envelope.
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

## Scope and licensing

Tutorials are Sail material — adaptive guidance with specific
parameter choices, not coastline constraints. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).

## Current learning path (until tutorials land)

- **Install + first run** — [Getting Started](../getting-started.md).
- **Architectural overview** — [Phase 1 Architecture](../phase-1-architecture.md).
- **Binding physics conventions** — [Conventions](../conventions.md) (rendered live from the repo root `CONVENTIONS.md`).
- **Contributor scope** — [Boundary Decision Tree](../boundary-decision-tree.md).
- **Runnable examples** — the four tools under `tools/run_demo_*.py` and the committed bundles under `benchmarks/data/<scenario>/` (each holding `manifest.json` + `arrays.npz` + `demo_report.json` + `plot.png`).
