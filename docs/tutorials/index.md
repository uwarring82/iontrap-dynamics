# Tutorials

Task-oriented walkthroughs of `iontrap-dynamics`. Each tutorial takes
~10–15 min to read and ~1–3 s to run. They are Sail material —
adaptive guidance with specific parameter choices, not coastline
constraints.

The track now spans eighteen tutorials, covering every
architectural layer of the library (configuration, Hamiltonians,
solve, observables, measurement, systematics, persistence,
entanglement diagnostics), an end-to-end reproduction of a
publication dataset against a legacy MATLAB bundle, and the two
service surfaces introduced at `v0.5.0` — parameter estimation /
quantum Darwinism and two-mode / motional open-system physics —
together with the `v0.6.0` motional completion (interferometric
observables, Lamb–Dicke regime helpers, and mode-frequency drift,
tutorials 14–17), and the `v0.6.0` reduced light–matter models
(JC / AJC / QRM vs full ion dynamics, Tutorial 18, anchored by
CONVENTIONS §25). The runnable demo tools under `tools/`
and their committed output bundles under
[`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data)
cover the same ground for users who prefer reading code to prose.

**First time?** Start with **[Tutorial 0](00_mental_model.md)** for the 30-second mental
model and your first runnable result, then follow the newcomer path **0 → 1 → 2 → 6** —
stop when you can build a Hamiltonian, read it out, and diagnose Fock truncation. Each
tutorial below is tagged `intro` (no prerequisites), `core` (the standard toolkit), or
`advanced` (a specialised or research-grade surface); do the `core` before diving into
`advanced`. Unfamiliar with a term? The **[Glossary](../glossary.md)** defines the
recurring vocabulary once.

Every tutorial is a variation on the same five-box pipeline:

--8<-- "docs/_snippets/pipeline.txt"

## Available

- `[intro]` [**Tutorial 0 — The 30-second mental model**](00_mental_model.md).
  The whole library in one breath: the physics‖code pipeline
  (`IonSystem → HilbertSpace → Hamiltonian → solve → readout`) and a single runnable cell
  that flops a spin and prints `⟨σ_z⟩` — a first result before Fock truncation, storage
  modes, or warnings. The on-ramp to the whole track.
- `[intro]` [**Tutorial 1 — Carrier Rabi flopping with finite-shot readout**](01_first_rabi_readout.md).
  End-to-end pipeline exercising every architectural layer through
  v0.2: configuration (`IonSystem`, `DriveConfig`, `ModeConfig`),
  `HilbertSpace`, `carrier_hamiltonian`, `sequences.solve`,
  `SpinReadout`, `binomial_summary`. Written as the canonical "Hello
  world" for the library post-v0.2 Convention Freeze.
- `[core]` [**Tutorial 2 — Red-sideband flopping from Fock ∣1⟩**](02_red_sideband_fock1.md).
  The four-step pattern from Tutorial 1 with the carrier swapped for
  the leading-order red-sideband Hamiltonian and motion initialised
  in `|n = 1⟩`. Introduces the `lamb_dicke_parameter` analytic helper,
  the `number` observable factory, and the `⟨σ_z⟩ + 2⟨n̂⟩ = 1`
  single-phonon-manifold conservation law as a sanity check.
  Parallels `tools/run_benchmark_sideband.py`.
- `[core]` [**Tutorial 3 — Gaussian π-pulse with `modulated_carrier_hamiltonian`**](03_gaussian_pi_pulse.md).
  First time-dependent Hamiltonian in the series. Swap the static
  carrier for a Gaussian-enveloped drive, normalise the envelope
  amplitude so the pulse area integrates to exactly π, and watch
  the Bloch vector trace a clean y–z meridian. Introduces the
  list-format dispatch through `sequences.solve` and the
  cumulative-integral analytic overlay
  `θ(t) = ∫₀^t Ω · f(t') dt'`. Closes with Blackman / stroboscopic
  / adiabatic-ramp envelope extensions. Parallels
  `tools/run_demo_gaussian_pulse.py`.
- `[core]` [**Tutorial 4 — Mølmer–Sørensen Bell gate**](04_ms_gate_bell.md).
  First two-ion scenario. Builds a two-²⁵Mg⁺ system sharing an
  axial COM mode, derives the Bell-closing detuning `δ = 2|Ωη|√K`
  and gate time `t_gate = π√K / |Ωη|` from physics inputs via the
  `ms_gate_closing_detuning` / `ms_gate_closing_time` analytic
  helpers, and verifies the four final-state invariants (loop
  closure `⟨n̂⟩ → 0`, equal Bell populations `P(|↓↓⟩) = P(|↑↑⟩) =
  1/2`, odd-parity `P_flip ≡ 0`, ion-exchange symmetry). First
  tutorial to wrap custom `qutip.Qobj` population projectors as
  `Observable` records. Parallels `tools/run_demo_ms_gate.py`.
- `[core]` [**Tutorial 5 — Custom observables**](05_custom_observables.md).
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
- `[core]` [**Tutorial 6 — Fock truncation diagnosis**](06_fock_truncation.md).
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
- `[core]` [**Tutorial 7 — Hash-verified cache round-trip**](07_cache_round_trip.md).
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
- `[advanced]` [**Tutorial 8 — Full Lamb–Dicke for hot-ion regimes**](08_full_lamb_dicke.md).
  When the `full_lamb_dicke=True` flag on the sideband builders
  matters. Covers the Wineland–Itano closed form
  `Ω_{n,n−1}^full = Ω·|η|·e^(−η²/2)·√((n−1)!/n!)·L_{n−1}^(1)(η²)`,
  the `η²·n ≳ 0.1` rule-of-thumb crossover, and a quantitative
  three-scenario comparison showing 3 % → 16 % → 30 % rate
  shortfall as `n` climbs from 1 to 10 at fixed `η = 0.26`.
  Closes with a when-to-flip decision tree covering
  thermal-start scenarios, MS-gate tuning, and sideband
  cooling cascades.
- `[core]` [**Tutorial 9 — Squeezed / coherent state preparation**](09_squeezed_coherent_prep.md).
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
- `[core]` [**Tutorial 10 — Finite-shot statistics**](10_finite_shot_statistics.md).
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
- `[core]` [**Tutorial 11 — Systematics: jitter ensembles**](11_jitter_ensembles.md).
  First systematics-layer tutorial. Layers a `RabiJitter(σ=3%)`
  onto the Tutorial 1 carrier-Rabi scenario and runs a
  200-trial ensemble through `solve_ensemble`. Verifies the
  ensemble mean against the analytic Gaussian-envelope
  dephasing prediction
  `⟨σ_z⟩ = −cos(Ω̄t)·exp(−(σΩ̄t)²/2)`; contrasts ensemble
  mean / std / SEM as three distinct error channels;
  illustrates the `n_jobs=1` default with a performance note
  and a DetuningJitter variation for the Lorentzian-dephasing
  analogue.
- `[advanced]` [**Tutorial 12 — Two-ion Bell-state entanglement**](12_bell_entanglement.md).
  Closes the tutorials track. Takes the Tutorial 4 MS-gate
  scenario and reads it out through both the `ParityScan`
  protocol (finite-shot parity estimator with explicit
  detector envelope) and the three registered entanglement
  trajectory evaluators (`concurrence_trajectory`,
  `entanglement_of_formation_trajectory`,
  `log_negativity_trajectory`). Three witnesses on one gate
  show complementary information: mid-gate spin-motion
  entanglement (log-negativity peaking at 1.31) disappears as
  the phase-space loop closes, while spin-spin concurrence
  rises from 0 to exactly 1.0 at `t_gate`. Explains the
  `OMITTED` vs `EAGER` storage-mode split required by the
  two surfaces and points the reader at the remaining
  library-surface learning paths beyond the tutorial track.
- `[advanced]` [**Tutorial 13 — Reproducing Clos 2016 (PRL 117, 170401)**](13_reproducing_clos_2016.md).
  End-to-end reproduction of a publication dataset against the
  legacy MATLAB bundle under
  [`legacy/clos 2016 prl/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/legacy/clos%202016%20prl).
  First tutorial to use the exact-diagonalisation entry point
  (`solve_spectrum`) instead of `sequences.solve`. Builds the
  **non-RWA** full-displacement spin–boson Hamiltonian (distinct
  from the carrier-RWA `carrier_hamiltonian_full_ld` covered in
  Tutorial 8), computes the legacy `IPR_av` quantity via
  `clos2016_averaged_effective_dimension` (a
  ρ₀-eigendecomposition–weighted average of pure-state effective
  dimensions, distinct from the textbook `effective_dimension`),
  and matches `theo_dim_N_1.dat` row-by-row. Documents the
  Raman-vs-single-photon wavelength provenance trap, then scales
  the same pipeline to N=2 and N=3 with achieved tolerances of
  ~6 % and ~4 % respectively. Closes with the dense-eigh
  envelope table that motivates the AAG / AAH benchmark
  dispatches.
- `[advanced]` [**Tutorial 14 — Quantum metrology: the Fisher-information limit**](14_quantum_metrology_qfi.md).
  First tutorial on the `iontrap_dynamics.information` estimation surface.
  Encodes a parameter unitarily and reads the precision ceiling with
  `quantum_fisher_information_trajectory`: the pure-state identity
  `F_Q = 4·Var(G)`, GHZ Heisenberg scaling `N²` versus the product
  standard quantum limit `N`, and `classical_fisher_information` /
  `cramer_rao_bound` showing the σ_y measurement saturates the bound while
  σ_x is phase-blind. Closes on continuous-variable probes (coherent
  `4|α|²`, sub-shot-noise squeezed `2e^{−2r}`). Embeds `qfi_scaling`,
  `cfi_linear_gaussian`, and `probe_qfi`.
- `[advanced]` [**Tutorial 15 — Quantum Darwinism: why the world looks classical**](15_quantum_darwinism.md).
  The companion estimation tutorial. Reads the partial-information plateau
  (`partial_information_plot`), the redundancy `R_δ = N`
  (`redundancy`), and the recoverability of quantum information from a
  Werner-mixed Bell pair (`recoverability`) off a GHZ cascade. Distinguishes
  *how much* each fragment knows (plateau height) from *how many* fragments
  know it (redundancy). Embeds `darwinism_redundancy` and `recoverability`.
- `[advanced]` [**Tutorial 16 — Two-mode SU(1,1) squeezing**](16_two_mode_squeezing.md).
  First two-mode motional Hilbert space. Builds the
  `two_mode_squeezed_vacuum` (per-mode `n̄ = sinh²|z|`, §23), reproduces it
  dynamically under `two_mode_squeezing_hamiltonian` (`n̄ = sinh²(gτ)`, the
  difference number pinned by the su(1,1) Casimir), and contrasts the SU(2)
  `beamsplitter_hamiltonian` that conserves the sum and swaps excitation
  instead. Heavy on the Fock-truncation gotcha. Embeds `two_mode_squeezing`.
- `[advanced]` [**Tutorial 17 — Motional decoherence and the Lamb–Dicke regime**](17_motional_decoherence_and_lamb_dicke.md).
  Bundles four motional-imperfection surfaces into one workflow: typed CPTP
  channels through `solve(channels=…)` (heating / damping / dephasing, with
  a windowed variant), reading the contrast loss off a fringe with
  `fringe_visibility` / `fit_fringe`, the Lamb–Dicke regime map
  (`debye_waller_factor`, `lamb_dicke_regime`, full-LD vs leading-order
  sideband), and budgeting a `ModeFrequencyDrift` (`η → η/√(1+δ)`). Embeds
  `motional_channels` and `lamb_dicke_regime`.
- `[advanced]` [**Tutorial 18 — Reduced models vs full dynamics**](18_reduced_models_vs_full_dynamics.md).
  Walks the four falsifiable cases of the reduced light–matter hierarchy
  (CONVENTIONS §25): the LOCK-3 identity makes JC and AJC spectra coincide
  (`jaynes_cummings_hamiltonian`, `anti_jaynes_cummings_hamiltonian`,
  `solve_spectrum`); the same label becomes a physical red-dark / blue-bright knob
  on the ion; `model_deviation` measures the rotating-wave approximation breaking
  down as `g/ω₀` grows into the `quantum_rabi_hamiltonian`; and the `2g√(n±1)`
  limit ties the reduced coupling back to the apparatus sideband rate. Embeds
  `reduced_models_comparison`.
- `[advanced]` [**Tutorial 19 — Squeezing by quenching the trap frequency**](19_squeezing_by_quenching.md).
  Generates motional squeezing from a time-dependent trap frequency `ω(t)` alone
  (CONVENTIONS §26): builds `nonadiabatic_squeezing_hamiltonian` from a
  `FrequencyWaveform`, reads squeezing back from the covariance matrix
  (`gaussian.squeezing_parameter` — the eigenvalue ratio `r = ¼ln(λ_max/λ_min)`, the
  purity `ν`, `n̄_sq`), checks the sudden kick and cyclic adiabatic limits, shows
  the Wigner ellipse on the vacuum-variance-1 grid (`phase_space.wigner`, `g = 1`),
  and optimises a single down/up pulse (`down_up_pulse`) whose squeezing oscillates
  with the hold time and peaks at ≈ twice a one-way ramp. Embeds
  `nonadiabatic_squeezing`.
- `[advanced]` [**Tutorial 20 — Phonon-pair creation and readout**](20_phonon_pair_creation.md).
  Reads a squeezed state out as a phonon-number distribution
  (`phonon_number_distribution`) and shows the even-`n`-only pair signature against
  the analytic `pure_squeezed_vacuum_pn`, grows the pairs by parametric modulation,
  and exercises the parity-aware Fock-truncation guard (`check_fock_truncation`,
  §13/§15) that keeps the readout honest. Shows how a parasitic displacement
  (`displacement_force_hamiltonian`) lifts the parity and how a two-pulse purifying
  echo removes it while amplifying the squeezing. Sketches the two-ion cosmology /
  Hawking analogue as an outlook.
- `[advanced]` [**Tutorial 21 — Two-ion motional entanglement: normal → local cut**](21_normal_to_local_entanglement.md).
  Builds the normal→local symplectic map (`ion_modes`, GT3b) from a `ModeConfig`-style
  basis and transports a Gaussian covariance into the local-ion frame, then reads the
  ion-cut entanglement with `log_negativity` and `entanglement_of_formation`. Shows
  that the two-ion motional *ground state* is entangled across the ions, that the
  Coulomb frequency splitting (`ω_stretch = √3 ω_COM`) is the entangler, and that the
  local-frequency gauge shifts each ion's effective temperature without touching the
  entanglement — the whole Gaussian toolbox in one pipeline.

## Scope and licensing

Tutorials are Sail material — adaptive guidance with specific
parameter choices, not coastline constraints. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).

## Complementary references

- **Install + first run** — [Getting Started](../getting-started.md).
- **Vocabulary** — [Glossary](../glossary.md) (plain-language definitions of the recurring terms).
- **Architectural overview** — [Phase 1 Architecture](../phase-1-architecture.md).
- **Binding physics conventions** — [Conventions](../conventions.md) (rendered live from the repo root `CONVENTIONS.md`).
- **Performance baselines** — [Benchmarks](../benchmarks.md).
- **Contributor scope** — [Boundary Decision Tree](../boundary-decision-tree.md).
- **Runnable examples** — the tools under `tools/run_demo_*.py` and the committed bundles under `benchmarks/data/<scenario>/` (each holding `manifest.json` + `arrays.npz` + `demo_report.json` + `plot.png`).
