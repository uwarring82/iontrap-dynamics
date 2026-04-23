# Workplan — Clos/Porras 2016 PRL integration (shipped; decisions recorded 2026-04-23)

> **Status: decisions recorded 2026-04-23. Retained for historical
> continuity per §9.** The capability shipped: Dispatches AAA–AAF
> and AAH–AAI landed on `main`; AAG is deferred on measurement,
> with an explicit re-activation trigger in
> [benchmarks.md § Exact-diagonalization envelope](benchmarks.md).
> The §7 open questions below are resolved in-line (see each Q for
> the decision and pointer to the shipped artefact). This document
> is no longer a deliberation surface — it is the integration's
> planning record. Coastline commitments (builders, result schemas,
> backend names, tolerances) live where they were recorded at ship
> time: `CHANGELOG.md`, `CONVENTIONS.md`, and the source / test /
> benchmark artefacts cited throughout.
>
> **Original framing (draft for deliberation, not Coastline).** This
> document scoped a cross-cutting initiative to port Diego Porras's
> MATLAB spin–boson machinery into `iontrap-dynamics`, reproduce the
> published observables, and measure the exact-diagonalization
> envelope on commodity hardware. Nothing here bound implementation;
> final Dispatch-level architecture decisions were made at the
> start of each Dispatch and recorded in `CHANGELOG.md` /
> `CONVENTIONS.md` at that time.

**Relates to:** `legacy/clos 2016 prl/INVENTORY.md` (source bundle),
`WORKPLAN_v0.3.md` §5 (adjacent to, but not part of, the v0.3.x
Phase-2 follow-up track),
`docs/phase-2-jax-backend-design.md` §4 Axis C (Lamb–Dicke
representation choice), `tests/regression/migration/test_migration_references.py:424`
(documented full-LD activation path).

**Classification.** The in-scope work produces new physics capability
(full-exponential Lamb–Dicke, IPR / ETH spectrum analyses, exact
diagonalization entry), new test tiers (Porras regression), and
a new benchmark family (exact-diag envelope). All become Coastline
the moment they ship. Implementation tactics (which eigensolver,
which iterative routine, which plot style) stay Sail.

---

## 1. Motivation

Three motivations, in increasing generality:

1. **Validation against a real experiment.** The legacy bundle at
   `legacy/clos 2016 prl/INVENTORY.md`
   contains pre-publication numerical results for a trapped-ion
   spin–boson simulator (PRL 117, 170401). The data tables —
   `theo_dim_N_{1..5}.dat`, `exp_ETH_N_{1..5}.dat` — are a
   regression reference with experimental cross-check, stronger than
   any analytic limit we currently test against.

2. **Close the full-Lamb–Dicke gap.** Our current Hamiltonian builders
   apply a leading-order Lamb–Dicke expansion. Porras's `ergodic.m`
   uses the full exponential operator
   $C = \exp\!\big(\eta_0 \sum_n \phi_n (a_n - a_n^\dagger)/\sqrt{\omega_n/\omega_1}\big)$.
   This gap is already documented as an activation path in
   `tests/regression/migration/test_migration_references.py:424`.
   Reproducing Porras at N = 1–3 forces us to close it.

3. **Know the hardware envelope.** IPR / ETH diagnostics need
   exact diagonalization. For N ions with per-mode cutoff $n_c$ on an
   axial chain, the spin–motion Hilbert space has dimension
   $d = 2^N \cdot n_c^N$. Because the full-exponential carrier
   operator is built from `expm(pol)`, the Hamiltonian is generically
   **dense in the Fock basis**; the envelope where dense
   diagonalization is tractable, and whether any iterative
   interior-window method is worthwhile on top of that dense
   structure, is currently **not measured** in this repo. Users
   hitting "how big a system can I diagonalize on my laptop?"
   deserve a concrete answer.

The three motivations together give the user's stated asks: (1)
integrate the Porras Hamiltonian + helpers, (2) benchmark, (3)
identify the PC-hardware exact-diag envelope.

---

## 2. Scope

### In scope — completion criteria at workplan closure

- **Full-exponential Lamb–Dicke Hamiltonian builder.** A public
  entry point that produces $H_\text{sph} + H_n + (\omega_z/2)\sigma^z$
  for N-ion axial chains with the full-exponential carrier operator.
  Leading-order builders remain available (Principle 5: separate
  entry, not an overload; no `ld_order=None` overload on the
  existing builder surface).
- **IPR, effective dimension, ETH-diagonal, and phonon-number-diagonal spectrum analyses.**
  New helpers in `src/iontrap_dynamics/spectrum_observables.py`
  (preferred) or colocated under `spectrum.py` if the module stays
  small. These consume a `SpectrumResult` plus an initial state and
  return typed analysis results. They do **not** extend the existing
  `Observable` contract, which is trajectory/operator-based.
- **Exact-diagonalization entry point.** A new module (proposed
  `src/iontrap_dynamics/spectrum.py`) with a `solve_spectrum(H, …)`
  API returning eigenvalues, eigenvectors, and derived diagnostics
  under a typed `SpectrumResult`. Dense `eigh` is the reference
  path; any iterative path must target an interior energy window
  around the initial state's mean energy rather than the lowest-k
  eigenpairs.
- **Reproduction against bundled Porras references at N ∈ {1, 2, 3}.**
  Exact-diag results agree with the bundled
  `theo_dim_N_{1,2,3}.dat` values within a declared tolerance
  (target: 1 %, documented if larger is needed — see §6 Risk 3).
- **PC-limits benchmark artefact.** A `tools/run_benchmark_spectrum_envelope.py`
  that measures wall-clock and peak memory for dense and any
  iterative interior-window
  exact diagonalization over a grid of (N, n_c). Report in
  `benchmarks/data/spectrum_envelope/`; section in
  `docs/benchmarks.md`.
- **Tutorial.** `docs/tutorials/T13-reproducing-clos-2016.md`
  walks through loading one of the bundled data tables,
  reproducing a single panel, and reading off the d_eff value.

### Out of scope — any design

- **Reproducing the paper PDF / figures.** We reproduce the
  *numbers* (d_eff, IPR, ETH diagonals, phonon-number diagonals); the figure-rendering
  side is Govinda Clos's `sbpaper` and stays upstream.
- **ETH theory development.** We *implement the spectrum analyses*
  Porras's code computes; we don't re-derive the ETH prediction
  or extend statistical-mechanics theory. If a user wants to
  study ETH-violating regimes, that's a Phase-4+ capability.
- **GPU exact diagonalization.** JAX's `jax.numpy.linalg.eigh`
  exists and works on GPU, but workplan §3 currently targets
  CPU-only (SciPy / NumPy) as the first cut — matches the
  "commodity PC" framing of the user's ask. GPU-diag becomes
  its own dispatch if and when the CPU envelope is characterised.
- **Time evolution on the full-LD Hamiltonian.** Phase-2 design
  (`docs/phase-2-jax-time-dep-design.md`) already handles
  time-dependent dynamics; full-LD builders here feed into those
  solvers unchanged. We do not add new time-evolution machinery.
- **Experimental fit / parameter-inference work.** The bundled
  `exp_*_N_{1..5}.dat` files are experiment-side; comparing them
  to theory is a research task, not a library feature. The
  tutorial demonstrates the path but does not ship a fitter.

### Deliberately ambiguous — to resolve in §7

- Whether an iterative interior-window solver is worth shipping in
  wave 2 once the dense full-LD benchmark is measured. If yes, the
  target is shift-invert / `sigma=meanE` style extraction, not
  lowest-k eigenpairs.

---

## 3. Architecture alignment

The work touches five existing modules and introduces two new ones.
The following table makes the touchpoints explicit.

| Piece                                      | Current state in repo                                                          | Change under this workplan                                                                 |
|--------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Axial-chain normal modes                   | `modes.axial_modes(N, center)` — leading order                                | No change if convention matches Porras's `cchain(N, center)`; else convention-reconciliation note in `CONVENTIONS.md` §X.                     |
| Lamb–Dicke parameter                       | `species.mg25_plus.lamb_dicke(axial_MHz)` — closed-form                        | Verify against Porras's `eta_calculator(25, 200, axial)` numerics; **helper is missing from legacy bundle** (see §7 Risk 1).                 |
| Carrier + sideband Hamiltonians            | `hamiltonians.{carrier,red_sideband,blue_sideband}_hamiltonian`                | Add `*_full_ld(...)` siblings; leading-order entries remain unchanged.                       |
| Observables                                | `observables.Observable` protocol + sz/sx/nbar/…                               | No change to the trajectory-observable contract; spectrum analyses land on a separate surface. |
| Solver entry                               | `sequences.solve` + `sequences.solve_ensemble` (time-evolution only)           | No change — exact diag is a different API (`spectrum.solve_spectrum`), not an overload. Dense `eigh` is the reference solver; any iterative path is interior-window only. |
| *(new)* Spectrum module                    | —                                                                              | `src/iontrap_dynamics/spectrum.py` with `solve_spectrum`, `SpectrumResult`.                 |
| *(new)* Spectrum analyses                  | —                                                                              | `src/iontrap_dynamics/spectrum_observables.py` for IPR, `d_eff`, ETH diagonals, phonon-number diagonals, and related helpers. |
| *(new)* Clos 2016 reproduction surface     | —                                                                              | `src/iontrap_dynamics/clos2016.py` carries the pieces that don't fit the textbook builder surface: the **non-RWA** full-displacement spin–boson Hamiltonian (`clos2016_spin_boson_hamiltonian`, parallel to AAB's RWA-projected `carrier_hamiltonian_full_ld`), the matching thermal initial state (`clos2016_initial_state`), the legacy `IPR_av` mixed-state quantity (`clos2016_averaged_effective_dimension`), and the pinned Raman wavelength constant (`CLOS2016_LEGACY_WAVELENGTH_M`). Plus a theory-surface loader (`load_clos2016_theory_dimension_surface`) added to `clos2016_references.py`. |

**CONVENTION_VERSION bump?** Adding a new public builder, a
new `solve_spectrum` entry, or a sibling spectrum-analysis module
does **not** change existing behaviour
→ no `CONVENTION_VERSION` bump required per §5.1 of
`CONVENTIONS.md`. Adding a `backend_name` for spectrum results
is a Coastline commitment per D5 (§4.2 of the JAX design note);
proposed value `"spectrum-scipy"` (CPU dense) /
`"spectrum-scipy-shift-invert"` (iterative interior window, if shipped).

**`SpectrumResult` schema decision.** This workplan no longer leaves
the result shape open. `SpectrumResult` is a frozen sibling of
`TrajectoryResult` in the existing result family: it subclasses
`Result`, carries metadata fields parallel to `ResultMetadata`
(`convention_version`, `request_hash`, `backend_name`,
`backend_version`, `fock_truncations`, `provenance_tags`), and stores
the spectrum-specific payload on named fields rather than positional
returns. The concrete payload is:

- `eigenvalues` — 1-D real array, ascending within the returned set.
- `eigenvectors` — 2-D complex array or `None` when lazily loaded.
- `eigenvectors_loader` — optional loader when eigenvectors are not
  materialised.
- `window_center_energy` — target energy used for interior-window
  extraction; `None` for dense full-spectrum solves.
- `window_width_energy` — declared physical width tied to the
  initial-state energy spread when used.
- `initial_state_mean_energy` / `initial_state_energy_std` — stored
  scalars for reproducibility and ETH-style weighting.
- `method` — `"dense"` or `"shift_invert"`.

Dense Hamiltonian snapshots and backend-specific initial-state objects
are **not** stored on-result or in cache; reproducibility binds through
the request hash plus the recorded energy-window scalars.

---

## 4. The three asks, as deliverables

### 4.1 Integrate the Porras Hamiltonian + helpers

**Inputs we have:**

- `legacy/clos 2016 prl/ergodic_2015_03_15.m` — the canonical driver.
  Documents all inputs/outputs; builds `H = Hsph + Hn + (ωz/2)σz`;
  computes IPR, IPR₀, ratio, σx/σy/σz diagonals, phonon diagonals.
- `legacy/clos 2016 prl/GC collection/matlabscripts/ergodic.m` —
  later successor variant (same interface, refined code).
- Multiple driver scripts (`ipr_*.m`, `sb_evolution.m`, `spectrum.m`)
  that exercise `ergodic` at different sweep parameterisations.

**Inputs that aren't standalone files but *are* present inline.**
The original draft of this workplan claimed `cchain.m`, `Operators.m`,
and `eta_calculator.m` were absent based on a `find` of top-level
filenames. That finding is incorrect: every `.m` driver in the bundle
(both `ergodic*.m` at the top level and the `_ipr_av*` family under
`GC collection/matlabscripts/`) carries its own private inline copies
of all three helpers as sub-functions — see e.g. `ergodic.m:155`
(`function eta = eta_calculator(m,lam,wx)`) and `ergodic_ipr_av.m:240`
(same helper, identical body) plus `ergodic_ipr_av.m:252`
(`function [Xmin,ener,center_wf,V] = cchain(n,center)`). We still
re-derive them in Python rather than transliterating MATLAB, but the
provenance check is cleaner: the helpers we re-derive can be cross-
checked against the bundle text, not just inferred from physics.

**Wavelength caveat — the `_ipr_av` family uses a different value.**
The two `ergodic*` driver families do *not* use the same Lamb–Dicke
calibration:

- `ergodic.m`, `ergodic_2015_03_15.m`, `sb_evolution*.m`, … →
  `eta_calculator(25, 200, axial)` (200 nm single-photon reference).
- `ergodic_ipr_av.m`, `ergodic_ipr_av_eta.m`, `ergodic_ipr_av_new*.m` →
  `eta_calculator(25, 279.63/sqrt(2), axial)` (≈ 197.73 nm Raman
  two-photon effective wavelength).

The `theo_dim_N_*.dat` regression anchor is downstream of the
`_ipr_av*` family — `IPR_av` is computed only there. Reproduction
code therefore **must** use the Raman two-photon wavelength, not
200 nm. `src/iontrap_dynamics/clos2016.py` pins
`CLOS2016_LEGACY_WAVELENGTH_M = 279.63e-9 / sqrt(2)` for this reason.

The re-derivations below match the standard physics, not the
inline helpers verbatim:

- **`cchain`** → the workplan §3 row claims this already exists as
  `modes.axial_modes`, but no such builder is present in the
  library yet (verified during AAF). The Coulomb-chain modes are
  re-derived numerically from `mypotential` (a quadratic axial
  trap with Coulomb repulsion at $\beta = 4$); for N = 2 and N = 3
  the AAA reference dataclass `clos2016_axial_mode_reference`
  pins them inline. **Verification step needed:** use explicit
  N = 2 and N = 3 axial references so the reconciliation is
  testable, not rhetorical. In the legacy `cchain(..., center=1)`
  convention — verified by re-implementing the Hessian eigenvalue
  problem in Python and matching the legacy MATLAB output — the
  dimensionless mode frequencies are:
  N = 2 → $\omega/\omega_1 = [1,\;\sqrt{3}]$ with first-ion
  participation weights $[-1/\sqrt{2},\;-1/\sqrt{2}]$;
  N = 3 → $\omega/\omega_1 = [1,\;\sqrt{3},\;\sqrt{29/5}]$
  with first-ion participation weights
  $[1/\sqrt{3},\;-1/\sqrt{2},\;1/\sqrt{6}]$ up to eigenvector-sign
  convention. (The original AAA dispatch landed `[1, 1.12267]`
  for N = 2 and `[1, 1.110697, 1.249302]` for N = 3; those values
  do not match `cchain` for any beta or trap convention found in
  the bundle and were corrected during AAF. Participation weights
  were already correct.)
- **`Operators`** → tensor products of qutip spin and phonon
  operators, already idiomatic in our codebase.
- **`eta_calculator`** → Lamb–Dicke parameter
  $\eta = k x_0 = (2\pi/\lambda)\sqrt{\hbar/(2 m \omega)}$,
  already implemented in `species.mg25_plus`. The Porras call
  signature parses as (mass / u, wavelength / nm, axial / MHz);
  the `_ipr_av*` family uses `(25, 279.63/sqrt(2), axial)` per the
  wavelength caveat above. `species.mg25_plus.lamb_dicke` takes a
  generic wavelength argument, so the regression path can pin the
  Raman value via `CLOS2016_LEGACY_WAVELENGTH_M` rather than baking
  the 200 nm single-photon reference into the helper.

**Dispatch shape.** See §5.

### 4.2 Run benchmarks

Three benchmark families, each with its own artefact:

1. **Full-LD vs. leading-order time-evolution cost** — how much does
   full-exponential Lamb–Dicke cost over our current leading-order
   builders? Measured at N ∈ {1, 2, 3}, n_c ∈ {3, 5, 8}, single
   Rabi period. Goal: quantify the performance penalty users
   pay for the full-LD path. Artefact:
   `tools/run_benchmark_lamb_dicke_order.py`.
2. **Exact-diagonalization envelope** (the user's ask 3, instantiated
   as benchmark). Dense `eigh` is the baseline at each `(N, n_c)`.
   If an iterative path is trialled, it must target the interior
   window around `meanE` (legacy `eigs(H, Neigens, meanE)` style),
   and the benchmark must compare it honestly against dense `eigh`
   on the same dense full-LD Hamiltonian rather than assuming a
   dense→sparse win.
   Artefact: `tools/run_benchmark_spectrum_envelope.py`. See §4.3.
3. **Spectrum-analysis cost** — the time to compute IPR, d_eff,
   ETH diagonals, and phonon-number diagonals once `H` is
   diagonalized. Small but worth
   measuring to know where the bottleneck sits (typically it's
   the diag, not the observable). Artefact rolls into (2).

All three report to `benchmarks/data/*/` with a JSON manifest
plus plot, matching the existing Dispatch X / Y / OO pattern.

### 4.3 PC exact-diagonalization envelope (user's ask 3)

This is the most concrete deliverable. Hilbert dimension for N
ions and per-mode cutoff $n_c$ is $d = 2^N \cdot n_c^N$.
For the Porras full-LD Hamiltonian, however, dimension is only half
the story: because `ergodic.m` builds `Hsph` through `expm(pol)`, the
matrix is generically dense in the truncated Fock basis. Any
iterative method therefore has to beat dense `eigh` on a **dense**
operator to earn its keep; that is a benchmark question, not a
planning assumption.

| N | $n_c$ | $d$      | Dense matrix storage lower bound (complex128, GB) | Feasible?              |
|---|-------|----------|--------------------------------|------------------------|
| 1 | 5     | 10       | 10⁻⁹                                        | Trivial                |
| 2 | 5     | 100      | 10⁻⁶                                        | Trivial                |
| 3 | 5     | 1 000    | 0.016                                       | Easy                   |
| 4 | 5     | 10 000   | 1.6                                         | Laptop-feasible        |
| 4 | 6     | 20 736   | 6.9                                         | Borderline 16 GB; OK 32 GB |
| 5 | 3     | 7 776    | 0.97                                        | Laptop-feasible        |
| 5 | 4     | 16 384   | 4.3                                         | Laptop-feasible        |
| 5 | 5     | 100 000  | 160                                         | **Infeasible dense**; iterative-only if benchmark proves value |
| 5 | 6     | 248 832  | 990                                         | **Infeasible dense**; iterative-only if benchmark proves value |
| 5 | 10    | 10 000 000 | 1 600 000                                 | **Infeasible without new structure exploitation** |

Cells above are order-of-magnitude back-of-envelope; the benchmark
measures the **actual** envelope on representative hardware
(consumer laptop, workstation). The memory column is a **dense-matrix
storage lower bound only**; actual peak RSS for `eigh` can be
materially higher once workspace and eigenvector storage are included.
Report includes:

- For dense (`scipy.linalg.eigh`): max $(N, n_c)$ that completes
  under a declared wall-clock budget (e.g. 10 min) and RAM
  budget (16 GB / 32 GB / 64 GB).
- If an iterative path is trialled: max $(N, n_c)$ that returns the
  interior window around `meanE` (e.g. 100–300 states nearest the
  target energy) within the same budgets, together with a direct
  comparison to dense `eigh` on the same dense matrix.
- Explicit table of "on a typical PC, you can exact-diagonalize up
  to this system size." This is the user-facing answer.

Deliverable lives in `docs/benchmarks.md` as a new subsection
**Exact-diagonalization envelope**, plus the raw data under
`benchmarks/data/spectrum_envelope/`.

---

## 5. Dispatch decomposition

This work is best treated as a **parallel capability track**, not as
Phase 3 and not as a sub-dispatch of the Phase-2 JAX line. The natural
dispatch sequence after `ZZ` is therefore `AAA`, `AAB`, … for this
track.

1. **Dispatch AAA — axial-mode convention reconciliation.**
   Before the full-LD builder lands, record the N = 3 axial-mode
   frequencies and participation weights from our `axial_modes`
   path and reconcile them against the Porras convention. Do not stop
   at prose: encode the explicit N = 2 / N = 3 reference values from
   §4.1 as regression expectations, with sign-insensitive eigenvector
   comparison. Also parse the `*ions_ipr_vs_nc.txt` convergence tables
   and record the minimal converged `n_c` used for N = 1–5 before any
   reproduction tolerance is declared. This is the cheap gate that
   blocks the rest of the track if either the mode convention or the
   truncation assumptions are already off. Cost: ~0.5 dispatch.
2. **Dispatch AAB — full-LD carrier builder.** Add
   `carrier_hamiltonian_full_ld(...)`. Quantitative test:
   for $\eta \in \{0.01, 0.05, 0.1\}$, require the relative
   Frobenius-norm difference between the full-LD and leading-order
   Hamiltonians to stay below `5 η²`; at $\eta = 0.3$, document the
   visible divergence rather than asserting agreement. Cost: ~1
   dispatch.

   **Important scope clarification (added during AAE).** The shipped
   `carrier_hamiltonian_full_ld` is the **carrier-RWA** all-orders
   builder: each mode contributes the diagonal Δn = 0 projection
   $\langle n|\hat M_0|n\rangle = e^{-\eta^2/2} L_n(\eta^2)$ of the
   displacement, i.e. the standard Debye–Waller / Laguerre dressing.
   That is the right object for textbook carrier dynamics under RWA.
   It is **not** the Porras spin–boson Hamiltonian: `ergodic_ipr_av.m`
   keeps the bare displacement $\sigma_+\,\exp(\hat P)$ (no Δn = 0
   projection), so off-diagonal Fock couplings — the very terms whose
   non-secular character drives ergodicity — survive. Reproducing the
   `theo_dim_N_*.dat` numbers therefore requires a separate non-RWA
   spin–boson assembler (shipped in AAE; see §3 architecture table
   for `src/iontrap_dynamics/clos2016.py`). AAB stays in scope as a
   library capability, but it is not the path the Porras regression
   takes.
3. **Dispatch AAC — spectrum module skeleton.**
   `src/iontrap_dynamics/spectrum.py` with `solve_spectrum(H, …)`
   returning `SpectrumResult(eigenvalues, eigenvectors, metadata)`.
   Dense full-spectrum solve via `scipy.linalg.eigh` is the reference
   entry. Unit tests for a 10-level harmonic-oscillator H (known
   spectrum). Cost: ~1 dispatch.
4. **Dispatch AAD — IPR, d_eff, ETH diagonals, and phonon-number diagonals analyses.** Four
   new spectrum-analysis helpers in
   `src/iontrap_dynamics/spectrum_observables.py`; each takes a
   `SpectrumResult` and an initial state, returns the scalar (or
   per-eigenstate array for ETH / phonon-number diagonals). Tests:
   known-state sanity (a single eigenstate
   has IPR = 1; a uniform superposition over $d$ eigenstates has
   IPR = $1/d$); thermal-state phonon diagonals against ergodic.m
   `nphdiag_cell` for a small N = 1 case. Cost: ~1 dispatch.

   **Naming note (added during AAE).** AAD ships the *textbook*
   `effective_dimension(spectrum, ρ) = 1 / Σ_α ⟨E_α|ρ|E_α⟩²`. The
   legacy MATLAB labels that quantity `IPR` (a misnomer) and ships a
   *separate* `IPR_av` quantity defined in `ergodic_ipr_av.m:121-132`
   as $\sum_j \lambda_j / \sum_\alpha |\langle E_\alpha|\psi_j\rangle|^4$
   — the mixed-state-weighted average of pure-state effective
   dimensions over the eigendecomposition $\rho_0 = \sum_j \lambda_j
   |\psi_j\rangle\langle\psi_j|$. The two coincide on pure states but
   diverge on mixed states. The Clos 2016 `theo_dim_N_*.dat` tables
   store `IPR_av`, *not* `effective_dimension`. AAD's helpers stay
   textbook; the legacy quantity lands in `clos2016.py` (AAE).
5. **Dispatch AAE — Porras N = 1 regression test.** Load
   `legacy/clos 2016 prl/DP num res_fig_1_2015_07_30/theo_dim_N_1.dat`
   as a regression reference; reproduce the `IPR_av` values using
   Dispatches AAA + AAC + AAD plus the new non-RWA assembler in
   `clos2016.py` (see AAB scope clarification). Workplan tolerance
   target was 1 %; achieved tolerance is `rtol = 0.10` against the
   3-significant-figure published table (max relative deviation
   ≈ 8.8 % at the sharp mid-resonance peaks at det ≈ 1.0, 2.0
   legacy units). The §6 Risk 3 guidance — "document whatever
   tolerance the pipeline actually achieves rather than forcing 1 %"
   — applies. Test lives at
   `tests/regression/reproduction/test_clos_2016_N1.py`. Cost: ~1
   dispatch (the AAE wave is larger than the original 0.5-dispatch
   estimate because the parallel non-RWA Hamiltonian, the `IPR_av`
   helper, the thermal initial-state builder, and the
   `theo_dim_N_*.dat` surface loader all land here, plus the
   wavelength provenance work).
6. **Dispatch AAF — Porras N = 2, N = 3 regression tests.** Same
   as AAE, extended. N = 4 and N = 5 deferred pending §4.3 results
   — they may exceed dense envelope. Cost: ~1 dispatch.

   **Shipped.** Test lives at
   `tests/regression/reproduction/test_clos_2016_N2_N3.py`,
   parametrized over `(N=2, cutoff=8)` and `(N=3, cutoff=6)`.
   Achieved tolerances on the 3-significant-figure published
   table: N = 2 ≈ 6.3 %, N = 3 ≈ 4.2 %. N = 3 uses cutoff = 6
   (below the inferred-converged value of 10) because cutoff = 10
   would put a single regression run over five minutes; the
   row-vs-row comparison at the same cutoff stays apples-to-apples.
   The AAA reference frequencies for N = 2 / N = 3 had to be
   corrected from the originally pinned (incorrect) values during
   this dispatch — see the §4.1 cchain note.
7. **Dispatch AAG — benchmark-gated interior-window iterative path.**
   Only if the `AAA`/`AAH` evidence suggests value, extend
   `solve_spectrum` with a `method="shift_invert"` option targeting the
   initial-state mean energy, mirroring legacy `eigs(H, Neigens, meanE)`.
   Success criterion is **not** "find lowest k"; it is "recover the
   same microcanonical observables as dense `eigh` from an interior
   window around `meanE`". `backend_name="spectrum-scipy-shift-invert"`.
   Cost: ~1 dispatch if it ships; zero if benchmark evidence says no.

   **Status: deferred.** AAH (measurement below) shows the dense
   path covers N ≤ 3 at fully converged cutoffs and N = 4 up to
   `n_c = 7` (`dim = 8 192`, measured: 7 min wall-clock, 3.6 GB
   peak RSS) on a 16 GB commodity laptop. The 16 GB envelope also
   reaches N = 5 at `n_c = 5` (`dim = 15 552`, projected). AAG
   becomes justified in two specific cases now backed by
   measurement: (1) N = 5 reproduction at fully converged cutoff
   (`n_c ≥ 6`, `dim ≥ 33 614`, crosses 64 GB dense); (2) detuning-
   sweep amortisation at `dim ≳ 5 000` on a 16 GB laptop, where
   the per-eigh cost grows from 53 s (one detuning) to 14 min
   (16-point sweep) and an iterative path that re-uses
   factorisation across detuning windows could cut this
   materially. Until a user hits one of these, this dispatch
   stays in reserve rather than scheduled — see
   `docs/benchmarks.md` "AAG gate status" for the threshold.
8. **Dispatch AAH — exact-diag envelope benchmark.**
   `tools/run_benchmark_spectrum_envelope.py` runs the dense and
   any iterative interior-window paths over a (N, n_c) grid; reports
   wall-clock + peak
   RSS; renders into `docs/benchmarks.md`. Cost: ~1 dispatch
   (benchmark tool + docs section; the `.json` + plots follow
   existing Dispatch-X/OO conventions).

   **Shipped.** Benchmark tool at
   `tools/run_benchmark_spectrum_envelope.py` (subprocess-isolated
   per grid point for clean peak-RSS measurement; `--include-large`
   flag adds a deliberate-swap probe at `dim ∈ {6 250, 6 750,
   8 192}`); report under
   `benchmarks/data/spectrum_envelope/{report.json,plot.png}`;
   documented in `docs/benchmarks.md` as "Exact-diagonalization
   envelope (Dispatch AAH)". The 23-point `(N, n_c)` grid spans
   `dim ∈ {12 … 8 192}` — the largest point reaches 1 GB matrix,
   3.6 GB peak RSS, and 7 min wall-clock on the reference 16 GB
   laptop. Headline: dense `eigh` follows $\mathcal{O}(d^{3})$
   wall-clock; peak RSS tracks the dense-matrix footprint with a
   ~4–5× workspace multiplier (less than the small-dim core fit
   suggested — scipy's `dsyevr` default is more memory-efficient
   than divide-and-conquer at large `d`). Re-fit with the
   large-dim probes via
   `tools/plot_spectrum_envelope_extrapolation.py`; the resulting
   16 GB envelope reaches `n_c = 8` for N = 4 and `n_c = 5` for
   N = 5. Iterative interior-window path is **not** scheduled
   (see AAG status).
9. **Dispatch AAI — tutorial.** `docs/tutorials/13_reproducing_clos_2016.md`
   end-to-end walk-through: load `theo_dim_N_1.dat`, build H,
   call `solve_spectrum`, compute `IPR_av`, reproduce one row of
   the table; then scale to N = 2 and N = 3 with
   `clos2016_axial_mode_reference`. Cost: ~1 dispatch.

   **Shipped.** Tutorial wired into `mkdocs.yml` and the
   tutorials index. Surfaces the three findings from the AAE
   wave (non-RWA vs carrier-RWA Hamiltonian, Raman vs single-
   photon wavelength, `IPR_av` vs textbook `effective_dimension`)
   and closes with the dense-eigh envelope table that motivates
   the AAG / AAH benchmark dispatches.

**Total:** 7–8 dispatches plus the `AAA` reconciliation gate; `AAG`
is explicitly benchmark-gated and may be skipped if the dense full-LD
operator makes it a net loss. Compressible to 5–6 by merging AAE+AAF
(if N=1/2/3 test ships together) and AAH+AAI (if benchmark and
tutorial land same dispatch). Expandable to 10+ if full-LD
sideband builders (RSB / BSB / MS) are added alongside the
carrier (§7 Q2).

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1. Porras helpers (`cchain`, `Operators`, `eta_calculator`) not standalone in legacy bundle | Certain (verified) | Low (re-graded) | The helpers are not standalone files but **are** present inline as sub-functions inside every `.m` driver (verified during AAE; see §4.1). Re-derive from physics anyway, but cross-check against the inline definitions rather than treating the helpers as missing. The wavelength provenance trap (200 nm vs `279.63/sqrt(2)` Raman effective; see §4.1) was the real instance of this risk and is now closed. |
| 2. Convention drift between our `axial_modes` and `cchain` | Medium | Medium | Make this a named pre-step (Dispatch `AAA`): encode explicit N = 2 / N = 3 analytic reference values before the full-LD builder lands. Mismatches surface immediately, not later. |
| 3. N ≥ 3 d_eff reproduction fails tolerance | Medium | Medium | Infer the converged `n_c` values from the `*ions_ipr_vs_nc.txt` tables before declaring the reproduction target. Only then document whatever tolerance the pipeline actually achieves rather than forcing 1 %. |
| 4. Dense diag at N = 4, $n_c \geq 6$ exceeds RAM on reference hardware | High | Medium | Expected; that's what §4.3 exists to measure. Do **not** assume an iterative method wins just because dense OOMs: the full-LD matrix is dense, so any interior-window path must earn its keep in benchmark data. |
| 5. Full-LD carrier builder doubles Hamiltonian-build cost | Medium | Low | Measured in §4.2 benchmark family 1; documented. Users opt in explicitly via the `_full_ld` entry. |
| 6. Scope creep into ETH theory | Medium | Medium | §2 Out-of-scope list is explicit. Spectrum analyses implement what Porras computes; we don't extend the theory. |
| 7. Interior-window iterative convergence issues at large N | Medium | Medium | Shift-invert / ARPACK can stall or become factorisation-bound on dense clusters of eigenvalues. Treat this as benchmark-gated, not assumed capability; document failure-modes in the report. |
| 8. Experimental data files (`exp_*_N_k.dat`) not used by the workplan | Low (by design) | Low | §2 Out-of-scope: we focus on numerical reproduction. Experimental-fit work is separate. |

---

## 7. Open questions — resolutions recorded 2026-04-23

1. **Full-LD coverage.** Carrier only for the first pass, or all
   four families (carrier, RSB, BSB, MS)? Porras's `ergodic.m`
   uses only the carrier — reproducing the paper needs only
   carrier. But half-finished coverage is a maintenance burden.
   — **Resolved: carrier-only for this integration.** Shipped as
   `carrier_hamiltonian_full_ld` in AAB; the Porras regression
   uses the non-RWA sibling `clos2016_spin_boson_hamiltonian`
   shipped in AAE (see §3 architecture table). RSB / BSB / MS
   full-LD builders are deferred — no active user demand, and
   the existing leading-order entries remain unchanged. Revisit
   only if a regression target explicitly needs them.
2. **Iterative interior-window backend.** If wave 2 ships an
   iterative solver at all, is SciPy/ARPACK shift-invert enough, or is
   PRIMME worth the extra dependency? Lowest-k ARPACK is explicitly
   out of scope; the only admissible target is the `meanE` window.
   — **Resolved: deferred (AAG gate status).** The AAH envelope
   benchmark shows dense `eigh` covers the publication-validated
   reproduction at N ∈ {1, 2, 3} at fully converged cutoffs and
   N = 4 up to `n_c = 7` on the 16 GB reference laptop. No
   iterative path shipped; re-activation trigger documented in
   [benchmarks.md § AAG gate status](benchmarks.md). If AAG ever
   ships, SciPy/ARPACK shift-invert is the planned first cut —
   PRIMME remains opt-in only.
3. **Benchmark hardware baseline.** Which PC is the "reference
   machine" for §4.3? (Consumer laptop 16 GB, workstation
   32 GB, or both?) This shapes the envelope tables.
   — **Resolved: 16 GB consumer laptop as primary; 32 / 64 / 128 GB
   tiers reported as extrapolation.** Measured grid lives at
   `benchmarks/data/spectrum_envelope/report.json`; the per-(N, RAM)
   envelope table in [benchmarks.md](benchmarks.md) projects from
   the measured wall-clock and peak-RSS scaling constants to
   larger RAM tiers honestly (distinguishing measurement from fit).
4. **JAX / Dynamiqs interaction.** `solve_spectrum` at Dispatch AAC
   targets scipy CPU. Later, could a JAX-backed `eigh` path
   land as `spectrum-jax-cuda`? Defer, but flag whether the
   API design should leave that door open.
   — **Resolved: deferred to Phase 3+; `SpectrumResult` schema
   leaves the door open.** Post-AAH measurement (see
   [benchmarks.md § scipy vs JAX on CPU for dense `eigh`](benchmarks.md))
   confirms no CPU speedup from `jax.numpy.linalg.eigh` and a
   ~130 MB higher baseline RSS. The remaining JAX value is GPU
   dispatch and autograd through the eigensolve — both Phase 3+.
   `SpectrumResult.backend_name` is the forward-compatibility hook
   (`"spectrum-scipy"` today; `"spectrum-jax-cuda"` / similar if
   ever shipped).

---

## 8. Recommendation (tentative)

Ship Dispatches `AAA` through `AAF` as the first wave — mode /
cutoff reconciliation, full-LD carrier, spectrum skeleton,
IPR/d_eff/ETH/phonon-number analyses, and N ∈ {1, 2, 3}
regression against Porras. This wave is end-to-end: by its
completion, the library demonstrably reproduces the
publication-validated numbers for three system sizes.

Ship Dispatches `AAG` through `AAI` as the second wave — benchmark-
gated interior-window solver, benchmark, tutorial. This wave opens
the "how big can I go?"
conversation honestly and gives users a concrete answer tied to
their hardware.

Treat N = 4 and N = 5 regression reproduction as a **stretch**
deliverable, contingent on the §4.3 benchmark showing the
envelope supports either dense diagonalization there or a benchmark-
validated interior-window solver around `meanE`. If the envelope
puts N = 4 squarely beyond dense RAM (likely), the
`theo_dim_N_{4,5}.dat` reproduction must still target the same
microcanonical window as legacy `eigs(H, Neigens, meanE)` rather
than falling back to the ground-state manifold.

The full-LD API-shape decision is taken **here**: use separate
`*_full_ld(...)` entries and leave existing leading-order builders
unchanged. The `SpectrumResult` schema decision is also taken
**here**, so `AAC` no longer waits on a schema debate.

---

## 9. Status at close-out (2026-04-23)

This document sits at
`docs/workplan-clos-2016-integration.md` as a **shipped-capability
record**. It is not linked into `mkdocs.yml`, matching the repository
convention for `phase-2-jax-backend-design.md` and
`phase-2-jax-time-dep-design.md`: design / workplan artefacts
stay in `docs/` for historical continuity but do not appear in
the user-facing navigation. User-facing content lives in
[docs/benchmarks.md](benchmarks.md) (the envelope and backend
comparison) and
[docs/tutorials/13_reproducing_clos_2016.md](tutorials/13_reproducing_clos_2016.md)
(the end-to-end walk-through).

**Ship status — what landed where.**

| Dispatch | Status   | Primary artefact |
|----------|----------|------------------|
| AAA      | Shipped  | `clos2016_axial_mode_reference` (corrected during AAF) |
| AAB      | Shipped  | `carrier_hamiltonian_full_ld` (carrier-RWA; see AAE note) |
| AAC      | Shipped  | `src/iontrap_dynamics/spectrum.py`, `SpectrumResult` |
| AAD      | Shipped  | `src/iontrap_dynamics/spectrum_observables.py` |
| AAE      | Shipped  | `tests/regression/reproduction/test_clos_2016_N1.py`; `clos2016.py` non-RWA assembler + `IPR_av` + wavelength pin |
| AAF      | Shipped  | `tests/regression/reproduction/test_clos_2016_N2_N3.py` |
| AAG      | Deferred | Benchmark-gated; re-activation trigger in [benchmarks.md § AAG gate status](benchmarks.md) |
| AAH      | Shipped  | `tools/run_benchmark_spectrum_envelope.py`, `benchmarks/data/spectrum_envelope/` |
| AAI      | Shipped  | `docs/tutorials/13_reproducing_clos_2016.md` |

**Post-close-out extensions.** After the main nine dispatches
closed, two measurement-only extensions landed: the 16 GB swap-
pushing probe grid (`--include-large` on the envelope benchmark)
and the scipy-vs-JAX CPU `eigh` comparison at
`tools/run_benchmark_spectrum_envelope_jax.py`. Both inform
the Q2 and Q4 resolutions in §7 above.

**If the integration is ever re-opened** — e.g. a user forces
AAG's re-activation trigger, or RSB / BSB / MS full-LD builders
become demand-driven — the new work gets its own kickoff note
and CHANGELOG entry at that time; this document is not updated
retroactively.
