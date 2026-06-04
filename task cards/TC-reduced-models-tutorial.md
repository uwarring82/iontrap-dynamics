# Task Card — Reduced light–matter models (JC / AJC / QRM) and a comparison tutorial against full trapped-ion dynamics

**Authored from:** the `ajc-provenance` side — the model-hierarchy note (`docs/hierarchy.md` v0.4), as an integration spec.
**Upstream target:** `uwarring82/iontrap-dynamics` @ v0.5.0 (`CONVENTIONS.md` frozen v0.3).
**ID:** TC-reduced-models-tutorial · **Status:** v0.2.1 reviewed draft. **Endorsed for WP-03 lift.** To be carried by WP-03 and mirrored as upstream issues once ratified. Authority derives from use (Lock–Key Rule).
**Licence:** this card is CC BY-SA 4.0 (spec/Coastline-adjacent). Deliverables carry their layer's licence: code = MIT (`src/`, `tests/`, `tools/`); the tutorial and the vendored note = Sail / CC BY-NC-SA 4.0 *or* CC BY-SA 4.0 as marked; any `CONVENTIONS.md` edit = Coastline / CC BY-SA 4.0.

---

## 1. Verdict (fit assessment)

`iontrap-dynamics` is a **strong fit** as the home for this tutorial. It already ships the full-ion side of the comparison — full-Lamb–Dicke carrier/sideband Hamiltonians and the exact (all-orders) sideband-Rabi analytic forms, with a Debye–Waller helper and a deep/intermediate/beyond regime classifier (Dispatch MCE) — plus exact-diagonalisation spectrum tools, the `solve(...)` surface on QuTiP and JAX, a finite-shot measurement layer, and a three-layer regression harness. The conceptual spine already exists too, as the reviewed-and-locked `hierarchy.md`.

**The gap is small and additive:** one new physics-layer module of *reduced* models (JC, AJC, QRM as abstract qubit–oscillator Hamiltonians, not tied to the sideband construction), a thin comparison harness, and the tutorial with reproducible figures and oracle-backed tests. No existing coastline is redefined; the work is additive capability plus one mandatory `CONVENTIONS.md` §25 gate for the reduced-model frame, signs, and term selection.

**One boundary must hold (and it mirrors the library's architecture):** the *reduced models* are physics-layer objects (what the apparatus approximates); the *sideband Hamiltonians* are apparatus-layer objects (how a real ion realises them). The tutorial's whole point is to compare the two — Axis A (model containment) against Axis B (physical realisation) in the language of `hierarchy.md`. Keeping that separation is what makes the tutorial a *falsifiable demonstration* of the note rather than an illustration of it.

**Non-apparatus:** every item is simulation/library/docs work — unblocked, parallelisable, no hardware dependency.

## 2. Ownership boundary

| Capability | Owner | Rationale |
|---|---|---|
| Reduced JC / AJC / QRM Hamiltonian builders (abstract qubit + 1 mode) | **library** | General light–matter physics; reusable beyond this tutorial |
| Comparison harness (reduced ↔ full-ion, deviation summary) | **library** | General "model vs realisation" tooling |
| Full-ion sideband dynamics + exact sideband-Rabi forms | **library (present)** | Already shipped (MCE; `hamiltonians`, `analytic`) — reuse, do not rebuild |
| The model **hierarchy / interpretation** and the locked wording (Axis A/B, the three regimes, the negative-$\omega_0$ trichotomy) | **`ajc-provenance` (vendored as a Sail companion)** | Authored interpretive work; provenance preserved |
| The tutorial narrative and figure choices | **library (Sail docs)** | Pedagogy, cites the note by section |

## 3. Already present — do not rebuild

Confirmed in the v0.5.0 tree and the `[Unreleased]` MCE dispatch:

- **Spin / Hilbert / states** — `operators.{sigma_z_ion, sigma_x_ion, sigma_plus_ion, sigma_minus_ion, spin_up, spin_down}`; `hilbert.HilbertSpace`; `system.IonSystem`; `states.{ground_state, coherent_mode, squeezed_vacuum_mode, cat_mode, compose_density}`.
- **Apparatus Hamiltonians** — `hamiltonians.{carrier_hamiltonian, carrier_hamiltonian_full_ld, detuned_carrier_hamiltonian, red_sideband_hamiltonian, blue_sideband_hamiltonian, detuned_red_sideband_hamiltonian, detuned_blue_sideband_hamiltonian}`; the red/blue sideband builders already expose `full_lamb_dicke=True` for the all-orders path (via internal `_full_ld_{carrier,raising,lowering}_single_mode` helpers).
- **Analytic oracles** — `analytic.{generalized_rabi_frequency, red_sideband_rabi_frequency, blue_sideband_rabi_frequency, red_sideband_rabi_frequency_full_ld, blue_sideband_rabi_frequency_full_ld, debye_waller_factor, lamb_dicke_confinement, lamb_dicke_regime (→ LambDickeRegime: deep/intermediate/beyond, thresholds 0.1 / 1.0), lamb_dicke_parameter, coherent_state_mean_n}`.
- **Solvers / spectra** — `solve(...)` (QuTiP default, `backend="jax"`), `solve_ensemble`, `TrajectoryResult`; `spectrum.solve_spectrum → SpectrumResult` (exact diagonalisation; `initial_state` is optional and only adds energy moments).
- **Convergence / conventions** — `conventions.{CONVENTION_VERSION, FOCK_CONVERGENCE_TOLERANCE}`; §13 Fock-truncation contract; §15 three-level warning/failure ladder.
- **Tutorial precedents** — 02 (red sideband from |1⟩), 06 (Fock-truncation diagnosis), 08 (full Lamb–Dicke), 13 (Clos 2016), 17 (motional decoherence + Lamb–Dicke). The new tutorial is **18**, sitting after 17.

## 4. Exemplary cases (the physics spine)

Each case is a runnable demonstration of one named claim in `hierarchy.md`, with a pre-committed oracle and a pass condition. This is what the tutorial walks through.

**Case A — JC ↔ AJC isolated equivalence ("only a label" is *true* here).** Maps to hierarchy §6 (LOCK-3 identity) and §8, regime 1; Q3.
- Setup: build reduced `H_JC(-ω₀,ω_f,g)` and `H_AJC(+ω₀,ω_f,g)` on the same one-ion / one-mode Hilbert space.
- Discriminator: the operator identity $H_{\mathrm{AJC}}(\omega_0)=\sigma_x H_{\mathrm{JC}}(-\omega_0)\sigma_x$, and coincidence of spectra.
- Oracle: eigenvalue sets from `solve_spectrum` (state-independent; call without `initial_state`) agree to machine precision; the identity holds as an exact embedded `Qobj` equality; an AJC observable trajectory equals the relabelled-JC trajectory.
- Pass: equality within numerical tolerance. **This is the "label true" anchor.**

**Case B — red vs blue sideband on the *ion* ("only a label" becomes a physical *knob*).** Maps to §8, regime 2; Q3.
- Setup: from $|\downarrow,0\rangle$ (ground-state notation per `CONVENTIONS.md` §3), evolve under `red_sideband_hamiltonian` and under `blue_sideband_hamiltonian` for the same ion and mode; the physical distinction is the red/blue sideband detuning sign, even though the exact-resonance builders themselves do not take `detuning_rad_s`.
- Discriminator: $|\downarrow,0\rangle$ is **dark** under the red sideband (JC; $a|0\rangle=0$, no flop) and **bright** under the blue sideband (AJC; $|\downarrow,0\rangle\leftrightarrow|\uparrow,1\rangle$ flopping).
- Oracle: blue-sideband flop frequency = `blue_sideband_rabi_frequency`; red-sideband population transfer from $|\downarrow,0\rangle$ is zero.
- Pass: the two trajectories are physically distinct as predicted. **This is the "knob" anchor — the crispest red≠blue statement.**

**Case C — JC vs QRM, rotating-wave breakdown ("only a label" is *false*).** Maps to §4 and §8, regime 3; Q3.
- Setup: sweep $g/\omega_0$ from weak to ultrastrong/deep-strong; compare reduced `H_JC` to reduced `H_QRM`.
- Discriminator: QRM ground state acquires virtual excitations $\langle a^\dagger a\rangle>0$ (and is not $|\downarrow,0\rangle$), parity replaces the $U(1)$ excitation number, dynamics diverge from JC.
- Oracle: weak-coupling agreement JC ≈ QRM as $g/\omega_0\to0$; QRM ground-state $\langle a^\dagger a\rangle$ from `solve_spectrum` and spectral/trajectory deviation against a committed reference curve at named control points.
- Pass: deviation (defined in §6) is below the pre-committed weak-coupling band and above the pre-committed USC/DSC separation band. Do **not** require pointwise monotonicity; QRM observables can have structured finite-truncation and resonance features. **This is the "label false" anchor.**

**Case D — full ion vs Lamb–Dicke-truncated JC (the Q4 payoff: Debye–Waller × Laguerre).** Maps to §5 nonlinear branch; Q4.
- Setup: drive the red sideband; compare full-LD sideband dynamics (`red_sideband_hamiltonian(..., full_lamb_dicke=True)` / `red_sideband_rabi_frequency_full_ld`) against the LD-truncated JC prediction (`red_sideband_hamiltonian(..., full_lamb_dicke=False)` / `red_sideband_rabi_frequency`), as a function of Fock $n$ and $\eta$ so that `lamb_dicke_confinement` $=\eta^2(2n+1)$ crosses deep → intermediate → beyond (`lamb_dicke_regime`, using `mean_phonon_number=n` for Fock-level examples).
- Discriminator: the per-Fock Rabi comb and the resulting trajectory deviate; the deviation tracks the classifier argument.
- Oracle: `red_sideband_rabi_frequency_full_ld` vs leading-order `red_sideband_rabi_frequency` (both already oracle-backed in the repo); deviation → 0 as $\eta^2(2n+1)\to0$.
- Pass: deviation is below the pre-committed deep-regime band and exceeds the pre-committed intermediate/beyond bands at selected control points, consistent with the classifier. Do **not** require pointwise monotonicity; the all-orders Laguerre structure bends, nulls, and revives by design. **This is the central deviation case.**

**Case E — bichromatic *simulated* QRM vs abstract QRM (Axis A ↔ Axis B bridge).** Maps to §5 laboratory note; Pedernales et al. 2015. **Deferred / future WP item (P2+)**.
- Prerequisite: a first-class bichromatic two-tone sideband builder, or an explicitly sanctioned composition helper, that sums red and blue near-sideband terms with pinned relative phases, detuning signs, and backend behaviour. Do not let this prerequisite block Cases A–D or Tutorial 18.
- Setup once prerequisite exists: drive two detuned sidebands together; show the effective dynamics match reduced `H_QRM`. Under the current `detuned_{red,blue}_sideband_hamiltonian` phase convention the likely mapping contains half-factors, e.g. $\omega_0^{\mathrm{eff}}\sim(\Delta_r+\Delta_b)/2$ and $\omega_f^{\mathrm{eff}}\sim(\Delta_b-\Delta_r)/2$ up to sign/phase choices; derive and pin the exact convention in the Case-E WP before coding.
- Oracle: agreement with reduced QRM over a chosen $(\Delta_r,\Delta_b)$ grid; reduces to JC/AJC when one tone is removed.
- Pass: cross-model agreement within band after the two-tone builder and effective-parameter map are independently convention-tested.

### Analytical expressions to carry into Tutorial 18

The tutorial should not re-derive the vendored hierarchy note, but it should display the minimal equations that make the numerical comparisons readable:

- **Reduced-model Hamiltonians:** $H_{\mathrm{QRM}}=\frac{\omega_0}{2}\sigma_z+\omega_f a^\dagger a+g\sigma_x(a+a^\dagger)$; $H_{\mathrm{JC}}=\frac{\omega_0}{2}\sigma_z+\omega_f a^\dagger a+g(a\sigma_+ + a^\dagger\sigma_-)$; $H_{\mathrm{AJC}}=\frac{\omega_0}{2}\sigma_z+\omega_f a^\dagger a+g(a^\dagger\sigma_+ + a\sigma_-)$. Mention QRM parity $\Pi=\sigma_z(-1)^{a^\dagger a}$ and the JC/AJC conserved $U(1)$ numbers $a^\dagger a+\sigma_+\sigma_-$ and $a^\dagger a-\sigma_+\sigma_-$.
- **LOCK-3 identity:** $H_{\mathrm{AJC}}(\omega_0)=\sigma_xH_{\mathrm{JC}}(-\omega_0)\sigma_x$, using $\sigma_x\sigma_z\sigma_x=-\sigma_z$ and $\sigma_x\sigma_\pm\sigma_x=\sigma_\mp$. State explicitly that the negative $\omega_0$ is an effective/model sign, not a negative physical ion splitting.
- **Ion-to-sideband source formulas:** include the schematic full-ion starting point $H=(\omega_{\mathrm{at}}/2)\sigma_z+\nu a^\dagger a+\Omega\sigma_x\cos(\eta(a+a^\dagger)-\omega_Lt+\phi)$ and the first-order Lamb-Dicke sideband terms after optical RWA, proportional to $\sigma_+a\,e^{-i(\delta+\nu)t}$ and $\sigma_+a^\dagger e^{-i(\delta-\nu)t}$ plus h.c. These formulas explain why red selects JC and blue selects AJC.
- **Full-LD sideband matrix elements:** for the blue branch, the all-orders sideband Rabi frequency/matrix-element scale is $\Omega_{n,n+1}^{\mathrm{full}}=\Omega e^{-\eta^2/2}\eta L_n^{(1)}(\eta^2)/\sqrt{n+1}$ before taking the rate magnitude; the library helper returns the magnitude. The red counterpart uses $L_{n-1}^{(1)}(\eta^2)/\sqrt{n}$ and is zero at $n=0$. In the Lamb-Dicke limit these reduce to $\Omega|\eta|\sqrt{n+1}=2g\sqrt{n+1}$ and $\Omega|\eta|\sqrt{n}=2g\sqrt n$ with $g=\eta\Omega/2$ up to sign/phase convention, so the reduced-model matrix element is the half-Rabi-rate scale $g\sqrt{n+1}$ or $g\sqrt n$.
- **Regime parameter:** keep $\eta^2(2n+1)$ visible for Fock examples and $\eta^2(2\bar n+1)$ for thermal/regime language. This is the bridge between Case D, `lamb_dicke_confinement`, and the deep/intermediate/beyond classifier.
- **Deferred bichromatic note:** if Case E is mentioned, include only the schematic retained interaction $H_I\simeq g(\sigma_+a e^{-i\Delta_rt}+\sigma_+a^\dagger e^{-i\Delta_bt}+\mathrm{h.c.})$ and the definitions $\delta_r=-\nu+\Delta_r$, $\delta_b=+\nu+\Delta_b$. Do not state a committed $\omega_0^{\mathrm{eff}}$ / $\omega_f^{\mathrm{eff}}$ map in Tutorial 18 until the separate two-tone convention derives it under the shipped sign and phase conventions.

## 5. Required feature updates

Each follows the library rule — *conventions before code* — so a conventions check is part of every set.

### RM0 — Conventions gate *(do first)*
- **Serves:** every reduced-model symbol.
- **Add:** an additive `CONVENTIONS.md` §25 "Reduced light–matter models" under a freeze amendment **before** code (Coastline gate; `CONVENTION_VERSION` bump). This is mandatory because current §5 says apparatus builders return interaction-picture Hamiltonians with the free atomic term removed, whereas reduced JC/AJC/QRM builders intentionally carry Schrödinger-picture bare terms.
- **Pin in §25:** reduced Hamiltonians are stored as $H/\hbar$ in rad·s⁻¹; the public builder inputs follow the library's existing angular-frequency convention (not a new dimensionless unit system); tutorial plots may normalise axes by $\omega_0$ or $\omega_f$. For $\omega_0>0$, $(\omega_0/2)\sigma_z$ makes $|\uparrow\rangle$ higher in energy than $|\downarrow\rangle$ under `CONVENTIONS.md` §3; negative $\omega_0$ is allowed as an effective/model parameter and is not a physical ion splitting. JC keeps $(a\sigma_+ + a^\dagger\sigma_-)$, AJC keeps $(a^\dagger\sigma_+ + a\sigma_-)$, QRM keeps $\sigma_x(a+a^\dagger)$. Reconcile notation with `hierarchy.md` §2 ($\omega_{\mathrm{at}}$ / $\omega_0$ / $\omega_0^{\mathrm{eff}}$).
- **Enforcement:** add a conventions test for the LOCK-3 identity $H_{\mathrm{AJC}}(\omega_0)=\sigma_x H_{\mathrm{JC}}(-\omega_0)\sigma_x$ using the library's embedded `sigma_x_ion`, `sigma_z_ion`, `sigma_plus_ion`, and `sigma_minus_ion`.
- **Priority:** P0. **Owner:** library.

### RM1 — Reduced-model Hamiltonian builders *(core new physics surface)*
- **Serves:** Cases A, C, E.
- **Add:** `src/iontrap_dynamics/reduced_models.py` with `jaynes_cummings_hamiltonian`, `anti_jaynes_cummings_hamiltonian`, `quantum_rabi_hamiltonian`, each on a single qubit + single bosonic mode.
- **Interface (sketch):** `quantum_rabi_hamiltonian(hilbert, mode_label, *, ion_index, omega_0, omega_f, g) -> qutip.Qobj` and analogous JC/AJC builders. Builders return static QuTiP `Qobj` Hamiltonians like the existing exact-resonance apparatus builders; there is **no** builder-level `backend=` kwarg. Backend parity is tested by passing the returned `Qobj` to `solve(..., backend="qutip"|"jax")` and `solve_spectrum(..., backend_name=...)` where supported. The API requires an explicit `mode_label` and `ion_index` so it remains unambiguous on larger `HilbertSpace` objects; tests cover the one-ion / one-mode tutorial case and explicit mode selection on multi-mode objects.
- **Operators:** build from `sigma_z_ion`, `sigma_x_ion`, `sigma_plus_ion`, `sigma_minus_ion`, `HilbertSpace.annihilation_for_mode`, `creation_for_mode`, and `number_for_mode`. Bare terms are $(\omega_0/2)\sigma_z + \omega_f a^\dagger a$; JC keeps $g(a\sigma_+ + a^\dagger\sigma_-)$, AJC keeps $g(a^\dagger\sigma_+ + a\sigma_-)$, QRM keeps $g\sigma_x(a+a^\dagger)$.
- **Acceptance:** JC/AJC dressed-state Rabi frequency $\propto |g|\sqrt{n+1}$ matches the closed form under the §25 convention; the Case-A identity holds exactly; QRM and JC agree in the weak-coupling/RWA reference band rather than by an artificial "drop terms" API; cross-backend trajectory agreement < 1e-3 for selected observables where the JAX solve path exists; unit + regression entries.
- **Conventions:** per RM0. **Priority:** P0. **Owner:** library.

### RM2 — Comparison harness + deviation summary
- **Serves:** Cases B, C, D, E.
- **Add:** thin helpers to run a reduced model and the matched full-ion build through `solve(...)` on a shared `HilbertSpace`/time grid and return a deviation summary. Add `model_deviation(traj_a, traj_b, *, observables=...)` only if existing observables cannot express it.
- **State fidelity scope:** state-fidelity deviation requires materialised states (`StorageMode.EAGER`, or a valid `states_loader` on a backend that supports `StorageMode.LAZY`). Pin the convention as `1 - qutip.fidelity(...)` using QuTiP's public fidelity value; a future squared-overlap convention would need its own convention note. If no materialised states are present, the helper computes observable/population RMS only and says so.
- **Backend scope:** RM2's state fidelity may be QuTiP-only initially; cross-backend parity is a separate RM6 gate based on shared observables and, where available, materialised state comparison after converting to `Qobj`.
- **Acceptance:** deviation → 0 for matched models in their common regime (weak coupling for C; deep LD for D); pre-committed control points separate the common and breakdown regimes; Fock-truncation convergence checked (§13, `FOCK_CONVERGENCE_TOLERANCE`).
- **Conventions:** none new. **Priority:** P1. **Owner:** library.

### RM3 — Tutorial 18
- **Serves:** the deliverable.
- **Add:** `docs/tutorials/18_reduced_models_vs_full_dynamics.md` walking Cases A→D, each anchored to the cited `hierarchy.md` section, with runnable snippets, the analytical-expression block above, and the deviation read-outs. Case E is named only as deferred future work unless its separate two-tone prerequisite has already landed. Register in `mkdocs.yml` nav after 17.
- **Acceptance:** snippets execute against the public API; pa11y WCAG 2 Level A gate passes; every figure generated by RM4 is cited in the final text; no physics claim beyond the locked note.
- **Conventions:** none new. **Priority:** P1. **Owner:** library (Sail / CC BY-NC-SA 4.0).

### RM4 — Reproducible figures
- **Serves:** RM3.
- **Add:** `tools/plot_reduced_models_comparison.py` (compute-only, like the existing `tools/plot_*`) generating the figures: the JC/AJC spectrum coincidence (A), the $|\downarrow,0\rangle$ dark/bright traces (B), the JC-vs-QRM deviation vs $g/\omega_0$ with the USC ground-state photon number (C), and the full-ion-vs-LD-JC deviation vs $\eta^2(2n+1)$ with the regime bands (D). Build under the corporate-design tokens; do not redefine token values.
- **Acceptance:** figures regenerate deterministically from committed parameters; the script writes arrays/report metadata as existing benchmark scripts do; the final tutorial cites the generated figures.
- **Conventions:** none new. **Priority:** P2. **Owner:** library.

### RM5 — Vendor the hierarchy note as the conceptual companion
- **Serves:** RM3 (the tutorial "combines the note with numerical results").
- **Add:** vendor `ajc-provenance/docs/hierarchy.md` (v0.4) into `docs/models-hierarchy.md` with a provenance header pointing back to `ajc-provenance` (commit hash + DOI once minted); preserve its CC BY-SA 4.0 licence; register in `mkdocs.yml`. The tutorial links it by section.
- **Acceptance:** provenance recorded; links resolve; rendering note retained.
- **Conventions:** none new. **Priority:** P1. **Owner:** library (vendored Sail).

### RM6 — Tests across the three layers
- **Serves:** all cases.
- **Add:** `tests/conventions/test_reduced_models_conventions.py` (§25 sign/frame/LOCK-3 identity); `tests/unit/test_reduced_models.py` (builders, Hermiticity, dimensions, explicit `mode_label` / `ion_index`, API rejection cases); `tests/regression/analytic/` oracles (JC/AJC Rabi, Case-A identity, QRM ground-state $\langle a^\dagger a\rangle$, Case-D full-LD vs leading-order, and the $2g\sqrt{n\pm1}$ reduced-limit relation to sideband Rabi frequencies); `tests/benchmarks/` deviation curves for C and D with stored references.
- **Backend parity:** add a dedicated QuTiP-vs-JAX trajectory parity test for reduced models by solving the same `Qobj` Hamiltonian through `solve(..., backend="qutip")` and `solve(..., backend="jax")`, asserting observable agreement < 1e-3 and state fidelity when materialised-state support is enabled.
- **Acceptance:** all green under ruff / ruff-format / mypy strict / pytest; SPDX headers on new modules; `[Unreleased]` CHANGELOG bullets, one per dispatch.
- **Conventions:** none new. **Priority:** P0–P1. **Owner:** library.

## 6. Acceptance gates / decision rules (Guardian)

- **Oracle-first.** No case "passes" on a plot; each pass condition above is a runnable oracle check.
- **Deviation is pre-defined.** Use a pinned state-fidelity convention (per RM2) and population/observable RMS between matched trajectories, reported against the control parameter ($g/\omega_0$ for C; $\eta^2(2n+1)$ for D). Add spectral deviation where it is sharper than a chosen trajectory (Cases A and C). Pre-commit the "agree" band in the common regime and named separation bands/control points in the breakdown regime; do **not** require pointwise monotonicity where QRM resonances or Laguerre structure make that physically over-specific.
- **Convergence gate.** Every trajectory clears the §13 Fock-truncation contract; report the truncation used.
- **Cross-backend.** Reduced-model solves agree QuTiP vs JAX under 1e-3 where a JAX path exists; builders themselves remain static QuTiP `Qobj` producers.
- **No claim beyond the note.** The tutorial introduces no physics statement absent from the locked `hierarchy.md`; it cites rungs.
- **Conventions before code (RM0).** `CONVENTIONS.md` §25 and its conventions test land *before* the reduced-model builders.
- **CI green + SPDX + CHANGELOG** on every landed item (the standard Coastline gates).

## 7. Sequencing and open questions

**Order:** RM0 → RM1 → (RM6 conventions/unit) → RM5 → RM2 → Cases A,B → draft RM3 → Cases C,D → RM4 → final RM3 → (RM6 regression/benchmark) → CHANGELOG/release. Case E is out of scope for this WP unless a separately scoped bichromatic two-tone builder and effective-parameter convention land first.

**Open questions for the breakout:**
- Which concrete dimensionless display normalisation the tutorial uses (`ω_f=1`, `ω_0=1`, or plotted ratios only). The builder interface itself stays in the library's angular-frequency units per RM0.
- $\eta$ grid and Fock cutoffs for Case D (which $(\eta,n)$ points straddle the 0.1 / 1.0 classifier thresholds while staying converged).
- Case-E effective-frequency and phase map, if the deferred two-tone WP is opened; do not encode the sum/difference formula without deriving it under the shipped detuned-sideband signs.
- Cheng et al. full author list to confirm before it is cited in the vendored note.

## 8. Execution

This card is the **spec**. Execution is a follow-on **Work Package (WP-03)** using `WP/TEMPLATE.md`: lift the §6 gates as quoted invariants, mint dispatch codes at ratification (registered in `WP/LOGBOOK.md`), and target an additive minor release (`backend=` defaults unchanged; every new symbol opt-in). Suggested release theme: *reduced light–matter models + model-vs-realisation tutorial*.

---

## Endorsement Marker

**Local candidate framework under active stewardship. No parity implied with externally validated laws.** This Task Card is a spec authored from `ajc-provenance` for `uwarring82/iontrap-dynamics`, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md` v0.3. Lock–Key rule applies: this card is a key built on the stable locks those documents specify. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian — pending: confirm no Coastline gate is relaxed, `CONVENTIONS.md` §25 lands before code, the deviation bands/control points are pre-committed, and no claim exceeds the locked note. Architect — pending: confirm RM1 keeps static `Qobj` builders with explicit `mode_label` / `ion_index`, and that RM2 does not conflate state fidelity with observable-only deviation. Scout — horizon: cross-repo provenance of the vendored note and the deferred Case-E two-tone convention. Integrator — sequenced per §7; release is additive.

**Convention version:** references `CONVENTIONS.md` v0.3 (frozen). This card specifies the need for an additive §25 and `CONVENTION_VERSION` bump, but the convention itself lands only through RM0.
**Source note:** `ajc-provenance/docs/hierarchy.md` v0.4 (commit to record; DOI on first release).

## Version history

| Version | Date | Change |
|---|---|---|
| 0.2.1 | 2026-06-04 | Added the analytical-expression block required for Tutorial 18: reduced Hamiltonians and symmetries, LOCK-3 identity, ion-to-sideband source formulas, full-LD Laguerre/Debye-Waller rates and Lamb-Dicke limits, regime parameter, and a scoped deferred bichromatic formula. |
| 0.2 | 2026-06-04 | Revised after repo-grounded review. Made RM0 §25 mandatory; removed builder-level `backend=` from reduced models; added explicit `mode_label` / `ion_index`; switched Case B to `|↓⟩` / `|↑⟩`; clarified `solve_spectrum` usage; replaced monotonicity gates with pre-committed bands/control points; scoped RM2 fidelity/materialised-state behaviour; moved RM5 before tutorial finalisation; deferred Case E behind a separate bichromatic two-tone convention. |
| 0.1 | 2026-06-03 | Initial draft. Fit verdict, ownership boundary, do-not-rebuild inventory, five exemplary cases (A–E), feature set RM0–RM6, acceptance gates, WP-03 execution hook. Not endorsed. |
