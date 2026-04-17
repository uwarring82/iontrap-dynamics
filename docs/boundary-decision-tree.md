# Boundary Decision Tree

## Purpose

This document is the contributor-facing form of the scope rules in
`WORKPLAN_v0.3.md` §1. Before proposing a feature, run it through the tree
below. If the tree says "out of scope", the feature belongs in a
complementary tool (`pylion`, `trical`, ARTIQ) or in a consumer of this
library such as `single-25Mg-plus` — not here.

The tree is a Coastline rule: it states a binding constraint. It is not a
preference. Features that fail it are rejected at PR review regardless of
how polished the implementation is.

## The tree

```text
Does the feature modify the spin or motional quantum state?
├── YES → Is it a Hamiltonian / Lindbladian / initial state / observable?
│         ├── YES → In scope (core / physics layer)
│         └── NO  → Is it a measurement channel or sampled observation?
│                   ├── YES → In scope (observation layer, Phase 1+)
│                   └── NO  → Out of scope
└── NO  → Does it model apparatus imperfections
          (drifts, jitter, detector, calibration)?
          ├── YES → In scope (apparatus layer, Phase 1+)
          └── NO  → Out of scope
                    (trap geometry, control stack, EM fields, lab IT)
```

## Layer semantics

In-scope features fall into exactly one of three layers, matching the
three-layer architecture in `CONVENTIONS.md` and
`WORKPLAN_v0.3.md` §2 Principle 9:

| Layer | What it models | Phase |
|---|---|---|
| Physics (core) | Hamiltonians, Lindbladians, state preparations, ideal observables | Phase 0+ |
| Apparatus | Drifts, timing jitter, amplitude miscalibration, SPAM as parameter-level | Phase 1+ |
| Observation | Finite-shot sampling, detector efficiency, dark counts, thresholding | Phase 1+ |

Noise and readout never contaminate the Hamiltonian layer. A feature that
would require doing so fails the tree.

## Apparatus-model clarification

Apparatus models are in scope **only** insofar as they modify Hamiltonian
parameters, dissipative channels, or measurement outcomes at the level of
the simulated quantum experiment. They are not a vehicle for importing lab
infrastructure, control-software emulation, or non-quantum environmental
modelling.

Concretely: a drift model that adjusts `omega_rabi(t)` during a gate is in
scope. A model of the voltage ramp on a given electrode that happens to
produce such drift via a first-principles electrode simulation is out of
scope — it belongs in `trical` or in a downstream consumer.

## Worked examples

Each example shows one feature, the branch the tree takes, and the
verdict. Examples in the first group are obviously in scope; the second
group shows real ambiguity cases; the third group shows explicit
rejections.

### Clearly in scope

**Example 1 — Red-sideband Rabi builder.**
Modifies the spin–motion state? **Yes.**
Is it a Hamiltonian? **Yes.**
→ Core physics layer. Ships in `hamiltonians.py` (Phase 1).

**Example 2 — Finite-shot parity measurement.**
Modifies the state? No, it samples observations of it.
Is it a measurement channel? **Yes.**
→ Observation layer (`measurement/`, Phase 1+).

**Example 3 — Global laser-frequency drift.**
Modifies the state? Not directly — it modifies the effective detuning.
Apparatus imperfection? **Yes.**
→ Apparatus layer (`systematics/`, Phase 1+).

### Ambiguous cases

**Example 4 — Generic SPAM error model** *(decision D8 worked example).*
A contributor proposes `StateInitError` and `MeasurementError` objects
that inject configurable bit-flip probabilities into the state-preparation
and readout steps.

Walk the tree:

- Does it modify the spin or motional quantum state? **Not directly** — it
  modifies the effective preparation fidelity and measurement outcome,
  parameterised by rates and matrices.
- Does it model apparatus imperfections? **Yes** — SPAM is a classical
  parameterisation of detector and state-prep imperfection.

Verdict: **in scope**, apparatus/observation layer (Phase 1+). Lives in
`systematics/` for the preparation half and `measurement/` for the readout
half. Ships as configurable, backend-agnostic rates — no hard-coded
species-specific numbers.

**Example 5 — Species-specific ²⁵Mg⁺ detection calibration.**
A contributor proposes importing the measured detection-efficiency curve
from a specific laboratory run on a specific ²⁵Mg⁺ apparatus, with
electrode voltages, photon-collection geometry, and PMT dark-count rates
baked in.

Walk the tree:

- Does it modify the spin or motional quantum state? No.
- Does it model apparatus imperfections? It depends. At the level of "a
  detection channel with some efficiency and dark-count rate," yes.
- **But**: the concrete laboratory calibration also encodes trap geometry,
  control-stack timing, electrode positions, PMT placement, and a specific
  experimental run's conditions.

Verdict: **split the feature.** The generic envelope — "Bernoulli readout
with efficiency η and dark-count rate d" — is in scope and lives in
`measurement/`. The ²⁵Mg⁺-specific calibration table, the electrode
geometry, the PMT placement, and the run-specific numbers are out of scope
*here* and belong in the consumer repository `single-25Mg-plus`, which
supplies them as configuration at simulation time.

This is the D8 split articulated: `iontrap-dynamics` provides the generic
measurement channel with named, typed parameters. The digital twin fills
those parameters from its own calibration data.

**Example 6 — Motional decoherence from anomalous heating.**
Contributor proposes a Lindblad collapse operator representing anomalous
heating at a given heating rate `ṅ`.

- Modifies the state? Yes.
- Hamiltonian / Lindbladian? **Lindbladian.**

Verdict: **in scope**, core physics layer. The physical source of the
heating (surface patches, Johnson noise, electrode voltage noise) is not
modelled here — the feature accepts `ṅ` as a configured parameter.

### Clearly out of scope

**Example 7 — ARTIQ pulse-sequence compiler.**
Modifies the state? No.
Apparatus imperfection? No, it's hardware control.
→ Out of scope. Belongs in ARTIQ itself, OxfordControl, or a lab-specific
control stack.

**Example 8 — Ion-crystal classical-trajectory simulator.**
Modifies the quantum state? No — it computes classical ion positions and
velocities under Coulomb + trap potential.
Apparatus imperfection? No.
→ Out of scope. Belongs in `pylion`.

**Example 9 — Trap-electrode finite-element field solver.**
Modifies the state? No.
Apparatus imperfection affecting the quantum model? No — it models the
classical electrostatic geometry from first principles.
→ Out of scope. Belongs in `trical`.

**Example 10 — Full lab digital twin with electrode-field + pulse
compiler + control-system emulation.**
The tree rejects this at the apparatus-clarification step: apparatus
models are in scope only when they modify Hamiltonian parameters,
dissipative channels, or measurement outcomes — not when they import
first-principles models of the hardware stack.
→ Out of scope here. Build it in a downstream consumer that imports
`iontrap-dynamics`, `trical`, `pylion`, and the control stack separately.

## When the tree doesn't resolve cleanly

Some proposals mix in-scope and out-of-scope pieces (Example 5 is one).
The rule is: **split the feature**. Keep the generic, parameterised,
configuration-driven part in this library. Push the species-specific,
calibration-specific, geometry-specific part into the consumer.

If after splitting the split feels awkward — the generic envelope is too
generic to be useful on its own — that is a signal the feature really
belongs in a downstream repo rather than here.

Unresolved cases that survive this rule should be raised in a GitHub
Discussion, referencing this document, before a PR is opened.

## Cross-references

- Full scope statement and non-scope list: `WORKPLAN_v0.3.md` §1
- Three-layer architecture: `CONVENTIONS.md` and `WORKPLAN_v0.3.md` §2
  Principle 9
- Complementary tools and their roles: `WORKPLAN_v0.3.md` §1, "Relationship
  to existing tools"

## Endorsement Marker

Local candidate framework under active stewardship. No external
endorsement is implied. This document is a Coastline rule within the
project's split-licence architecture (CC BY-SA 4.0).
