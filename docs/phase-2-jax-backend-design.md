# Phase 2 JAX backend — design note (draft for deliberation)

> **Status: draft for deliberation, not Coastline.** This document is a
> deliberately pre-decision exploration of the remaining Phase 2
> deliverable. It exposes design tensions and open questions so they can
> be resolved before code is written. Nothing in this document binds a
> future implementation; the eventual JAX-backend dispatch will come
> with its own Coastline updates (CHANGELOG entry, extension to
> `docs/phase-1-architecture.md`, and — if behaviour is user-visible —
> a `CONVENTIONS.md` version bump per §5.1's freeze protocol).

**Relates to:** `WORKPLAN_v0.3.md` §5 Phase 2 (the open JAX-backend
deliverable), §1 Scope ("Dynamiqs … future backend target"), §3
(richer backend contract), D5 (result family vs backend variety, now
recorded in `docs/phase-1-architecture.md`).

**Classification reminder.** Once a design is chosen the resulting
interface commitments become Coastline (backend contract, conversion
rules, tolerance thresholds). Implementation tactics (which library,
which JIT strategy, which benchmark dashboard) stay Sail.

---

## 1. The prior question: why a JAX backend at all?

Dispatch X, Y, and OO together measured the library's performance
against QuTiP 5.2. The findings were uncomfortable for anyone
expecting a dramatic Phase 2 win:

- `sesolve` / `mesolve` parity at library-typical Hilbert sizes
  (dim ≤ 48, mean wall-clock ratio 1.00×).
- `joblib` parallel dispatch wins only once single-solve cost exceeds
  ~15 ms — which happens at larger Fock truncations or long time
  grids, not at library-typical workloads.
- CSR-vs-dense operator dtype ties within 5 % at dim ≤ 60; CSR pulls
  ahead to ~1.4× only at single-ion Fock ≥ 60 (dim 120).

The pattern is consistent: **at the scales this library is actually
used, QuTiP 5 is already near the floor of what a Python-dispatched
ODE integrator can deliver**. The SciPy stepper is the bottleneck,
not matrix–vector products.

This raises the prior question before any JAX-specific design work:
**is the JAX backend a performance deliverable, a capability
deliverable, or a positioning deliverable?** The answer shapes every
downstream decision. Three plausible answers:

1. **Performance.** Win at large Hilbert spaces (dim ≥ 300, long
   trajectories, ensemble sweeps). Needs hard evidence that JAX
   beats QuTiP at the scales the library *will* reach as users hit
   bigger systems — not the scales it reaches today.
2. **Capability.** Autograd for gradient-based control pulse design,
   Bayesian inference on systematics parameters, GPU access for
   Monte-Carlo unravellings. These are things QuTiP cannot do at all;
   JAX opens them categorically.
3. **Positioning.** `WORKPLAN_v0.3.md` §1 names Dynamiqs as a future
   backend target. Shipping a JAX backend keeps the architectural
   commitment honest and unblocks Dynamiqs adoption later. Low
   user-visible benefit today, high credibility benefit.

These are not mutually exclusive, but they lead to different
architectures. Deliberation axis **A** below is: *what problem is this
backend actually solving?*

---

## 2. Scope boundaries

### In scope (any design)

- Forward-compute on pure-state and Lindblad trajectories for the
  four canonical Hamiltonian families (carrier, RSB, BSB, MS).
- Cross-backend numerical equivalence within a declared tolerance
  against the QuTiP reference on ≥ 1 worked example. (Phase 2 exit
  criterion named in workplan §5 Phase 2.)
- `ResultMetadata.backend_name` set to a distinct string (D5
  rule): proposed `"jax-diffrax"`, `"jax-dynamiqs"`, or
  `"jax-manual"` depending on design choice.
- Reuse of the existing `TrajectoryResult` schema per D5. Any
  JAX-native representation of states lives inside the backend
  module and is converted to the canonical tuple-of-`Qobj` (or
  equivalent) representation at the `sequences.solve` boundary.
- Opt-in installation via an extras group: `pip install
  iontrap-dynamics[jax]`. Not a base dependency.

### Out of scope (any design)

- Replacing QuTiP as the reference backend. The convention-version
  binding and analytic-regression tier are anchored to QuTiP; JAX
  is a sibling, not a successor.
- Implicit backend selection based on heuristics. Users opt in via
  an explicit `backend=` kwarg or equivalent. (Design Principle 5:
  one way to do it at the public API level.)
- A general-purpose autograd exposure in the first dispatch.
  Autograd on solver parameters is a real Option-B capability but
  deserves its own deliberation; this note frames the first
  dispatch as forward-compute only.
- Training-wheels for JAX pitfalls (NaN propagation, JIT cache
  invalidation on shape change, platform-specific float32 defaults).
  These need a tutorial, not a library feature.

### Deliberately ambiguous (to resolve below)

- GPU execution as a first-class target vs. "the user can opt in
  if JAX is installed with CUDA".
- Whether JAX-native Hamiltonian builders are a new layer or whether
  QuTiP-built operators get converted once at the solver boundary.

---

## 3. Architecture axes

Four axes, mostly independent. Choices along each affect the
implementation cost, the test surface, and the user-visible API.

### Axis A — Problem framing (from §1)

| Framing       | Consequence for design                                                        |
|---------------|--------------------------------------------------------------------------------|
| Performance   | Emphasis on JIT-compilation + `jax.lax.scan`; benchmark at dim ≥ 300; GPU optional but valued. |
| Capability    | Emphasis on autograd-ready state shape; benchmark secondary; GPU optional.     |
| Positioning   | Emphasis on integration cleanliness + Dynamiqs-compatibility; benchmark minimal. |

**Open question A1.** Which framing does the first JAX dispatch
prioritise? Default recommendation: **positioning + minimum viable
performance**. Ship a clean backend-contract realisation that other
Phase-3 work (autograd, GPU, Dynamiqs adoption) can build on,
without over-promising a performance win at current-library scale.

### Axis B — Integrator choice

| Option                                        | Maturity | Autograd | GPU | Phase-2 cost |
|-----------------------------------------------|----------|----------|-----|--------------|
| `diffrax` (Patrick Kidger)                    | Mature   | Native   | Yes | Low–medium   |
| `Dynamiqs` (quantum-specific JAX library)     | Growing  | Native   | Yes | Medium       |
| Hand-rolled `jax.scipy.linalg.expm + lax.scan` | n/a      | Native   | Yes | Medium–high  |

**`diffrax`** is the pragmatic default: it's a well-maintained JAX
ODE library with a public API that closely mirrors SciPy's; the
existing `sesolve` / `mesolve` call structure ports with minimal
conceptual translation. Tradeoffs: an extra dependency in the
`[jax]` extras, and — since diffrax is a general ODE library — no
quantum-specific convenience (trace preservation, Hermiticity
tracking) comes for free.

**`Dynamiqs`** is explicitly named as a future target in
`WORKPLAN_v0.3.md` §1. It provides quantum-specific primitives
(master equations, MC unravellings, expectation values) natively
on JAX arrays. Adopting Dynamiqs means the JAX backend is a thin
adapter; the tradeoff is a larger dependency and the need to track
Dynamiqs's own API churn.

**Hand-rolled** gives maximum control and no third-party API
risk, but requires writing the integrator, step-size controller,
and error reporting from scratch. This is probably too much for a
first dispatch.

**Open question B1.** Diffrax, Dynamiqs, or hand-rolled? The answer
depends on how much trust we want to place in a third-party JAX
library's long-term stability vs. the upfront cost of writing our
own stepper. Default recommendation: **diffrax**, with a design
note recorded on *why* diffrax over Dynamiqs so a future re-
evaluation is easy.

**Open question B2.** If diffrax, pinned to a specific version
range? JAX + diffrax version churn is real. Need a policy akin to
the QuTiP-version pin in `pyproject.toml`.

### Axis C — Operator representation and the conversion boundary

The library's Hamiltonian builders return QuTiP `Qobj` instances.
JAX solvers need `jax.Array`. Where does the conversion happen?

| Option                                      | Pros                                                                 | Cons                                                                |
|---------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------|
| Convert at `solve` boundary (one-shot)      | No builder duplication; existing test surface unchanged.             | Conversion cost paid on every call; conversion is `Qobj.full()` → `jnp.asarray`, i.e. dense. |
| JAX-native builders (parallel layer)        | Native dtype end-to-end; supports autograd on Hamiltonian parameters.| Duplicates the builder family; test surface doubles; maintenance cost ≈ 2× on builder code. |
| Hybrid: lazy JAX views on Qobj outputs      | Deferred conversion; avoids duplication.                             | Complex; leaks QuTiP internals into the JAX path.                   |

**Design Principle 5 tension.** A parallel JAX-native builder layer
is the cleanest technical path for autograd (Axis D) but violates
the spirit of "one way to do it at the public API level" — users
would have two sets of builders to choose from. If we go parallel-
builders, the public API has to remain single-entry; the choice of
representation is an internal backend concern.

**Open question C1.** One-shot conversion at `solve`, or parallel
JAX-native builders? Default recommendation: **one-shot conversion**
for the first dispatch. It decouples the JAX deliverable from a
builder-layer refactor. Parallel builders become a later dispatch
tied to Axis D (autograd) if and when that capability is requested.

**Open question C2.** Dense or sparse JAX representation? JAX has
experimental sparse support (`jax.experimental.sparse`) but it's
less mature than dense. Dispatch OO showed QuTiP's CSR wins over
dense only at dim ≥ 120; for JAX the right default is probably
**dense** until JAX sparse support hardens.

### Axis D — Autograd as a capability deliverable

Autograd is the genuine Option-B capability win: it unlocks
gradient-based pulse optimisation, Fisher-information calculations
on systematics parameters, and variational circuit training. QuTiP
cannot do any of these.

**Not in scope for the first dispatch** (per §2 Out of scope). But
the design choice on Axis C matters here: a one-shot Qobj→jax
conversion at the solve boundary **closes the door on autograd
through Hamiltonian parameters**, because the parameters have
already been consumed by QuTiP before JAX sees the array. A
parallel JAX-native builder layer keeps that door open at the cost
of builder duplication.

**Open question D1.** Is the first JAX dispatch's Axis-C choice a
permanent commitment, or is it staged? Proposal: **stage**. Ship
Axis C as one-shot conversion; if and when autograd is prioritised,
a second dispatch adds JAX-native builders alongside. This means
the first dispatch is narrow and shippable; the second has known
scope ahead of time.

---

## 4. Integration with the existing surface

### 4.1 `sequences.solve` API extension

Two candidates:

```python
# (i) new kwarg, discriminator value
solve(..., backend="jax")            # or "qutip" (default)

# (ii) new function
jax_solve(...)                       # parallel entry point
```

**Recommendation: option (i).** `sequences.solve` is the single
public solver entry point (see `docs/phase-1-architecture.md`);
adding a `backend` discriminator preserves that single entry. Option
(ii) would fragment the public surface for no user benefit.

When `backend="jax"` is selected, `solver=` is reinterpreted: the
current sesolve / mesolve dispatcher is QuTiP-specific. On the JAX
path there is no sesolve vs mesolve at the QuTiP level; the
distinction becomes Schrödinger-vs-Lindblad inside the JAX module.

### 4.2 Backend-name tagging

Per D5, `ResultMetadata.backend_name` uniquely identifies the
backend. Proposed values:

| Integrator choice | `backend_name`     |
|-------------------|--------------------|
| diffrax           | `"jax-diffrax"`    |
| Dynamiqs          | `"jax-dynamiqs"`   |
| Hand-rolled       | `"jax-manual"`     |

`backend_version` records the version of the underlying library
(e.g. diffrax version + JAX version; pattern TBD).

### 4.3 Convention-version binding

The QuTiP backend binds a `TrajectoryResult` to `CONVENTION_VERSION`
at solve time. The JAX backend must do the same — a JAX result is
only comparable to a QuTiP result if they share a convention
version. This is a hard invariant to hit: the existing `sequences.py`
machinery for recording the convention version should apply
unchanged.

### 4.4 Cache I/O

The cache layer (`cache.save_trajectory` / `load_trajectory`) writes
`.npz` + JSON manifest. The schema is backend-agnostic: `times`,
`expectations`, optional `states` (serialised to NumPy). A JAX
result should round-trip through the cache identically — the states
are materialised to NumPy at write time regardless of backend.

**Open question 4.4.1.** Does the cache record the backend name in
the manifest? Yes — it already does via `metadata.backend_name`.
No code change needed; just verify round-trip parity in a new test.

### 4.5 Installation

`pyproject.toml` extras block:

```toml
[project.optional-dependencies]
jax = [
  "jax>=0.4.30",
  "diffrax>=0.6.0",     # or "dynamiqs>=0.3.0"
]
```

`iontrap_dynamics.backends.jax` raises a clear `ImportError` with
install instructions if the user calls `backend="jax"` without the
extras installed. Pattern mirrors how optional dependencies are
handled elsewhere in the scientific-Python stack.

---

## 5. Candidate designs

Combining the axes, three coherent candidate designs emerge. Each
is internally consistent; each makes different tradeoffs.

### Design α — "Minimum viable JAX"

- **Framing (A):** Positioning + demonstration.
- **Integrator (B):** diffrax.
- **Representation (C):** one-shot `Qobj → jnp.asarray(dense)` at
  the solve boundary.
- **Autograd (D):** explicitly deferred.
- **Scope:** one Hamiltonian family (carrier Rabi) demonstrates
  cross-backend equivalence within 10⁻⁶ over 4 Rabi periods at
  dim 24; benchmark on the same canonical scenario.
- **Cost estimate:** single dispatch. Backend module + `backend=`
  kwarg wiring + one round-trip test + one benchmark artefact +
  `docs/benchmarks.md` section.
- **Risk:** We might measure no performance win at current scales
  — that's fine and documented honestly, same shape as Dispatch X
  and OO.

### Design β — "Dynamiqs-aligned"

- **Framing (A):** Positioning for the future-backend-target line
  in workplan §1.
- **Integrator (B):** Dynamiqs.
- **Representation (C):** one-shot conversion; Dynamiqs accepts
  NumPy / JAX arrays as input.
- **Autograd (D):** deferred, but the path is clearer since
  Dynamiqs bakes autograd into its API.
- **Scope:** two Hamiltonian families (carrier + RSB), one
  Lindblad case with a dissipator, benchmark at one large Fock
  truncation to show where JAX begins to win over QuTiP.
- **Cost estimate:** 1–2 dispatches; larger dependency surface;
  ties the library's Phase 2 capability to Dynamiqs's API
  stability.
- **Risk:** Dynamiqs API churn. Mitigated by a strict version
  pin + a policy that Dynamiqs upgrades land as their own
  dispatches with regression checks.

### Design γ — "Autograd-ready parallel builders"

- **Framing (A):** Capability.
- **Integrator (B):** diffrax.
- **Representation (C):** JAX-native Hamiltonian builders as a
  parallel layer.
- **Autograd (D):** in scope for this dispatch; a worked example
  demonstrates gradient of a fidelity w.r.t. a Rabi frequency.
- **Scope:** parallel builder family for all four Hamiltonian
  types; autograd test; JIT cache correctness tests.
- **Cost estimate:** 3–4 dispatches minimum. Builder surface
  doubles. Test count doubles on the builder layer.
- **Risk:** High. The library's Phase 2 deadline (target v0.3)
  becomes strained. Autograd introduces JAX-pitfall maintenance
  (shape polymorphism, float32/64 defaults, trace-time
  correctness) the team hasn't hit yet.

### Comparison table

| Criterion                 | α (minimum)           | β (Dynamiqs)             | γ (autograd)              |
|---------------------------|-----------------------|--------------------------|---------------------------|
| Code cost                 | 1 dispatch            | 1–2 dispatches           | 3–4 dispatches            |
| Test-surface growth       | ~10 new tests         | ~20 new tests            | ~80 new tests             |
| Dependency surface        | `jax + diffrax`       | `jax + dynamiqs`         | `jax + diffrax`           |
| Performance win at dim≤60 | None measurable       | None measurable          | None measurable           |
| Performance win at dim≥300| Likely modest         | Likely modest–good       | Likely modest             |
| Capability win            | None                  | Lindblad-native JAX      | Autograd on parameters    |
| Positioning payoff        | High                  | Highest                  | Medium                    |
| Long-term API risk        | Low                   | Medium–high              | Low                       |

---

## 6. Open questions for the maintainer

Consolidated from the axes above. Each is a decision that blocks
implementation start.

1. **Framing (A1).** Performance, capability, or positioning? Most
   consequential choice; shapes everything downstream.
2. **Integrator (B1).** Diffrax, Dynamiqs, or hand-rolled?
3. **Version pinning (B2).** Policy for JAX / diffrax / Dynamiqs
   version bumps — especially given JAX's aggressive major-version
   cadence.
4. **Representation (C1).** One-shot conversion, JAX-native builders,
   or staged?
5. **Sparse support (C2).** Dense JAX arrays in the first cut
   (recommended) or opt-in sparse?
6. **Autograd (D1).** Deferred (recommended for α / β) or in-scope
   (required for γ)?
7. **Benchmark scenario(s).** At what Hilbert size / trajectory
   length do we claim the first "JAX measurably wins over QuTiP"
   number? This is a workplan §5 exit criterion — "≥ 1 worked
   example matching QuTiP reference within cross-platform
   tolerance" — but the *performance* threshold is currently
   unspecified.
8. **GPU.** First-class deliverable (CI runs a GPU smoke test) or
   opt-in-for-users-who-install-it-themselves?
9. **Design choice α / β / γ.** Or a hybrid.

---

## 7. Staging proposal (conditional on Design α)

If Design α is chosen — the minimum-viable path — the dispatch
decomposition could look like:

1. **Dispatch α.1 — backend skeleton.** `iontrap_dynamics.backends.jax`
   package; `pyproject.toml` `[jax]` extras; `backend=` kwarg on
   `sequences.solve` with NotImplemented fallback. No solver code
   yet; just the wiring + a unit test that the kwarg raises cleanly
   when the extras aren't installed.
2. **Dispatch α.2 — carrier Rabi end-to-end.** diffrax integrator
   for the carrier Hamiltonian, one-shot Qobj→jnp conversion,
   cross-backend equivalence test at 10⁻⁶ over 4 Rabi periods at
   dim 24. `ResultMetadata.backend_name = "jax-diffrax"` in place.
3. **Dispatch α.3 — remaining Hamiltonian families.** RSB / BSB /
   MS gate ported; cross-backend tolerance tests added for each.
4. **Dispatch α.4 — benchmark artefact.** `tools/run_benchmark_
   jax_vs_qutip.py`; report + plot in `benchmarks/data/`; new
   section in `docs/benchmarks.md`. Headline finding recorded
   honestly regardless of whether JAX wins at current scales.

Total estimated cost: **4 dispatches**, each sized like OO or PP
(a few files each, one CHANGELOG entry each). Could compress to
three by merging α.3 and α.4.

---

## 8. Risks and their mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| JAX measures no win at library scale | High | Low | Document honestly; the backend's positioning / capability value is the primary rationale, not raw speed. |
| diffrax / Dynamiqs API churn | Medium | Medium | Strict version pin; upgrade-as-its-own-dispatch policy. |
| Float32/64 defaults trip up convention-version test | Medium | Medium | Explicit `jax.config.update("jax_enable_x64", True)` inside the backend; tolerance in cross-backend test. |
| NaN / inf propagation under JIT is silent | Medium | High | Mandatory `jnp.isfinite` check on returned states / expectations at the solve boundary; typed exception on failure. |
| JIT cache blow-up on shape polymorphism | Medium | Medium | Scenario benchmarks fix the shape; advanced use cases (ensemble sweeps) warn about recompilation cost. |
| GPU / Metal support is platform-specific | High (for users) | Low (for design) | Document `[jax]` extras cover CPU; GPU / Metal are user responsibility; test only on CPU in CI. |
| First dispatch lands before D2 (cd-rules tag) | High | Low | JAX-backend work is independent of the release-gate D2. Ship on `main` under `[Unreleased]` as with OO / PP. |

---

## 9. Non-decisions (explicitly out of scope for this note)

Listed so future readers do not mistake silence for implicit choice.

- **A Dynamiqs-vs-diffrax bake-off.** This note names both; if
  Design β is chosen the bake-off can happen inside the first
  dispatch as a private benchmark and inform the final choice.
- **A JAX tutorial in `docs/tutorials/`.** When the first JAX
  dispatch lands, a tutorial mirrors Tutorial 1's shape. That
  tutorial is a Phase-2 closeout artefact, not a design decision.
- **Autograd examples beyond the single gradient test in Design γ.**
  Pulse optimisation, Fisher-information, variational circuits —
  all are real use cases but belong to Phase 3+.
- **Replacing the QuTiP analytic-regression tier.** Analytic
  references are convention-level; a JAX result compares *against*
  the analytic tier, not in place of it.

---

## 10. Recommendation (tentative, subject to deliberation)

Ship **Design α** as four dispatches, with the understanding that
**(a)** the performance story is likely to be a null result at
current library scale and should be reported as such, and **(b)**
the primary Phase 2 value is opening the architectural slot that
future autograd / Dynamiqs / GPU work will fill. Design γ's
autograd capability is the long-term prize but costs too much for
a first cut; revisit after Design α ships and we have real
operational experience with a JAX code path.

Defer Design β (Dynamiqs) until after Design α lands — the
Dynamiqs dependency is a larger commitment that's easier to make
once the `backend=` discriminator and the cross-backend test
surface already exist from α.

**If the framing answer (A1) is instead "capability," invert the
recommendation: Design γ first, staged.**

---

## 11. What happens next

This document sits at `docs/phase-2-jax-backend-design.md` as a
deliberation artefact. It is not linked into `mkdocs.yml` because it
is internal planning, not user-facing documentation. When the
maintainer returns answers to §6 open questions, the document is
either:

- **Superseded by a design decision** — annotated with a "Decisions
  recorded 2026-XX-XX" banner, the chosen design marked, and
  retained for historical continuity; or
- **Discarded** — moved under `archive/` per CD 0.8 (deprecation,
  not deletion) if the JAX backend is dropped or postponed beyond
  v0.3.

In either case, the first code-bearing dispatch following this
deliberation updates `docs/phase-1-architecture.md` with the
realised backend contract and adds a `CHANGELOG.md` `[Unreleased]`
entry under the usual dispatch-letter sequence.
