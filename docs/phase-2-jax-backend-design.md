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

**Caveat on "A1 = performance".** If the framing answer is
performance, then Design α below is *not* the right first step. The
prior gap is a **QuTiP-only scaling benchmark** at dim ≥ 300 with
long trajectories (≥ 5000 steps) to measure where the QuTiP floor
actually sits outside the library's current tested range. Without
that baseline, a JAX-vs-QuTiP benchmark at library-typical scales
can only restate what Dispatches X / Y / OO already established —
parity, because we are below the floor where either backend's
implementation cost dominates.

---

## 2. Scope boundaries

### In scope — Phase 2 exit (not any single dispatch)

These are the deliverables that must be on `main` before the JAX
backend can be considered closed at the Phase 2 / `v0.3` level.
No single dispatch below ships all of them; the staging in §7
defines incremental per-dispatch scope.

- Forward-compute on pure-state and Lindblad trajectories for the
  four canonical Hamiltonian families (carrier, RSB, BSB, MS).
- Cross-backend numerical equivalence within a declared tolerance
  against the QuTiP reference on ≥ 1 worked example per family.
  (Phase 2 exit criterion named in workplan §5 Phase 2.)
- `ResultMetadata.backend_name` set to a distinct string (D5 rule
  + schema-commitment implications per §4.2).
- Reuse of the existing `TrajectoryResult` schema per D5. The JAX
  backend converts its output to the canonical `TrajectoryResult`
  **schema** at the `sequences.solve` boundary — *not* eagerly to
  `Qobj`. What happens to states is governed by `storage_mode`:
  * `StorageMode.OMITTED` — no host materialisation of states;
    expectations are computed JAX-side and only the expectation
    NumPy arrays cross the boundary.
  * `StorageMode.EAGER` — states materialise to the canonical
    tuple-of-`Qobj` at the boundary (necessary for downstream
    analysis that uses QuTiP primitives).
  * `StorageMode.LAZY` — the `states_loader` can remain JAX-lazy,
    with per-index `Qobj` conversion deferred until a specific
    state is requested.
- Opt-in installation via the existing `[jax]` extras group
  already declared in `pyproject.toml` (see §4.5). Not a base
  dependency.

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

**Open question B1.** Diffrax, Dynamiqs, or hand-rolled? The
answer depends on how much trust we want to place in a third-party
JAX library's long-term stability vs. the upfront cost of writing
our own stepper, **and** on whether A1 is positioning (where
inheriting the existing `[jax] = dynamiqs>=0.2` extras block
favours Dynamiqs) or capability (where diffrax's lower
third-party-surface helps autograd). See §4.5 and §5 for the
extras-alignment finding; §10 picks the integrator by A1 branch.

**Open question B2.** Whichever is chosen, pinned to a specific
version range? JAX + diffrax + Dynamiqs version churn is real.
Need a policy akin to the QuTiP-version pin in `pyproject.toml`
(the current `>=0.4` / `>=0.2` ranges are too loose).

### Axis C — Operator representation and the conversion boundary

The library's Hamiltonian builders return QuTiP `Qobj` instances.
JAX solvers need `jax.Array`. Where does the conversion happen?

| Option                                      | Pros                                                                 | Cons                                                                |
|---------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------|
| Convert at `solve` boundary (one-shot)      | No builder duplication; existing test surface unchanged.             | Conversion cost paid on every call; conversion is `Qobj.full()` → `jnp.asarray`, i.e. dense. Closes the door on autograd through Hamiltonian parameters. |
| JAX-native builders (parallel layer)        | Native dtype end-to-end; supports autograd on Hamiltonian parameters.| Duplicates the builder family; test surface doubles; maintenance cost ≈ 2× on builder code. |

A third option — a lazy-view bridge that wraps `Qobj.data` in a
JAX-compatible view — was considered and **rejected**. It couples to
QuTiP's internal `Qobj.data` representation (which changed between
QuTiP 4 and QuTiP 5; see the CSR default finding in Dispatch OO), so
every QuTiP release carries a non-trivial maintenance risk against
our bridge. The two options above are sufficient; the hybrid is not
considered further.

**Storage-mode interaction.** Whichever Axis-C choice is made, the
states returned on the canonical `TrajectoryResult` must honour the
declared `storage_mode` (see §2 In scope). In particular, the
one-shot-conversion path must *not* materialise states when
`storage_mode=OMITTED` — only the expectation arrays cross the
boundary in that case.

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

The `backend=` kwarg is a **family** discriminator (`"qutip"`,
`"jax"`) — separate from `ResultMetadata.backend_name`, which is a
specific **identifier** for provenance (see §4.2). One user-facing
kwarg, one metadata string; they don't share a vocabulary.

**The existing `solver=` kwarg.** `solver=` already has a stable
public meaning on the QuTiP backend (`"auto" | "sesolve" |
"mesolve"`, validated in `sequences.py` `_choose_solver`); it is
**not reinterpreted** on the JAX backend. On `backend="jax"`,
`solver=` must be `"auto"` (or omitted); explicit values
(`"sesolve"` / `"mesolve"`) are QuTiP-specific identifiers and
passing them with `backend="jax"` raises `ConventionError`. The
Schrödinger-vs-Lindblad decision on the JAX path is made from the
input dtype (pure ket vs density matrix) — the same signal QuTiP's
`"auto"` already uses. If JAX-specific solver control is ever
needed, it is introduced as a new kwarg (e.g. `jax_integrator=`),
not as an overload on `solver=`. Backend choice changes the
implementation, not the semantic contract of existing public kwargs.

### 4.2 Backend-name tagging

Per D5, `ResultMetadata.backend_name` uniquely identifies the
backend. Proposed values:

| Integrator choice | `backend_name`     |
|-------------------|--------------------|
| diffrax           | `"jax-diffrax"`    |
| Dynamiqs          | `"jax-dynamiqs"`   |
| Hand-rolled       | `"jax-manual"`     |

`backend_version` records the version of the underlying library
(e.g. diffrax version + JAX version; pattern TBD per §6 Q3).

**Schema-commitment implications.** `backend_name` is not free-form
metadata. It appears in user `.npz` + JSON cache manifests (via
`cache.save_trajectory`), and is cross-checked on load. Once a
first JAX dispatch ships `"jax-dynamiqs"` (say), that string is
written into every user's cache artefacts. Swapping the under-the-
hood integrator later — e.g. moving from Dynamiqs to diffrax under
the same `backend="jax"` kwarg — either **invalidates those
cached results** or requires a versioned tag (e.g.
`"jax-dynamiqs-v0.5"` → `"jax-diffrax-v0.6"`). This is a Coastline
commitment, not a label choice; the first JAX dispatch should
treat the chosen string as a user-visible contract and document the
versioning policy at the same time (§6 Q10).

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

The cache already records `metadata.backend_name` in the manifest
(see `cache.save_trajectory`) and cross-checks it on load. The
round-trip through `.npz` + JSON must therefore preserve the JAX
backend's identification; a round-trip parity test is added as
part of the dispatch that first emits JAX results (α.2 in the §7
staging).

### 4.5 Installation

**The `[jax]` extras block already exists** in `pyproject.toml`:

```toml
# from pyproject.toml (excerpt, line 85)
jax = [
    "jax>=0.4",
    "jaxlib>=0.4",
    "dynamiqs>=0.2",
]
```

This is a pre-existing declaration of intent toward the **Dynamiqs**
integrator (Axis B) — committed to the repo before this
deliberation. Design choice implications:

- **Designs β / γ** inherit the declared extras; no
  `pyproject.toml` change needed to start implementation.
- **Designs α / α′** (which favour diffrax) require the extras to
  change in the same dispatch that lands the backend — either
  *replacing* `dynamiqs` with `diffrax`, or *adding* `diffrax`
  alongside. Mismatch between the declared extras and the
  implementation is a correctness issue, not a documentation
  issue.
- Version ranges are currently loose (`>=0.4` on JAX) and were
  last reviewed when the extras block was first added. Before
  any first-JAX dispatch, the ranges should be re-pinned to the
  release actually exercised in CI, matching the policy answer
  to §6 Q3.

`iontrap_dynamics.backends.jax` raises a clear `ImportError` with
install instructions if the user calls `backend="jax"` without the
extras installed. Pattern mirrors how optional dependencies are
handled elsewhere in the scientific-Python stack. Before the solver
code lands (§7, α.1), the same entry point raises
`NotImplementedError` when the extras *are* installed, so the two
failure modes stay distinguishable by exception type.

---

## 5. Candidate designs

Combining the axes, four coherent candidate designs emerge. Each
is internally consistent; each makes different tradeoffs.

**Integrator-default alignment with `pyproject.toml`.** The existing
`[jax]` extras block declares `dynamiqs>=0.2`, not diffrax. Design β
is therefore the *default-aligned* option — it inherits the
declared dependency. Designs α, α′, and γ all specify diffrax (see
each design's Axis-B bullet below) and therefore require the extras
to change in the same dispatch (see §4.5).

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

### Design α′ — "α plus a narrow autograd slice"

Between α and γ: the full α (four families, one-shot conversion,
deferred autograd) plus one parallel JAX-native builder exercised
through autograd on a single gradient test.

- **Framing (A):** Positioning + minimal capability probe.
- **Integrator (B):** diffrax.
- **Representation (C):** one-shot conversion on three families
  (RSB / BSB / MS); parallel JAX-native builder `carrier_jax`
  kept in lockstep with `carrier_hamiltonian` via a per-dispatch
  regression test: the dense NumPy view of `carrier_jax(...)` must
  agree element-wise with `carrier_hamiltonian(...).full()` to
  within 10⁻¹² at a representative configuration.
- **Autograd (D):** single proof — `d fidelity / d Ω_Rabi` through
  diffrax's `solve_adjoint`. One test, one worked example in
  `docs/tutorials/`.
- **Scope:** α.1–α.3 unchanged; adds α.5 — `carrier_jax` parallel
  builder, autograd gradient test, one tutorial example.
- **Cost estimate:** ~5 dispatches (α.1–α.5); parallel-builder
  scope limited to one Hamiltonian family.
- **Risk:** Medium. The `carrier_jax` builder must stay in
  lockstep with `carrier_hamiltonian`; divergence between the
  QuTiP and JAX definitions of the same Hamiltonian would
  silently fracture the library's "canonical form" guarantee.
  Mitigated by the per-dispatch regression test above plus an
  invariant check in CI. The pitfalls γ would hit across the
  full builder surface — shape polymorphism, float32/64
  defaults, trace-time correctness — are exercised on a
  controlled single-family scope.

α′ is the bridge design: it produces γ-transferable operational
experience that α alone cannot (α's one-shot-conversion path does
not exercise autograd), at a cost bounded by ~5 dispatches
rather than γ's 3–4 open-scope dispatches.

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

| Criterion                   | α (minimum)            | α′ (α + narrow autograd) | β (Dynamiqs)             | γ (autograd-full)         |
|-----------------------------|------------------------|--------------------------|--------------------------|---------------------------|
| Code cost                   | 4 dispatches           | ~5 dispatches            | 1–2 dispatches           | 3–4 dispatches            |
| Test-surface growth         | ~10 new tests          | ~25 new tests            | ~20 new tests            | ~80 new tests             |
| Dependency surface          | `jax + diffrax`*       | `jax + diffrax`*         | `jax + dynamiqs` ✓        | `jax + diffrax`*          |
| Aligned with existing extras | No (change required)   | No (change required)     | **Yes**                  | No (change required)      |
| Performance win at dim ≤ 60 | None measurable        | None measurable          | None measurable          | None measurable           |
| Performance win at dim ≥ 300| Likely modest          | Likely modest            | Likely modest–good       | Likely modest             |
| Capability win              | None                   | One-family autograd proof| Lindblad-native JAX      | Autograd on parameters    |
| γ-transferable experience   | Plumbing only          | Plumbing + autograd      | Plumbing only            | N/A (γ itself)            |
| Positioning payoff          | High                   | High                     | Highest                  | Medium                    |
| Long-term API risk          | Low                    | Low                      | Medium–high              | Low                       |

*requires a change to the existing `[jax]` extras block (§4.5).

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
9. **Design choice α / α′ / β / γ.** Or a hybrid.
10. **`backend_name` versioning policy (from §4.2).** Is
    `backend_name` frozen per backend family (so changing the
    under-the-hood integrator invalidates cached results), or
    versioned per implementation swap (e.g.
    `"jax-dynamiqs-v0.5"` → `"jax-diffrax-v0.6"` when swapping)?
    Must be answered before α.2 / β.1 writes a tag into any
    cache artefact.
11. **QuTiP-only prior benchmark (from §1).** If A1 = performance,
    is a QuTiP-only scaling benchmark at dim ≥ 300 / long
    trajectories a precondition? Per §10, treated as a standalone
    dispatch independent of — and possibly superseding — any JAX
    dispatch: the measured QuTiP ceiling may justify or retire the
    JAX-backend deliverable entirely before any JAX code is written.

---

## 7. Staging proposal (conditional on Design α)

If Design α is chosen — the minimum-viable path — the dispatch
decomposition could look like:

1. **Dispatch α.1 — backend skeleton.** `iontrap_dynamics.backends.jax`
   package; pinned update to the `[jax]` extras (swap `dynamiqs` for
   `diffrax` per §4.5); `backend=` kwarg on `sequences.solve` with the
   error paths from §4.5 wired in. Two unit tests: (i) calling
   `backend="jax"` without the extras raises `ImportError` with
   install instructions; (ii) calling it with the extras installed
   raises `NotImplementedError` until α.2 lands the solver code. The
   two failure modes stay distinguishable by exception type.
2. **Dispatch α.2 — carrier Rabi end-to-end.** diffrax integrator
   for the carrier Hamiltonian, one-shot Qobj→jnp conversion,
   cross-backend equivalence test at 10⁻⁶ over 4 Rabi periods at
   dim 24. `ResultMetadata.backend_name = "jax-diffrax"` in place.
   A `backends.jax._materialise(result, storage_mode)` helper
   implements the OMITTED / EAGER / LAZY contract from §2 at the
   solve boundary; tests cover all three modes.
3. **Dispatch α.3 — remaining Hamiltonian families.** RSB / BSB /
   MS gate ported; cross-backend tolerance tests added for each.
4. **Dispatch α.4 — benchmark artefact.**
   `tools/run_benchmark_jax_vs_qutip.py` at a deliberately
   **informative** scale: dim ≥ 300 (single-ion Fock truncations
   up to ~150) *and* trajectory length ≥ 5000 steps. Dispatches
   X / Y / OO already established that library-typical scales
   (dim ≤ 60) are a wash; α.4 at dim 24 would only restate that
   finding. If A1 = performance and §6 Q11 is answered with a
   QuTiP-only prior (α.0), α.4 measures the delta from that
   prior. Report + plot in `benchmarks/data/jax_vs_qutip/`;
   new section in `docs/benchmarks.md`. Headline finding
   recorded honestly regardless of direction.

Total estimated cost: **4 dispatches**, each sized like OO or PP
(a few files each, one CHANGELOG entry each). Could compress to
three by merging α.3 and α.4; could expand to five if α.0 is
required by §6 Q11 or if α′'s α.5 autograd dispatch is included.

### 7.1 α → γ migration seams

If α (or α′) ships and γ is later adopted, which α artefacts are
retained and which are superseded? Naming this bounds γ's future
cost and makes α's investment/yield ratio auditable.

**Retained under γ unchanged.** These survive a γ promotion and
γ builds on top of them:

- `backend=` discriminator kwarg on `sequences.solve` (§4.1).
- `ResultMetadata.backend_name` string and its schema-commitment
  policy (§4.2).
- The `[jax]` extras block once pinned (§4.5).
- Cross-backend equivalence test harness (tolerance thresholds,
  per-family worked examples) introduced in α.2 / α.3.
- The benchmark tool at α.4 — γ extends it with larger Hilbert
  spaces and autograd-specific timings, not replaces it.
- The `ConventionError` path on `solver="sesolve"` +
  `backend="jax"` from §4.1.

**Superseded under γ.** These are specifically α-path artefacts
that γ replaces:

- One-shot `Qobj → jnp.asarray` conversion at the solve boundary
  (§4 / Axis C choice). Under γ, parallel JAX-native builders
  produce `jax.Array` end-to-end; the one-shot helpers become
  vestigial.
- The α.2 / α.3 cross-backend tests that compare one-shot-
  converted JAX output against QuTiP. Under γ, the comparison
  is between JAX-native-builder output and QuTiP-builder output
  (converted); the test target shifts.

**Written off.** The one-shot conversion helpers are α-specific
code (est. ~100–200 lines in `backends/jax/conversion.py`) that γ
deletes. α's test surface around them is similarly written off.
This is the honest cost of α-before-γ staging.

Under **α′**, the `carrier_jax` parallel builder from α.5 is the
first γ-reusable artefact; the γ-transferable operational
experience it produces is the reason to consider α′ over α when
γ is a likely follow-on.

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

The right answer depends on §6 Q1 (framing) and whether γ is
likely to follow.

**If A1 = positioning and γ is unlikely in the next release cycle:**
Ship **Design β** (Dynamiqs). It is the *default-aligned* option —
no `pyproject.toml` change, no extras mismatch, and it realises the
workplan §1 commitment to Dynamiqs as a future backend target
directly. Size: 1–2 dispatches. What α buys that β does not — the
"one-shot conversion helpers" — is **written off under γ anyway**
(§7.1), so α's detour has no γ-retained value that β lacks.

**If A1 = positioning and γ is a likely follow-on:**
Ship **Design α′** (α + narrow autograd slice). α′ produces
γ-transferable operational experience that α alone does not —
specifically the JAX pitfalls (shape polymorphism, x64 defaults,
trace-time correctness) that γ will hit across the full builder
surface. α′ exercises them on a controlled single-builder scope
first. Requires changing `[jax]` extras from Dynamiqs to diffrax
(§4.5). Size: ~5 dispatches.

**If A1 = capability:**
Ship **Design γ** directly, staged. α and α′ are both the wrong
first step because the one-shot-conversion path closes the door on
autograd through Hamiltonian parameters (§3 Axis C Cons column).
Staged γ starts with a single-family parallel builder and
expands — effectively α′'s α.5 promoted to the headline.

**If A1 = performance:**
Do **not** start a JAX dispatch yet. The prior is a QuTiP-only
scaling benchmark at dim ≥ 300 with long trajectories (§1 caveat,
§6 Q11). Answer *that* first; a JAX dispatch can only be
justified against a measured QuTiP ceiling, not an assumed one.

**Honest accounting.** The earlier draft of this note recommended
α "because it buys operational experience for γ." That claim is
false (§7.1): α's one-shot-conversion code is written off by γ.
α's *plumbing* (discriminator, metadata, test harness, benchmark
tool) is retained by every design — but so is β's, at lower cost.
α is the right choice **only** when γ is genuinely unlikely *and*
there is a reason to avoid β's Dynamiqs dependency (e.g. a
specific concern about Dynamiqs's API stability). In practice,
the choice collapses to **β or α′**, with γ a possibility if A1
is capability.

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
