# Phase 2 JAX backend β.4 — time-dependent Hamiltonians via Option X

> **Status: design note, not yet Coastline.** Sibling to
> `docs/phase-2-jax-backend-design.md`. β.4 is the remaining JAX-
> backend deliverable — the library's time-dependent Hamiltonian
> builders don't yet reach the JAX backend because their coefficient
> callables are scipy-traced (`numpy` / `math`), incompatible with
> Dynamiqs's `dq.modulated(f, Q)` which requires `f` to be JAX-
> traceable (`jnp`). β.3 surfaced this finding and deferred here.
>
> **Pre-decided:** the design family is **Option X — parallel
> JAX-native builders** (confirmed by the maintainer ahead of this
> note). This document scopes X concretely; it does not re-open the
> X / Y / Z deliberation.

**Relates to:**
`docs/phase-2-jax-backend-design.md` §3 Axis C (operator
representation), §6 Q4 (representation choice), §7.1 (α → γ
migration seams; applies by analogy to β → β.4). See also the
design note's Axis D (autograd) — the parallel builders β.4 adds
are the same scaffolding γ / α′ would have used for autograd, so
β.4 is γ-transferable at the builder-surface level.

---

## 1. What Option X is (and isn't)

**Option X:** for each time-dependent Hamiltonian builder in
`src/iontrap_dynamics/hamiltonians.py`, add a JAX-native code path
that returns a Dynamiqs `TimeQArray` (sum of `dq.modulated(f_jax,
Q)` pieces), alongside the existing QuTiP list-format path. The
builders' underlying operator construction (Pauli embeddings,
Lamb–Dicke factors, MS-gate coupling prefactors) is shared; only
the coefficient callable differs — `math.cos(δt)` on the QuTiP
path, `jnp.cos(δt)` on the JAX path.

**Option X is not:**

- **JAX-native everything.** Time-independent builders
  (`carrier_hamiltonian`, `red_sideband_hamiltonian`, …) already
  reach the JAX backend unchanged via β.2's one-shot
  `Qobj → QArrayLike` conversion at the `solve` boundary. Option X
  is specifically the *time-dependent* half of the builder surface.
- **γ (full autograd-ready parallel builder family).** γ needs
  parallel JAX builders for every Hamiltonian so the backend can
  do `jax.grad` through operator parameters. Option X covers only
  the subset that emits `[Qobj, callable]` lists today. γ is a
  strictly larger superset; the X scaffolding is directly
  reusable under γ but not sufficient.
- **A coefficient-callable replacement library.** Users who
  write their own time-dep coefficients (e.g. custom
  `modulated_carrier_hamiltonian` envelopes) must supply their own
  JAX-traceable callable. The library provides the plumbing but
  not a pre-baked library of every envelope shape a user might
  want — that's a `tutorials/` concern, not a `src/` concern.

---

## 2. Builder inventory (the actual scope)

Five call sites in `src/iontrap_dynamics/hamiltonians.py` emit
QuTiP time-dependent list format. Two flavours.

### 2.1 Structured cos / sin detuning envelopes (4 builders)

All four follow the same pattern: two Hermitian operators `(H_1,
H_2)` are paired with `cos(δt)` and `sin(δt)` respectively, where
`δ` is a closed-over scalar detuning.

| Builder                                      | Line (current) | H_1                | H_2              | Closure |
|----------------------------------------------|---------------:|--------------------|------------------|---------|
| `detuned_carrier_hamiltonian`                | 186–320        | `(Ω/2) σ_φ`       | `(Ω/2) σ_⊥`      | `δ = drive.detuning_rad_s` |
| `detuned_red_sideband_hamiltonian` (via `_list_format_cos_sin`) | 683–864 | `H_static` | `H_quadrature` | same |
| `detuned_blue_sideband_hamiltonian` (via `_list_format_cos_sin`) | 810–864 | `H_static` | `H_quadrature` | same |
| `detuned_ms_gate_hamiltonian`                | 1022–1200      | `A_X`              | `A_P`            | detuning internal to MS gate |

**Translation pattern** (common to all four):

```python
# QuTiP path (current):
def cos_coeff(t: float, args: object) -> float:
    return math.cos(delta * t)
def sin_coeff(t: float, args: object) -> float:
    return math.sin(delta * t)
return [[H_1, cos_coeff], [H_2, sin_coeff]]

# JAX path (β.4):
import dynamiqs as dq
import jax.numpy as jnp
def cos_coeff_jax(t):
    return jnp.cos(delta * t)
def sin_coeff_jax(t):
    return jnp.sin(delta * t)
return dq.modulated(cos_coeff_jax, H_1) + dq.modulated(sin_coeff_jax, H_2)
```

The operator construction (`H_1`, `H_2`) is identical in both
branches; only the final assembly differs. The helper
`_list_format_cos_sin(H_static, H_quadrature, detuning_rad_s)` at
line 867 already factors out the QuTiP-side assembly; β.4 adds a
sibling `_timeqarray_cos_sin(H_static, H_quadrature,
detuning_rad_s)` that emits the Dynamiqs form.

### 2.2 User-supplied envelope (1 builder)

`modulated_carrier_hamiltonian(hilbert, drive, *, ion_index,
envelope)` at line 1208 wraps an arbitrary user callable:

```python
# QuTiP path:
def coeff(t: float, args: object) -> float:
    return envelope(t)
return [[base_H, coeff]]
```

The user's `envelope` is scipy-traced by default — they wrote it
using whatever they wanted. The library can't introspect or
auto-translate arbitrary user code.

**Option X resolution for this case:** add a parallel
`envelope_jax` keyword-only parameter. The caller supplies both a
QuTiP-friendly envelope (used by the QuTiP backend) and a
JAX-friendly envelope (used by the JAX backend). If only one is
supplied, the builder can only produce the matching backend's
output.

```python
def modulated_carrier_hamiltonian(
    hilbert,
    drive,
    *,
    ion_index,
    envelope,                 # QuTiP-facing callable, scipy-traceable
    envelope_jax=None,        # optional JAX-traceable mirror
    backend="qutip",          # which branch to emit
):
    ...
    if backend == "qutip":
        # existing path unchanged
        return [[base_H, lambda t, args: envelope(t)]]
    elif backend == "jax":
        if envelope_jax is None:
            raise ConventionError(
                "modulated_carrier_hamiltonian(backend='jax') requires "
                "envelope_jax= — a JAX-traceable mirror of envelope. "
                "Use jax.numpy operations; see tutorials/… for examples."
            )
        return dq.modulated(envelope_jax, base_H)
```

Clear failure mode, no silent translation hack.

---

## 3. API pattern — backend kwarg on existing builders

**Chosen:** `backend: str = "qutip"` kwarg on every time-dep
builder, mirroring `sequences.solve`'s existing pattern.
`backend="qutip"` returns the current QuTiP list format;
`backend="jax"` returns the Dynamiqs `TimeQArray`. Unknown
backend strings raise `ConventionError` with the same validator
surface `sequences._validate_backend` already uses.

### 3.1 Why not parallel function names (`*_jax` suffix)?

Considered and rejected. Rationale:

- Doubles the top-level API surface (12 time-dep builders instead
  of 6). Design Principle 5 ("one way to do it at the public API
  level") applies — the `_jax` suffix is an implementation detail
  that shouldn't leak.
- Users who switch backends mid-project would have to rename
  calls everywhere; the `backend=` kwarg is a one-character
  change.
- Parallel names would complicate tutorial prose: every tutorial
  with a time-dep Hamiltonian would fork into QuTiP / JAX
  variants. The kwarg lets one tutorial serve both.

### 3.2 Why not solver-side auto-detection?

Also considered. The `sequences.solve` function could, in
principle, inspect the Hamiltonian's runtime type and dispatch
accordingly (QuTiP list → QuTiP path; `TimeQArray` → JAX path).
Rejected because:

- The user's `backend=` kwarg on `solve` would become
  informational-only when a `TimeQArray` is passed, violating
  the invariant "`backend=` is the single authoritative
  discriminator."
- Error mode on mismatch (user passes QuTiP list with
  `backend="jax"`) is confusing: either the solver silently
  overrides the kwarg or it raises — both are worse than the
  builder rejecting at construction time.

### 3.3 Validation contract

When `backend="jax"` is passed to a time-dep builder:

- The builder computes `H_1`, `H_2` as QuTiP `Qobj`s (same code
  path as `backend="qutip"`); Dynamiqs accepts these as
  `QArrayLike`.
- The builder imports `dynamiqs` and `jax.numpy` lazily inside
  the `backend == "jax"` branch, so importing the builder module
  does not require `[jax]` extras.
- If `dynamiqs` / `jax` import fails inside the branch, `BackendError`
  fires with the same install hint as `solve(backend="jax")`.
  Users get one consistent failure mode.

---

## 4. Shared coefficient-callable module

To avoid duplicating `jnp.cos(delta * t)` / `jnp.sin(delta * t)`
across four builders, β.4 introduces:

```
src/iontrap_dynamics/backends/jax/
├── __init__.py
├── _core.py                    # existing β.1 – β.3
└── _coefficients.py            # new in β.4
```

`_coefficients.py` exports JAX-traceable closure factories:

```python
def cos_detuning_jax(delta: float) -> Callable[[float], Array]:
    """Return a JAX-traceable callable t ↦ jnp.cos(delta * t)."""
    def coeff(t):
        return jnp.cos(delta * t)
    return coeff

def sin_detuning_jax(delta: float) -> Callable[[float], Array]:
    ...
```

Each builder's `backend="jax"` branch imports these lazily and
passes the returned closure to `dq.modulated`. Single source of
truth for the detuning coefficient shape; zero duplication.

**Out of scope for β.4:** a library of pre-baked envelope shapes
(Gaussian, sech, Blackman, …) for `modulated_carrier_hamiltonian`
users. If demand emerges that's a tutorial-side or separate
`envelopes/` module; not a β.4 blocker.

---

## 5. Staging proposal

Five dispatches, each OO / PP-sized.

### β.4.1 — coefficient-callable module + detuned_carrier

- New `_coefficients.py` module with `cos_detuning_jax` +
  `sin_detuning_jax`.
- `detuned_carrier_hamiltonian` gains `backend: str = "qutip"`
  kwarg; `backend="jax"` branch emits `dq.modulated` sum.
- Cross-backend equivalence test on detuned carrier at dim 24, 4
  Rabi periods, 1e-3 tolerance (mirroring β.2's carrier test).
- Smallest scope — validates the pattern.

### β.4.2 — detuned red / blue sideband

- `_list_format_cos_sin` gains a sibling
  `_timeqarray_cos_sin(H_static, H_quadrature, delta)` in
  `backends/jax/_builders.py` (new module).
- `detuned_red_sideband_hamiltonian` + `detuned_blue_sideband_hamiltonian`
  gain the same `backend=` kwarg; dispatch to the new helper.
- Cross-backend equivalence tests for both builders.

### β.4.3 — detuned MS gate

- `detuned_ms_gate_hamiltonian` gains `backend=` kwarg.
- The MS gate's bichromatic structure may require two
  `dq.modulated` pieces per spin pair; verify that
  `dq.modulated + dq.modulated + ...` scales correctly for the
  4-piece MS Hamiltonian form.
- Cross-backend equivalence test.

### β.4.4 — modulated_carrier_hamiltonian (user envelope)

- `envelope_jax` keyword parameter added.
- Validation: `backend="jax"` + `envelope_jax=None` →
  `ConventionError` with the actionable message from §2.2.
- Two tutorials updated: one that demonstrates the JAX path on a
  Gaussian envelope (`jnp.exp(-t**2 / 2 / sigma**2)`), one that
  flags the dual-callable requirement.
- Largest β.4 dispatch because it touches user-facing docs.

### β.4.5 — cross-backend equivalence benchmark

- `tools/run_benchmark_jax_timedep.py` — runs each of the four
  structured builders through both backends and records the
  cross-backend delta over a long trajectory (≥ 5000 steps) at
  dim ≥ 100. Records in `benchmarks/data/jax_timedep/` with the
  same report.json + plot.png convention as OO.
- `docs/benchmarks.md` section.

Total: 5 dispatches. Can compress to 4 by merging β.4.1 + β.4.2
(the detuned-structured pattern is nearly identical).

---

## 6. Open questions for the maintainer

Decisions that still block implementation start even with Option X
chosen.

1. **`modulated_carrier_hamiltonian` — dual-callable or split
   function?** The note proposes `envelope_jax=None` kwarg. An
   alternative is a separate `modulated_carrier_hamiltonian_jax`
   function that takes only `envelope_jax`. The kwarg form is
   more consistent with other builders; the split form is
   slightly safer (can't accidentally pass a QuTiP envelope as
   JAX by forgetting the kwarg). Default recommendation: kwarg
   form, since the builder's other kwargs already encode the
   asymmetry.

2. **`backend_name` tagging for time-dep JAX results.** Results
   from JAX-backend time-dep solves still tag as
   `"jax-dynamiqs"` per β.2's schema-commitment contract
   (design note §4.2). Time-dep doesn't warrant a distinct tag
   (`"jax-dynamiqs-timedep"` would be noise).
   Default recommendation: keep single `"jax-dynamiqs"` tag.

3. **Cross-backend tolerance for time-dep scenarios.** β.2
   measured 2e-5 library-default agreement for time-independent
   carrier Rabi. Time-dep trajectories integrate oscillating
   coefficients; cross-backend disagreement may be larger if
   QuTiP and Dynamiqs handle step-size adaptation differently
   around coefficient discontinuities. Assert 1e-3 as in β.2 (safe
   margin) and tighten later if empirically warranted? Default
   recommendation: 1e-3 initially, same as β.2.

4. **Migration-scenario 4 unblock.** The scope-4 `qc.py`
   regression (stroboscopic AC-π/2) is currently skipped because
   the library's `modulated_carrier_hamiltonian` applies the
   envelope to the RWA carrier; `qc.py` uses the full-exponential
   Lamb–Dicke operator. β.4 ships JAX support for
   `modulated_carrier_hamiltonian` but does **not** change the
   physics (still RWA). Migration-scenario 4 stays skipped post-
   β.4. Flag, don't fix — separate dispatch.

5. **γ transferability — does β.4's parallel-builder scaffolding
   serve γ?** Yes, mostly. γ needs autograd through Hamiltonian
   parameters (e.g. `d fidelity / d Ω_Rabi`). The β.4 builders
   close over scalars (`delta`, `omega`, …) that γ can promote
   to traced JAX arrays with minimal code changes. The β.4
   commitment to `dq.modulated(f, Q)` is γ-aligned because `Q`
   (the Hermitian operator) is not traced through but `f`
   (the coefficient callable) can be — matching γ's autograd
   needs exactly. β.4 delivers ~60 % of γ's builder surface for
   ~20 % of γ's cost.

6. **Timing — does β.4 need to land before `v0.3`?** Workplan
   §5 Phase 2 targets `v0.3` for the JAX backend. β.1 / β.2 /
   β.3 cover the time-independent Phase 2 exit criteria; β.4
   extends the surface but is not strictly gating if the
   cross-backend example in the workplan is a time-independent
   scenario. Maintainer call: is β.4 a v0.3-blocker or a
   v0.3.x follow-up?

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| JAX closure over a Python float captures trace-time value; late-bound `delta` mutation breaks | Low | Medium | Factory pattern in `_coefficients.py` explicitly snapshots `delta` in a closure. Tests cover "different `delta` values produce different trajectories" — cheap. |
| `dq.modulated` discontinuity-ts missing on piecewise coefficients | Medium | Medium | β.4.1 – β.4.3 use continuous cos / sin (no discontinuities). β.4.4 user envelope may have discontinuities — pass `discontinuity_ts` through on `modulated_carrier_hamiltonian(backend="jax")`, exposed as a kwarg. |
| `dq.modulated + dq.modulated` sum performance at dim ≥ 100 | Low | Low | β.2 measured 2e-5 agreement at dim 24; β.4.5 benchmark verifies at dim ≥ 100. Dynamiqs's `TimeQArray` sum is a shallow wrapper, not a full matmul cost. |
| User-envelope `envelope_jax` trace-time errors (ConcretizationTypeError, etc.) | High (for novice users) | Medium | Clear error messages when the user's envelope can't be traced. Tutorial β.4.4 includes common pitfalls (`float(...)` on traced values, Python control flow, etc.). |
| Coefficient-callable closure captures stale `hilbert` or `drive` reference | Low | Low | Coefficients in `_coefficients.py` close over scalars only — no `hilbert` / `drive` capture. |
| Builder's `backend="jax"` branch imports dynamiqs at call time; cost in hot loops | Low | Low | Same pattern as `_core.solve_via_jax`. Python caches the module after first import; cost per subsequent call is a dict lookup. |

---

## 8. Non-decisions (out of scope for β.4)

- **Parallel JAX-native Hamiltonians for time-independent
  builders.** Time-independent builders already reach JAX via
  β.2's one-shot conversion; γ's autograd-ready versions are
  orthogonal to β.4.
- **GPU execution as a test target.** JAX can run on GPU / TPU
  given CUDA / Metal builds; β.4 tests run on CPU only in CI.
  GPU is a user-install concern per design note §4.5.
- **Dynamiqs `TimeQArray` serialisation through cache I/O.**
  `cache.save_trajectory` operates on `TrajectoryResult`, not on
  Hamiltonians. Cached results don't embed Hamiltonians, so
  `TimeQArray` never hits disk. No cache-schema change.

---

## 9. What happens next

This document sits at `docs/phase-2-jax-time-dep-design.md` as a
deliberation artefact. Unlike the parent note, it is pre-decided on
the Option-X family; the remaining open questions (§6) are
implementation-detail choices rather than architectural.

When §6 answers return from the maintainer, the β.4.1 dispatch
opens with:

1. Create `src/iontrap_dynamics/backends/jax/_coefficients.py`.
2. Extend `detuned_carrier_hamiltonian` with `backend=` kwarg.
3. Mirror the β.2 test shape in
   `tests/unit/test_backends_jax_dynamiqs.py::TestTimeDependent`
   (or a sibling file if it grows large).
4. CHANGELOG entry + `docs/phase-1-architecture.md` cross-ref.

β.4 does **not** require a new `pyproject.toml` extras change or a
new module-level public export. All changes are additive (new
kwarg values) and per-builder. Existing callers using only the
QuTiP path observe no behaviour change.
