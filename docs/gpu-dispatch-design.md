# GPU dispatch — design note (draft for deliberation)

> **Status: draft for deliberation, not Coastline.** This document is a
> pre-decision exploration of the GPU / CUDA extensions already flagged
> in `docs/phase-2-jax-backend-design.md` Axis B and
> `docs/benchmarks.md § scipy vs JAX on CPU for dense eigh`. Nothing
> here binds an implementation. The eventual dispatch that opens this
> work will come with its own Coastline updates (CHANGELOG entry,
> extension to `docs/phase-1-architecture.md`, and — if behaviour is
> user-visible — `CONVENTIONS.md` version bump per §5.1's freeze
> protocol).

**Relates to:** `docs/phase-2-jax-backend-design.md` Axis B (GPU /
TPU as future backend target), `docs/phase-2-jax-backend-design.md`
Axis D (autograd as capability deliverable),
`docs/phase-2-jax-time-dep-design.md` (time-dependent Hamiltonian
plumbing on JAX), `docs/benchmarks.md § Exact-diagonalization
envelope (Dispatch AAH)`, `docs/benchmarks.md § AAG gate status`,
`docs/benchmarks.md § scipy vs JAX on CPU for dense eigh`,
`docs/workplan-clos-2016-integration.md` §7 Q4 (JAX / Dynamiqs
interaction, resolved 2026-04-23 for CPU only).

**Classification reminder.** Once a direction is chosen, the
resulting commitments become Coastline: the `backend_name` string
assigned to each GPU path, the `SpectrumResult` / `TrajectoryResult`
fields that must round-trip across backends, the tolerance thresholds
against reference runs. Implementation tactics (which JIT strategy,
which benchmark dashboard, which CUDA version pin) stay Sail.

---

## 1. Why now

Three findings from the Phase 2 / Clos 2016 waves converge on the
GPU question at the same moment.

1. **CPU null results, twice.** Dispatch YY (β.4.5) measured
   QuTiP 5 as ~2.8× faster than Dynamiqs + JAX on CPU at
   `dim ≥ 100`. The post-AAH comparison measured
   `jax.numpy.linalg.eigh` as ~22 % slower than
   `scipy.linalg.eigh` asymptotically on CPU with a ~130 MB higher
   baseline. The JAX backend's value on CPU is positioning and
   forward-compatibility, not speed — every Phase 2 benchmark
   that could have shown a CPU win measured the opposite.

2. **A concrete user-case where GPU would matter.** The AAH
   envelope shows dense `eigh` on a 16 GB consumer laptop
   reaching N = 4 at `n_c = 7` (`dim = 8 192`; 7 min per eigh;
   1.9 hr for a 16-detuning sweep) and N = 5 at `n_c = 5`
   (`dim = 15 552`, projected). Both sit well inside the
   measured scaling but past the point where the binding
   constraint flips from RAM to wall-clock. A consumer-class
   GPU (16–24 GB VRAM) has roughly the same reach envelope
   as the 16 GB CPU tier — dense complex128 matrix storage is
   `dim² × 16` bytes before workspace, so `dim ≈ 15 000` is
   the boundary at 16 GB on both sides — but cuSOLVER `eigh`
   wall-clock at `dim ≈ 8 000` is well under a minute in
   published benchmarks, vs 7 min on CPU. **The GPU win this
   note pitches is wall-clock, not reach.**

3. **AAG is one of three ways to reach past dim ≈ 15 000.**
   The workplan left AAG (interior-window shift-invert around
   `meanE`) deferred because dense on CPU still covers the
   publication-validated reproduction. A GPU dense path
   compresses CPU wall-clock at the same envelope but does
   not expand it — a fully-converged N = 5 at `n_c = 6`
   (`dim = 33 614`) needs ~18 GB for the dense matrix alone
   plus workspace, past the 16 / 24 / 40 GB consumer- and
   lab-class tiers. GPU data therefore refines AAG's gate
   status rather than displacing it: AAG's value shifts from
   "cover the N = 5 `n_c = 5` boundary" to "cover
   fully-converged N = 5 past `dim ≈ 35 000` where no
   single GPU card reaches."

The question the note scopes: can the library usefully ship a GPU
dispatch today, given that CI is CPU-only, and if so, where does it
start?

---

## 2. Scope boundaries

### In scope — the body of this note

- A GPU-capable `solve_spectrum` path. The proposed new
  backend identity is **device-neutral**: one
  `backend_name = "spectrum-jax"` covering both CPU-JAX and
  GPU-JAX dispatch, with device recorded in
  `metadata.provenance_tags`. Gated on a JAX install at
  runtime; otherwise no behaviour change.
- A GPU-capable time-evolution path through the **existing**
  `backend_name = "jax-dynamiqs"` contract. The shipped
  identity is preserved; device selection becomes a new
  solver kwarg (`device="gpu"|"cpu"|None`), with the
  chosen device recorded in `provenance_tags`. No new
  `backend_name` is minted for GPU time-evolution.
- CI / testing model for GPU-only code paths — how we
  assert correctness without GitHub-hosted GPU runners.
- Reference-hardware policy — what counts as a baseline
  machine, how sweep-benchmarks are stored, how we catch
  regressions.
- Autograd-through-the-eigensolve as an *enabler*, not a
  deliverable — the architecture should leave it open without
  forcing it.

### Out of scope (any design)

- **TPU support.** JAX dispatches to XLA:TPU transparently, but
  the library has no TPU user. Cost/value is wrong; revisit if
  and when a TPU user emerges.
- **AMD / ROCm.** Same reasoning — no known user. `jax[cuda12]`
  on NVIDIA is the only platform considered here. Apple Metal
  (`jax[metal]`) is called out separately in §3 Axis A because
  the AG Schätz lab has Apple Silicon machines in active use.
- **QuTiP GPU.** QuTiP 5 has no first-class GPU backend;
  Dynamiqs is the only practical path. This note does not
  re-open the backend-choice question from Phase 2.
- **Distributed / multi-GPU.** `jax.pmap` / `jax.distribute`
  are capability-accessible but have no library use case at
  current scales. Single-device GPU only.
- **GPU-first defaults.** `backend="qutip"` (CPU, QuTiP 5)
  remains the library default. Every GPU path is opt-in via
  an explicit `backend=` string. Same contract as Phase 2.

### Deliberately ambiguous — to resolve below

- Whether `solve_ensemble`'s parameter-sweep dispatch benefits
  more from GPU-per-point or from CPU parallelism via joblib
  (the current v0.3.0 behaviour). §4 Axis C opens this.
- Whether the first GPU dispatch targets **spectrum** (narrow,
  AAH-extending, well-bounded by the existing `SpectrumResult`
  schema) or **time-evolution** (wider, matches Phase 2's own
  framing, amortises across more user flows). §5 compares.

---

## 3. Architecture axes

Four axes matter. The §5 candidate designs sit at different
points along them.

### Axis A — Hardware target

- **A.1: CUDA only.** `jax[cuda12]` on NVIDIA. Smallest cost,
  matches every prior JAX-GPU paper in the trapped-ion / AMO
  space.
- **A.2: CUDA + Apple Metal.** `jax[cuda12]` plus `jax[metal]`
  (currently experimental — limited LAPACK coverage). Adds a
  development-machine path (lab Apple Silicon) but runtime
  support for `eigh` on Metal is not guaranteed at `jaxlib`'s
  current release cadence.
- **A.3: CUDA only, Metal as advisory.** Document Metal as a
  known-working-for-time-evolution path once JAX adds it, but
  don't test / benchmark it until LAPACK coverage lands
  upstream.

A.3 is the sensible middle ground: users on Metal can still run
their code (the `device="gpu"` kwarg resolves to whatever JAX
platform the installed `jaxlib` provides — `jax[metal]` on
Apple Silicon, `jax[cuda12]` on NVIDIA), but the library does
not make any performance or correctness claim on the Metal
path. Metal users who hit `eigh`-unavailable errors get the
raw JAX diagnostic; the library does not intercept.

### Axis B — Correctness testing without CI GPU runners

Three honest options; the first is the one Phase 2 chose for
Dynamiqs-gated tests, extended to GPU.

- **B.1: Import-gated skips.** Tests that need a GPU use
  `pytest.importorskip("jax")` plus a `jax.devices()[0].platform
  == "gpu"` predicate; skip cleanly on CI. Developer runs them
  locally on GPU hardware before merging. Same pattern as
  `tests/unit/test_backends_jax_dynamiqs.py`. Honest, zero CI
  cost, documented blind spot.
- **B.2: Self-hosted runner.** Someone in AG Schätz runs a
  GitHub Actions self-hosted runner pinned to a known GPU.
  Pros: CI actually exercises GPU paths. Cons: runner becomes
  a bus-factor dependency; upstream JAX / CUDA bumps can silently
  break the runner; stewardship cost is non-trivial.
- **B.3: Artefact-based regression.** Developer commits a
  reference JSON (per-(N, n_c) wall-clock and RSS at a named
  hardware tier) into `benchmarks/data/gpu/`. CI re-runs the
  benchmark *only* on matching hardware (skip otherwise), but
  the artefact is checked into git as the ground truth that
  any future run compares against. This is the AAH pattern
  extended.

**B.1 + B.3** is the recommendation: tests are import-gated;
benchmarks produce artefacts that anyone on matching hardware
can regenerate and diff.

### Axis C — Spectrum vs. time-evolution priority

- **C.1: Spectrum-first.** Open
  `solve_spectrum(..., backend_name="spectrum-jax", device="gpu")`.
  Scope is bounded by the existing `SpectrumResult` schema —
  no schema changes, just a second method="dense" implementation
  under a new `backend_name` ("spectrum-jax") and one new optional
  `device` kwarg. Extend the existing
  `tools/run_benchmark_spectrum_envelope_jax.py` with a
  `--device` flag rather than shipping a CUDA-only clone; add a
  new section in `docs/benchmarks.md`. Addresses AAH's N = 4 /
  N = 5 wall-clock tier and gives concrete data against which
  AAG can be re-evaluated.
- **C.2: Time-evolution-first.** Extend `backend="jax"` to
  auto-detect GPU and dispatch there; Dynamiqs on CUDA for
  `sesolve` / `mesolve`. Wider impact (every tutorial using
  JAX benefits), but the AAH null-result-on-CPU data suggests
  the GPU win for time-evolution at library-typical
  `dim ≤ 100` is unproven — small Hilbert spaces pay the GPU
  launch latency without amortising.
- **C.3: Both, in that order.** Spectrum ships first as a
  self-contained dispatch; time-evolution follows once the
  measurement discipline and artefact-storage pattern are
  established.

The data favours C.1 first: AAH gives a clear N = 4 /
`n_c = 7` target where CPU costs 7 min per eigh, so a 2× GPU
win is user-visible and measurable. The time-evolution case is
murkier — the β.4.5 benchmark has already shown that for
`dim ≤ 100` on CPU, QuTiP beats JAX on wall-clock; it's not
obvious that the same library sizes flip on GPU.

### Axis D — Autograd as an enabler, not a deliverable

The JAX design note's Axis D (autograd through the solver) is
explicitly Phase 3+ territory. This note does **not** ship
autograd — but every design decision here should leave the
door open, because the most user-visible future motivation for
GPU is exactly "gradient-based fitting of experimental data to
simulation." Concretely:

- `SpectrumResult` and `TrajectoryResult` fields must remain
  convertible to JAX arrays without breaking the dense `eigh`
  / `sesolve` return contract.
- Any new GPU builder must accept JAX-traced inputs
  transparently (the Phase 2 β.4 builders already do this).
- `backend_name = "spectrum-jax"` does not promise autograd; a
  future autograd-aware path is recorded via a provenance tag
  (e.g. `"autograd"`) on the same backend_name rather than by
  minting a new identity.

---

## 4. Integration with the existing surface

### 4.1 `solve_spectrum` API

The shipped signature (`src/iontrap_dynamics/spectrum.py`)
already accepts `backend_name: str | None`. GPU support adds
one new accepted value, no new kwarg:

```python
solve_spectrum(H, method="dense", backend_name="spectrum-jax")
```

The new `backend_name = "spectrum-jax"` is **device-neutral**.
Device selection happens through a new optional `device` kwarg
(`"gpu" | "cpu" | None`, where `None` means "use JAX's default
device"). The chosen device is recorded in
`metadata.provenance_tags` alongside `cuda:<version>` /
`jaxlib:<version>` strings when the device is a GPU. Allowed
`backend_name` values become: `"spectrum-scipy"` (shipped),
`"spectrum-jax"` (new, this note), and the AAG-gated
`"spectrum-scipy-shift-invert"` (deferred).

### 4.2 `sequences.solve` / `solve_ensemble` API

The shipped trajectory contract fixes
`backend_name = "jax-dynamiqs"` for the JAX path (see
`tests/unit/test_backends_jax_dynamiqs.py:169`). **This note
does not change that identity.** GPU for time-evolution is
device routing, not a new backend:

- Existing caller contract unchanged — `solve(..., backend="jax")`
  on a GPU-enabled JAX install dispatches to GPU transparently,
  exactly as it already does today. `backend_name` stays
  `"jax-dynamiqs"`.
- New optional kwarg `device="gpu" | "cpu" | None` accepted on
  `solve`, `solve_ensemble`, and time-dependent builders. `None`
  (default) means "JAX default device"; explicit `"gpu"` on a
  machine without GPU raises a clear diagnostic. The selected
  device is recorded in `ResultMetadata.provenance_tags`.
- **Why `"gpu"` and not `"cuda"`.** The kwarg value matches JAX's
  own platform strings (`"cpu"` / `"gpu"` / `"tpu"`, as returned
  by `jax.default_backend()` and accepted by `jax.devices()`).
  A JAX install built against `jax[cuda12]` resolves `"gpu"` to
  CUDA; `jax[metal]` resolves the same string to Metal. The
  library does not need to (and deliberately does not) encode
  the vendor. Vendor-specific information (CUDA version,
  cuSOLVER version, jaxlib build) is recorded as provenance,
  not as part of the device kwarg contract.
- No new `backend_name` string is introduced. The three
  positions in an earlier draft of this note (new identity /
  device kwarg / provenance tag) collapse to one: **device is
  provenance**, not identity, for both spectrum and
  time-evolution.

### 4.3 Convention version

Adding the `spectrum-jax` backend_name and the `device` kwarg
does not change `CONVENTION_VERSION` — existing callers
(`backend="qutip"` / `"jax"`, `backend_name="spectrum-scipy"`)
see no behaviour change, no result-schema extension. GPU-specific
provenance (device string, VRAM tier, CUDA version, cuSOLVER
version, jaxlib version) lives in `provenance_tags` on the
metadata — same mechanism used for notebook / sweep / ci tagging
today.

### 4.4 Installation

One extras group proposed, composing with the existing `[jax]`:

- `gpu` — `jax[cuda12]>=0.4`, plus `dynamiqs` (so that
  `iontrap-dynamics[gpu]` is a complete install for **both** the
  spectrum and the time-evolution GPU paths). The `jax[cuda12]`
  build replaces the CPU `jax` + `jaxlib` that come in via
  `[jax]`; co-installing both groups is safe because pip
  resolves to the CUDA `jaxlib` when present.
- Users who only want GPU spectrum (no time-evolution) can
  install `iontrap-dynamics[gpu]` and skip the extra Dynamiqs
  import at runtime — `solve_spectrum` has no Dynamiqs
  dependency.

`jax[metal]` stays out of optional-dependencies — users can
install it themselves if they want the experimental path.

---

## 5. Candidate designs

### Design α — "Spectrum-only GPU"

Scope: Ship `solve_spectrum` on JAX with GPU dispatch. One new
`backend_name = "spectrum-jax"` (device-neutral). One new
`device` kwarg on `solve_spectrum`. **Extend** the existing
`tools/run_benchmark_spectrum_envelope_jax.py` — which already
replays the AAH grid through JAX on CPU — with a `--device`
flag, rather than shipping a separate CUDA-only script. One reference-hardware artefact in
`benchmarks/data/gpu/spectrum_envelope/`. Tests are
import-gated GPU-skips on CI. No change to time-evolution.

Cost: ~2 dispatches. Risk: low.

Deliverable readable: "On a single NVIDIA RTX 40-series GPU
with 16 GB VRAM, dense `eigh` across the measured AAH grid
(up to `dim = 8 192`) completes each point in
single-digit-second wall-clock, against 7 min on 16 GB CPU at
the same `dim`. Reach is the same as 16 GB CPU
(`dim ≈ 15 500` at the workspace boundary); fully-converged
N = 5 at `n_c ≥ 6` (`dim ≥ 33 614`) remains outside the
dense-on-a-single-GPU envelope and stays on the AAG path."

### Design β — "α plus GPU time-evolution"

Design α, plus `device="gpu" | "cpu" | None` kwarg added to
`solve` / `solve_ensemble` / time-dependent builders. **No
new `backend_name`** — the shipped `"jax-dynamiqs"` identity
is preserved; device lives in `provenance_tags`. New section
in `docs/benchmarks.md` comparing `"qutip"`, `"jax-dynamiqs"`
(CPU), `"jax-dynamiqs"` (GPU via `device="gpu"`) across the
β.4.5 grid.

Cost: ~4–5 dispatches. Risk: medium (Dynamiqs-on-CUDA is less
exercised in the literature than cuSOLVER `eigh`; launch-
latency noise at `dim ≤ 100` may dominate at small sizes and
bloat the reference artefacts).

### Design γ — "β plus autograd seam"

Design β, plus an explicit autograd-compatible builder wrap
(`carrier_hamiltonian_full_ld` taking JAX-traced `η` / detuning
inputs end-to-end with a `jax.grad` test). Not a full
autograd release — just enough to prove the door isn't closed.

Cost: ~5–6 dispatches. Risk: medium-high (autograd brings its
own convention questions — do we expose `jax.grad` directly or
wrap it? Does the `ResultMetadata` record the differentiation
target?).

### Comparison table

| Criterion                        | α (spectrum only)       | β (α + time-evolution)   | γ (β + autograd seam)         |
|----------------------------------|-------------------------|--------------------------|-------------------------------|
| New public `backend_name`        | 1 (`spectrum-jax`)      | 1 (no new evolution name)| 1 (autograd uses provenance)  |
| New public kwargs                | `device=` on `solve_spectrum` | +`device=` on `solve` / ensemble / builders | +autograd-scoped provenance tags |
| Dispatch count                   | ~2                      | ~4–5                     | ~5–6                          |
| AAH user case addressed          | Yes (wall-clock only)   | Yes                      | Yes                           |
| β.4.5 user case addressed        | No                      | Yes                      | Yes                           |
| Phase 3 autograd enabled         | Door left open          | Door left open           | Door cracked open             |
| CI cost                          | Skips only              | Skips only               | Skips only                    |
| Reference-hardware burden        | One tier                | One tier                 | One tier                      |
| Risk profile                     | Low                     | Medium                   | Medium-high                   |

---

## 6. Open questions for the maintainer

1. **Reference GPU.** Which machine becomes the baseline — an
   RTX 40-series consumer card? An NVIDIA L4 / A10 lab card?
   A lab workstation the maintainer has physical access to?
   This shapes what "AAH on GPU" looks like as an artefact.
2. **CI posture on GPU skips.** Document as "tests
   import-gated; see `docs/gpu-dispatch-design.md` for
   regeneration" — or keep the skips silent and rely on
   human discipline? Phase 2's precedent is silent skips,
   but the GPU surface will be larger.
3. **Metal posture.** Advisory-only (§3 A.3) is the
   recommended framing. Does the maintainer want to
   actually test on Apple Silicon, or is "transparent JAX
   device dispatch, no warranty" enough?
4. **α vs β vs γ.** The main decision. §7 assumes α; if
   the maintainer picks β or γ the staging changes.
5. **Dependency posture.** `gpu` optional-deps group with
   `jax[cuda12]>=0.4`, or leave CUDA install to the user
   entirely and just test against whatever they have?

---

## 7. Staging proposal (conditional on Design α)

Five dispatch-shape units, in order. Only the first three ship
capability; BBI and BBJ close the documentation loop.

1. **Dispatch BBA — JAX spectrum backend (device-neutral).**
   Wire `solve_spectrum(..., backend_name="spectrum-jax",
   device=...)`. Reuse the existing `SpectrumResult` schema;
   add `"spectrum-jax"` to the allowed `backend_name` set; add
   the `device` kwarg with default `None`. Tests: CPU-JAX path
   runs on CI and asserts numeric equivalence against
   `backend_name="spectrum-scipy"` at a small problem
   (`N = 2, n_c = 4`); GPU-device tests are import-gated and
   skipped on CI. Cost: ~1 dispatch.
2. **Dispatch BBB — GPU envelope benchmark (tool extension).**
   Extend `tools/run_benchmark_spectrum_envelope_jax.py` with a
   `--device` flag that routes the existing per-point logic
   through JAX's device API. Subprocess-isolated wall-clock
   capture is unchanged. VRAM capture is **out of scope for the
   library proper** — the benchmark shells out to `nvidia-smi
   --query-gpu=memory.used --format=csv,noheader,nounits` at
   start / peak / end of each grid point and records the three
   values. This is the only portable-across-jaxlib-versions
   source; JAX's `Device.memory_stats()` exists but its
   dict-shape varies with jaxlib minor versions and is not a
   stable contract to build a benchmark tool on top of. Metal
   machines get an honest `"device_memory_source": "unavailable"`
   field in the report (no equivalent tool) and wall-clock
   numbers only. Produces `benchmarks/data/gpu/spectrum_envelope/
   report.json` and a plot overlaying GPU wall-clock against the
   CPU baseline. Cost: ~1 dispatch.
3. **Dispatch BBC — AAG re-scoping from GPU wall-clock data.**
   With BBB's numbers in hand, refine the AAG gate-status
   language in `docs/benchmarks.md`. **Not an "is AAG still
   needed?" decision** — the revised §1 premise and the
   benchmarks.md "AAG unambiguously valuable at N = 5 `n_c = 6`"
   text already agree that fully-converged N ≥ 5 stays outside
   any single-GPU dense envelope. Instead, BBC records: (a)
   the measured GPU vs CPU wall-clock ratio across the AAH
   grid, (b) the `(N, n_c, device)` point at which wall-clock
   becomes the binding constraint on each tier, and (c) the
   updated `(N, n_c)` boundary at which AAG's interior-window
   path becomes the only option — unchanged from
   benchmarks.md today for CPU, newly measured for
   consumer-GPU and lab-GPU tiers. Cost: ~0.5 dispatch
   (documentation + measurement record).
4. **Dispatch BBD — consumer-facing benchmark doc refresh.**
   Update `docs/benchmarks.md` to include the "Which envelope
   am I in?" decision tree: CPU dense (16 / 32 / 64 / 128 GB
   tiers) vs. GPU dense (16 / 24 / 40 / 80 GB VRAM tiers)
   vs. AAG interior-window (any tier, `dim ≥ 35 000`). Cost:
   ~0.5 dispatch.
5. **Dispatch BBE — tutorial addendum.** A supplementary
   page under `docs/tutorials/13_reproducing_clos_2016.md`
   (or a new Tutorial 14) showing the reproduction on GPU
   for users who have it. Cost: ~0.5 dispatch.

**Total:** ~3.5 dispatches if β / γ are deferred,
expandable by 2–3 dispatches if β follows immediately
(time-evolution GPU + its benchmark + its integration into
the cross-backend comparison at scale).

---

## 8. Risks and their mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **GPU drift between dev machines** (same code, different `jaxlib` CUDA compile, different numbers) | Medium | Medium | Artefact-based regression (§3 B.3); declare tolerance per backend in `docs/benchmarks.md`; do not test GPU numerics for bit-exact equality against CPU. |
| **CUDA / jaxlib upstream churn** (install breaks on minor-version bumps) | High | Low | Pin `jax[cuda12]` to a tested range in `pyproject.toml [project.optional-dependencies].gpu`; bump explicitly when the benchmark re-runs cleanly. |
| **No lab GPU available** when the dispatch is scheduled | Medium | High (blocks the work) | Staging §7 assumes a named reference machine; resolve Q1 before BBA starts. |
| **Small-dim GPU launch latency** overwhelms spectrum wins | Low | Low | The AAH envelope already shows dense CPU dominating small-dim costs; GPU win is pitched at `dim ≥ 4 000`. Small-dim comparison will show CPU winning and is recorded honestly. |
| **Autograd promises creep in** (users treat "JAX-backed" as "differentiable") | Medium | Medium | `backend_name = "spectrum-jax"` does not imply autograd; autograd status (when/if shipped) is recorded as a provenance tag, not by minting a new backend identity. Documentation distinguishes "runs on GPU" from "differentiable through the solver". |
| **Metal path silently diverges from CUDA** | Low | Low | §3 A.3 says Metal is advisory-only; no correctness claim; user bug reports get routed to "reproduce on CUDA first". |
| **CI blindness** grows as GPU surface expands | Medium | Medium | Dispatch-level CHANGELOG entries must explicitly note "GPU tests are import-skipped on CI; regenerate locally via X." |

---

## 9. Non-decisions (explicitly out of scope for this note)

- Cutting a `v0.5.0` release simultaneously with BBA. Release
  cadence is a separate decision from capability design; if
  BBA ships into `[Unreleased]` first and `v0.5.0` bundles
  later-dispatch work with it, that's fine.
- Renaming `backend="jax"` or minting a new GPU-specific
  identity such as `backend="jax-cuda"` /
  `backend_name="jax-dynamiqs-cuda"`. The shipped contract
  (`backend="jax"` → `backend_name="jax-dynamiqs"`, regardless
  of device) is preserved; device is provenance, not identity.
  Callers that need to know whether a result came from a GPU
  read `provenance_tags`, not `backend_name`.
- Deprecating the CPU JAX path. It's the autograd-forward
  scaffolding and the one CI actually exercises.

---

## 10. Recommendation (tentative)

**Design α, staged per §7.** Start with the spectrum path
because it is the narrowest honest answer to the "can the
library usefully ship GPU today?" question. The AAH envelope
gives a concrete wall-clock target (N = 4, `n_c = 7`, 7 min
on CPU → measured GPU seconds on the reference hardware),
and BBC lands the per-tier wall-clock-vs-reach record that
the AAG gate-status text currently flags as the open
measurement.

**Defer β and γ until α ships and BBC's wall-clock record
exists in repo.** The Phase 2 CPU null results are warning
enough against opening the time-evolution surface before a
first GPU data point lands — GPU time-evolution might pay off
at `dim ≥ 1 000` and lose at `dim ≤ 100`, and we don't want to
re-litigate that with hand-waving.

**Hardware baseline (conditional recommendation, pending
Q1):** an RTX 40-series consumer GPU with ≥ 16 GB VRAM.
Consumer-tier matches the "on a typical PC" framing that
AAH committed to; lab A10 / L4 cards can be added as a
second tier if and when a lab user hits the envelope.

**Classification at ship-time.** The new
`backend_name = "spectrum-jax"` and the `device` kwarg on
`solve_spectrum` become Coastline as soon as BBA ships. The
scaling observations from BBB are Sail (future optimization
work can move them). The "AAG deferred or scheduled?"
decision from BBC is Coastline — it refines the gate
status last touched on the Clos 2016 workplan.

---

## 11. What happens next

This document sits at `docs/gpu-dispatch-design.md` as a
deliberation artefact, following the repository convention
for `docs/phase-2-jax-backend-design.md` /
`docs/phase-2-jax-time-dep-design.md`: not linked into
`mkdocs.yml`, retained in `docs/` for historical continuity.

When the §6 questions are answered, this document becomes
either:

- **Superseded by a Dispatch BBA kickoff note** annotated
  with a "Decisions recorded YYYY-MM-DD" banner and retained
  for historical continuity; or
- **Retired** if GPU work is postponed beyond the current
  cycle, archived per the same precedent that
  `docs/workplan-clos-2016-integration.md` followed when it
  transitioned to shipped-record status.

---

## 12. Open-system reframing (2026-04-28 addendum)

This addendum extends §1's framing — which is implicitly
**closed-system** (dense `eigh`, unitary `sesolve` /
time-dependent builders) — to the dissipative case. It does
**not** change the §10 recommendation. Design α (spectrum
first, BBA → BBE per §7) remains the right narrow first move.
What changes is the *post-α* roadmap: where §5 staged β
("α plus GPU time-evolution") as the natural follow-up under
the implicit assumption that time-evolution means
density-matrix `mesolve`, the open-system landscape inverts
that priority.

### 12.1 Why the framing matters

§1's null-result evidence (Dispatch YY / β.4.5; the
post-AAH `eigh` comparison) is for **closed-system**
workloads. Dissipative dynamics changes the constraint
structure — and consequently the GPU value proposition —
along two axes that the closed-system measurements do not
exercise:

1. **Memory.** Density-matrix evolution stores `dim²` complex
   amplitudes; state-vector evolution stores `dim`. For the
   trapped-ion regime the library ships into
   (single-ion `dim ≤ 100`, two-ion full-Lamb–Dicke
   `dim ≲ 10³`), the closed/open memory ratio is benign on
   CPU but bites earlier on GPU because VRAM tiers are
   smaller than the corresponding RAM tiers. Published
   QuTiP 5 + A100 measurements illustrate the same regime
   shift in a different setting: a closed-system spin-chain
   that fits 22 spins on the GPU collapses to 11 spins
   open-system before saturation.
2. **Parallelism.** `mcsolve` / quantum-trajectories Monte
   Carlo (QTMC) reverts to state-vector memory scaling per
   trajectory and is **embarrassingly parallel** across
   trajectories — the canonical workload shape for which GPUs
   were designed. The library's existing `systematics` layer
   (jitter / drift / SPAM scans, `perturb_carrier_rabi`,
   `solve_ensemble`) already produces parameter ensembles;
   trajectory ensembles fit the same architectural slot.

| Method                    | Memory / instance | Parallel structure       | GPU suitability at ion-trap scales       |
|---------------------------|-------------------|--------------------------|------------------------------------------|
| `mesolve` (density matrix)| `dim²`            | Single dense ODE         | Decreased vs. closed-system. Memory wall hits earlier. |
| `mcsolve` / QTMC          | `dim` per trajectory | Independent trajectories | High — the only path with consistent literature wins at our scales. |

The implication for §5 is concrete: **β as currently scoped
(GPU `mesolve` / `sesolve`) is the wrong place to stage the
post-α work.** Density-matrix GPU dispatch saturates the
VRAM envelope before it amortises launch latency at our
typical `dim`. Trajectory-ensemble GPU dispatch — not in §5
at all — is the path the literature consistently shows
winning at the scales we ship to.

### 12.2 Strategic options under the open-system frame

Three branches, ordered by effort and confidence-of-payoff:

| Branch | Path                                  | Effort | GPU payoff at our scales | Recommendation |
|--------|---------------------------------------|--------|--------------------------|----------------|
| A      | Full `mesolve` on GPU (density matrix) | High   | Low. VRAM saturates before speedup; closed-system A100 data already shows the shrunk envelope. | **Defer.** Negative-return at current ion-trap dimensions. |
| B      | `mcsolve` / QTMC trajectory ensembles   | Medium | High. Embarrassingly parallel, state-vector memory per trajectory, aligns with the existing `systematics` ensemble surface. | **Pursue as Phase 3 feasibility branch.** Replaces β as the natural post-α direction. |
| C      | Batched parameter sweeps on JAX (`vmap`) | Low    | Moderate. Already half-built — `solve_ensemble` is the structural precedent; `vmap` over jitter / detuning grids is incremental. | **Near-term, post-BBA.** Lowest-cost path to a measurable open-system GPU data point. |

This re-orders the post-α roadmap from "β (GPU
time-evolution `mesolve`) → γ (autograd seam)" to
"C (batched sweeps via `vmap`) → B (trajectory ensembles) →
γ (autograd seam)". A stays deferred indefinitely at
ion-trap scales; if a future user case lands at
`dim ≳ 10⁴` open-system, the deferral is re-litigated then.

### 12.3 Identified benchmark gap

**We have not found a published cross-backend benchmark**
for trapped-ion open-system dynamics that compares full
`mesolve` against QTMC trajectory ensembles on a common
hardware tier. The GPU benchmarks we have surveyed
(Dynamiqs paper, QuantumToolbox.jl release notes, QuTiP 5
A100 measurements, CUDA-Q examples) use generic spin
chains or harmonic-oscillator test cases. The
spin ⊗ motion tensor structure of trapped-ion systems —
with its sparse, mode-selective jump operators and
strongly anisotropic dim contributions from spin (small)
versus motional (large) factors — is not represented in
the literature surveyed. A reader who knows of one should
flag it; the gap claim is held provisionally.

**Implication.** If branch B or C is pursued, the resulting
artefact closes a gap in the broader benchmark literature,
not just an internal performance question. A targeted
trapped-ion-specific cross-backend record (CPU `mcsolve`
vs GPU trajectory ensembles, on representative
sideband / MS-gate / stroboscopic builders with realistic
heating + dephasing jump operators) is the
publication-shaped output of that branch.

### 12.4 Architectural alignment with the shipped surface

Branch B does not require new top-level abstractions. The
jump-operator surface is the only genuinely new public API:

- **Jump operators.** Spin dephasing (`σ_z`-coupled),
  motional heating (`a + a†` on the heating mode), motional
  decoherence (`a†a`) and SPAM-style Lindblad channels are
  the four families a trapped-ion `mcsolve` interface must
  expose. The `Lindbladian` slot already contemplated in
  `phase-1-architecture.md` §"Stochastic unravellings" is
  the natural anchor.
- **Trajectory aggregator.** `StochasticTrajectoryResult`
  alongside the shipped `TrajectoryResult`, also
  pre-flagged in `phase-1-architecture.md` §"Non-goals for
  Phase 1." The aggregator owns trajectory-count provenance,
  RNG seed-stream metadata, and observables-with-error-bars
  semantics.
- **Backend dispatch.** Re-uses the §4.2 device-as-provenance
  contract from `backend_name = "jax-dynamiqs"`. No new
  `backend_name` is minted for the trajectory path beyond
  whatever the underlying integrator (QuTiP `mcsolve`,
  Dynamiqs `mcsolve`-equivalent) requires.
- **RNG stream management.** The one genuinely new
  cross-backend convention question: how seeds compose
  across (a) trajectory index within an ensemble and
  (b) parameter index within a `systematics` sweep over
  trajectory ensembles. Resolving this is the
  Phase 3 feasibility study's primary Coastline output.

### 12.5 Action

The §6 open questions remain answer-pending; this addendum
adds a sixth:

6. **Post-α direction.** Does the maintainer accept the
   re-ordering to `C → B → γ` (batched sweeps → trajectory
   ensembles → autograd seam), with full `mesolve` on GPU
   deferred indefinitely? The §5 comparison table predates
   this reframing and would be regenerated in the kickoff
   note that supersedes this design.

If accepted, the next deliberation artefact is a
**Phase 3 feasibility note** scoping branch B:
jump-operator definitions for the four canonical trapped-ion
dissipation channels, trajectory aggregator and RNG
stream-composition convention, and the cross-backend
benchmark plan (`qutip.mcsolve` CPU vs `jax`-based
trajectory ensembles with explicit GPU/CPU device split).
The feasibility note is **not** part of the current GPU
dispatch's §7 staging; it is a downstream successor that
opens once BBA–BBE close and BBC's wall-clock record is in
repo.

**Status:** addendum recorded; Phase 3 feasibility study
scoping awaits maintainer review of §12.5 question 6.
