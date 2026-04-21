# Tutorial 7 — Hash-verified cache round-trip

**Goal.** Every `tools/run_benchmark_*.py` and
`tools/run_demo_*.py` in this repo writes a
`benchmarks/data/<scenario>/` bundle that another process — or
you, tomorrow — can reload without re-solving. The persistence
layer is opinionated: no pickle, strict hash verification at load
time, and a refusal to silently return a cache whose inputs don't
match what you passed. This tutorial walks through the three
functions that make that work —
[`compute_request_hash`](../phase-1-architecture.md),
`save_trajectory`, and `load_trajectory` — applied to the
Tutorial 2 red-sideband scenario, and shows what happens at each
way the cache can diverge from the request that's asking for it.

By the end you will have:

1. Generated a canonical request hash from a parameter dict and
   stamped it onto a `TrajectoryResult` through `solve()`.
2. Written the result to a cache directory with `save_trajectory`
   and inspected the two files that get produced (`manifest.json`
   + `arrays.npz`).
3. Loaded it back through `load_trajectory` and verified that
   every expectation array round-trips bit-identical.
4. Triggered an `IntegrityError` on four independent mismatch
   modes: wrong hash, missing file, tampered manifest, tampered
   npz array.
5. Understood the `StorageMode.OMITTED`-only scope and why the
   library refuses to pickle quantum states.

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) for the
scenario used throughout. [Tutorial 6](06_fock_truncation.md) is
a useful background for how `result.warnings` survives the
round-trip (cached results preserve their warnings tuple).

---

## The request-hash contract in one picture

```
   caller's parameter dict
   { "scenario": ..., "N_fock": ..., ... }
             │
             │  compute_request_hash
             ▼
     SHA-256 hex (64 chars)
             │
             ├──────► solve(..., request_hash=hex)
             │           │
             │           └──► TrajectoryResult
             │                 ├── metadata.request_hash = hex
             │                 └── save_trajectory(result, dir)
             │                       │
             │                       ├── manifest.json ── embeds the hex
             │                       └── arrays.npz   ── times + expectations
             │
             ▼
         load_trajectory(dir, expected_request_hash=hex)
                 │
                 ├── manifest hex == expected → restore
                 └── manifest hex ≠ expected  → IntegrityError
```

The hash is the **only** acceptance criterion. The manifest
carries the parameter-dependent hash and the load path carries
what the caller *thinks* those parameters are. Any drift in
either triggers a refusal — no "try anyway and see if the
numbers look close" fallback.

## Step 1 — Set up the scenario and hash its parameters

Pure Tutorial 2 — RSB flop from `|↓, 1⟩`:

```python
import numpy as np
import qutip

from iontrap_dynamics.analytic import lamb_dicke_parameter
from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

N_FOCK = 30

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": N_FOCK})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
    phase_rad=0.0,
)
hamiltonian = red_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)

eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
sideband_rabi_rate = drive.carrier_rabi_frequency_rad_s * eta
flop_period = 2 * np.pi / sideband_rabi_rate

parameters = {
    "scenario": "tutorial_07_cache_roundtrip",
    "N_fock": N_FOCK,
    "n_steps": 200,
    "rabi_over_2pi_MHz": 0.1,
    "mode_freq_over_2pi_MHz": 1.5,
    "laser_wavelength_nm": 280.0,
    "initial_state": "|↓, n=1⟩",
    "duration_units": "two_flop_periods",
}
request_hash = compute_request_hash(parameters)
print(f"request_hash = {request_hash}")
# request_hash = d3c81eef… (64 hex chars, deterministic for this exact dict)
```

!!! note "What belongs in the parameter dict"

    The rule of thumb: any input whose change would produce a
    different `TrajectoryResult` belongs in the dict. The
    Hamiltonian builder choice, the Fock truncation, the initial
    state, the drive parameters — all material. But *derived*
    quantities (η, gate time, the actual ω_mode in rad/s) do not
    belong: they're bookkeeping on top of the primary inputs, and
    including them just lets two equivalent parameter dicts hash
    to different values (a false-negative cache miss). Derived
    numbers go in the demo report (your `demo_report.json`), not
    in the hash payload.

    `compute_request_hash` requires JSON-serialisable values
    only — numpy scalars and `Path` objects raise `TypeError`.
    Convert to primitives (`float`, `str`, `int`) before hashing.

## Step 2 — Solve and save

`solve(..., request_hash=hex)` stamps the hash onto
`result.metadata.request_hash`. `save_trajectory` then reads it
back off the result and embeds it into the manifest — so the
hash baked into the bundle is derived from the result, not
re-passed:

```python
from pathlib import Path
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.cache import save_trajectory

psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, 1))
times = np.linspace(0.0, 2 * flop_period, 200)

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    observables=[spin_z(hilbert, 0), number(hilbert, "axial")],
    request_hash=request_hash,
    storage_mode=StorageMode.OMITTED,   # ← cache requirement
)

cache_dir = Path("/tmp/iontrap_cache_tutorial7")
save_trajectory(result, cache_dir, overwrite=True)
```

!!! warning "`StorageMode.OMITTED` is the only supported cache input"

    Quantum states are backend-specific (QuTiP's `Qobj` carries a
    backend-tagged numeric representation), and this library
    refuses to pickle them — every persisted object must
    round-trip through JSON + `np.savez`, which the `Qobj`
    representation doesn't. So `save_trajectory` raises
    `ConventionError` on `EAGER` or `LAZY` results. The
    expectation values *are* cacheable (they're plain
    `np.ndarray`), just not the underlying state trajectory.
    Phase 2+ will add backend-annotated state serialisation if
    there's demand.

## Step 3 — Inspect the bundle

The cache dir has exactly two files. `manifest.json` is
human-readable:

```bash
$ cat /tmp/iontrap_cache_tutorial7/manifest.json
```

```json
{
  "cache_format_version": 1,
  "expectation_labels": ["n_axial", "sigma_z_0"],
  "metadata": {
    "backend_name": "qutip-sesolve",
    "backend_version": "5.2.3",
    "convention_version": "0.1-draft",
    "fock_truncations": {"axial": 30},
    "provenance_tags": [],
    "request_hash": "d3c81eef…",
    "storage_mode": "omitted"
  },
  "warnings": []
}
```

`arrays.npz` is the numeric payload — `times` plus one array per
expectation, prefixed `expectation__<label>`:

```python
with np.load(cache_dir / "arrays.npz") as npz:
    print(sorted(npz.files))
# ['expectation__n_axial', 'expectation__sigma_z_0', 'times']
```

Bundle shape is strict by design — `load_trajectory` checks that
every label in `expectation_labels` has a matching npz array and
rejects both missing and extra arrays.

!!! tip "Adding demo-specific sidecars"

    The repo's `tools/run_demo_*.py` scripts write two *extra*
    files alongside the canonical bundle: `demo_report.json`
    (narrative report, analytic overlays, environment capture)
    and `plot.png` (rendered figure). These live inside the same
    directory but **outside** the cache contract — `load_trajectory`
    ignores anything that isn't `manifest.json` or `arrays.npz`,
    so downstream consumers can store whatever sidecars they
    want without breaking round-trip. See `run_benchmark_sideband.py`
    for the pattern.

## Step 4 — Load and verify

`load_trajectory` takes the expected hash and refuses to return
the cache if it doesn't match the manifest:

```python
from iontrap_dynamics.cache import load_trajectory

restored = load_trajectory(cache_dir, expected_request_hash=request_hash)

# Every numeric array round-trips bit-identical.
assert np.array_equal(restored.times, result.times)
for label in result.expectations:
    assert np.array_equal(
        restored.expectations[label], result.expectations[label]
    ), f"mismatch on {label}"

# Metadata restores too.
assert restored.metadata.request_hash == request_hash
assert restored.metadata.fock_truncations == {"axial": 30}
assert restored.metadata.storage_mode.value == "omitted"

# Warnings tuple survives the round-trip (empty here, but would
# carry through for a scenario that raised FockConvergenceWarning).
assert restored.warnings == ()
```

## Step 5 — The four failure modes

The `IntegrityError` ladder is where the value of hash
verification actually shows up. Each of these scenarios is a
separate way a cache can go wrong; each produces a distinct
diagnostic.

### Mismatched expected hash

The most common failure — a caller passes the wrong hash because
they've drifted their parameter dict since the cache was written:

```python
from iontrap_dynamics.exceptions import IntegrityError

try:
    load_trajectory(
        cache_dir,
        expected_request_hash="0" * 64,   # ← deliberately wrong
    )
except IntegrityError as exc:
    print(exc)
# request_hash mismatch: expected '000…', cache recorded 'd3c81eef…'.
```

### Missing files

Delete one of the two canonical files and the loader will
complain about the one it can't find — not the one that's still
there:

```python
(cache_dir / "arrays.npz").unlink()
try:
    load_trajectory(cache_dir, expected_request_hash=request_hash)
except IntegrityError as exc:
    print(exc)
# cache arrays file missing: /tmp/iontrap_cache_tutorial7/arrays.npz
```

`manifest.json` missing produces the symmetric error. (Restore
the file before the next failure demo: `save_trajectory(result,
cache_dir, overwrite=True)`.)

### Tampered manifest (hash changed after write)

Edit the hash field in the manifest and re-save it, *without*
re-running the solve. Now the manifest disagrees with whatever
the caller expects:

```python
import json

# Corrupt the manifest's recorded hash
manifest_path = cache_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["metadata"]["request_hash"] = "f" * 64
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

try:
    load_trajectory(cache_dir, expected_request_hash=request_hash)
except IntegrityError as exc:
    print(exc)
# request_hash mismatch: expected 'd3c81eef…', cache recorded 'fff…'.
```

This is the same `IntegrityError` class as the first failure —
the distinction between "wrong hash in caller" and "wrong hash in
manifest" isn't surfaced, because from the library's perspective
they're the same bug: the request doesn't match the cache.

### Extra or missing npz arrays

If an npz has an array not declared in `expectation_labels` (or
vice versa), the loader rejects it:

```python
# Re-save clean, then tamper with arrays.npz
save_trajectory(result, cache_dir, overwrite=True)
with np.load(cache_dir / "arrays.npz") as npz:
    tampered = {k: npz[k] for k in npz.files}
tampered["expectation__unexpected_label"] = np.zeros(10)
np.savez(cache_dir / "arrays.npz", **tampered)

try:
    load_trajectory(cache_dir, expected_request_hash=request_hash)
except IntegrityError as exc:
    print(exc)
# arrays.npz has unexpected keys: ['expectation__unexpected_label']
```

The reason this is an error rather than a warning: an extra
array means either the manifest and the npz were written at
different times (a build-system race condition), or the npz has
been tampered with since. Either way, the cache is not what the
manifest says it is — and the library's contract is to refuse
rather than silently pass a potentially-inconsistent result.

## When to actually use the cache

Three common patterns:

1. **Skip recomputation in a notebook loop.** Attempt a load;
   if `IntegrityError` (or `FileNotFoundError`), run the solve
   and save. Pattern:
   ```python
   try:
       result = load_trajectory(cache_dir, expected_request_hash=request_hash)
   except (IntegrityError, FileNotFoundError):
       result = solve(...)
       save_trajectory(result, cache_dir, overwrite=True)
   ```

2. **Commit reference results alongside the code.** `tools/run_demo_*.py`
   writes bundles under `benchmarks/data/<scenario>/`; CI diff-checks
   them against what the current code regenerates. Catch drift in
   physics builders the moment a PR lands. Bundles are git-tracked
   because they're ~10 KB for typical scenarios.

3. **Share across processes / machines.** Same rules apply; the
   hash is a reproducibility contract. A collaborator loading
   your bundle and getting `IntegrityError` has a precisely-named
   reason to ask you what changed.

!!! warning "Don't commit caches for parameter sweeps"

    Bundling one cache per trial in a 1000-trial jitter ensemble
    means 1000 directories in the repo — acceptable for archival
    reference results, noise for exploratory work. Cache to
    `/tmp/` for sweeps and use the bundle layout only for
    scenarios that get committed alongside the code.

## Where to next

- [Tutorial 2](02_red_sideband_fock1.md) — the scenario this
  tutorial caches.
- [Tutorial 6](06_fock_truncation.md) — `result.warnings` that
  survives the cache round-trip.
- [Phase 1 Architecture](../phase-1-architecture.md) —
  `cache.save_trajectory` / `load_trajectory` /
  `compute_request_hash` API reference and the `TrajectoryResult`
  / `ResultMetadata` / `StorageMode` schema this all runs on.
- [`src/iontrap_dynamics/cache.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/cache.py)
  — reference implementation; every `IntegrityError` path is
  exercised by the unit-test suite under
  `tests/unit/test_cache.py`.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
