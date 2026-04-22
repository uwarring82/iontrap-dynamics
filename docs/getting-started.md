# Getting Started

## Install

`iontrap-dynamics` currently targets Python 3.11+.

```sh
python -m pip install -e ".[dev]"
```

Documentation tooling:

```sh
python -m pip install -e ".[dev,docs]"
```

## What you can import today

As of `v0.2.0` + the Phase 2 JAX-backend track on `main`, the library
ships the full Phase 0 + Phase 1 surface plus the JAX backend:

- `iontrap_dynamics.exceptions`, `.conventions`, `.results`, `.cache`,
  `.analytic`, `.invariants` (Phase 0 foundations)
- `.species`, `.drives`, `.modes`, `.system`, `.hilbert`, `.states`,
  `.operators`, `.hamiltonians`, `.observables`, `.sequences`
  (Phase 1 physics)
- `.measurement`, `.systematics`, `.entanglement` (Phase 1 v0.2
  layers: finite-shot sampling + apparatus systematics + entanglement
  observables)
- `.backends.jax` (Phase 2 on `main`: opt in via `backend="jax"` on
  `solve` and on the time-dependent Hamiltonian builders; install
  with the `[jax]` extras)

## Example

```python
from iontrap_dynamics import CONVENTION_VERSION, StorageMode
from iontrap_dynamics.results import ResultMetadata, TrajectoryResult
import numpy as np

metadata = ResultMetadata(
    convention_version=CONVENTION_VERSION,
    request_hash="0" * 64,
    backend_name="demo",
    backend_version="0.0.0",
    storage_mode=StorageMode.OMITTED,
)

result = TrajectoryResult(
    metadata=metadata,
    times=np.linspace(0.0, 1.0e-6, 5),
    expectations={"sigma_z": np.array([-1.0, -0.5, 0.0, 0.5, 1.0])},
)
```

## Read in this order

1. `CONVENTIONS.md` for units, basis ordering, detuning sign, Lamb-Dicke
   definition, and failure policy.
2. `WORKPLAN_v0.3.md` for scope, architecture, and Phase 0 milestones.
3. `src/iontrap_dynamics/results.py` and `cache.py` for the stable result and
   persistence contracts.

## Build the docs site locally

```sh
mkdocs build --strict
```

The site configuration lives in `mkdocs.yml`, and custom presentation styles
live in `docs/stylesheets/extra.css`.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
