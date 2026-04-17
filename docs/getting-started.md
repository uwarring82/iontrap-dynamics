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

The implemented surface is still Phase 0 infrastructure rather than full solver
coverage, but the foundational contracts are live:

- `iontrap_dynamics.exceptions`
- `iontrap_dynamics.conventions`
- `iontrap_dynamics.results`
- `iontrap_dynamics.cache`
- `iontrap_dynamics.analytic`
- `iontrap_dynamics.invariants`

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
