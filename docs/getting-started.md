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

## Run your first simulation

The whole library is one five-box pipeline — a *description of an experiment* in, a
*simulated measurement* out:

--8<-- "docs/_snippets/pipeline.txt"

Here it is end to end: drive a single ²⁵Mg⁺ ion on resonance and watch its spin flop from
down (`⟨σ_z⟩ = −1`) to up (`+1`) and back. It needs only the base install.

```python
import numpy as np
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# what's in the lab: one Mg-25 ion, one axial mode, one resonant laser
mode = ModeConfig(label="axial", frequency_rad_s=2 * np.pi * 1.5e6,
                  eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
drive = DriveConfig(k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
                    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6, phase_rad=0.0)

# a basis, the drive Hamiltonian, and one Rabi period of evolution from |↓, 0⟩
hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})
result = solve(
    hilbert=hilbert,
    hamiltonian=carrier_hamiltonian(hilbert, drive, ion_index=0),
    initial_state=qutip.tensor(spin_down(), qutip.basis(3, 0)),
    times=np.linspace(0.0, 2 * np.pi / drive.carrier_rabi_frequency_rad_s, 50),
    observables=[spin_z(hilbert, 0)],
)
sz = np.array(result.expectations["sigma_z_0"])
print(f"⟨σ_z⟩:  start {sz[0]:+.2f}  →  half-way {sz[len(sz) // 2]:+.2f}  →  end {sz[-1]:+.2f}")
# → ⟨σ_z⟩:  start -1.00  →  half-way +1.00  →  end -1.00   (one Rabi cycle)
```

[Tutorial 0](tutorials/00_mental_model.md) unpacks this line by line;
[Tutorial 1](tutorials/01_first_rabi_readout.md) adds a finite-shot detector and
confidence intervals.

## The result object, by hand

Every solve returns a frozen `TrajectoryResult`. You rarely build one yourself, but it
shows the stable output contract — metadata (convention version, backend, storage mode),
the time grid, and the per-observable expectation arrays:

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
