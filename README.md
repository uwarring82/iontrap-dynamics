# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is being built around a
hard separation between physics, apparatus, and observation layers, with QuTiP
as the Phase 0 reference backend.

## Status

Phase 0 is complete and Phase 1 is underway. The configuration layer and the
first Hamiltonian builder are in place; end-to-end dynamics works
(`DriveConfig` → `carrier_hamiltonian` → `qutip.mesolve` → expected π-pulse
flip).

Phase 0 artefacts (all done):

- Public conventions locked in `CONVENTIONS.md` v0.1-draft.
- Three-layer regression harness populated: migration (5 / 5 scenarios with
  legacy `qc.py`-generated references, bit-identical across three runs),
  analytic (6 closed-form formulas), invariant (9 checks).
- Cache-integrity contract + the corresponding tests.
- CI with ruff, mypy strict, pytest, pa11y accessibility report.

Today the importable code surface covers:

**Foundation (Phase 0)**

- `iontrap_dynamics.exceptions` — canonical exception hierarchy
  (`IonTrapError`, `ConventionError`, `BackendError`, `IntegrityError`,
  `ConvergenceError`)
- `iontrap_dynamics.results` — frozen `TrajectoryResult` schema with
  storage-mode consistency enforcement
- `iontrap_dynamics.cache` — hash-verified `.npz` + JSON persistence
- `iontrap_dynamics.conventions` — `CONVENTION_VERSION` marker
- `iontrap_dynamics.invariants` — density-matrix / state-vector validators
- `iontrap_dynamics.analytic` — closed-form reference formulas (carrier
  Rabi, sideband rates, Lamb–Dicke parameter, coherent-state occupation)

**Configuration layer (Phase 1)**

- `iontrap_dynamics.operators` — single-ion Pauli set in the atomic-physics
  convention (`sigma_z_ion`, `sigma_plus_ion`, ...; see CONVENTIONS.md §3)
- `iontrap_dynamics.species` — `IonSpecies`, `Transition`, `TransitionType`
  and factories for ²⁵Mg⁺, ⁴⁰Ca⁺, ⁴³Ca⁺
- `iontrap_dynamics.drives` — `DriveConfig` (wavevector, Rabi, detuning, ...)
- `iontrap_dynamics.modes` — `ModeConfig` with CONVENTIONS.md §11
  normalisation enforced at construction
- `iontrap_dynamics.system` — `IonSystem` composition with cross-validation
- `iontrap_dynamics.hilbert` — `HilbertSpace` implementing the §2 tensor
  ordering, operator embedding helpers, motional primitives (a, a†, n̂)
- `iontrap_dynamics.states` — `ground_state` ket + `compose_density`
  general composition

**Dynamics (Phase 1, full builder family)**

The public Hamiltonian surface is symmetric across four families:

|                | exact (time-indep. Qobj)   | detuned (list format)                |
|----------------|----------------------------|--------------------------------------|
| carrier        | `carrier_hamiltonian`      | `detuned_carrier_hamiltonian`        |
| red sideband   | `red_sideband_hamiltonian` | `detuned_red_sideband_hamiltonian`   |
| blue sideband  | `blue_sideband_hamiltonian`| `detuned_blue_sideband_hamiltonian`  |
| MS gate        | `ms_gate_hamiltonian`      | `detuned_ms_gate_hamiltonian`        |

Plus `modulated_carrier_hamiltonian` (time-dependent envelope primitive),
`two_ion_{red,blue}_sideband_hamiltonian` (single-tone shared-mode), and a
`full_lamb_dicke: bool` flag on the sideband builders (Wineland–Itano
all-orders operator via matrix exponentiation). Solver entry point:
`iontrap_dynamics.sequences.solve(...)` — accepts both Qobj and QuTiP
list-format Hamiltonians, enforces the §13 Fock-saturation ladder on
every call.

Test suite: **497 passed, 3 skipped**. Skips are migration-tier builder-
comparison slots with probe-informed blockers (see `CHANGELOG.md`).

Docs site scaffold:

- `mkdocs.yml` configures the public-facing documentation build
- `docs/index.md` — welcome page
- `docs/getting-started.md` — install + first run
- `docs/framework.md` — high-level design rules
- `docs/conventions.md` — rendered live from root `CONVENTIONS.md`
  (single source of truth via `pymdownx.snippets`)
- `docs/phase-1-architecture.md` — concrete public-API reference
  (module map, per-module surface, extension points, non-goals)
- `docs/boundary-decision-tree.md` — contributor scope rules (closes D8)
- `docs/stylesheets/tokens.css` — vendored from `threehouse-plus-ec/cd-rules`

The authoritative project documents are:

- `WORKPLAN_v0.3.md` for scope, architecture, milestones, and governance
- `CONVENTIONS.md` for physical, numerical, and notational rules
- `LICENCE` for the repository split-licence declaration

## Scope

Planned capabilities include:

- Unitary and dissipative dynamics for coupled spin-motion systems
- Standard ion-trap Hamiltonians: carrier, sideband, Mølmer-Sørensen,
  parametric modulation, and stroboscopic drives
- Standard state preparations and observables for spins and motional modes
- Backend-agnostic architecture, with QuTiP first and JAX/Dynamiqs later

Explicitly out of scope:

- Trap geometry simulation
- Molecular dynamics of ion crystals
- Pulse-sequence compilers and hardware-control stacks
- Electromagnetic field modelling

## Development

Python 3.11+ is required.

Editable install:

```sh
python -m pip install -e ".[dev]"
```

Optional groups:

- `.[docs]` for documentation tooling
- `.[plot]` for plotting helpers used by examples and tutorials
- `.[jax]` for the future JAX/Dynamiqs backend track

Example:

```sh
python -m pip install -e ".[dev,docs]"
```

## Repository Layout

- `src/iontrap_dynamics/` - Python package
- `tools/` - maintenance scripts for asset fetch and checksum verification
- `assets/` - design assets consumed from `threehouse-plus-ec/cd-rules`
- `WORKPLAN_v0.3.md` - project workplan
- `CONVENTIONS.md` - binding conventions document

## Licence

The distributable Python package is MIT-licensed. The repository as a whole
uses a split-licence architecture declared in `LICENCE`; design documents and
tutorial material do not all share the same terms.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
