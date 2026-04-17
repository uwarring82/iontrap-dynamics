# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is being built around a
hard separation between physics, apparatus, and observation layers, with QuTiP
as the Phase 0 reference backend.

## Status

This repository is in Phase 0 scaffold work.

- Public conventions are locked in `CONVENTIONS.md` v0.1-draft.
- The package scaffold, result schema, and cache-integrity contract are in
  place; invariant and analytic regression tiers are populated.
- Core solver builders, observables, and worked examples follow next; the docs
  landing site scaffold is now in place.

Today the importable code surface covers:

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

Test suite: 80 tests pass (20 schema, 24 cache, 9 invariants, 27 analytic).

Docs site scaffold:

- `mkdocs.yml` configures the public-facing documentation build
- `docs/index.md` provides the welcome page
- `docs/getting-started.md` and `docs/framework.md` give the first navigation
  layer

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
