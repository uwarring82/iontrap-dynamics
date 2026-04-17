# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is being built around a
hard separation between physics, apparatus, and observation layers, with QuTiP
as the Phase 0 reference backend.

## Status

This repository is in Phase 0 scaffold work.

- Public conventions are being locked before solver code lands.
- The package metadata and exception hierarchy are in place.
- Core dynamics modules, tests, examples, and docs site scaffolding follow next.

Today the importable code surface is intentionally small:

- `iontrap_dynamics.exceptions` provides the canonical exception hierarchy named
  in `CONVENTIONS.md`.

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
