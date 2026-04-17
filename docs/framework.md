# Project Framework

## Design Rules

The public design is driven by a few hard constraints:

- Conventions before code
- No hidden laboratory assumptions
- Stratified reproducibility rather than vague reproducibility claims
- Named dataclasses and typed records instead of positional returns
- Plotting downstream of physics, not mixed into it
- Cache integrity as a hard failure boundary

## Architecture in one view

```text
physics layer      -> Hamiltonians, states, observables, deterministic solvers
apparatus layer    -> drifts, calibration errors, jitter, later Phase 1+
observation layer  -> finite-shot sampling, detectors, readout models
```

The separation matters because it prevents noise or measurement shortcuts from
quietly deforming the public Hamiltonian model.

## Result Contract

Every deterministic solver is expected to return a `TrajectoryResult` with:

- `times`
- `expectations`
- `metadata`
- optional structured warnings
- a declared storage mode for state retention

The result object is frozen and keyword-only. That is deliberate: downstream
analysis should receive a stable, named schema rather than an unpacked tuple.

## Provenance and Integrity

The project already enforces two strong constraints before broad solver work:

- Results record the `CONVENTIONS.md` version they were produced under.
- Cached trajectories are loaded only when the expected request hash matches the
  recorded one.

This is the line between “reproducible enough” and “quietly wrong”.

## Documentation Layers

- Coastline material states binding constraints.
- Sail material carries adaptive guidance and tutorials.

This docs folder currently contains only Coastline-facing entry pages; richer
tutorial material lands later under `docs/tutorials/`.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
