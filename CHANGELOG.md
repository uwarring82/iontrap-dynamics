# Changelog

All notable changes to `iontrap-dynamics` will be documented in this file.

The format follows Keep a Changelog, and the project aims to follow Semantic
Versioning once the public package surface reaches its first alpha release.

## [Unreleased]

### Added

- Initial repository workplan and conventions documents
- Split-licence declarations at repository root and in `assets/`
- Asset provenance record plus fetch/hash maintenance scripts
- `pyproject.toml` with hatchling build metadata and tool configuration
- Package scaffold at `src/iontrap_dynamics/`
- Canonical exception hierarchy in `src/iontrap_dynamics/exceptions.py`
- Backend-agnostic invariant diagnostics for density matrices and state vectors
- Hash-verified cache I/O for `TrajectoryResult` (`.npz` + JSON manifest)
- Permanent `regression_invariant` tests for trace, Hermiticity, positivity,
  norm conservation, and swap symmetry
- MkDocs landing site scaffold with a styled welcome page and first
  navigation layer under `docs/`
