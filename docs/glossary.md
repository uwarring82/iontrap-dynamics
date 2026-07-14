# Glossary

Plain-language definitions of the terms that recur across the tutorials and the
API. Each entry is deliberately short; follow the linked tutorial or convention
section for the full treatment. Terms are cross-linked with *(see …)*.

This page is **Sail** reference material — orienting guidance, not a binding
definition of record. Where a term has a pinned convention, that convention in
[`CONVENTIONS.md`](conventions.md) is authoritative.

---

## Adiabatic

A change slow enough that the system stays in its instantaneous ground (or
energy) state and picks up no extra excitation. The opposite of a *(see
[Quench](#quench))*. Ramp a trap frequency adiabatically and nothing happens;
ramp it fast and you create *(see [Squeezing](#squeezing))*. Tutorials 18–19.

## Carrier

A laser drive tuned exactly to the ion's internal `|↓⟩ ↔ |↑⟩` transition (zero
detuning), which flips the spin without changing the motion. Contrast the *(see
[Sideband](#sideband))*, which couples spin and motion. Tutorial 1.

## Coastline

Governance term for the project's **binding, slow-to-change** layer:
`CONVENTIONS.md`, the workplan, licences. Coastline edits are versioned and
sealed by the maintainer. Contrast *(see [Sail](#sail))*. Tutorials are Sail, not
Coastline.

## Covariance matrix

For a motional mode, the 2×2 matrix `V` of quadrature variances
`⟨{x̂, p̂}⟩ − ⟨x̂⟩⟨p̂⟩` that fully describes a Gaussian state's width and shape. Its
eigenvalues give the *(see [Squeezing](#squeezing))* `r = ¼ ln(λ_max/λ_min)` and
its determinant the *(see [Symplectic eigenvalue](#symplectic-eigenvalue))*.
Tutorial 19; `CONVENTIONS.md` §26.4.

## Debye–Waller factor

The `e^{−η²(2n̄+1)/2}` suppression of a sideband coupling as the mode heats up —
the motional wavefunction spreads over the laser's phase, weakening the drive.
Depends on the *(see [Lamb–Dicke parameter](#lambdicke-parameter))* `η`.
Tutorials 8, 17.

## Fock truncation

The library represents a motional mode on a **finite** ladder of number states
`|0⟩ … |N−1⟩`. `fock=N` is that cutoff. Choose it too small and a spreading state
runs off the top of the ladder, silently biasing the result — so every solve
carries a convergence guard. Tutorial 6; `CONVENTIONS.md` §13/§15.

## Lamb–Dicke parameter

The dimensionless `η = k·x₀` measuring how far the ion's zero-point motion `x₀`
reaches along the laser wavevector `k`. Small `η` (the *Lamb–Dicke regime*) makes
sideband couplings simple and linear; large `η·√n̄` needs the full treatment.
Tutorials 2, 8, 17.

## Mode (motional mode)

A collective vibration of the trapped ion(s) — a quantum harmonic oscillator.
Its quanta are *phonons*. A single ion has one axial and two radial modes; `N`
ions share `N` modes per direction. Everything motional in the library lives on a
`ModeConfig`.

## Phonon

One quantum of a motional *(see [Mode](#mode-motional-mode))* — the vibrational
analogue of a photon. `⟨n̂⟩` is the mean phonon number. *(See
[Squeezing](#squeezing))* creates phonons in pairs.

## Quench

A **fast** change of a control parameter (here, the trap frequency `ω(t)`) —
fast enough to be non-adiabatic, so it leaves the state excited. A quench of the
trap creates *(see [Squeezing](#squeezing))*. The opposite of *(see
[Adiabatic](#adiabatic))*. Tutorial 19.

## Rabi frequency

The rate `Ω` at which a resonant drive cycles the spin between `|↓⟩` and `|↑⟩`;
one full flop-and-back takes a Rabi period `T = 2π/Ω`. Sideband and carrier
drives each have their own effective Rabi rate. Tutorials 0, 1.

## RWA (rotating-wave approximation)

Dropping the fast-oscillating counter-rotating terms of a drive Hamiltonian,
valid when the coupling is weak compared to the transition frequency. It is what
makes the *(see [Sideband](#sideband))* and carrier builders simple; Tutorial 18
shows where it breaks down (the quantum Rabi model).

## Sail

Governance term for the project's **adaptive, freely-revisable** layer: tutorials,
demo tools, this glossary. Sail material gives specific, helpful choices without
constraining the Coastline. Licensed CC BY-NC-SA 4.0. Contrast *(see
[Coastline](#coastline))*.

## Sideband

A laser drive detuned by one motional frequency from the carrier, so absorbing a
photon also adds (**blue**) or removes (**red**) one *(see [Phonon](#phonon))* —
the workhorse of cooling, state preparation, and gates. Contrast the *(see
[Carrier](#carrier))*. Tutorial 2.

## Squeezing

A Gaussian state whose noise is reduced below the vacuum floor in one quadrature
(and raised in the other), quantified by `r` (`r = 0` is no squeezing). It shows
up as an elliptical *(see [Wigner function](#wigner-function))* and, in the number
basis, as *(see [Phonon](#phonon))* population on **even `n` only** (pairs).
Tutorials 9, 16, 19–20.

## Symplectic eigenvalue

For a single mode, `ν = √(det V)` of the *(see [Covariance
matrix](#covariance-matrix))* — a purity indicator: `ν = 1` is a pure state,
`ν > 1` is mixed (`ν = 2n̄ + 1` for a thermal state). It is **not** the purity
itself (purity `= 1/ν`). Tutorial 19; `CONVENTIONS.md` §26.4.

## Wavevector

The laser's `k` (direction and magnitude `2π/λ`). Only its projection onto a
mode's eigenvector couples the drive to that motion, which is what sets the *(see
[Lamb–Dicke parameter](#lambdicke-parameter))*. Tutorial 1; `CONVENTIONS.md` §10.

## Wigner function

A phase-space quasi-probability picture of a motional state. The vacuum is a unit
circle; *(see [Squeezing](#squeezing))* flattens it into an ellipse whose axes are
the *(see [Covariance matrix](#covariance-matrix))* eigenvalues. Tutorial 19;
`CONVENTIONS.md` §26.3.

## WKB phase

The accumulated phase `∫ω dt` a motional state picks up as it evolves. It sets
whether two squeezing kicks add or cancel — the reason a down/up trap pulse's
squeezing oscillates with the hold time. Tutorial 19.

---

*This glossary is Sail material — CC BY-NC-SA 4.0, per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).*
