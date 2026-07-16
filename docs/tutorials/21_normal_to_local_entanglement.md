# Tutorial 21 — Two-ion motional entanglement: normal modes to the local-ion cut

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/21_normal_to_local_entanglement.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Two trapped ions share *collective* normal modes (centre-of-mass and
stretch), but their entanglement lives across the *ion* partition — invisible in
the mode basis. By the end you will have built the normal→local symplectic map
from a `ModeConfig`-style basis with the GT3b adapter, transported a Gaussian
covariance into the local-ion frame, and read out the ion-cut entanglement with
`log_negativity` and `entanglement_of_formation` — discovering that the two ions'
*motional ground state* is already entangled, that the entanglement is created by
the Coulomb frequency splitting, and that the local-frequency gauge changes each
ion's apparent temperature without touching the entanglement.

**Reference implementation.** The Gaussian toolbox in
[`gaussian.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/gaussian.py)
(covariance functionals, log-negativity, entanglement of formation) and the
ion adapter in
[`ion_modes.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/ion_modes.py)
(the normal→local symplectic map). The cross-repo handshake is specified in
`task cards/TC-gt3b-ion-symplectic-adapter.md`.

**Expected time.** ~15 min reading; ~2 s runtime.

**Level.** `advanced` — a research-grade surface that ties the whole Gaussian
toolbox together. Do the core (Steps 1–2) first.

**Prerequisites.** [Tutorial 16](16_two_mode_squeezing.md) (two-mode Gaussian
states and the `sinh²` occupation) and [Tutorial 12](12_bell_entanglement.md)
(two-ion entanglement in the spin sector — here the entanglement is *motional*).
No cross-repo install is needed: we build the mode basis by hand so every cell
runs from a bare notebook.

---

!!! note "New here? Read this first"

    - A pair of ions in one trap does not have "ion 1's oscillator" and "ion 2's
      oscillator" as independent objects — Coulomb repulsion couples them into
      two **normal modes**: the **centre-of-mass** (both ions swing together) and
      the **stretch** (they swing against each other), at different frequencies.
    - A Gaussian state is fully described by its **covariance matrix** `V` — the
      `2N×2N` table of quadrature variances/correlations. Everything below is a
      function of `V`; no wavefunctions.
    - Entanglement is always defined *across a cut*. The interesting cut here is
      **ion 0 vs ion 1** (a physical partition you could address separately), not
      mode-vs-mode. To see it you must re-express `V` in the **local** basis — one
      oscillator per ion — which is what the GT3b map `S` does: `V_local = S V Sᵀ`.
    - `S` is **symplectic** (`S Ω Sᵀ = Ω`): it is a legitimate change of canonical
      coordinates, so it preserves purity and every physical property — it just
      *reveals* the ion-cut structure the mode basis hides.
    - Two entanglement measures appear: **log-negativity** `E_N` (an easy-to-compute
      witness, non-zero ⇔ entangled) and **entanglement of formation** `E_F` (the
      cost to create the state; for a pure state it equals the entropy of one side).

    **In a hurry?** Step 1 builds the map; Step 2 shows the two-ion *ground state*
    is entangled across the ions — that pair is the core. Step 3 explains *why*
    (the frequency splitting) and Step 4 shows the local-frequency gauge is
    entanglement-neutral.

**Symbols in this tutorial**

| symbol | plain meaning |
| --- | --- |
| `B` | mass-weighted normal-mode matrix; column `m` is mode `m`'s displacement pattern over the ions |
| `ω_COM`, `ω_stretch` | the two mode frequencies; for two equal-mass ions `ω_stretch = √3 · ω_COM` |
| `ω_local` | the per-ion **local reference frequency** — a gauge choice (Step 4), not a physical observable |
| `S` | the normal→local symplectic map; `V_local = S V_normal Sᵀ` (GT3b) |
| `V` | a `2N×2N` covariance matrix (§27, vacuum variance 1) |
| `E_N` | ion-cut **log-negativity** (bits); `0` ⇔ separable across that cut |
| `E_F` | ion-cut **entanglement of formation** (bits); for a pure state = the entropy of one side |
| `ν̃₋` | smaller partial-transpose symplectic eigenvalue; `< 1` ⇔ entangled |

## The scenario

For two equal-mass ions in a linear trap the axial motion has two normal modes:
the centre-of-mass at `ω_COM` and the stretch at `ω_stretch = √3 · ω_COM`, with the
familiar patterns `(1, 1)/√2` and `(1, −1)/√2`. A Gaussian motional state is a
covariance `V` in the *normal-mode* basis. But if you want to know how entangled
**ion 0** is with **ion 1** — the quantity a two-site experiment measures — you
must transport `V` into the *local* basis, one oscillator per ion. That map is a
symplectic congruence `V_local = S V Sᵀ`; the GT3b adapter builds `S` from the
mode data, the generic congruence applies it, and the toolbox reads out the cut.

### Step 1 — the two mode bases and the map between them

We build the analytic two-ion axial basis and hand it to the GT3b adapter as an
`IonModeBasis` payload. (In production the payload is emitted by the sibling
`iontrap-structure` package; here we construct it directly so the notebook is
self-contained.) The adapter returns the normal→local symplectic `S`.

```python
import numpy as np
from scipy.constants import atomic_mass

from iontrap_dynamics import gaussian as g
from iontrap_dynamics import ion_modes as im

# Two equal-mass ²⁵Mg⁺ ions, axial motion only (one coordinate per ion, so 2 modes).
w_com = 2 * np.pi * 1.0e6  # centre-of-mass frequency, rad/s
w_stretch = np.sqrt(3.0) * w_com  # equal-mass stretch mode: √3 above the COM
B = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)  # cols: COM (1,1)/√2, stretch (1,-1)/√2

# The wire-format ordering tag the consumer enforces byte-for-byte (see the GT3b card).
FRAME = "ion-major-axis-minor;row=axes_per_ion*i+c;col=mode"


def make_basis(local_reference_rad_s):
    """Materialize an IonModeBasis for our two-ion axial system at a chosen local gauge."""
    return im.materialize_ion_mode_basis(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=np.array([w_com, w_stretch]),
        mass_weighted_eigenvectors=B,
        masses_kg=np.full(2, 25.0 * atomic_mass),
        local_reference_frequencies_rad_s=local_reference_rad_s,
        coordinate_frame=FRAME,
    )


basis = make_basis(np.array([w_com, w_com]))  # local reference = COM frequency (the natural gauge)
S = im.normal_to_local_symplectic(basis)
residual = np.max(np.abs(S @ g.symplectic_form(2) @ S.T - g.symplectic_form(2)))
print(f"normal→local map S is symplectic (S Ω Sᵀ = Ω): residual {residual:.1e}")
print(f"axes per ion = {basis.axes_per_ion}, ions = {basis.n_ions}")
```

`S` is symplectic to machine precision, so it is a genuine change of canonical
coordinates — the ion-cut readout it enables is physics, not an artefact.

### Step 2 — the coupled ground state is entangled across the ions

The two-ion *motional ground state* is the normal-mode **vacuum**: `V_normal = 𝟙`.
Transport it to the local basis and read the ion-cut entanglement.

```python
V_ground = im.to_local_covariance(basis, np.eye(4))  # map the normal-mode vacuum
ion0 = im.ion_mode_indices(basis, 0)  # the local coordinate(s) of ion 0

print("local-basis ground-state covariance V_local:")
print(np.round(V_ground, 4))
print(f"\nion-cut log-negativity      E_N = {g.log_negativity(V_ground, ion0):.4f} bits")
print(f"ion-cut entanglement of form. E_F = {g.entanglement_of_formation(V_ground):.4f} bits")
print(f"global purity                μ   = {g.purity(V_ground):.4f}  (a pure state — globally)")
```

The off-diagonal blocks of `V_local` are non-zero: the two ions are correlated
even though the collective modes are in their lowest state. `E_N = 0.40` bits and
`E_F = 0.14` bits both certify entanglement, while the global purity stays `1` —
the state is *globally pure but locally mixed*, the hallmark of entanglement. This
is the ground-state entanglement of two coupled oscillators, made visible only in
the local frame.

### Step 3 — the frequency splitting is what entangles

Why is the ground state entangled? Because the map is **active** — it squeezes.
If the two modes had the *same* frequency (`ω_stretch = ω_COM`, i.e. no Coulomb
coupling) the map would be a passive **rotation**, and rotating the vacuum leaves
it separable. The real `√3` splitting makes `S` a squeeze, and squeezing the
vacuum across the cut creates entanglement. Sweep the ratio to see it grow.

```python
import matplotlib.pyplot as plt

GREY, BLUE, RED = "#888888", "#1f77b4", "#d62728"

# passive (equal frequencies → rotation) vs active (√3 → squeeze), same vacuum input
passive = im.materialize_ion_mode_basis(
    schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
    frequencies_rad_s=np.array([w_com, w_com]),  # no splitting: a passive map
    mass_weighted_eigenvectors=B,
    masses_kg=np.full(2, 25.0 * atomic_mass),
    local_reference_frequencies_rad_s=np.array([w_com, w_com]),
    coordinate_frame=FRAME,
)
e_passive = g.log_negativity(im.to_local_covariance(passive, np.eye(4)), [0])
e_active = g.log_negativity(V_ground, [0])
print(f"passive (equal ω) ground-state E_N = {e_passive:.4f}  (rotation cannot entangle the vacuum)")
print(f"active  (√3 ratio) ground-state E_N = {e_active:.4f}  (squeezing does)")

# E_N of the ground state vs the mode-frequency ratio ω_stretch / ω_COM
ratios = np.linspace(1.0, 3.0, 21)
e_of_ratio = []
for ratio in ratios:
    b = im.materialize_ion_mode_basis(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=np.array([w_com, ratio * w_com]),
        mass_weighted_eigenvectors=B,
        masses_kg=np.full(2, 25.0 * atomic_mass),
        local_reference_frequencies_rad_s=np.array([w_com, w_com]),
        coordinate_frame=FRAME,
    )
    e_of_ratio.append(g.log_negativity(im.to_local_covariance(b, np.eye(4)), [0]))

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(ratios, e_of_ratio, color=BLUE, linewidth=1.4)
ax.axvline(np.sqrt(3.0), color=RED, linestyle="--", linewidth=1.0, label=r"real 2-ion $\sqrt{3}$")
ax.axhline(0.0, color=GREY, linewidth=0.8)
ax.set_xlabel(r"mode-frequency ratio $\omega_\mathrm{stretch}/\omega_\mathrm{COM}$")
ax.set_ylabel(r"ground-state ion-cut $E_N$ (bits)")
ax.set_title("Coulomb splitting entangles the motional ground state")
ax.legend(frameon=False)
plt.show()
```

At ratio `1` (decoupled) the ground state is separable; as the Coulomb coupling
splits the modes the ground-state ion-cut entanglement climbs monotonically. The
physical two-ion value sits at `√3`.

### Step 4 — the local-frequency gauge is entanglement-neutral

The **local reference frequency** `ω_local` sets the scale of each ion's local
quadratures. Changing it is a per-ion single-mode squeeze — a *local* operation —
so it re-labels the covariance and shifts each ion's apparent occupation and
effective temperature, but it **cannot** change entanglement across the ion cut.

```python
for label, local in (
    ("ω_local = ω_COM  (natural)", np.array([w_com, w_com])),
    ("ω_local = [3.7, 0.4]·ω_COM (skewed)", np.array([3.7 * w_com, 0.4 * w_com])),
):
    b = make_basis(local)
    v = im.to_local_covariance(b, np.eye(4))
    nbar = g.mean_occupation(v[:2, :2], np.zeros(2))  # ion-0 marginal occupation
    print(
        f"{label:38s}  E_N = {g.log_negativity(v, im.ion_mode_indices(b, 0)):.4f}  "
        f"ion-0 n̄ = {nbar:.4f}  T_eff = {g.effective_temperature(nbar, w_com) * 1e6:.2f} µK"
    )
```

`E_N` is identical in both rows — the entanglement is gauge-invariant — while the
ion-0 occupation and effective temperature change with the gauge. This is exactly
why the GT3b card treats `ω_local` as an explicit, tagged *gauge* (decision D3):
it is a representation choice, not a physical prediction, and it never touches the
ion-cut entanglement.

**Takeaway.** Entanglement is a property of a *cut*, and the physically meaningful
cut for two trapped ions is ion-vs-ion — which the normal-mode covariance hides.
The GT3b symplectic map `S` re-expresses the state in the local basis without
changing any physics (`S Ω Sᵀ = Ω`), and then the Gaussian toolbox reads the cut:
the ground state is entangled, the Coulomb frequency splitting is the entangler,
and the local-frequency gauge is invisible to the entanglement. The whole chain —
`ion_modes.normal_to_local_symplectic` → `gaussian.congruence` →
`gaussian.log_negativity` / `entanglement_of_formation` — is pure covariance
algebra over the sealed §27 conventions.

## Where to next

- The producer side of the handshake — building the `IonModeBasis` payload from a
  real equilibrium-and-modes calculation — lives in the sibling
  [`iontrap-structure`](https://github.com/uwarring82/iontrap-structure) package;
  this tutorial's hand-built basis is the same wire contract.
- [Tutorial 16 — Two-mode SU(1,1) squeezing](16_two_mode_squeezing.md) builds the
  *normal-mode* squeezed states you can feed into Step 2 in place of the vacuum to
  drive the ion-cut entanglement higher.

---

## Licence

Sail material — adaptive guidance with specific parameter choices, not a
coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
