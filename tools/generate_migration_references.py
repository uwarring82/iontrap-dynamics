# SPDX-License-Identifier: MIT
"""Generate migration-regression reference data from legacy/qc.py.

Purpose
-------

The three-layer regression harness in ``WORKPLAN_v0.3.md`` §0.B defines
migration regressions as frozen outputs of the legacy ``qc.py`` script
(tagged ``qc-legacy-v1.0``) for five canonical scenarios. Phase 1 builders
must reproduce these arrays to 10⁻¹⁰ on the same platform, subject to the
caveat that ``qc.py`` is a *check*, not a truth criterion — analytic and
invariant tests are the permanent physics anchors.

This script performs two workplan actions:

- **§0.B item 1 / Week 1 item 3 — legacy-stability check**: run each
  scenario three times and verify bit-identical output. Gate for
  proceeding with reference generation. Invoke with ``--stability-check``.

- **§0.B items 3-4 / Week 1 item 7 — reference generation**: run each
  scenario once and save ``.npz`` arrays + ``metadata.json`` into
  ``tests/regression/migration/references/<scenario>/``. Default mode.

Scope
-----

All five canonical scenarios are implemented. Scenarios 1, 2, 3, 5
dispatch straight into qc.py methods; scenario 4 is a Path-A
duplicate of qc.py's ``single_spin_and_mode_ACpi2`` with a single
spline-callable fix for QuTiP 5 compatibility (see that scenario's
section for the rationale). The reference bundles for all five
scenarios are committed under
``tests/regression/migration/references/``.

Regenerating the bundles requires the ``[legacy]`` extras
(``pip install -e ".[legacy]"``) — qc.py pulls in a heavy non-core
stack (pandas, seaborn, sympy, allantools, requests, ipython,
ipywidgets, matplotlib) via ``from qutip import *`` and explicit
imports. Normal users reading the migration-regression tests do
not need the extras — the tests load the committed bundles
directly from disk.

QuTiP compat
------------

``qc.py`` was written against QuTiP 4.x and uses ``from qutip import *``
which, under QuTiP 5, does not expose the rotation-operator helpers
``rx``/``ry``/``rz`` or ``concurrence``. These are injected into the ``qc`` module's globals
before any scenario runs. Two ``FutureWarning`` messages about ``e_ops``
and ``progress_bar`` keyword-only transitions in QuTiP 5.3 are expected
and ignored — we accept the current deprecation noise since the script
targets exactly one frozen legacy-tag version of ``qc.py``.

Usage
-----

    python tools/generate_migration_references.py                 # generate
    python tools/generate_migration_references.py --stability-check  # gate
    python tools/generate_migration_references.py --scenario 1       # just one
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Suppress the QuTiP 5.3 deprecations and the SyntaxWarnings that come from
# docstring escape sequences inside the legacy script — they are informational
# only and would otherwise drown out the reference-generation log.
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="qc")


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "legacy"
REFERENCES_DIR = REPO_ROOT / "tests" / "regression" / "migration" / "references"


# ----------------------------------------------------------------------------
# QuTiP-5 compat shim for the legacy qc.py
# ----------------------------------------------------------------------------


def _load_qc_module() -> Any:
    """Import ``legacy/qc.py`` and patch the QuTiP-5 compat gap."""
    sys.path.insert(0, str(LEGACY_DIR))
    from qutip.core.gates import rx, ry, rz

    try:
        from qutip.metrics import concurrence
    except ImportError:
        from qutip import concurrence

    qc = importlib.import_module("qc")
    qc.rx = rx
    qc.ry = ry
    qc.rz = rz
    qc.concurrence = concurrence
    return qc


# ----------------------------------------------------------------------------
# Environment capture (platform-identifying context for the metadata)
# ----------------------------------------------------------------------------


def _environment_context() -> dict[str, str]:
    import qutip
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
    }


def _parameters_hash(parameters: dict[str, Any]) -> str:
    """Stable SHA-256 of a parameter bundle — same contract as
    :func:`iontrap_dynamics.cache.compute_request_hash`."""
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Shared helper: single-spin, single-mode scenarios
# ----------------------------------------------------------------------------
#
# Scenarios 1, 2, and (eventually) 5 all drive a single spin against a single
# motional mode with different regimes (on-resonance carrier; red-sideband
# from Fock |1⟩; squeeze + displace off-resonance). They share:
#
#   - the qc.py method `squeeze_to_entangle_singleSpin_singleMode`
#   - the same reduced-state tracing (1 spin + 1 mode) and therefore the
#     same 11 observable keys in the output bundle
#
# The regime-specific choices live in each scenario's parameters dict.


def _run_single_spin_single_mode(qc_module: Any, p: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run one single-spin / single-mode scenario through qc.py and return
    an arrays-by-observable dict.

    Output keys (same for every single-spin single-mode scenario):

    * ``times`` — 1-D, μs (qc.py's native time units).
    * ``sigma_x``, ``sigma_y``, ``sigma_z`` — spin expectations.
    * ``spin_entropy`` — ⟨−Tr(ρ log ρ)⟩ on the spin reduced state.
    * ``n_mode`` — ⟨n⟩ motional occupation.
    * ``var_x``, ``var_p`` — position and momentum variances.
    * ``mode_entropy`` — entropy of the mode reduced state.
    * ``mode_X``, ``mode_P`` — ⟨X⟩ and ⟨P⟩ quadratures.
    """
    q = qc_module.QC()
    output, _mS, _mM = q.squeeze_to_entangle_singleSpin_singleMode(
        Omega=p["Omega_over_2pi_MHz"] * 2 * np.pi,
        omega_1=p["omega_mode_over_2pi_MHz"] * 2 * np.pi,
        omega_z=p["omega_spin_over_2pi_MHz"] * 2 * np.pi,
        r_spin=p["r_spin_rad"],
        n_th=p["n_thermal"],
        Fck=p["Fck"],
        sq_a=p["sq_ampl"],
        sq_phi=p["sq_phi_rad"],
        dis_a=p["dis_ampl"],
        dis_phi=p["dis_phi_rad"],
        phi_drive=p["phi_drive_rad"],
        tmax=p["tmax_periods"],
        nosteps=p["nosteps"],
        FockPrec=p["FockPrec"],
        state_in=None,
        do_plot=False,
        LD_regime=p["LD_regime"],
    )

    spin_props, _ = q.trace_spin_props(output, ptrace_sel=[1], verbose=False)
    mode_props, _ = q.trace_motional_props(output, ptrace_sel=[0], verbose=False)
    spin_props = np.asarray(spin_props, dtype=np.complex128)
    mode_props = np.asarray(mode_props, dtype=np.complex128)

    # trace_spin_props (1 spin) columns: [<sx>, <sy>, <sz>, S]
    # trace_motional_props (1 mode) columns: [n_cut, <n>, Var_x, Var_p, S, <X>, <P>]
    times = np.asarray(output.times, dtype=np.float64)
    return {
        "times": times,
        "sigma_x": np.real(spin_props[:, 0]),
        "sigma_y": np.real(spin_props[:, 1]),
        "sigma_z": np.real(spin_props[:, 2]),
        "spin_entropy": np.real(spin_props[:, 3]),
        "n_mode": np.real(mode_props[:, 1]),
        "var_x": np.real(mode_props[:, 2]),
        "var_p": np.real(mode_props[:, 3]),
        "mode_entropy": np.real(mode_props[:, 4]),
        "mode_X": np.real(mode_props[:, 5]),
        "mode_P": np.real(mode_props[:, 6]),
    }


# ----------------------------------------------------------------------------
# Scenario 1 — single-ion carrier flopping, thermal
# ----------------------------------------------------------------------------
#
# Regime: on-resonance carrier drive (omega_z = 0), thermal initial motion,
# Lamb–Dicke regime expansion. The workplan §0.B entry for this scenario
# reads "single-ion carrier flopping (thermal)".

SCENARIO_1_PARAMETERS: dict[str, Any] = {
    "Omega_over_2pi_MHz": 0.05,
    "omega_mode_over_2pi_MHz": 2.2,
    "omega_spin_over_2pi_MHz": 0.0,
    "r_spin_rad": [0.0, 0.0, 0.0],
    "n_thermal": 0.5,
    "Fck": 0,
    "sq_ampl": 0.0,
    "sq_phi_rad": 0.0,
    "dis_ampl": 0.0,
    "dis_phi_rad": 0.0,
    "phi_drive_rad": 0.0,
    "tmax_periods": 1,
    "nosteps": 1,
    "FockPrec": 0.005,
    "LD_regime": True,
}


def _run_scenario_1(qc_module: Any) -> dict[str, np.ndarray]:
    return _run_single_spin_single_mode(qc_module, SCENARIO_1_PARAMETERS)


# ----------------------------------------------------------------------------
# Scenario 2 — single-ion red-sideband flopping, Fock |1⟩
# ----------------------------------------------------------------------------
#
# Regime: laser red-detuned by one mode frequency (omega_z = −omega_1),
# near-vacuum thermal occupation so the Fock |1⟩ state dominates, no
# squeezing or displacement. Couples |↓, 1⟩ ↔ |↑, 0⟩ at leading-order
# rate |η|·Ω (analytic/formula in src/iontrap_dynamics/analytic.py,
# verified there at pytest-approx). The workplan §0.B entry for this
# scenario reads "red-sideband flopping (Fock |1⟩)".

SCENARIO_2_PARAMETERS: dict[str, Any] = {
    "Omega_over_2pi_MHz": 0.05,
    "omega_mode_over_2pi_MHz": 2.2,
    "omega_spin_over_2pi_MHz": -2.2,  # red-detuned by one mode frequency
    "r_spin_rad": [0.0, 0.0, 0.0],
    "n_thermal": 0.001,  # near-vacuum to isolate the Fock state
    "Fck": 1,  # initial motional state |1⟩
    "sq_ampl": 0.0,
    "sq_phi_rad": 0.0,
    "dis_ampl": 0.0,
    "dis_phi_rad": 0.0,
    "phi_drive_rad": 0.0,
    "tmax_periods": 1,
    "nosteps": 1,
    "FockPrec": 0.005,
    "LD_regime": True,
}


def _run_scenario_2(qc_module: Any) -> dict[str, np.ndarray]:
    return _run_single_spin_single_mode(qc_module, SCENARIO_2_PARAMETERS)


# ----------------------------------------------------------------------------
# Scenarios 3–5 — implemented
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Scenario 3 — two-ion MS-like gate (blue-sideband single-tone drive)
# ----------------------------------------------------------------------------
#
# Note on nomenclature: the qc.py method `squeeze_to_entangle_twoSpins_singleMode`
# drives both spins with a single laser tone at omega_z = +omega_mode (blue
# sideband). A proper Mølmer–Sørensen gate uses two tones at omega_atom ± δ,
# with δ ≈ omega_mode — the single-tone version here is a simplified
# relative of MS that still produces spin–motion entanglement via the
# sideband coupling. The workplan §0.B entry calls this the "two-ion MS
# gate" scenario and treats qc.py's output as the regression target; the
# labelling is looser than the textbook definition but matches the
# legacy script's intent.

SCENARIO_3_PARAMETERS: dict[str, Any] = {
    "Omega_over_2pi_MHz": 1.0,
    "omega_mode_over_2pi_MHz": 2.5,
    "omega_spin_over_2pi_MHz": 2.5,  # blue sideband (omega_z = +omega_mode)
    "r_spin_rad": [0.0, 0.0, 0.0],
    "n_thermal": 0.001,
    "Fck": 0,
    "sq_ampl": 0.0,
    "sq_phi_rad": 0.0,
    "dis_ampl": 0.0,
    "dis_phi_rad": 0.0,
    "phi_drive_rad": 0.0,
    "tmax_periods": 1,
    "nosteps": 50,  # 50 samples / μs → ~250 samples over tend
    "FockPrec": 0.0025,
    "LD_regime": True,
}


def _run_scenario_3(qc_module: Any) -> dict[str, np.ndarray]:
    """Two-spin, one-mode scenario. Output has 21 arrays:

    * ``times`` — 1-D, μs.
    * Joint-spin: ``spin_joint_entropy``, ``eof``.
    * Per-ion spin: ``sigma_{x,y,z}_A``, ``spin_entropy_A``, ``sigma_{x,y,z}_B``,
      ``spin_entropy_B``.
    * Two-ion populations: ``p_down_down``, ``p_single_flip``, ``p_up_up``.
    * Bell fidelity: ``bell_fidelity`` (qc.py's ``(|dd⟩ + i|uu⟩)/√2``
      convention — see CONVENTIONS.md §9 for the adoption note).
    * Mode: ``n_mode``, ``var_x``, ``var_p``, ``mode_entropy``, ``mode_X``,
      ``mode_P``.
    """
    q = qc_module.QC()
    p = SCENARIO_3_PARAMETERS
    output, _mS, _mM = q.squeeze_to_entangle_twoSpins_singleMode(
        Omega=p["Omega_over_2pi_MHz"] * 2 * np.pi,
        omega_1=p["omega_mode_over_2pi_MHz"] * 2 * np.pi,
        omega_z=p["omega_spin_over_2pi_MHz"] * 2 * np.pi,
        r_spin=p["r_spin_rad"],
        n_th=p["n_thermal"],
        Fck=p["Fck"],
        sq_a=p["sq_ampl"],
        sq_phi=p["sq_phi_rad"],
        dis_a=p["dis_ampl"],
        dis_phi=p["dis_phi_rad"],
        phi_drive=p["phi_drive_rad"],
        tmax=p["tmax_periods"],
        nosteps=p["nosteps"],
        FockPrec=p["FockPrec"],
        state_in=None,
        do_plot=False,
        LD_regime=p["LD_regime"],
    )

    spin_props, _ = q.trace_spin_props(output, ptrace_sel=[1, 2], verbose=False)
    mode_props, _ = q.trace_motional_props(output, ptrace_sel=[0], verbose=False)
    spin_props = np.asarray(spin_props, dtype=np.complex128)
    mode_props = np.asarray(mode_props, dtype=np.complex128)

    # Spin columns (2 spins, 14 total):
    # [S, EoF, <sx>_A, <sy>_A, <sz>_A, S_A, <sx>_B, <sy>_B, <sz>_B, S_B,
    #  P(|dd>), P(|du>+|ud>), P(|uu>), Bell-state fidelity]
    # Mode columns (1 mode): [n_cut, <n>, Var_x, Var_p, S, <X>, <P>]
    times = np.asarray(output.times, dtype=np.float64)
    return {
        "times": times,
        "spin_joint_entropy": np.real(spin_props[:, 0]),
        "eof": np.real(spin_props[:, 1]),
        "sigma_x_A": np.real(spin_props[:, 2]),
        "sigma_y_A": np.real(spin_props[:, 3]),
        "sigma_z_A": np.real(spin_props[:, 4]),
        "spin_entropy_A": np.real(spin_props[:, 5]),
        "sigma_x_B": np.real(spin_props[:, 6]),
        "sigma_y_B": np.real(spin_props[:, 7]),
        "sigma_z_B": np.real(spin_props[:, 8]),
        "spin_entropy_B": np.real(spin_props[:, 9]),
        "p_down_down": np.real(spin_props[:, 10]),
        "p_single_flip": np.real(spin_props[:, 11]),
        "p_up_up": np.real(spin_props[:, 12]),
        "bell_fidelity": np.real(spin_props[:, 13]),
        "n_mode": np.real(mode_props[:, 1]),
        "var_x": np.real(mode_props[:, 2]),
        "var_p": np.real(mode_props[:, 3]),
        "mode_entropy": np.real(mode_props[:, 4]),
        "mode_X": np.real(mode_props[:, 5]),
        "mode_P": np.real(mode_props[:, 6]),
    }


# ----------------------------------------------------------------------------
# Scenario 4 — single-ion stroboscopic AC-π/2  (Path-A QuTiP-5 compat duplicate)
# ----------------------------------------------------------------------------
#
# Background: qc.py's single_spin_and_mode_ACpi2 builds a time-dependent
# Hamiltonian whose modulation coefficient is a scipy CubicSpline. On QuTiP
# 5.1+ the callable's 0-d-array return is rejected with
# "TypeError: The coefficient function must return a number". Since the
# offending `mod(t, args)` function is a closure inside the legacy method,
# there is no external-patch fix; the only way to invoke the stroboscopic
# path is to re-author the body here.
#
# This implementation is a byte-for-byte duplicate of the legacy code
# flow, with ONE minimal change: the spline wrapper returns float(cs(t))
# instead of cs(t) directly. Everything else — Butterworth low-pass
# filter, filtered-square-wave pulse shape, scipy CubicSpline
# interpolation, Lamb-Dicke operator, Hamiltonian structure, mesolve
# invocation — matches qc.py exactly. Reusable qc.py helpers
# (initialise_spins, initialise_single_mode, trace_{spin,motional}_props)
# are still called on the passed QC instance; only the evolution loop
# whose callable semantics changed is re-authored.
#
# Per the previous dispatch, scenario 4 sits on the "legacy with fixes"
# side of the faithful-replay boundary. Scenarios 1, 2, 3, 5 still
# dispatch straight into qc.py methods unchanged.

SCENARIO_4_PARAMETERS: dict[str, Any] = {
    # Spin initial rotation (qc.py convention: [s_ini, s_phs] in units of 2π,
    # applied through initialise_spins).
    "spin_initial_rotation": 0.0,
    "spin_initial_phase": 0.0,
    # Motion (qc.py mode_para unpacked)
    "n_thermal": 0.001,
    "dis_ampl": 0.0,
    "dis_phi_fraction": 0.0,
    "sq_ampl": 0.0,
    "sq_phi_fraction": 0.0,
    "mode_freq_MHz": 1.3,
    "theta_m_deg": 0.0,
    # Drive (qc.py drive_para unpacked)
    "Omega_over_2pi_MHz": 0.125,
    "detuning_fraction": 0.0,  # omega_z = detuning_fraction · omega_mode; 0 = on resonance
    # Stroboscopic modulation (qc.py: mod_on=True, mod_type=0)
    "mod_fac": 1,  # modulation frequency as multiple of omega_mode
    "mod_amp": 1.0,
    "strobo_dur_us": 0.1,  # pulse length inside each mod period
}


def _run_scenario_4(qc_module: Any) -> dict[str, np.ndarray]:
    """Path-A QuTiP-5-compat duplicate of qc.single_spin_and_mode_ACpi2.

    Produces the 11-observable single-spin / single-mode layout (identical
    keys to scenarios 1, 2, 5).
    """
    import qutip
    import scipy.constants as cst
    from scipy.interpolate import CubicSpline
    from scipy.signal import butter, filtfilt, square

    q = qc_module.QC()
    p = SCENARIO_4_PARAMETERS

    # Parameter unpacking — mirrors qc.py
    omega_1 = 2 * np.pi * p["mode_freq_MHz"]
    omega_z = p["detuning_fraction"] * omega_1
    Omega = p["Omega_over_2pi_MHz"] * 2 * np.pi

    s_ini = p["spin_initial_rotation"]
    s_phs = p["spin_initial_phase"]
    r_spin = [s_ini * 2 * np.pi, 0.0, (0.677 / 2 + s_phs) * 2 * np.pi]

    n_th = p["n_thermal"]
    Fck = 0
    dis_a = p["dis_ampl"]
    dis_phs = p["dis_phi_fraction"] * 2 * np.pi
    sq_a = p["sq_ampl"]
    sq_phs = p["sq_phi_fraction"] * 2 * np.pi

    Degree = np.pi / 180
    theta_m = p["theta_m_deg"]
    eta_1 = q.lamb_dicke(
        [0, -np.sqrt(2) / np.sqrt(2), np.sqrt(2) / np.sqrt(2)],
        [0, -np.sin(theta_m * Degree), np.cos(theta_m * Degree)],
        omega_1 * cst.mega,
        25 * cst.atomic_mass,
    )
    Omega_eff = np.exp(-(eta_1**2) / 2) * Omega

    # Initial state (qc.py helpers — deterministic and QuTiP-5 clean)
    rho_spin_0, _ps = q.initialise_spins(no=1, angle=r_spin, verbose=False)
    rho_motion_0, props_m = q.initialise_single_mode(
        n_th=n_th,
        Fck=Fck,
        sq_ampl=sq_a,
        sq_phi=sq_phs,
        dis_ampl=dis_a,
        dis_phi=dis_phs,
        Prec=1e-10,
        Ncut=int(2 * dis_a**2 + 10),
        verbose=False,
    )
    N = props_m[0]
    rho_0 = qutip.tensor(rho_motion_0, rho_spin_0)

    # Operators
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sm = qutip.tensor(qutip.qeye(N), qutip.destroy(2))
    sz = qutip.tensor(qutip.qeye(N), qutip.sigmaz())

    # Hamiltonian (full-exponential, non-LD-truncated — matches qc.py)
    H0 = omega_z / 2 * sz + omega_1 * a.dag() * a
    C = (1j * eta_1 * (a.dag() + a) + 1j).expm()
    HI = Omega / 2 * (sm.dag() * C + sm * C.dag())

    # Time grid for the stroboscopic branch
    omega_mod = p["mod_fac"] * omega_1
    strobo_dur = p["strobo_dur_us"]
    duty_cycle = strobo_dur / (2 * np.pi / omega_mod)
    tend = 0.925 * 2 * np.pi / Omega_eff / 4 / duty_cycle
    tlist = np.linspace(0, tend, int(tend * 200))

    # Stroboscopic envelope — identical to qc.py except for the float() fix
    def mod_fct(t: np.ndarray) -> np.ndarray:
        t_off = 2 * np.pi / omega_mod / 4 - strobo_dur / 2
        return (square(omega_mod * (t - t_off), duty=duty_cycle) + 1) / 2

    def aom_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
        nyq = 3
        normal_cutoff = cutoff / nyq
        b, a_coef = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a_coef, data)

    f_sampl = 1 / (tend / len(tlist))
    spline_samples = np.abs(aom_filter(mod_fct(tlist), cutoff=0.2, fs=f_sampl, order=1))
    cs = CubicSpline(tlist, spline_samples)

    def mod_coeff(t: float, args: Any) -> float:
        # Path-A compat fix: float() wraps the 0-d ndarray that CubicSpline
        # returns on scalar input so QuTiP 5 accepts it as a coefficient.
        return float(cs(t))

    H = [H0, [HI, mod_coeff]]

    output = qutip.mesolve(H, rho_0, tlist, [], [], options={"progress_bar": False})

    spin_props, _ = q.trace_spin_props(output, ptrace_sel=[1], verbose=False)
    mode_props, _ = q.trace_motional_props(output, ptrace_sel=[0], verbose=False)
    spin_props = np.asarray(spin_props, dtype=np.complex128)
    mode_props = np.asarray(mode_props, dtype=np.complex128)

    times = np.asarray(output.times, dtype=np.float64)
    return {
        "times": times,
        "sigma_x": np.real(spin_props[:, 0]),
        "sigma_y": np.real(spin_props[:, 1]),
        "sigma_z": np.real(spin_props[:, 2]),
        "spin_entropy": np.real(spin_props[:, 3]),
        "n_mode": np.real(mode_props[:, 1]),
        "var_x": np.real(mode_props[:, 2]),
        "var_p": np.real(mode_props[:, 3]),
        "mode_entropy": np.real(mode_props[:, 4]),
        "mode_X": np.real(mode_props[:, 5]),
        "mode_P": np.real(mode_props[:, 6]),
    }


# ----------------------------------------------------------------------------
# Scenario 5 — single-mode squeezing + displacement
# ----------------------------------------------------------------------------
#
# Regime: on-resonance carrier drive on a squeezed + displaced initial
# motional state (no thermal admixture, no Fock offset). The non-trivial
# sq_ampl and dis_ampl put the motional state away from vacuum with both
# a first-moment (⟨X⟩, ⟨P⟩) and second-moment (Var_x, Var_p) signature,
# so the reference exercises the squeezing and displacement operators
# that Phase 1 state-prep builders will provide. The workplan §0.B entry
# for this scenario reads "single-mode squeezing + displacement".

SCENARIO_5_PARAMETERS: dict[str, Any] = {
    "Omega_over_2pi_MHz": 0.05,
    "omega_mode_over_2pi_MHz": 2.2,
    "omega_spin_over_2pi_MHz": 0.0,  # on-resonance carrier
    "r_spin_rad": [0.0, 0.0, 0.0],
    "n_thermal": 0.0,  # pure vacuum baseline before sq/dis
    "Fck": 0,
    "sq_ampl": 0.5,  # moderate squeezing (r = 0.5)
    "sq_phi_rad": 0.0,
    "dis_ampl": 1.0,  # displace by α = 1 → coherent ⟨n⟩ = 1
    "dis_phi_rad": 0.0,
    "phi_drive_rad": 0.0,
    "tmax_periods": 1,
    "nosteps": 1,
    "FockPrec": 0.005,
    "LD_regime": True,
}


def _run_scenario_5(qc_module: Any) -> dict[str, np.ndarray]:
    return _run_single_spin_single_mode(qc_module, SCENARIO_5_PARAMETERS)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "01_single_ion_carrier_thermal": {
        "index": 1,
        "description": "Single-ion carrier flopping from thermal motion (on-resonance)",
        "runner": _run_scenario_1,
        "parameters": SCENARIO_1_PARAMETERS,
    },
    "02_single_ion_red_sideband_fock1": {
        "index": 2,
        "description": "Single-ion red-sideband flopping from Fock |1⟩",
        "runner": _run_scenario_2,
        "parameters": SCENARIO_2_PARAMETERS,
    },
    "03_two_ion_ms_gate": {
        "index": 3,
        "description": "Two-ion MS-like gate (single-tone blue-sideband drive)",
        "runner": _run_scenario_3,
        "parameters": SCENARIO_3_PARAMETERS,
    },
    "04_single_ion_stroboscopic_ac_halfpi": {
        "index": 4,
        "description": "Single-ion stroboscopic AC-π/2 drive (Path-A QuTiP-5 compat)",
        "runner": _run_scenario_4,
        "parameters": SCENARIO_4_PARAMETERS,
    },
    "05_single_mode_squeeze_displace": {
        "index": 5,
        "description": "Single-mode squeezing + displacement of a single spin",
        "runner": _run_scenario_5,
        "parameters": SCENARIO_5_PARAMETERS,
    },
}


# ----------------------------------------------------------------------------
# Main operations: stability check + reference generation
# ----------------------------------------------------------------------------


def _arrays_are_bit_identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def stability_check(qc_module: Any, scenario_name: str, *, runs: int = 3) -> bool:
    """Run a scenario ``runs`` times; return True iff all outputs are bit-identical.

    Corresponds to workplan §0.B item 1 / Week 1 item 3. Output is bit-identical
    when the deterministic Lindblad solver produces the same floating-point
    arrays on the current platform with the current dependency stack.
    """
    spec = SCENARIOS[scenario_name]
    print(f"  stability: running {scenario_name} x{runs}")
    first = spec["runner"](qc_module)
    for i in range(1, runs):
        again = spec["runner"](qc_module)
        if not _arrays_are_bit_identical(first, again):
            print(
                f"  stability: FAIL on run {i + 1} — outputs drift between runs.", file=sys.stderr
            )
            return False
    print(f"  stability: OK — {runs} identical runs")
    return True


def generate_reference(qc_module: Any, scenario_name: str) -> Path:
    """Run a scenario once and write ``.npz`` + ``metadata.json`` to disk.

    Returns the path to the created scenario directory.
    """
    spec = SCENARIOS[scenario_name]
    target = REFERENCES_DIR / scenario_name
    target.mkdir(parents=True, exist_ok=True)

    arrays = spec["runner"](qc_module)
    np.savez(target / "arrays.npz", **arrays)

    metadata = {
        "scenario_name": scenario_name,
        "scenario_index": spec["index"],
        "description": spec["description"],
        "qc_legacy_tag": "qc-legacy-v1.0",  # see workplan §8 item 2
        "qc_module_path": str(LEGACY_DIR.relative_to(REPO_ROOT) / "qc.py"),
        "parameters": spec["parameters"],
        "parameters_hash": _parameters_hash(spec["parameters"]),
        "environment": _environment_context(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "observable_keys": sorted(arrays.keys()),
        "n_samples": int(arrays["times"].size),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"  wrote {target.relative_to(REPO_ROOT)} ({len(arrays)} arrays, {metadata['n_samples']} samples)"
    )
    return target


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stability-check",
        action="store_true",
        help="Run each scenario 3 times and verify bit-identical output. Does not write references.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Restrict to one scenario (default: all implemented).",
    )
    args = parser.parse_args(argv)

    qc_module = _load_qc_module()

    scenarios = [
        name
        for name, spec in SCENARIOS.items()
        if args.scenario is None or spec["index"] == args.scenario
    ]

    exit_code = 0
    for name in scenarios:
        try:
            if args.stability_check:
                ok = stability_check(qc_module, name, runs=3)
                if not ok:
                    exit_code = 1
            else:
                generate_reference(qc_module, name)
        except NotImplementedError as exc:
            print(f"  SKIP {name}: {exc}")
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
