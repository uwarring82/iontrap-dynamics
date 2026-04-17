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

Scenario 1 (single-ion carrier flopping, thermal) is fully implemented.
Scenarios 2–5 are stubbed with ``NotImplementedError`` and a comment
pointing at the qc.py method they will call; each un-stubbing is its
own small PR.

QuTiP compat
------------

``qc.py`` was written against QuTiP 4.x and uses ``from qutip import *``
which, under QuTiP 5, does not expose the rotation-operator helpers
``rx``/``ry``/``rz``. These are injected into the ``qc`` module's globals
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
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
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

    import qc  # noqa: PLC0415 — deferred import is deliberate
    qc.rx = rx
    qc.ry = ry
    qc.rz = rz
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
        "sigma_x":      np.real(spin_props[:, 0]),
        "sigma_y":      np.real(spin_props[:, 1]),
        "sigma_z":      np.real(spin_props[:, 2]),
        "spin_entropy": np.real(spin_props[:, 3]),
        "n_mode":       np.real(mode_props[:, 1]),
        "var_x":        np.real(mode_props[:, 2]),
        "var_p":        np.real(mode_props[:, 3]),
        "mode_entropy": np.real(mode_props[:, 4]),
        "mode_X":       np.real(mode_props[:, 5]),
        "mode_P":       np.real(mode_props[:, 6]),
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
    "n_thermal": 0.001,               # near-vacuum to isolate the Fock state
    "Fck": 1,                         # initial motional state |1⟩
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
# Scenarios 3–5 — stubbed (one-PR-per-scenario activation plan)
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
    "nosteps": 50,                   # 50 samples / μs → ~250 samples over tend
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
        "eof":                np.real(spin_props[:, 1]),
        "sigma_x_A":          np.real(spin_props[:, 2]),
        "sigma_y_A":          np.real(spin_props[:, 3]),
        "sigma_z_A":          np.real(spin_props[:, 4]),
        "spin_entropy_A":     np.real(spin_props[:, 5]),
        "sigma_x_B":          np.real(spin_props[:, 6]),
        "sigma_y_B":          np.real(spin_props[:, 7]),
        "sigma_z_B":          np.real(spin_props[:, 8]),
        "spin_entropy_B":     np.real(spin_props[:, 9]),
        "p_down_down":        np.real(spin_props[:, 10]),
        "p_single_flip":      np.real(spin_props[:, 11]),
        "p_up_up":            np.real(spin_props[:, 12]),
        "bell_fidelity":      np.real(spin_props[:, 13]),
        "n_mode":             np.real(mode_props[:, 1]),
        "var_x":              np.real(mode_props[:, 2]),
        "var_p":              np.real(mode_props[:, 3]),
        "mode_entropy":       np.real(mode_props[:, 4]),
        "mode_X":             np.real(mode_props[:, 5]),
        "mode_P":             np.real(mode_props[:, 6]),
    }


def _run_scenario_4(qc_module: Any) -> dict[str, np.ndarray]:
    # BLOCKED on QuTiP 4 → 5 coefficient-callable semantics.
    #
    # qc.single_spin_and_mode_ACpi2 with mod_on=True, mod_type=0 (stroboscopic)
    # builds a time-dependent Hamiltonian H = H0 + mod(t) · HI where
    # mod(t) = spline_mod_fct_fil(t) — a scipy.interpolate.CubicSpline over
    # a filtered square-wave pulse train. In QuTiP 4 this worked because
    # coefficient callables could return a 0-d numpy array; QuTiP 5.1+
    # enforces:
    #
    #     TypeError: The coefficient function must return a number
    #
    # Two resolution paths (neither pure-invoke — both require re-wrapping
    # legacy logic in this generator):
    #
    #   (A) Duplicate `single_spin_and_mode_ACpi2`'s body here and change
    #       `return cs(t)` to `return float(cs(t))` inside spline_mod_fct_fil.
    #       Preserves exact stroboscopic envelope behaviour.
    #
    #   (B) Precompute mod_values = np.array([mod(t, []) for t in tlist])
    #       and pass H = [H0, [HI, mod_values]] to mesolve — the QuTiP 5
    #       idiomatic array-coefficient form. Uses QuTiP's internal
    #       interpolation rather than scipy's CubicSpline; small numerical
    #       drift possible but the physics is the same.
    #
    # Either choice crosses from "faithful legacy replay" into "legacy with
    # fixes". Deferring until the user picks (A) vs (B).
    #
    # Sinusoidal mode (mod_type=1) does NOT trigger the bug — its mod(t) is
    # a scalar-returning cosine — but that's different physics from what
    # workplan §0.B item 4 specifies ("stroboscopic AC-π/2"), so using it as
    # a substitute would mislabel the regression target.
    raise NotImplementedError(
        "scenario 4 (stroboscopic AC-π/2) blocked on QuTiP 5 coefficient-callable "
        "compat; see comments in _run_scenario_4 for resolution paths A/B."
    )


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
    "omega_spin_over_2pi_MHz": 0.0,    # on-resonance carrier
    "r_spin_rad": [0.0, 0.0, 0.0],
    "n_thermal": 0.0,                  # pure vacuum baseline before sq/dis
    "Fck": 0,
    "sq_ampl": 0.5,                    # moderate squeezing (r = 0.5)
    "sq_phi_rad": 0.0,
    "dis_ampl": 1.0,                   # displace by α = 1 → coherent ⟨n⟩ = 1
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
        "description": "Single-ion stroboscopic AC-π/2 drive",
        "runner": _run_scenario_4,
        "parameters": {},
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
            print(f"  stability: FAIL on run {i + 1} — outputs drift between runs.", file=sys.stderr)
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "observable_keys": sorted(arrays.keys()),
        "n_samples": int(arrays["times"].size),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"  wrote {target.relative_to(REPO_ROOT)} ({len(arrays)} arrays, {metadata['n_samples']} samples)")
    return target


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stability-check", action="store_true",
        help="Run each scenario 3 times and verify bit-identical output. Does not write references.",
    )
    parser.add_argument(
        "--scenario", type=int, default=None, choices=[1, 2, 3, 4, 5],
        help="Restrict to one scenario (default: all implemented).",
    )
    args = parser.parse_args(argv)

    qc_module = _load_qc_module()

    scenarios = [
        name for name, spec in SCENARIOS.items()
        if args.scenario is None or spec["index"] == args.scenario
    ]

    exit_code = 0
    for name in scenarios:
        spec = SCENARIOS[name]
        try:
            if args.stability_check:
                ok = stability_check(qc_module, name, runs=3)
                if not ok:
                    exit_code = 1
            else:
                generate_reference(qc_module, name)
        except NotImplementedError as exc:
            print(f"  SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — summarise per-scenario failures
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
