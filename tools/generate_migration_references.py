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
# Scenario 1 — single-ion carrier flopping, thermal
# ----------------------------------------------------------------------------
#
# qc.py method: `QC.squeeze_to_entangle_singleSpin_singleMode`
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
    """Invoke qc.py for scenario 1 and return arrays keyed by observable.

    Output keys:

    * ``times`` — 1-D, μs (qc.py's native time units).
    * ``sigma_x``, ``sigma_y``, ``sigma_z`` — spin expectations.
    * ``spin_entropy`` — ⟨−Tr(ρ log ρ)⟩ on the spin reduced state.
    * ``n_mode`` — ⟨n⟩ motional occupation.
    * ``var_x``, ``var_p`` — position and momentum variances.
    * ``mode_entropy`` — entropy of the mode reduced state.
    * ``mode_X``, ``mode_P`` — ⟨X⟩ and ⟨P⟩ quadratures.
    """
    q = qc_module.QC()
    p = SCENARIO_1_PARAMETERS
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
# Scenarios 2–5 — stubbed (one-PR-per-scenario activation plan)
# ----------------------------------------------------------------------------

def _run_scenario_2(qc_module: Any) -> dict[str, np.ndarray]:
    # TODO: red-sideband flopping from Fock |1⟩. Invokes
    # qc.squeeze_to_entangle_singleSpin_singleMode with Fck=1,
    # omega_z = -omega_1 (red detune), n_th ≈ 0.
    raise NotImplementedError("scenario 2 (red-sideband Fock|1⟩) not yet ported")


def _run_scenario_3(qc_module: Any) -> dict[str, np.ndarray]:
    # TODO: two-ion Mølmer–Sørensen gate. Invokes
    # qc.squeeze_to_entangle_twoSpins_singleMode with symmetric red/blue
    # detunings producing the MS interaction.
    raise NotImplementedError("scenario 3 (two-ion MS gate) not yet ported")


def _run_scenario_4(qc_module: Any) -> dict[str, np.ndarray]:
    # TODO: single-ion stroboscopic AC-π/2. Invokes
    # qc.single_spin_and_mode_ACpi2 with mod_on=True and the canonical
    # strobo_dur.
    raise NotImplementedError("scenario 4 (stroboscopic AC-π/2) not yet ported")


def _run_scenario_5(qc_module: Any) -> dict[str, np.ndarray]:
    # TODO: single-mode squeezing + displacement. Invokes
    # qc.squeeze_to_entangle_singleSpin_singleMode with non-trivial sq_a
    # and dis_a; drive off-resonance so the squeezing is observable.
    raise NotImplementedError("scenario 5 (squeeze+displace) not yet ported")


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
        "parameters": {},
    },
    "03_two_ion_ms_gate": {
        "index": 3,
        "description": "Two-ion Mølmer–Sørensen gate (bichromatic MS interaction)",
        "runner": _run_scenario_3,
        "parameters": {},
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
        "parameters": {},
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
