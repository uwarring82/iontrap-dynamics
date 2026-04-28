# SPDX-License-Identifier: MIT
"""Generate the two schematic figures embedded in docs/overview.md.

These are pedagogical schematics, not data plots. They are committed to
docs/figures/ so the rendered mkdocs site has them available without a
separate generation step at build time.

Run with:

    python tools/generate_overview_figures.py

Outputs:
    docs/figures/ion_trap_schematic.png
    docs/figures/open_system_memory.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

# Neutral palette — readable in light and dark mkdocs themes.
COLOR_ION = "#1f4f8b"
COLOR_LASER = "#c45a00"
COLOR_TRAP = "#888888"
COLOR_SPIN = "#2a2a2a"
COLOR_MOTION = "#2a2a2a"
COLOR_DENSITY = "#1f4f8b"
COLOR_TRAJ = "#c45a00"
COLOR_GRID = "#dddddd"


def _ion_trap_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Trap potential (parabola) along the bottom.
    x_trap = np.linspace(2.5, 7.5, 200)
    y_trap = 0.7 + 0.18 * (x_trap - 5.0) ** 2
    ax.plot(x_trap, y_trap, color=COLOR_TRAP, linewidth=2.0)
    ax.text(
        7.7,
        1.0,
        "trap potential",
        ha="left",
        va="center",
        color=COLOR_TRAP,
        fontsize=10,
        style="italic",
    )

    # The ion sits at the bottom of the well.
    ion_x, ion_y = 5.0, 0.85
    ax.add_patch(plt.Circle((ion_x, ion_y), 0.20, color=COLOR_ION, zorder=5))
    ax.text(ion_x + 0.32, ion_y, "²⁵Mg⁺", ha="left", va="center", color=COLOR_ION, fontsize=10)

    # Two laser beams approaching the ion.
    ax.annotate(
        "",
        xy=(ion_x - 0.18, ion_y + 0.15),
        xytext=(2.6, 3.4),
        arrowprops=dict(arrowstyle="->", color=COLOR_LASER, linewidth=1.6),
    )
    ax.text(2.45, 3.55, "laser 1", ha="left", va="bottom", color=COLOR_LASER, fontsize=10)
    ax.annotate(
        "",
        xy=(ion_x + 0.18, ion_y + 0.15),
        xytext=(7.4, 3.4),
        arrowprops=dict(arrowstyle="->", color=COLOR_LASER, linewidth=1.6),
    )
    ax.text(7.55, 3.55, "laser 2", ha="right", va="bottom", color=COLOR_LASER, fontsize=10)

    # Spin levels — right-hand panel.
    spin_x0, spin_x1 = 8.6, 9.6
    g_y, e_y = 1.6, 4.4
    ax.hlines([g_y, e_y], spin_x0, spin_x1, colors=COLOR_SPIN, linewidth=2.0)
    ax.text(spin_x1 + 0.1, g_y, r"$|g\rangle$", va="center", fontsize=11)
    ax.text(spin_x1 + 0.1, e_y, r"$|e\rangle$", va="center", fontsize=11)
    ax.annotate(
        "",
        xy=((spin_x0 + spin_x1) / 2, e_y - 0.12),
        xytext=((spin_x0 + spin_x1) / 2, g_y + 0.12),
        arrowprops=dict(arrowstyle="<->", color=COLOR_SPIN, linewidth=1.2),
    )
    ax.text(
        8.55,
        (g_y + e_y) / 2,
        "spin",
        ha="right",
        va="center",
        color=COLOR_SPIN,
        fontsize=10,
        style="italic",
    )

    # Motional ladder — left-hand panel.
    mot_x0, mot_x1 = 0.4, 1.4
    mot_levels = np.array([1.6, 2.4, 3.2, 4.0, 4.8])
    mot_labels = [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$", r"$|3\rangle$", r"$|4\rangle$"]
    ax.hlines(mot_levels, mot_x0, mot_x1, colors=COLOR_MOTION, linewidth=1.5)
    for y_level, lab in zip(mot_levels, mot_labels, strict=True):
        ax.text(mot_x0 - 0.15, y_level, lab, ha="right", va="center", fontsize=10)
    ax.text(
        (mot_x0 + mot_x1) / 2,
        5.3,
        "motion",
        ha="center",
        va="bottom",
        color=COLOR_MOTION,
        fontsize=10,
        style="italic",
    )

    # Title.
    ax.text(
        5.0,
        5.7,
        "A single trapped ion: spin coupled to motion via lasers",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _open_system_memory(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ----- Left panel: density-matrix mesolve, dim^2 footprint -----
    ax = axes[0]
    ax.set_aspect("equal")
    n_grid = 8
    for i in range(n_grid):
        for j in range(n_grid):
            ax.add_patch(
                plt.Rectangle(
                    (j, n_grid - 1 - i),
                    1,
                    1,
                    facecolor=COLOR_DENSITY,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.85,
                )
            )
    ax.set_xlim(-0.5, n_grid + 0.5)
    ax.set_ylim(-1.5, n_grid + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        n_grid / 2,
        n_grid + 0.5,
        r"density matrix $\rho$",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(n_grid / 2, -0.5, r"memory $\propto$ dim$^{2}$", ha="center", va="top", fontsize=11)
    ax.text(
        n_grid / 2,
        -1.15,
        "single dense ODE; saturates GPU VRAM\nearlier than CPU saturates RAM",
        ha="center",
        va="top",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )

    # ----- Right panel: trajectory ensemble, dim per trajectory -----
    ax = axes[1]
    ax.set_aspect("equal")
    n_traj = 6
    n_states = 8
    spacing = 1.4
    for k in range(n_traj):
        x0 = k * spacing
        for i in range(n_states):
            ax.add_patch(
                plt.Rectangle(
                    (x0, n_states - 1 - i),
                    1,
                    1,
                    facecolor=COLOR_TRAJ,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.85,
                )
            )
        ax.text(x0 + 0.5, -0.3, f"#{k + 1}", ha="center", va="top", fontsize=8.5, color="#666666")
    ax.set_xlim(-0.7, n_traj * spacing - 0.4)
    ax.set_ylim(-1.6, n_states + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        n_traj * spacing / 2 - 0.4,
        n_states + 0.5,
        "trajectory ensemble (Monte Carlo)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        n_traj * spacing / 2 - 0.4,
        -0.7,
        r"memory $\propto$ dim per trajectory",
        ha="center",
        va="top",
        fontsize=11,
    )
    ax.text(
        n_traj * spacing / 2 - 0.4,
        -1.25,
        "trajectories run independently in parallel\n— the workload shape GPUs are designed for",
        ha="center",
        va="top",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )

    # Top-level title across both panels.
    fig.suptitle(
        "Two ways to simulate dissipative dynamics — and why the GPU answer differs",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ion_trap_schematic(OUT_DIR / "ion_trap_schematic.png")
    _open_system_memory(OUT_DIR / "open_system_memory.png")
    print(f"Wrote {OUT_DIR / 'ion_trap_schematic.png'}")
    print(f"Wrote {OUT_DIR / 'open_system_memory.png'}")


if __name__ == "__main__":
    main()
