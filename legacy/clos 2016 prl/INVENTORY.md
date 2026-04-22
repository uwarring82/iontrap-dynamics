# Inventory — `clos 2016 prl`

Reference bundle for the 2016 spin-boson thermalization paper. This folder
is a **legacy archive** (read-only, provenance-only) kept alongside the
current `iontrap-dynamics` package so that future work — reproducing
figures, benchmarking against a known quantum-simulator dataset, or
extending the IPR / ETH diagnostics — has a single place to look.

## 1. Paper

- **Reference (by DOI):** [10.1103/PhysRevLett.117.170401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.170401) — *Phys. Rev. Lett.* **117**, 170401 (2016).
- **Author prefixes recoverable from the bundle**
  - *GC* — Govinda Clos (experiment; drafted `sbpaper`; primary email `govinda.clos@physik.uni-freiburg.de` on `.git/logs/HEAD`).
  - *DP* — Diego Porras (numerics; folder prefix `DP num res_*`).
  - *U. Warring* — named on this repository; co-author on the paper.
- **Physics, as recoverable from the data / scripts.** Trapped-ion
  spin–boson quantum simulator. For `N = 1 … 5` ions, the Hamiltonian
  is the canonical carrier + Lamb–Dicke displacement,
  $H = (\Omega/2)\,\sigma^{+}\,e^{\eta_0\sum_n \phi_n(a_n - a_n^{\dagger})/\sqrt{\omega_n/\omega_1}} + \text{h.c.} + \sum_n \omega_n a_n^{\dagger}a_n + (\omega_z/2)\sigma^z,$
  with the spin–boson coupling built from the axial normal modes
  (`cchain(N, center)` in the MATLAB scripts). Observables of interest
  are IPR (inverse participation ratio), the effective dimension
  $d_{\rm eff} = 1/\text{IPR}$, and eigenstate-thermalization
  diagonal overlaps $\langle\sigma^{x,y,z}\rangle_{\alpha}$.

> The DOI link above is authoritative for citation. This note only
> summarises *what the bundle contains*, not the paper's full physics.

## 2. Provenance

- **Compiled by:** U. Warring (assembled the combined folder for archival).
- **Numerical results:** Diego Porras (MATLAB + `.dat`), dated 2015-07-04 through 2015-07-30.
- **Paper drafting:** Govinda Clos (LaTeX, inside `GC collection/sbpaper/`), March–April 2015 per the git reflog.
- **Dates on inventory:** 2015-03-15 (earliest MATLAB script) through 2015-07-30 (latest numerical-result folder). The PRL was published 2016-10-21; this bundle is pre-submission working material.

## 3. Top-level contents

| Path                                                           | Type      | Purpose                                                                                 |
|----------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------|
| `1ions_ipr_vs_nc.txt`                                          | TSV (small) | IPR vs. phonon-cutoff `nc` convergence data, N = 1 ion.                                |
| `2ions_ipr_vs_nc.txt`                                          | TSV       | Same, N = 2.                                                                            |
| `3ions_ipr_vs_nc.txt`                                          | TSV       | Same, N = 3.                                                                            |
| `4ions_ipr_vs_nc.txt`                                          | TSV       | Same, N = 4.                                                                            |
| `5ions_ipr_vs_nc.txt`                                          | TSV       | Same, N = 5.                                                                            |
| `Emergence of statistical physics in a trapped ion.pptx`       | MS PowerPoint (~1 MB) | Talk slides; presentation version of the paper.                                  |
| `SummaryETHVsDet_AllIons_fi3and4.dat`                          | binary `.dat` (7.5 KB) | Combined ETH-vs-detuning summary for Figs. 3 and 4, all ion numbers.       |
| `ergodic_2015_03_15.m`                                         | MATLAB script | Earliest standalone driver — full calculation of `[meanE, DE, enerf, szdiag, sxdiag, nphdiag_cell, sz, sx, sy, IPR, IPR0, ratio_IPR]`. Signature documents the ETH inputs. |
| `ergodic_band_matrix_2015_05_07.m`                             | MATLAB script | May 2015 revision exploring band-matrix structure of the Hamiltonian.                |
| `ergodic_inf_cutoff_2015_04_21.m`                              | MATLAB script | April 2015 revision probing large-`nc` limit / cutoff convergence.                   |

The script signature in `ergodic_2015_03_15.m` (lines 1–28) is the
clearest entry point to the whole bundle. Inputs: `N`, `nc`, `n_val`
(mean phonon occupations, assumed thermal per mode), `Omega`,
`omegaz`, `axial`, `theta`, `phi`. Output: level-resolved expectation
values plus IPR diagnostics. The numerical-result `.dat` folders below
are all downstream of the same driver (or closely related variants).

## 4. Subfolders

### 4.1 `DP num res_fig_1_2015_07_10/`

Theoretical effective-dimension data for Fig. 1, cut on 2015-07-10.

- `theo_dim_N_1.dat` … `theo_dim_N_5.dat` — per-ion-count `d_eff` vs. detuning / coupling axis. Sample first line of `theo_dim_N_1.dat`: six columns, `Var1_1 … Var1_6`, with `Var1_2` ≈ 2.28 saturating at large values — consistent with `d_eff` tending to a plateau as IPR → 1/d.
- `theo_dim_3d_final.dat` — 3D surface over `(N, det, coupling)` assembled from the per-N slices for the final Fig. 1 rendering.

### 4.2 `DP num res_fig_1_2015_07_30/`

Revised Fig. 1 data, 2015-07-30 (three weeks after §4.1).

Same file set as §4.1 plus `SummaryDeffVsDetAllIons.dat`. This supersedes §4.1 for any Fig. 1 reproduction — use 07-30 not 07-10.

### 4.3 `DP num res_ETH_fig_3_2015_07_30/`

Experimental ETH overlay data for Fig. 3 ("ETH holds / violates"
discriminator), 2015-07-30.

- `exp_ETH_N_1.dat` … `exp_ETH_N_5.dat` — per-ion-count experimental traces aligned with the theoretical ETH prediction.
- `SummaryETHVsDet_AllIons_fi3and4.dat` — duplicate of the top-level summary file, scoped to this figure's source.

### 4.4 `DP num res_fig_4_2015_07_04/`

Experimental effective-dimension data for Fig. 4, 2015-07-04.

- `exp_dim_N_1.dat` … `exp_dim_N_5.dat` — experimental `d_eff` traces per ion number.
- `SummaryDeffVsDetAllIons.{csv,dat,txt}` — three formats of the same combined summary (Porras exported all three for downstream convenience).

### 4.5 `DP num res_fig_4_2015_07_17/`

Revised Fig. 4 data, 2015-07-17. Supersedes §4.4 — `exp_dim_N_*` values refit after two weeks of analysis. `SummaryDeffVsDetAllIons.dat` only (no `.csv`/`.txt` variants).

### 4.6 `GC collection/matlabscripts/`

113 files total. Mix of MATLAB scripts (`.m`) and saved workspaces (`.mat`). Ordering by likely role:

- **Drivers** (per-ion-count sweep scripts)
  - `ipr_N.m`, `ipr_N_atBest.m`, `ipr_N_atDiego.m` — IPR over N, different parameterisations.
  - `ipr_eta.m`, `ipr_nbar.m`, `ipr_omegaz.m`, `ipr_omegazN5.m`, `ipr_omegaz_eta1.m` — IPR over single-parameter sweeps (Lamb–Dicke parameter η, mean occupation `nbar`, carrier detuning `ωz`).
  - `ipr_3d.m`, `ipr_3d_av.m`, `ipr_3d_av_Ns.m` — 3D-grid IPR sweeps.
  - `ipr_script.m`, `ipr0.m` — IPR and baseline-IPR (zero coupling).
- **Core calculator**
  - `ergodic.m` — successor to the top-level `ergodic_2015_03_15.m`.
  - `ergodic_ipr.m`, `ergodic_ipr_av.m`, `ergodic_ipr_av_eta.m`, `ergodic_ipr_av_new.m`, `ergodic_ipr_av_new_eigs.m`, `ergodic_ipr_gc.m` — variants with different averaging schemes; `_gc` is Govinda's personal copy.
  - `diffcutoffs.m` — cutoff-convergence sweep; directly produces the `*ions_ipr_vs_nc.txt` files at the top level.
  - `integration_times.m` — time-grid choices for the dynamics.
  - `etatest.m` — Lamb–Dicke parameter sanity probe.
- **Paper-integration script**
  - `Warring-jh.m` — named for U. Warring (one-off integration / figure-prep script).
- **Saved workspaces (`.mat`)** — pre-computed results for fast figure regeneration. Naming convention `N{ions}Ax{axial-idx}nval{nbar}.mat` or `{...}integrals.mat` for intermediate integrals; `nions{N}_allinone_nbar0.5_t{T}fix.mat` for time-averaged runs. `axials_n123_allinone.mat` combines per-N axial-frequency sweeps for N=1,2,3 into one workspace. `nions3_allinone_nbar05_t{10,20,30,50}fix.mat` are the same calculation at different total integration times (convergence study).

### 4.7 `GC collection/sbpaper/` — local-only

**The entire `sbpaper/` subtree is excluded from outer-repo tracking
via `.gitignore` (`legacy/**/sbpaper/`) and is not pushed to `origin`.**

Why excluded:

- `sbpaper/` contains its own `.git/` — an active nested repository.
  Committing it alongside the outer `iontrap-dynamics` repo would
  either shadow the nested repo (its `.git/` contents get ignored
  but the folder is added as a gitlink, creating a submodule-ish
  stub with no registered URL) or add the nested tree as loose files
  and bloat this repo by ~75 MB of pre-publication PDFs / `.dat`
  traces. Neither is appropriate for a reference bundle.
- Everything downstream in `iontrap-dynamics` that actually
  cross-references the paper (data tables for Figs. 1 / 3 / 4, the
  MATLAB drivers, the ergodic / IPR observable definitions) already
  lives *outside* `sbpaper/` — under `DP num res_*` and
  `GC collection/matlabscripts/`, which **are** tracked.

What is on disk locally (informational — not on `origin`):

- `sbpaper/.git/` — intact git repo (308 files at `HEAD`) with the
  paper-drafting history (13 commits, 2015-03 through 2015-04).
- `sbpaper/.gitignore`, `sbpaper/comments/`, `sbpaper/data/`,
  `sbpaper/figures/` — checked-out working tree. The top-level
  paper sources (`main.tex`, `main.pdf`, `main.bbl`, `main.blg`)
  plus `literature/` and `sources/` are referenced by
  `git ls-tree HEAD` but are not present in the local checkout
  (partial clone).

To restore / access full content locally: check out `sbpaper` from
its original upstream (likely tied to
`govinda.clos@physik.uni-freiburg.de`) rather than from this bundle.
This inventory captures the *provenance* (dates, authors, structure)
but not the commit blobs.

Reflog summary (read at ingest time from `sbpaper/.git/logs/HEAD`,
preserved here since the `.git/` itself does not ship to origin):

| Date (approx.) | Commit                   | Message                                             |
|----------------|--------------------------|-----------------------------------------------------|
| 2015-03-24     | `a3b33402…`              | Initialer Commit. Spin Boson Papier Stand 24.03.2015. |
| 2015-03-24     | `22ca5b33…`              | Add literature.                                     |
| 2015-03-24     | `f4c9c201…`              | Text for Figs. 1--3.                                |
| 2015-03-25     | `8e69ddf1…`              | Started introduction.                               |
| 2015-03-26     | `6e7a3b3f…`              | Minor changes on main.                              |
| 2015-03-26     | `f7732f13…`              | Worked on Supp. (× 3)                               |
| 2015-03-30     | `5eec07cc…`              | Worked on Supp.                                     |
| 2015-03-30     | `99a6506c…`              | Enhanced transitions in Fig. 1.                     |
| 2015-03-30     | `a6b76733…`              | Minor changes. Version v2 for Diego.                |
| 2015-04-03     | `7fb16039…`              | New fig.1 with IPR.                                 |
| 2015-04-10     | `48e8b336…`              | Lots of Fig. changes.                               |
| 2015-04-13     | `228908e6…`              | Refined figures. *(last reflog entry)*              |

The last `COMMIT_EDITMSG` shows the staged payload for that final
commit: adds `data/1ion_ETH.txt`, `data/1ion_close_to_ETH.txt`,
`data/1ion_far_from_ETH.txt`, `data/2ion_ETH.txt`, `…`, plus updates
to the `*_stats_from*period.txt` and `wz*_{exp,num}.dat` data files.
This is paper-side data layout; the numerical sources live in
`DP num res_*` above, not here.

This reflog is the only part of `sbpaper/` that ships with the outer
repo (via this inventory). The tree and object database are
intentionally untracked per the rationale above; to rebuild the paper
PDF, work against the upstream repository, not against `origin`
`iontrap-dynamics`.

## 5. Connection to `iontrap-dynamics` (current repo)

Points of contact that future work might exercise:

- **Hamiltonian families.** `ergodic.m` builds the carrier + full
  exponential Lamb–Dicke operator $\exp(\eta_0 \sum_n \phi_n (a_n -
  a_n^\dagger)/\sqrt{\omega_n/\omega_1})$. Current builders in
  `src/iontrap_dynamics/hamiltonians.py` apply a *leading-order*
  Lamb–Dicke expansion; reproducing Fig. 1 quantitatively requires
  either extending `modulated_carrier_hamiltonian` with an
  `ld_order=None` mode (discussed in
  `tests/regression/migration/test_migration_references.py:424`
  as an open activation path), or implementing a dedicated
  full-exponential carrier builder.
- **Multi-mode bath construction.** `cchain(N, center)` in the MATLAB
  driver computes axial-chain normal modes directly. The
  `iontrap-dynamics` equivalent is `src/iontrap_dynamics/modes.py`
  → `axial_modes(...)`. Cross-check: same convention for COM mode
  participation weights $\phi_n(i)$ at `center` index.
- **Observables not currently in the library.** IPR and effective
  dimension $d_{\rm eff} = 1/\text{IPR}$ are not part of
  `src/iontrap_dynamics/observables.py`. Adding them (as level-basis
  projectors summed over the evolved state) would unlock reproduction
  of Figs. 1 and 4 directly. The ETH diagonal overlaps $\langle \sigma^{x,y,z}\rangle_\alpha$
  are also absent; they require diagonalising $H$ and reading off the
  expectation values per eigenstate — currently one would have to
  compute this outside the solver path.
- **Ensemble sweeps.** Fig. 4's detuning scans at fixed `nbar`
  parameterise exactly the workload that
  `src/iontrap_dynamics/sequences.py` → `solve_ensemble` targets —
  see Dispatch Y's joblib parallel dispatch. A future reproducibility
  dispatch would load these `.dat` files as regression references and
  re-compute the theoretical traces through `solve_ensemble`.
- **Paper-to-library cross-references.** If a dispatch that reproduces
  Fig. 1 or Fig. 4 lands in `iontrap-dynamics`, it should cite back
  to this inventory (stable path: `legacy/clos 2016 prl/INVENTORY.md`).

## 6. Known gaps and caveats

- `sbpaper/.git/refs/` is missing; the git directory is reduced and cannot be browsed with normal git commands. Reflog-only recovery (§4.7).
- The `.pptx`, the summary `.dat`, and several `.mat` files have restrictive permissions (`-rw-------` or `-rwx------`) that may confuse tooling; adjust locally if needed for read access, leave repository permissions untouched.
- No explicit `LICENCE` or re-use statement accompanies this bundle. Treat all contents as the authors' pre-publication material — cite via the DOI in §1, do not redistribute the raw files externally.
- The paper authoritative citation (title, author list, full abstract) is not reproduced here deliberately; this note is an inventory, not a paper summary. Follow the DOI link for the canonical record.
