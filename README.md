# EDMA — Affinity-Aware Embedding Division Multiple Access

Reproducibility package for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Affinity-Aware Embedding
> Division Multiple Access for Multi-User Semantic Communications,"
> submitted to *IEEE Transactions on Vehicular Technology*, 2026.

This repository contains the simulation code, the raw result data, and
the figure files behind every numerical claim in the paper. It is
private during peer review and will be made public upon publication.

## Layout

| Folder | Contents |
|---|---|
| `code/` | Simulation and plotting scripts (Python, CPU only) |
| `data/` | Raw results written by the scripts, one CSV per experiment |
| `fig/` | Figure PDFs included in the manuscript |

## Requirements

Python 3.10 or later with `numpy` and `matplotlib`. The Fig. 4
experiment additionally uses `torch` (CPU build is sufficient). No GPU
is required. Every script fixes the seed 2026 and writes its raw output
to `data/`, so plotting is decoupled from simulation.

## Reproducing the figures

Run the scripts from inside `code/`.

| Figure | Content | Script | Data |
|---|---|---|---|
| Fig. 2 | Per-user MSE and self-interference floor | `revision_sims.py E1` | `floor_validation.csv` |
| Fig. 3 | Effective sum rate at the CLIP affinity | `revision_sims.py E7a` | `rate_corrected.csv` |
| Fig. 4 | Cosine recovery on real BERT+ViT pairs | `fig_real_merged.py`, then `refine_matched.py`; replot with `replot_merged.py` | `bertvit_merged.csv` |
| Fig. 5 | Realizable versus genie-aided SIC | `revision_sims.py E2`; replot with `replot_sic.py` | `sic_comparison.csv` |
| Fig. 6 | Affinity sweep and crossover | `revision_sims.py E7a` | `beta_sweep_corrected.csv` |
| Fig. 7 | Multi-user scaling | `revision_sims.py E7c` | `multiuser_corrected.csv` |

Fig. 1 is a system diagram and has no simulation behind it.

Quantities quoted in the text but not plotted come from the same
driver: `revision_sims.py E0` writes `theorem_check.csv` (Theorem 1
constants), `E4` writes `csi_error.csv` (imperfect-CSI robustness), and
`E5` writes `mask_family_rev.csv` (Walsh–Hadamard versus Haar masks).
`revision_sims.py` with no argument runs every experiment.

## Verifying the analysis

`verify_math.py` re-derives every closed-form expression in the paper
numerically and prints one PASS/FAIL line per item, covering the
per-realization Gram identity, Theorem 1 and its self-interference
constants, the effective-SINR corollary, the MAC-consistency
proposition, the wideband limit, both crossover conditions, the
affinity-mismatch bound, the CSI-invariance identity, the multi-user
inverse formula, and the Walsh–Hadamard construction. It depends only
on `numpy`.

## Conventions

The scripts follow the manuscript exactly: unit per-block transmit
energy `E_b = 1` per user, `rho = E_b / sigma_n^2` as the per-block
SNR with per-symbol SNR `rho/d`, complex block-Rayleigh gains unless
the evaluation point `h_u = 1` is stated, real unit-norm embeddings,
and masks drawn fresh from the Haar mixture on every realization.

`fig_real_merged.py` also produces columns for a retrained
attention-based receiver. Those columns are kept in `bertvit_merged.csv`
for completeness but are not used by any figure in the paper.
`revision_sims.py E6` covers a high-affinity combining mode that is
outside the scope of this paper.

## Citation and license

Citation details and a license will be added when the paper is
published.
