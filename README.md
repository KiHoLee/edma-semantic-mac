# EDMA — Affinity-Aware Embedding Division Multiple Access

Reproducibility package for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Affinity-Aware Embedding
> Division Multiple Access for Multi-User Semantic Communications,"
> submitted to *IEEE Transactions on Vehicular Technology*, 2026.

This repository contains the simulation code, the raw result data, and
the figure files behind every numerical claim in the paper. It is
publicly available during peer review so that the editors and
reviewers can inspect and rerun every experiment.

The design under test: each user applies an independent Haar
orthogonal mask, and the receiver runs a matched filter followed by
the closed-form affinity-aware Wiener demultiplexer, which harvests
the coherent interference component that the measured pairwise
affinity `beta` predicts. The affinity-blind reference sets `beta = 0`
in the same filter.

## Layout

| Folder | Contents |
|---|---|
| `code/` | Simulation and plotting scripts (Python) |
| `data/` | Raw results written by the scripts, one CSV per experiment |
| `fig/`  | Figure PDFs included in the manuscript (`block_diagram_src.tex` is the TikZ source of Fig. 1) |

## Requirements

Python 3.10 or later with `numpy` and `matplotlib`. The Monte Carlo
experiments in `revision_sims_gpu.py`, `fig_real_merged.py`, and
`refine_matched.py` use `torch` (CUDA when available; the scripts fall
back to CPU). All random draws come from the numpy generator with the
fixed seed 2026 — torch only accelerates QR, matrix products, and
linear solves — and every script writes its raw output to `data/`, so
plotting is fully decoupled from simulation.

## Reproducing the figures

Run the scripts from inside `code/`. All plots are rendered from
`data/` only, by `replot_all.py` (Figs. 2, 3, 5, 6, 7) and
`replot_merged.py` (Fig. 4).

| Figure | Content | Simulation | Data |
|---|---|---|---|
| Fig. 1 | System diagram | `latexmk -pdf fig/block_diagram_src.tex` | — |
| Fig. 2 | Per-user MSE, aware vs blind floor | `revision_sims.py E1` | `floor_validation.csv` |
| Fig. 3 | Effective sum rate at the CLIP affinity | `revision_sims.py E7a` | `rate_corrected.csv` |
| Fig. 4 | Cosine recovery on real BERT+ViT pairs | `fig_real_merged.py`, then `refine_matched.py` | `bertvit_merged.csv` |
| Fig. 5 | Receiver comparison under Rayleigh fading | `revision_sims_gpu.py E2` | `sic_comparison.csv` |
| Fig. 6 | Value of the measured affinity | `revision_sims.py E7a` | `beta_sweep_corrected.csv` |
| Fig. 7 | Multi-user scaling (joint Wiener) | `revision_sims_gpu.py E7c` | `multiuser_corrected.csv` |

Quantities quoted in the text but not plotted come from the same
drivers: `revision_sims.py E0` writes `theorem_check.csv` (Theorem 1
validation across affinities and channel phases),
`revision_sims_gpu.py E3` writes `rayleigh_mse.csv` (unconditional
Rayleigh MSE), `E4` writes `csi_error.csv` (imperfect-CSI
robustness), `E5` writes `mask_family_rev.csv` (Walsh–Hadamard versus
Haar), `E8` writes `mismatch.csv` (affinity mismatch and
quantization), and `E9` writes `cosine_ceiling.csv` (cosine-ceiling
corollary check). The empirical affinity statistics quoted in the
manuscript are recomputable from `clip_realdata_beta.csv` and
`bert_vit_beta.csv` (32 paired and 32 unpaired samples per encoder
family), and the trained refinement gates behind the capacity-check
claim are stored in `refine_gates.npz`.

## Verifying the analysis

`verify_math.py` re-derives every closed-form claim numerically and
prints one PASS/FAIL line per item: Theorem 1 at the equal-gain point
and under random channel phases for both users, the aware and blind
error floors and the value-of-affinity ratio, the cosine-ceiling
corollary, the blind-receiver/matched-filter cosine equivalence, the
monotonicity proposition, the full-cooperation bound with its
equality case at `beta = 1`, the finite-SNR MAC-condition boundary,
the exact Walsh–Hadamard closed form, the quadratic mismatch
stationarity, and the dominated floor of the correlated-mask
alternative from the Appendix. It depends only on `numpy`.

## Conventions

The scripts follow the manuscript exactly: unit per-block transmit
energy `E_b = 1` per user, `rho = E_b / sigma_n^2` as the per-block
SNR with per-symbol SNR `rho/d`, complex block-Rayleigh gains unless
the evaluation point `h_u = 1` is stated, real unit-norm embeddings
with the orientation `<e1, e2> = +beta`, and independent Haar masks
drawn fresh on every realization.

## Citation and license

Until the paper is published, cite the submitted manuscript listed
at the top of this file. A formal citation entry and a license
will be added upon publication.
