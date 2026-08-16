"""
Simulations for the EDMA TVT manuscript (v2 design).
=======================================================
Design v2: each user applies an independent orthogonal mask; the
receiver runs one matched filter per user followed by the
affinity-aware linear MMSE demultiplexer, which exploits the
coherent interference component that the pairwise affinity beta
predicts.  Per-realization statistic for user 1 (c1 = h2/h1):

    t1 = (I + beta*c1*Q) e1 + sqrt(g)*c1*Q w + n_t,   Q = M1^T M2,

and the demultiplexer is the Wiener filter

    e1_hat = (1/d) A^H (A A^H/d + (g|c1|^2/d + sig^2/|h1|^2) I)^{-1} t1,
    A = I + beta*c1*Q,  g = 1 - beta^2.

Closed form (Theorem 1, d -> inf, per channel realization):

    MSE_1 = rho_e / sqrt((1 + beta^2|c1|^2 + rho_e)^2 - 4 beta^2|c1|^2),
    rho_e = g|c1|^2 + d sig^2/|h1|^2;   floor at |c1| = 1: sqrt(g)/2.

The affinity-blind receiver (beta = 0 in the filter) reduces to a
scalar shrinkage of the matched filter with floor 1/2, so the entire
cosine gain of the aware receiver is attributable to the predicted
affinity.  Effective SINR: eta = 1/MSE - 1 (biased MMSE convention).

Experiments in this file (CPU, numpy):
  E0  theorem_check      : closed form vs Monte Carlo, both users
  E1  fig_floor          : per-user MSE vs block SNR, aware vs blind floor
  E7a rate_corrected + beta_sweep_corrected : closed-form rate curves

The Monte Carlo experiments E2, E3, E4, E5, E7c, E8, E9 are canonical
in revision_sims_gpu.py (torch backend, run under WSL); figures are
rendered from data/ by replot_all.py and replot_merged.py.

Conventions (identical to the manuscript):
  * unit per-block transmit energy E_b = 1 per user
  * rho = E_b / sigma_n^2  (per-block received SNR; per-symbol SNR rho/d)
  * block-Rayleigh h ~ CN(0,1) unless the AWGN point |h|=1 is stated
  * complex AWGN CN(0, sigma^2 I_d); embeddings real, unit norm
  * orientation convention <e1,e2> = +beta
Fixed seed 2026.  CSVs -> ../data, PDFs -> ../fig.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data";    CSV_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "fig";     FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "grid.alpha": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "figure.figsize": (3.15, 2.36), "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.205, right=0.965, top=0.955, bottom=0.185)

# shared legend-label dictionary (single source for every figure)
LBL = {
    "edma":  "EDMA",
    "blind": "Affinity-blind",
    "oma":   "OMA",
    "genie": "Genie-aided SIC bound",
    "sic":   "Realizable analog SIC",
    "todma": "ToDMA-adapted",
    "mac":   "MAC sum capacity",
    "hybrid": "EDMA + refinement stage",
    "haar":  "Haar masks",
    "wh":    "Walsh-Hadamard masks",
}

rng = np.random.default_rng(2026)


def save_fig(fig, name):
    p = FIG_DIR / f"{name}.pdf"
    fig.subplots_adjust(**AXES_RECT)
    fig.savefig(p)
    plt.close(fig)
    print(f"[OK] wrote {p}")


def write_csv(name, header, rows):
    p = CSV_DIR / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[OK] wrote {p}")


# ------------------------------------------------------------------
# Core constructions
# ------------------------------------------------------------------
def haar(d):
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))


def unit(v):
    return v / np.linalg.norm(v)


def embed_pair(d, beta):
    """e1, e2 real unit vectors with <e1,e2> = +beta."""
    e1 = unit(rng.standard_normal(d))
    w  = rng.standard_normal(d)
    w  = unit(w - (w @ e1) * e1)
    e2 = beta * e1 + math.sqrt(1.0 - beta**2) * w
    return e1, e2


def rayleigh(n=1):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)


def cnoise(d):
    return (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)


def cosine(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def mse_theory(beta, c1, rho_e):
    """Theorem 1: per-realization MSE of the aware demultiplexer."""
    a0 = 1.0 + beta**2 * abs(c1)**2 + rho_e
    return rho_e / math.sqrt(a0 * a0 - 4.0 * beta**2 * abs(c1)**2)


def mse_blind(c1, dsig2_h):
    """Affinity-blind scalar-shrinkage MSE (beta = 0 in the filter)."""
    r0 = abs(c1)**2 + dsig2_h
    return r0 / (1.0 + r0)


def eta_of(mse):
    """Effective SINR of a (possibly biased) estimator with unit signal."""
    return 1.0 / mse - 1.0


def aware(t1, Q, beta, c1, nvar, d):
    """Affinity-aware Wiener demultiplexer applied to t1 = M1^T r / h1.

    Uses A A^H = (1+beta^2|c1|^2) I + beta(c1 Q + conj(c1) Q^T), so the
    system matrix is assembled in O(d^2) and solved with one LU."""
    g = 1.0 - beta * beta
    rho = g * abs(c1)**2 / d + nvar
    S = beta * (c1 * Q + np.conj(c1) * Q.T) / d
    S[np.diag_indices(d)] += (1.0 + beta**2 * abs(c1)**2) / d + rho
    x = np.linalg.solve(S, t1)
    return (x + beta * np.conj(c1) * (Q.T @ x)) / d


def blind(t1, c1, nvar, d):
    """Affinity-blind receiver: scalar shrinkage of the matched filter."""
    lam = (1.0 / d) / (1.0 / d + abs(c1)**2 / d + nvar)
    return lam * t1


# ------------------------------------------------------------------
# E0 : Theorem-1 verification (both users, random phases)
# ------------------------------------------------------------------
def E0_theorem_check(d=512, betas=(0.0, 0.311, 0.5, 0.7), ntr=200, snr=20.0):
    print("\n=== E0: Theorem 1 (aware-demultiplexer MSE) verification ===")
    sig = 10 ** (-snr / 20.0)
    rows = []
    worst = 0.0
    for beta in betas:
        g = 1.0 - beta**2
        r1s, r2s = [], []
        for _ in range(ntr):
            h1 = np.exp(1j * rng.uniform(0, 2 * np.pi))
            h2 = np.exp(1j * rng.uniform(0, 2 * np.pi))
            e1, e2 = embed_pair(d, beta)
            M1, M2 = haar(d), haar(d)
            Q = M1.T @ M2
            r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + sig * cnoise(d)
            c1, c2 = h2 / h1, h1 / h2
            g1 = aware(M1.T @ r / h1, Q, beta, c1, sig**2 / abs(h1)**2, d)
            g2 = aware(M2.T @ r / h2, Q.T, beta, c2, sig**2 / abs(h2)**2, d)
            th1 = mse_theory(beta, c1, g * abs(c1)**2 + d * sig**2 / abs(h1)**2)
            th2 = mse_theory(beta, c2, g * abs(c2)**2 + d * sig**2 / abs(h2)**2)
            r1s.append(np.linalg.norm(g1 - e1)**2 / th1)
            r2s.append(np.linalg.norm(g2 - e2)**2 / th2)
        r1, r2 = float(np.mean(r1s)), float(np.mean(r2s))
        dev = max(abs(r1 - 1.0), abs(r2 - 1.0)) * 100
        worst = max(worst, dev)
        print(f"  beta={beta:.3f}  MC/theory user1 = {r1:.4f}, user2 = {r2:.4f}"
              f"  (max dev {dev:.2f}%)")
        rows.append([beta, r1, r2, dev])
    write_csv("theorem_check", ["beta", "user1_mc_over_theory",
                                "user2_mc_over_theory", "max_dev_pct"], rows)
    print(f"  worst-case deviation {worst:.2f}%")
    return worst


# ------------------------------------------------------------------
# E1 : MSE vs block SNR at |h|=1 -- aware floor sqrt(g)/2 vs blind 1/2
# ------------------------------------------------------------------
def E1_floor(beta=0.311, dims=(256, 768), snr_db=np.arange(0, 41, 2.5), ntr=120):
    print("\n=== E1: finite-d validation, aware vs blind floor ===")
    g = 1.0 - beta**2
    fig, ax = plt.subplots()
    colors = {256: "C0", 768: "C3"}
    rows = []
    for d in dims:
        mc = np.zeros(len(snr_db))
        for _ in range(ntr):
            e1, e2 = embed_pair(d, beta)
            M1, M2 = haar(d), haar(d)
            Q = M1.T @ M2
            r0 = (M1 @ e1) + (M2 @ e2)
            n = cnoise(d)
            for k, s in enumerate(snr_db):
                sig = 10 ** (-s / 20.0)
                g1 = aware(M1.T @ (r0 + sig * n), Q, beta, 1.0, sig**2, d)
                mc[k] += np.linalg.norm(g1 - e1)**2
        mc /= ntr
        rho = 10 ** (snr_db / 10.0)
        th = np.array([mse_theory(beta, 1.0, g + d / r) for r in rho])
        bl = np.array([mse_blind(1.0, d / r) for r in rho])
        ax.semilogy(snr_db, mc, "o", ms=3.5, color=colors[d], mfc="none",
                    label=rf"Monte Carlo, $d={d}$")
        ax.semilogy(snr_db, th, "-", color=colors[d],
                    label=rf"Theorem 1, $d={d}$")
        if d == dims[-1]:
            ax.semilogy(snr_db, bl, "--", color="C1", lw=1.1,
                        label=LBL["blind"])
        for s, m, t, b in zip(snr_db, mc, th, bl):
            rows.append([d, s, m, t, b])
        dev = 100 * max(abs(mc / th - 1))
        print(f"  d={d}: max MC/theory dev {dev:.1f}%")
    ax.axhline(math.sqrt(g) / 2, color="gray", lw=0.8, ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.text(1.0, 0.52, r"blind floor $1/2$", fontsize=7, color="gray")
    ax.text(22.0, 0.40, r"aware floor $\sqrt{1-\beta^2}/2$",
            fontsize=7, color="gray")
    ax.set_yscale("linear")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel(r"Per-user MSE $\mathbb{E}\|\hat{\mathbf{e}}_u-\mathbf{e}_u\|_2^2$")
    ax.set_xlim(0, 40); ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper right", ncol=1)
    save_fig(fig, "fig_floor")
    write_csv("floor_validation",
              ["d", "snr_db", "mse_mc", "mse_theory", "mse_blind"], rows)
    print(f"  aware floor {math.sqrt(g)/2:.4f} vs blind floor 0.5000 "
          f"(ratio {0.5/(math.sqrt(g)/2):.4f} = 1/sqrt(1-beta^2))")


# ------------------------------------------------------------------
# E7 : effective-rate figures (eta = 1/MSE - 1)
# ------------------------------------------------------------------
def T_edma(rho, d, beta):
    m = mse_theory(beta, 1.0, (1.0 - beta**2) + d / rho)
    return 2.0 * math.log2(1.0 + eta_of(m))


def T_blind(rho, d):
    return 2.0 * math.log2(1.0 + 1.0 / (1.0 + d / rho))


def E7_rates(beta=0.311, d=512):
    print("\n=== E7a: effective-rate comparison ===")
    snr_db = np.arange(0, 41, 0.5)
    rho = 10 ** (snr_db / 10.0)
    Te = np.array([T_edma(r, d, beta) for r in rho])
    Tb = np.array([T_blind(r, d) for r in rho])
    To = 2 * np.log2(1 + rho / (2 * d))
    Tg = 2 * np.log2(1 + rho / d)
    Cm = np.log2(1 + 2 * rho / d)
    fig, ax = plt.subplots()
    ax.plot(snr_db, Te, "-",  color="C3", label=LBL["edma"])
    ax.plot(snr_db, Tb, ":",  color="C4", lw=1.2, label=LBL["blind"])
    ax.plot(snr_db, To, "--", color="C1", label=LBL["oma"])
    ax.plot(snr_db, Tg, "-.", color="C0", label=LBL["genie"])
    ax.plot(snr_db, Cm, "-",  color="k", lw=1.0, label=LBL["mac"])
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 40); ax.set_ylim(0, 3.2)
    ax.legend(loc="upper left")
    save_fig(fig, "fig_rate_corrected")
    rows = [[s, Te[i], Tb[i], To[i], Tg[i], Cm[i]]
            for i, s in enumerate(snr_db)]
    write_csv("rate_corrected",
              ["snr_db", "edma", "blind", "oma", "genie", "mac"], rows)
    i20 = list(snr_db).index(20.0)
    print(f"  at 20 dB: EDMA {Te[i20]:.3f}, blind {Tb[i20]:.3f}, "
          f"OMA {To[i20]:.3f} (gain {Te[i20]/To[i20]:.2f}x), "
          f"MAC {Cm[i20]:.3f}, EDMA/MAC {Te[i20]/Cm[i20]:.3f}")
    g = 1.0 - beta**2
    rho_c = 2 * d * (2 / math.sqrt(g) - 1)
    ix = np.where(To >= Te)[0]
    rc_num = snr_db[ix[0]] if len(ix) else float("nan")
    print(f"  OMA re-crossover: floor formula {10*math.log10(rho_c):.1f} dB, "
          f"numerical {rc_num:.1f} dB "
          f"(blind: {10*math.log10(2*d):.1f} dB)")

    print("\n=== E7b: value-of-affinity sweep ===")
    betas = np.linspace(0.0, 0.98, 99)
    fig, ax = plt.subplots()
    rows = []
    for s, col in ((10, "C0"), (20, "C3")):
        rho_s = 10 ** (s / 10.0)
        Te = np.array([T_edma(rho_s, d, b) for b in betas])
        Tb = T_blind(rho_s, d)
        To = 2 * math.log2(1 + rho_s / (2 * d))
        Tg = 2 * math.log2(1 + rho_s / d)
        ax.plot(betas, Te, "-", color=col, label=rf"EDMA, $\rho={s}$ dB")
        ax.axhline(Tb, color=col, ls=":", lw=1.0)
        ax.axhline(To, color=col, ls="--", lw=1.0)
        ax.axhline(Tg, color=col, ls="-.", lw=0.8)
        ixg = np.where(Te >= Tg)[0]
        bg = betas[ixg[0]] if len(ixg) else float("nan")
        print(f"  rho={s} dB: EDMA(0)/blind = {Te[0]/Tb:.3f}, "
              f"EDMA(0.311) gain over blind "
              f"{Te[np.argmin(abs(betas-0.311))]/Tb:.3f}x, "
              f"crosses genie at beta ~ {bg:.2f}")
        for i, b in enumerate(betas):
            rows.append([s, b, Te[i], Tb, To, Tg])
    for b0 in (0.030, 0.311):
        ax.axvline(b0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"Pairwise affinity $\beta$")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    save_fig(fig, "fig_beta_sweep_corrected")
    write_csv("beta_sweep_corrected",
              ["snr_db", "beta", "edma", "blind", "oma", "genie"], rows)


if __name__ == "__main__":
    import sys
    todo = set(sys.argv[1:])
    ALL = {
        "E0": E0_theorem_check, "E1": E1_floor, "E7a": E7_rates,
    }
    for name, fn in ALL.items():
        if not todo or name in todo:
            fn()
    print("\nAll requested simulations complete.")
