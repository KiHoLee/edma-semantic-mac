"""Canonical figure rendering. Reads ONLY data/*.csv, writes fig/*.pdf.

Figures: fig_floor, fig_rate_corrected, fig_beta_sweep_corrected,
fig_sic, fig_multiuser_corrected. (fig_bertvit_merged is rendered by
replot_merged.py; block_diagram.pdf comes from block_diagram_src.tex.)
One physical geometry and one label dictionary for every plot.
"""
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "fig"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 6.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "grid.alpha": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "figure.figsize": (3.15, 2.36), "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.205, right=0.965, top=0.955, bottom=0.185)

LBL = {
    "edma":  "EDMA",
    "blind": "Affinity-blind",
    "oma":   "OMA",
    "genie": "Genie-aided SIC bound",
    "sic":   "Realizable analog SIC",
    "todma": "ToDMA-adapted",
    "mac":   "MAC sum capacity",
    "coop":  "Full-cooperation bound",
    "hybrid": "EDMA + refinement stage",
}


def rows_of(name):
    return list(csv.DictReader(open(DATA / f"{name}.csv")))


def col(rows, k):
    return [float(r[k]) for r in rows]


def save(fig, name):
    fig.subplots_adjust(**AXES_RECT)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"[OK] wrote {name}.pdf")


# ------------------------------------------------------ fig_floor
def fig_floor():
    from matplotlib.lines import Line2D
    rows = rows_of("floor_validation")
    fig, ax = plt.subplots()
    colors = {"256": "C0", "768": "C3"}
    beta = 0.311
    for d in ("256", "768"):
        rd = [r for r in rows if r["d"] == d or r["d"] == f"{d}.0"
              or float(r["d"]) == float(d)]
        snr = col(rd, "snr_db")
        ax.plot(snr, col(rd, "mse_mc"), "o", ms=3.5, color=colors[d],
                mfc="none")
        ax.plot(snr, col(rd, "mse_theory"), "-", color=colors[d])
        if d == "768":
            ax.plot(snr, col(rd, "mse_blind"), "--", color="C1", lw=1.2)
    g = 1.0 - beta**2
    ax.axhline(math.sqrt(g) / 2, color="gray", lw=0.8, ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.annotate("blind floor $1/2$", xy=(17.0, 0.512), fontsize=7,
                color="gray")
    ax.annotate(r"aware floor $\sqrt{1-\beta^2}/2$", xy=(14.0, 0.432),
                fontsize=7, color="gray")
    ax.set_xlabel("SNR $\\rho$ [dB]")
    ax.set_ylabel(r"Per-user MSE $\mathbb{E}\|\hat{\mathbf{e}}_u-\mathbf{e}_u\|_2^2$")
    ax.set_xlim(0, 40); ax.set_ylim(0.4, 1.05)
    # framed in-axes legend like every other result figure; composite
    # handles (marker = Monte Carlo, line = Theorem 1; the convention
    # is stated in the caption) keep it to three entries
    handles = [
        Line2D([], [], color="C0", marker="o", mfc="none", ms=3.5,
               ls="-", label=rf"{LBL['edma']}, $d=256$"),
        Line2D([], [], color="C3", marker="o", mfc="none", ms=3.5,
               ls="-", label=rf"{LBL['edma']}, $d=768$"),
        Line2D([], [], color="C1", ls="--", lw=1.2, label=LBL["blind"]),
    ]
    ax.legend(handles=handles, loc="upper right")
    save(fig, "fig_floor")


# ------------------------------------------------ fig_rate_corrected
def fig_rate():
    rows = rows_of("rate_corrected")
    snr = col(rows, "snr_db")
    fig, ax = plt.subplots()
    ax.plot(snr, col(rows, "edma"), "-", color="C3", label=LBL["edma"])
    ax.plot(snr, col(rows, "blind"), ":", color="C4", lw=1.2,
            label=LBL["blind"])
    ax.plot(snr, col(rows, "oma"), "--", color="C1", label=LBL["oma"])
    ax.plot(snr, col(rows, "genie"), "-.", color="C0", label=LBL["genie"])
    ax.plot(snr, col(rows, "mac"), "-", color="k", lw=1.0, label=LBL["mac"])
    ax.set_xlabel("SNR $\\rho$ [dB]")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 40); ax.set_ylim(0, 3.2)
    ax.legend(loc="upper left")
    save(fig, "fig_rate_corrected")


# ------------------------------------------ fig_beta_sweep_corrected
def fig_beta_sweep():
    rows = rows_of("beta_sweep_corrected")
    fig, ax = plt.subplots()
    for s, cc in (("10", "C0"), ("20", "C3")):
        rd = [r for r in rows if float(r["snr_db"]) == float(s)]
        b = col(rd, "beta")
        ax.plot(b, col(rd, "edma"), "-", color=cc,
                label=rf"EDMA, $\rho={s}$ dB")
        ax.axhline(float(rd[0]["blind"]), color=cc, ls=":", lw=1.0)
        ax.axhline(float(rd[0]["oma"]), color=cc, ls="--", lw=1.0)
        ax.axhline(float(rd[0]["genie"]), color=cc, ls="-.", lw=0.8)
    # one legend entry per reference style (color-independent)
    ax.plot([], [], ls=":", color="gray", label=LBL["blind"])
    ax.plot([], [], ls="--", color="gray", label=LBL["oma"])
    ax.plot([], [], ls="-.", color="gray", label=LBL["genie"])
    for b0 in (0.030, 0.311):
        ax.axvline(b0, ymax=0.54, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"Pairwise affinity $\beta$")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    save(fig, "fig_beta_sweep_corrected")


# ------------------------------------------------------- fig_sic
def fig_sic():
    rows = rows_of("sic_comparison")
    snr = col(rows, "snr_db")
    fig, ax = plt.subplots()
    ax.plot(snr, col(rows, "edma"), "o-", color="C3", label=LBL["edma"])
    ax.plot(snr, col(rows, "blind"), "d:", color="C4", label=LBL["blind"])
    ax.plot(snr, col(rows, "sic"), "^-.", color="C2", label=LBL["sic"])
    ax.plot(snr, col(rows, "oma"), "v--", color="C1", label=LBL["oma"])
    ax.plot(snr, col(rows, "genie"), "-", color="gray", lw=1.0,
            label=LBL["genie"])
    ax.set_xlabel("SNR $\\rho$ [dB]")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(snr[0], snr[-1]); ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left")
    save(fig, "fig_sic")


# ------------------------------------------ fig_multiuser_corrected
def fig_multiuser():
    rows = rows_of("multiuser_corrected")
    fig, ax = plt.subplots()
    colors = {"2": "C0", "3": "C2", "4": "C3"}
    for U in ("2", "3", "4"):
        rd = [r for r in rows if float(r["U"]) == float(U)]
        snr = col(rd, "snr_db")
        ax.plot(snr, col(rd, "edma_mc"), "-", color=colors[U],
                label=rf"EDMA, $U={U}$")
        ax.plot(snr, col(rd, "oma"), "--", color=colors[U], lw=1.0,
                label=rf"OMA, $U={U}$")
        mk = [i for i, s in enumerate(snr) if s % 5 == 0]
        ax.plot([snr[i] for i in mk], [col(rd, "edma_mc")[i] for i in mk],
                "o", color=colors[U], ms=4, mfc="none")
    ax.set_xlabel("SNR $\\rho$ [dB]")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, "fig_multiuser_corrected")


if __name__ == "__main__":
    import sys
    todo = set(sys.argv[1:])
    ALL = {"floor": fig_floor, "rate": fig_rate, "beta": fig_beta_sweep,
           "sic": fig_sic, "multi": fig_multiuser}
    for name, fn in ALL.items():
        if not todo or name in todo:
            fn()
