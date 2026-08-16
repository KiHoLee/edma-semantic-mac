"""Canonical replot of fig_bertvit_merged.pdf from data/bertvit_merged.csv.
Curves: EDMA, EDMA + refinement (hybrid), ToDMA-adapted, OMA, genie bound.
The capacity-check column edma_ref2 remains in the CSV but is not
plotted (it tracks edma_ref; quoted in the text only)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
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

rows = list(csv.DictReader(open(ROOT / "data" / "bertvit_merged.csv")))
snr = [float(r["snr_db"]) for r in rows]
col = lambda k: [float(r[k]) for r in rows]

fig, ax = plt.subplots()
ax.plot(snr, col("edma"), "o-", color="C3", label="EDMA")
ax.plot(snr, col("edma_ref"), "^-", color="C2",
        label="EDMA + refinement stage")
ax.plot(snr, col("todma"), "d-.", color="C4", label="ToDMA-adapted")
ax.plot(snr, col("oma"), "v:", color="C1", label="OMA")
ax.plot(snr, col("genie"), "-", color="gray", lw=1.0,
        label="Genie-aided SIC bound")
ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
ax.set_ylabel("Mean cosine similarity")
ax.set_xlim(snr[0], snr[-1])
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left")
fig.subplots_adjust(**AXES_RECT)
fig.savefig(ROOT / "fig" / "fig_bertvit_merged.pdf")
print("[OK] wrote fig_bertvit_merged.pdf (no attention curve)")
