"""Canonical replot of fig_sic.pdf from data/sic_comparison.csv
(realizable analog SIC vs genie SIC vs EDMA vs OMA, beta = 0.311,
d = 512, block-Rayleigh). US-spelling labels, uniform geometry."""
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

rows = list(csv.DictReader(open(ROOT / "data" / "sic_comparison.csv")))
snr = [float(r["snr_db"]) for r in rows]
col = lambda k: [float(r[k]) for r in rows]

fig, ax = plt.subplots()
ax.plot(snr, col("edma"), "o-", color="C3", label="EDMA (closed form)")
ax.plot(snr, col("sic"), "^-.", color="C2", label="Realizable analog SIC")
ax.plot(snr, col("oma"), "v:", color="C1", label="OMA")
ax.plot(snr, col("genie"), "-", color="gray", lw=1.0,
        label="Genie-aided SIC bound")
ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
ax.set_ylabel("Mean cosine similarity")
ax.set_xlim(snr[0], snr[-1])
ax.set_ylim(0, 0.7)
ax.legend(loc="upper left")
fig.subplots_adjust(**AXES_RECT)
fig.savefig(ROOT / "fig" / "fig_sic.pdf")
print("[OK] wrote fig_sic.pdf")
for r in rows:
    print(f"  {float(r['snr_db']):4.0f} dB  EDMA {float(r['edma']):.3f}  "
          f"SIC {float(r['sic']):.3f}  genie {float(r['genie']):.3f}  "
          f"OMA {float(r['oma']):.3f}")
