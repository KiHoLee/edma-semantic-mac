"""
Revision simulations for the EDMA TCOM resubmission.
=======================================================
Implements the per-realisation (finite-d) analysis and the corrected
energy-normalised rate accounting, plus the reviewer-requested
experiments:

  E0  Theorem-1 verification: exact self-interference constant C_SI
  E1  fig_floor          : per-user MSE vs block SNR, interference floor
  E2  fig_sic            : realisable SIC vs genie SIC vs EDMA vs OMA
  E3  (text numbers)     : Rayleigh unconditional MSE, ZF vs regularised
  E4  fig_csi            : imperfect-CSI robustness
  E5  fig_maskfam        : Walsh-Hadamard structured masks vs Haar
  E6  fig_coop           : high-affinity combining-mode crossover
  E7  fig_rate_corrected, fig_beta_sweep_corrected, fig_multiuser_corrected

Conventions (identical to the revised manuscript):
  * unit per-block transmit energy E_b = 1 per user
  * rho = E_b / sigma_n^2  (per-block received SNR; per-symbol SNR rho/d)
  * block-Rayleigh h ~ CN(0,1) unless the AWGN point |h|=1 is stated
  * complex AWGN CN(0, sigma^2 I_d); embeddings real, unit norm
  * orientation convention <e1,e2> = +beta
Fixed seed. CSVs -> ../fig, PDFs -> ../fig_toc.
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
    "legend.fontsize": 7.0, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "grid.alpha": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "figure.figsize": (3.15, 2.36), "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.205, right=0.965, top=0.955, bottom=0.185)

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


def two_user_masks(d, beta, U1=None, U2=None):
    if U1 is None: U1 = haar(d)
    if U2 is None: U2 = haar(d)
    g = math.sqrt(1.0 - beta**2)
    return U1, beta * U1 + g * U2


def rayleigh(n=1):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)


def C_SI(beta, c):
    """User-1 self-interference constant (exact to O(1/d)), <e1,e2>=+beta."""
    g = 1.0 - beta**2
    return (g**2 * abs(c)**2 + beta**2 + beta**4 * abs(c)**2
            + 2.0 * beta**4 * np.real(c)) / g


def C_SI2(beta, c2):
    """User-2 self-interference constant (deterministic), c2 = h1/h2."""
    g = 1.0 - beta**2
    return (abs(c2)**2 + beta**2 + 2.0 * beta**2 * np.real(c2)) / g


def C_bar(beta):
    """Symmetrised constant at |h|=1 (block-alternating mask roles)."""
    return 0.5 * (C_SI(beta, 1.0 + 0j) + C_SI2(beta, 1.0 + 0j))


def demux(r, M1, M2, h1, h2, beta):
    """beta-aware demultiplexer (13); returns (e1_hat, e2_hat)."""
    g = 1.0 - beta**2
    t1 = (M1.T @ r) / h1
    t2 = (M2.T @ r) / h2
    e1 = (t1 - beta * (h2 / h1) * t2) / g
    e2 = (t2 - beta * (h1 / h2) * t1) / g
    return e1, e2


def cosine(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------------
# E0 : Theorem-1 verification
# ------------------------------------------------------------------
def E0_theorem_check(d=512, betas=(0.0, 0.311, 0.5, 0.7), ntr=300):
    print("\n=== E0: Theorem 1 (self-interference constant) verification ===")
    rows = []
    worst = 0.0
    for beta in betas:
        # random unit-modulus channels (AWGN-type magnitude, random phase)
        errs1, errs2 = [], []
        for _ in range(ntr):
            h1 = np.exp(1j * rng.uniform(0, 2 * np.pi))
            h2 = np.exp(1j * rng.uniform(0, 2 * np.pi))
            e1, e2 = embed_pair(d, beta)
            M1, M2 = two_user_masks(d, beta)
            r = h1 * (M1 @ e1) + h2 * (M2 @ e2)          # noise-free
            g1, g2 = demux(r, M1, M2, h1, h2, beta)
            errs1.append(np.linalg.norm(g1 - e1)**2 / C_SI(beta, h2 / h1))
            errs2.append(np.linalg.norm(g2 - e2)**2 / C_SI2(beta, h1 / h2))
        r1, r2 = float(np.mean(errs1)), float(np.mean(errs2))
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
# E1 : interference floor (MSE vs block SNR), AWGN point |h|=1
# ------------------------------------------------------------------
def E1_floor(beta=0.311, dims=(256, 768), snr_db=np.arange(0, 41, 2.5), ntr=150):
    print("\n=== E1: finite-d interference floor ===")
    g = 1.0 - beta**2
    csi = C_SI(beta, 1.0 + 0j)
    fig, ax = plt.subplots()
    colors = {256: "C0", 768: "C3"}
    rows = []
    for d in dims:
        mc = np.zeros(len(snr_db))
        for _ in range(ntr):
            e1, e2 = embed_pair(d, beta)
            M1, M2 = two_user_masks(d, beta)
            r0 = (M1 @ e1) + (M2 @ e2)
            n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
            for k, s in enumerate(snr_db):
                sig = 10 ** (-s / 20.0)
                g1, _ = demux(r0 + sig * n, M1, M2, 1.0, 1.0, beta)
                mc[k] += np.linalg.norm(g1 - e1)**2
        mc /= ntr
        rho = 10 ** (snr_db / 10.0)
        th   = d / (rho * g) + csi
        ideal = d / (rho * g)
        ax.semilogy(snr_db, mc, "o", ms=3.5, color=colors[d], mfc="none",
                    label=rf"MC, $d={d}$")
        ax.semilogy(snr_db, th, "-", color=colors[d],
                    label=rf"Theorem 1, $d={d}$")
        if d == dims[-1]:
            ax.semilogy(snr_db, ideal, ":", color="k", lw=1.1,
                        label="Idealized (no floor)")
        for s, m, t, i in zip(snr_db, mc, th, ideal):
            rows.append([d, s, m, t, i])
        onset = 10 * math.log10(d / (g * csi))
        print(f"  d={d}: floor C_SI={csi:.4f}, onset ~{onset:.1f} dB, "
              f"max MC/theory dev "
              f"{100*max(abs(mc/th-1)):.1f}%")
    ax.axhline(csi, color="gray", lw=0.8, ls="--")
    ax.text(1.0, csi * 1.15, r"floor $C_{\mathrm{SI}}$", fontsize=7, color="gray")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel(r"Per-user MSE $\mathbb{E}\|\hat{\mathbf{e}}_u-\mathbf{e}_u\|_2^2$")
    ax.set_xlim(0, 40); ax.set_ylim(0.5, 2000)
    ax.legend(loc="upper right", ncol=1)
    save_fig(fig, "fig_floor")
    write_csv("floor_validation", ["d", "snr_db", "mse_mc", "mse_theory", "mse_ideal"], rows)


# ------------------------------------------------------------------
# E2 : realisable SIC vs genie SIC vs EDMA vs OMA (Rayleigh)
# ------------------------------------------------------------------
def E2_sic(beta=0.311, d=512, snr_db=np.arange(0, 31, 5), ntr=400):
    print("\n=== E2: realisable vs genie SIC (Rayleigh) ===")
    res = {k: np.zeros(len(snr_db)) for k in
           ("edma", "oma", "genie", "sic")}
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = two_user_masks(d, beta)
        h1, h2 = rayleigh(2)
        r0 = h1 * (M1 @ e1) + h2 * (M2 @ e2)
        n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
        n2 = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
        for k, s in enumerate(snr_db):
            sig = 10 ** (-s / 20.0)
            r = r0 + sig * n
            # EDMA
            g1, g2 = demux(r, M1, M2, h1, h2, beta)
            res["edma"][k] += 0.5 * (cosine(g1, e1) + cosine(g2, e2))
            # OMA equivalent-bandwidth model: interference-free, noise x sqrt(2)
            o1 = e1 + math.sqrt(2) * sig * n / h1
            o2 = e2 + math.sqrt(2) * sig * n2 / h2
            res["oma"][k] += 0.5 * (cosine(o1, e1) + cosine(o2, e2))
            # genie SIC: perfect removal of the other user for BOTH users
            ge1 = M1.T @ (r - h2 * (M2 @ e2)) / h1
            ge2 = M2.T @ (r - h1 * (M1 @ e1)) / h2
            res["genie"][k] += 0.5 * (cosine(ge1, e1) + cosine(ge2, e2))
            # realisable SIC: stronger user first (matched filter),
            # unit-norm projection as the analog decision, then subtract
            if abs(h1) >= abs(h2):
                hs, hw, Ms, Mw, es, ew = h1, h2, M1, M2, e1, e2
            else:
                hs, hw, Ms, Mw, es, ew = h2, h1, M2, M1, e2, e1
            d_s = Ms.T @ r / hs
            dec_s = d_s / np.linalg.norm(d_s)          # analog decision
            r_res = r - hs * (Ms @ dec_s)
            d_w = Mw.T @ r_res / hw
            res["sic"][k] += 0.5 * (cosine(d_s, es) + cosine(d_w, ew))
    for k in res:
        res[k] /= ntr
    fig, ax = plt.subplots()
    ax.plot(snr_db, res["edma"], "o-",  color="C3", label="EDMA")
    ax.plot(snr_db, res["genie"], "s--", color="C0", label="Genie-aided SIC")
    ax.plot(snr_db, res["sic"],  "^-.", color="C2", label="Realisable SIC")
    ax.plot(snr_db, res["oma"],  "v:",  color="C1", label="OMA")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(snr_db[0], snr_db[-1]); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    save_fig(fig, "fig_sic")
    rows = [[s] + [res[k][i] for k in ("edma", "oma", "genie", "sic")]
            for i, s in enumerate(snr_db)]
    write_csv("sic_comparison", ["snr_db", "edma", "oma", "genie", "sic"], rows)
    i20 = list(snr_db).index(20)
    print(f"  at 20 dB: EDMA {res['edma'][i20]:.3f}, realisable SIC "
          f"{res['sic'][i20]:.3f}, genie {res['genie'][i20]:.3f}, "
          f"OMA {res['oma'][i20]:.3f}")


# ------------------------------------------------------------------
# E3 : Rayleigh unconditional MSE ??ZF inversion vs regularised
# ------------------------------------------------------------------
def E3_regularised(beta=0.311, d=512, snrs=(10, 20), ntr=4000):
    print("\n=== E3: Rayleigh unconditional MSE, ZF vs regularised ===")
    rows = []
    for s in snrs:
        sig = 10 ** (-s / 20.0)
        sig2 = sig**2
        mse_zf, mse_rg = [], []
        for _ in range(ntr):
            e1, e2 = embed_pair(d, beta)
            M1, M2 = two_user_masks(d, beta)
            h1, h2 = rayleigh(2)
            r = h1 * (M1 @ e1) + h2 * (M2 @ e2) \
                + sig * (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
            g1, _ = demux(r, M1, M2, h1, h2, beta)
            mse_zf.append(np.linalg.norm(g1 - e1)**2)
            # regularised inversion: 1/h -> h*/(|h|^2 + d sigma^2)
            eps = d * sig2
            f1 = (abs(h1)**2 + eps) / np.conj(h1)
            f2 = (abs(h2)**2 + eps) / np.conj(h2)
            g1r, _ = demux(r, M1, M2, f1, f2, beta)
            mse_rg.append(np.linalg.norm(g1r - e1)**2)
        zf_mean, zf_med = float(np.mean(mse_zf)), float(np.median(mse_zf))
        rg_mean, rg_med = float(np.mean(mse_rg)), float(np.median(mse_rg))
        print(f"  {s} dB: ZF mean {zf_mean:9.2f} (median {zf_med:6.2f}) | "
              f"regularised mean {rg_mean:6.3f} (median {rg_med:6.3f})")
        rows.append([s, zf_mean, zf_med, rg_mean, rg_med])
    write_csv("rayleigh_mse", ["snr_db", "zf_mean", "zf_median",
                               "reg_mean", "reg_median"], rows)


# ------------------------------------------------------------------
# E4 : imperfect CSI
# ------------------------------------------------------------------
def E4_csi(beta=0.311, d=512, snr=30.0,
           sh2=np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]), ntr=400):
    """EDMA cosine is CSI-direction-invariant (h-estimates cancel in the
    demux direction); realisable SIC degrades through its subtraction stage."""
    print("\n=== E4: imperfect CSI robustness (EDMA vs realisable SIC) ===")
    sig = 10 ** (-snr / 20.0)
    res_e = np.zeros(len(sh2)); res_s = np.zeros(len(sh2))
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = two_user_masks(d, beta)
        h1, h2 = rayleigh(2)
        r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + sig * (
            rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
        eps1, eps2 = rayleigh(2)
        for j, v in enumerate(sh2):
            hh1 = h1 + math.sqrt(v) * eps1
            hh2 = h2 + math.sqrt(v) * eps2
            g1, g2 = demux(r, M1, M2, hh1, hh2, beta)
            res_e[j] += 0.5 * (cosine(g1, e1) + cosine(g2, e2))
            # realisable SIC with the same imperfect estimates
            if abs(hh1) >= abs(hh2):
                hs, hw, Ms, Mw, es, ew = hh1, hh2, M1, M2, e1, e2
            else:
                hs, hw, Ms, Mw, es, ew = hh2, hh1, M2, M1, e2, e1
            d_s = Ms.T @ r / hs
            dec_s = d_s / np.linalg.norm(d_s)
            r_res = r - hs * (Ms @ dec_s)
            d_w = Mw.T @ r_res / hw
            res_s[j] += 0.5 * (cosine(d_s, es) + cosine(d_w, ew))
    res_e /= ntr; res_s /= ntr
    print(f"  EDMA:  {res_e[0]:.4f} -> {res_e[-1]:.4f} "
          f"(delta {100*(res_e[0]-res_e[-1]):.2f} points)")
    print(f"  SIC :  {res_s[0]:.4f} -> {res_s[-1]:.4f} "
          f"(delta {100*(res_s[0]-res_s[-1]):.2f} points)")
    fig, ax = plt.subplots()
    ax.plot(sh2, res_e, "o-", color="C3", label="EDMA")
    ax.plot(sh2, res_s, "^-.", color="C2", label="Realisable SIC")
    ax.set_xlabel(r"CSI error variance $\sigma_h^2$")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(0, sh2[-1]); ax.set_ylim(0, 0.7)
    ax.legend(loc="lower left")
    save_fig(fig, "fig_csi")
    rows = [[v, res_e[j], res_s[j]] for j, v in enumerate(sh2)]
    write_csv("csi_error", ["sigma_h2", "edma", "sic"], rows)


# ------------------------------------------------------------------
# E5 : Walsh-Hadamard structured masks vs Haar
# ------------------------------------------------------------------
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def E5_maskfam(beta=0.311, d=512, snr_db=np.arange(0, 41, 5), ntr=200):
    print("\n=== E5: Walsh-Hadamard masks vs Haar mixture ===")
    H = hadamard(d)
    g = math.sqrt(1.0 - beta**2)
    res = {"haar": np.zeros(len(snr_db)), "wh": np.zeros(len(snr_db))}
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = two_user_masks(d, beta)
        D1 = np.diag(rng.choice([-1.0, 1.0], d))
        D2 = np.diag(rng.choice([-1.0, 1.0], d))
        W1 = H @ D1
        W2 = beta * W1 + g * (H @ D2)
        r0h = (M1 @ e1) + (M2 @ e2)
        r0w = (W1 @ e1) + (W2 @ e2)
        n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
        for k, s in enumerate(snr_db):
            sig = 10 ** (-s / 20.0)
            g1, _ = demux(r0h + sig * n, M1, M2, 1.0, 1.0, beta)
            w1, _ = demux(r0w + sig * n, W1, W2, 1.0, 1.0, beta)
            res["haar"][k] += cosine(g1, e1)
            res["wh"][k] += cosine(w1, e1)
    for k in res:
        res[k] /= ntr
    dev = 100 * np.max(np.abs(res["wh"] - res["haar"]))
    print(f"  max |WH - Haar| cosine deviation: {dev:.2f} points")
    fig, ax = plt.subplots()
    ax.plot(snr_db, res["haar"], "o-", color="C3",
            label=r"Haar mixture, $\mathcal{O}(d^2)$")
    ax.plot(snr_db, res["wh"], "s--", color="C0",
            label=r"Walsh-Hadamard, $\mathcal{O}(d\log d)$")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(snr_db[0], snr_db[-1]); ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left")
    save_fig(fig, "fig_maskfam")
    rows = [[s, res["haar"][i], res["wh"][i]] for i, s in enumerate(snr_db)]
    write_csv("mask_family_rev", ["snr_db", "haar", "wh"], rows)


# ------------------------------------------------------------------
# E6 : high-affinity combining mode
# ------------------------------------------------------------------
def E6_coop(d=512, snr=20.0, betas=np.linspace(0.0, 0.98, 21), ntr=100):
    print("\n=== E6: high-affinity combining-mode crossover ===")
    sig = 10 ** (-snr / 20.0)
    pairs = [(haar(d), haar(d)) for _ in range(ntr)]
    chans = [rayleigh(2) for _ in range(ntr)]
    cos_dx = np.zeros(len(betas)); cos_cb = np.zeros(len(betas))
    for j, beta in enumerate(betas):
        for t in range(ntr):
            U1, U2 = pairs[t]
            h1, h2 = chans[t]
            e1, e2 = embed_pair(d, beta)
            M1, M2 = two_user_masks(d, beta, U1, U2)
            r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + sig * (
                rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
            g1, _ = demux(r, M1, M2, h1, h2, beta)
            cos_dx[j] += cosine(g1, e1)
            # affinity combining: coherent weights for the e1 component
            a1 = h1 + beta**2 * h2
            a2 = beta * (h1 + h2)
            comb = np.conj(a1) * (M1.T @ r) + np.conj(a2) * (M2.T @ r)
            cos_cb[j] += cosine(comb, e1)
    cos_dx /= ntr; cos_cb /= ntr
    ix = np.where(cos_cb >= cos_dx)[0]
    cross = betas[ix[0]] if len(ix) else float("nan")
    print(f"  crossover affinity ~ {cross:.2f} at rho={snr:.0f} dB")
    fig, ax = plt.subplots()
    ax.plot(betas, cos_dx, "o-", color="C3", label="Separation mode (demux)")
    ax.plot(betas, cos_cb, "s--", color="C0", label="Combining mode")
    ax.set_xlabel(r"Pairwise affinity $\beta$")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.8)
    ax.legend(loc="lower left")
    save_fig(fig, "fig_coop")
    rows = [[b, cos_dx[i], cos_cb[i]] for i, b in enumerate(betas)]
    write_csv("coop_mode", ["beta", "cos_demux", "cos_combine"], rows)
    return cross


# ------------------------------------------------------------------
# E7 : corrected effective-rate figures
# ------------------------------------------------------------------
def eta_edma(rho, d, beta, csi=None):
    g = 1.0 - beta**2
    if csi is None:
        csi = C_bar(beta)      # symmetrised constant (alternating masks)
    return 1.0 / (d / (rho * g) + csi)


def E7_rates(beta=0.311, d=512):
    print("\n=== E7a: corrected effective-rate comparison ===")
    snr_db = np.arange(0, 31, 1.0)
    rho = 10 ** (snr_db / 10.0)
    g = 1.0 - beta**2
    T_edma  = 2 * np.log2(1 + eta_edma(rho, d, beta))
    T_ideal = 2 * np.log2(1 + rho * g / d)
    T_oma   = 2 * np.log2(1 + rho / (2 * d))
    T_genie = 2 * np.log2(1 + rho / d)
    C_mac   = np.log2(1 + 2 * rho / d)
    fig, ax = plt.subplots()
    ax.plot(snr_db, T_edma, "-",  color="C3", label="EDMA (Theorem 1)")
    ax.plot(snr_db, T_ideal, ":", color="C3", lw=1.1,
            label="EDMA idealized (infeasible)")
    ax.plot(snr_db, T_oma, "--",  color="C1", label="OMA")
    ax.plot(snr_db, T_genie, "-.", color="C0", label="Genie-aided SIC bound")
    ax.plot(snr_db, C_mac, "-", color="k", lw=1.0, label="MAC sum capacity")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 30); ax.set_ylim(0, 3.2)
    ax.legend(loc="upper left")
    save_fig(fig, "fig_rate_corrected")
    rows = [[s, T_edma[i], T_ideal[i], T_oma[i], T_genie[i], C_mac[i]]
            for i, s in enumerate(snr_db)]
    write_csv("rate_corrected",
              ["snr_db", "edma", "edma_ideal", "oma", "genie", "mac"], rows)
    i20 = list(snr_db).index(20.0)
    csi = C_bar(beta)
    rho_c = d * (2 - 1 / g) / csi
    print(f"  at 20 dB: EDMA {T_edma[i20]:.3f}, OMA {T_oma[i20]:.3f} "
          f"(gain {T_edma[i20]/T_oma[i20]:.2f}x), MAC {C_mac[i20]:.3f}, "
          f"EDMA/MAC {T_edma[i20]/C_mac[i20]:.3f} (gamma={g:.3f})")
    print(f"  OMA re-crossover rho_c = {10*math.log10(rho_c):.1f} dB")

    print("\n=== E7b: corrected beta sweep ===")
    betas = np.linspace(0.0, 0.98, 99)
    fig, ax = plt.subplots()
    rows = []
    for s, col in ((10, "C0"), (20, "C3")):
        rho_s = 10 ** (s / 10.0)
        Te = np.array([2 * np.log2(1 + eta_edma(rho_s, d, b)) for b in betas])
        To = 2 * np.log2(1 + rho_s / (2 * d))
        Tg = 2 * np.log2(1 + rho_s / d)
        ax.plot(betas, Te, "-", color=col, label=rf"EDMA, $\rho={s}$ dB")
        ax.axhline(To, color=col, ls="--", lw=1.0,
                   label=rf"OMA, $\rho={s}$ dB")
        ax.axhline(Tg, color=col, ls="-.", lw=0.8,
                   label=rf"Genie-aided SIC, $\rho={s}$ dB")
        ix = np.where(Te <= To)[0]
        bstar = betas[ix[0]] if len(ix) else float("nan")
        print(f"  rho={s} dB: crossover beta* = {bstar:.3f} "
              f"(wideband limit 1/sqrt(2)=0.707)")
        for i, b in enumerate(betas):
            rows.append([s, b, Te[i], To, Tg])
    for b0 in (0.031, 0.311):
        ax.axvline(b0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"Pairwise affinity $\beta$")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", ncol=1, fontsize=5.8,
              handlelength=1.5, borderaxespad=0.2)
    save_fig(fig, "fig_beta_sweep_corrected")
    write_csv("beta_sweep_corrected",
              ["snr_db", "beta", "edma", "oma", "genie"], rows)


def E7_multiuser(beta=0.311, d=512, Us=(2, 3, 4), ntr_cal=80, ntr_mc=120):
    print("\n=== E7c: corrected multi-user scaling ===")
    snr_db = np.arange(0, 31, 2.5)
    snr_mk = np.arange(0, 31, 5)
    rho = 10 ** (snr_db / 10.0)
    fig, ax = plt.subplots()
    colors = {2: "C0", 3: "C2", 4: "C3"}
    rows = []
    csi2 = C_SI(beta, 1.0 + 0j)
    for U in Us:
        B = (1 - beta) * np.eye(U) + beta * np.ones((U, U))
        Binv_uu = np.linalg.inv(B)[0, 0]
        gU = 1.0 / Binv_uu
        # calibrate C_SI^(U) by noise-free MC at h_u = 1 (the same
        # evaluation convention as the two-user rate curves, so the
        # U = 2 curve reduces exactly to T_EDMA with C_bar),
        # averaged over all users (mask roles are asymmetric)
        acc = 0.0
        for _ in range(ntr_cal):
            A = np.linalg.cholesky(B)
            Uks = [haar(d) for _ in range(U)]
            Ms = [sum(A[u, k] * Uks[k] for k in range(U)) for u in range(U)]
            h = np.ones(U, dtype=complex)
            # symmetric equal-affinity embeddings: e_u = beta-mixed set
            base = unit(rng.standard_normal(d))
            es = []
            for u in range(U):
                w = rng.standard_normal(d)
                w = unit(w - (w @ base) * base)
                # construct so that <e_u,e_v> ~ beta pairwise
                es.append(unit(math.sqrt(beta) * base
                               + math.sqrt(1 - beta) * w))
            r = sum(h[u] * (Ms[u] @ es[u]) for u in range(U))
            Binv = np.linalg.inv(B)
            # block demux e_hat_u = (1/h_u) sum_v Binv[u,v] M_v^T r
            for u in range(U):
                eh = sum(Binv[u, v] * (Ms[v].T @ r) for v in range(U)) / h[u]
                acc += np.linalg.norm(eh - es[u])**2
        csiU = acc / (ntr_cal * U)
        print(f"  U={U}: C_SI^(U) = {csiU:.3f}  "
              f"((U-1)*C_bar = {(U-1)*C_bar(beta):.3f}), gamma_U = {gU:.3f}")
        eta = 1.0 / (d * Binv_uu / rho + csiU)
        T_th = U * np.log2(1 + eta)
        T_oma = U * np.log2(1 + rho / (U * d))
        ax.plot(snr_db, T_th, "-", color=colors[U], label=rf"EDMA, $U={U}$")
        ax.plot(snr_db, T_oma, "--", color=colors[U], lw=1.0,
                label=rf"OMA, $U={U}$")
        # MC markers (with noise, h_u = 1, per-realization real masks)
        err_mc = np.zeros(len(snr_mk))
        for _ in range(ntr_mc):
            A = np.linalg.cholesky(B)
            Uks = [haar(d) for _ in range(U)]
            Ms = [sum(A[u, k] * Uks[k] for k in range(U)) for u in range(U)]
            h = np.ones(U, dtype=complex)
            base = unit(rng.standard_normal(d))
            es = []
            for u in range(U):
                w = rng.standard_normal(d)
                w = unit(w - (w @ base) * base)
                es.append(unit(math.sqrt(beta) * base
                               + math.sqrt(1 - beta) * w))
            r0 = sum(h[u] * (Ms[u] @ es[u]) for u in range(U))
            n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
            Binv = np.linalg.inv(B)
            for k, s in enumerate(snr_mk):
                sig = 10 ** (-s / 20.0)
                r = r0 + sig * n
                for u in range(U):
                    eh = sum(Binv[u, v] * (Ms[v].T @ r) for v in range(U)) / h[u]
                    err_mc[k] += np.linalg.norm(eh - es[u])**2
        err_mc /= ntr_mc * U
        T_mc = U * np.log2(1 + 1.0 / err_mc)
        ax.plot(snr_mk, T_mc, "o", color=colors[U], ms=4, mfc="none")
        for i, s in enumerate(snr_db):
            rows.append([U, s, T_th[i], T_oma[i]])
        i20 = list(snr_db).index(20.0)
        print(f"     at 20 dB: EDMA {T_th[i20]:.3f} vs OMA {T_oma[i20]:.3f} "
              f"(gain {T_th[i20]/T_oma[i20]:.2f}x)")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Effective sum rate [bps/Hz]")
    ax.set_xlim(0, 30); ax.set_ylim(0, 1.5)
    ax.legend(loc="upper left", ncol=1, fontsize=6.2)
    save_fig(fig, "fig_multiuser_corrected")
    write_csv("multiuser_corrected", ["U", "snr_db", "edma", "oma"], rows)


if __name__ == "__main__":
    import sys
    todo = set(sys.argv[1:])
    ALL = {
        "E0": E0_theorem_check, "E1": E1_floor, "E2": E2_sic,
        "E3": E3_regularised, "E4": E4_csi, "E5": E5_maskfam,
        "E6": E6_coop, "E7a": E7_rates, "E7c": E7_multiuser,
    }
    for name, fn in ALL.items():
        if not todo or name in todo:
            fn()
    print("\nAll requested revision simulations complete.")
