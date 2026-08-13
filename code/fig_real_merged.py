"""
Merged real-data comparison figure (replaces separate Figs 4 and 5).
===================================================================
Evaluates ALL schemes on the cached real BERT (text) + ViT (image)
embedding pairs (16 pairs, d = 768, measured mean affinity ~0.028)
under the manuscript's complex block-Rayleigh channel:

  r = h1 M1 e1 + h2 M2 e2 + n,  n ~ CN(0, sigma^2 I),  h_u ~ CN(0,1),
  per-block energy E_b = 1, rho = 1/sigma^2 (per-block SNR).

Schemes:
  1. EDMA      : per-realisation Haar-mixture masks with the per-pair
                    measured beta_i, closed-form demux (13).
  2. OMA          : equivalent-bandwidth model, noise std x sqrt(2).
  3. Genie SIC    : perfect removal of the other user's waveform.
  4. Attention    : retrained reproduction of the learned predecessor,
                    d = 768, trained on parametric pairs at the measured
                    mean affinity with Rayleigh channels and
                    channel-equalised matched-filter inputs
                    x_u = Re(M_u^T r / h_u); evaluated on the REAL pairs.
  5. ToDMA-adapted: OMP sparse coding of the real embedding (T = 16
                    atoms, V = 1024), T slots x L = 48 signatures,
                    per-slot OMP detection on the complex observation,
                    genie association, true coefficients granted.

Outputs: fig/fig_bertvit_merged.pdf, data/bertvit_merged.csv.
200 fading realisations per pair -> 3,200 Monte-Carlo samples per SNR.
Seed fixed.
"""
from __future__ import annotations
import csv
import math
import pickle
import time
from pathlib import Path
import numpy as np
import torch
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

SEED = 2026
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

D = 768
SNRS = np.arange(0.0, 31.0, 5.0)
NFADE = 200                      # fading realisations per pair


def haar(d):
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))


def unit(v):
    return v / np.linalg.norm(v)


def cosine(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_pairs():
    a, b = pickle.load(open(DATA / "bert_vit_cached.pkl", "rb"))
    a = np.stack([unit(x - x.mean()) for x in a])
    b = np.stack([unit(x - x.mean()) for x in b])
    betas = np.abs((a * b).sum(1))
    print(f"[pairs] {len(a)} cached BERT/ViT pairs, d={a.shape[1]}, "
          f"beta mean {betas.mean():.4f} std {betas.std():.4f}")
    return a, b, betas


# ------------------------------------------------------------------
# attention model: trained at the measured mean affinity, d=768,
# Rayleigh channels, channel-equalised MF inputs
# ------------------------------------------------------------------
EPS_EQ = 0.1     # regularised equalisation h*/(|h|^2+EPS_EQ):
                 # caps deep-fade amplification for the learned readout


def train_attention(beta0, epochs=150, steps=20, batch=48, lr=5e-4,
                    l1=1.0, l2=0.5, l3=0.5):
    print(f"=== training attention reproduction (d={D}, beta={beta0:.3f}, "
          f"{epochs} epochs, Rayleigh) ===", flush=True)
    gen = torch.Generator().manual_seed(SEED)
    g0 = math.sqrt(1.0 - beta0**2)

    def torch_pairs(n):
        e1 = torch.nn.functional.normalize(
            torch.randn(n, D, generator=gen), dim=1)
        w = torch.randn(n, D, generator=gen)
        w = w - (w * e1).sum(1, keepdim=True) * e1
        w = torch.nn.functional.normalize(w, dim=1)
        return e1, beta0 * e1 + g0 * w

    M1 = torch.nn.Parameter(torch.linalg.qr(
        torch.randn(D, D, generator=gen))[0])
    M2 = torch.nn.Parameter(beta0 * M1.detach()
                            + g0 * torch.linalg.qr(
                                torch.randn(D, D, generator=gen))[0])
    Q1 = torch.nn.Parameter(torch.randn(D, D, generator=gen) / math.sqrt(D))
    Q2 = torch.nn.Parameter(torch.randn(D, D, generator=gen) / math.sqrt(D))
    opt = torch.optim.Adam([M1, M2, Q1, Q2], lr=lr)
    eye = torch.eye(D)

    t0 = time.time()
    for ep in range(epochs):
        for _ in range(steps):
            e1, e2 = torch_pairs(batch)
            snr_db = 5.0 + 20.0 * torch.rand(batch, 1, generator=gen)
            sig = 10 ** (-snr_db / 20.0)
            hr = torch.randn(batch, 2, generator=gen)
            hi = torch.randn(batch, 2, generator=gen)
            # complex channel on real signals; equalised MF real part:
            # x_u = Re(M_u^T r / h_u); build via real/imag components
            s1 = e1 @ M1.T
            s2 = e2 @ M2.T
            nr = sig * torch.randn(batch, D, generator=gen) / math.sqrt(2)
            ni = sig * torch.randn(batch, D, generator=gen) / math.sqrt(2)
            rr = (hr[:, :1] * s1 + hr[:, 1:2] * s2) / math.sqrt(2) + nr
            ri = (hi[:, :1] * s1 + hi[:, 1:2] * s2) / math.sqrt(2) + ni
            outs = []
            for u, (Mu, Qu) in enumerate(((M1, Q1), (M2, Q2))):
                hu_r = hr[:, u:u+1] / math.sqrt(2)
                hu_i = hi[:, u:u+1] / math.sqrt(2)
                mag = hu_r**2 + hu_i**2 + EPS_EQ
                xr = (rr @ Mu)
                xi = (ri @ Mu)
                xu = (xr * hu_r + xi * hu_i) / mag  # Re(h* r'/(|h|^2+eps))
                sc = (xu @ Qu.T) / math.sqrt(D)
                outs.append(D * torch.softmax(sc, dim=1) * xu)
            gram = ((M1.T @ M1 - eye)**2).mean() \
                 + ((M2.T @ M2 - eye)**2).mean() \
                 + ((M1.T @ M2 - beta0 * eye)**2).mean()
            mse = ((outs[0] - e1)**2).mean() + ((outs[1] - e2)**2).mean()
            cs = torch.nn.functional.cosine_similarity(
                outs[0], e1, dim=1).mean() \
                + torch.nn.functional.cosine_similarity(
                outs[1], e2, dim=1).mean()
            loss = l1 * gram + l2 * mse + l3 * (2.0 - cs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([M1, M2, Q1, Q2], 1.0)
            opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}: loss {float(loss.detach()):.4f}",
                  flush=True)
    print(f"  trained in {time.time()-t0:.0f}s, "
          f"{4*D*D/1e6:.2f}M parameters")
    return (M1.detach().numpy(), M2.detach().numpy(),
            Q1.detach().numpy(), Q2.detach().numpy())


def att_apply(model, r, h1, h2):
    M1, M2, Q1, Q2 = model
    outs = []
    for u, (Mu, Qu, hu) in enumerate(((M1, Q1, h1), (M2, Q2, h2))):
        xu = np.real(np.conj(hu) * (Mu.T @ r)) / (abs(hu)**2 + EPS_EQ)
        sc = (Qu @ xu) / math.sqrt(D)
        sc = sc - sc.max()
        w = np.exp(sc); w /= w.sum()
        outs.append(D * w * xu)
    return outs


# ------------------------------------------------------------------
# ToDMA-adapted on real embeddings (complex channel)
# ------------------------------------------------------------------
def todma_prepare(V=1024, T=16):
    L = D // T
    Dict = rng.standard_normal((V, D))
    Dict /= np.linalg.norm(Dict, axis=1, keepdims=True)
    Sig = rng.standard_normal((V, L))
    Sig /= np.linalg.norm(Sig, axis=1, keepdims=True)
    amp = math.sqrt(1.0 / T)          # E_b = 1 per user per block
    return Dict, Sig, amp, T, L


def omp_code(Dict, e, T):
    resid = e.copy(); idx = []
    for _ in range(T):
        corr = np.abs(Dict @ resid)
        if idx:
            corr[idx] = -1
        k = int(corr.argmax()); idx.append(k)
        A = Dict[idx].T
        coef, *_ = np.linalg.lstsq(A, e, rcond=None)
        resid = e - A @ coef
    return idx, coef


def todma_run(tod, codes, h, sig, noise_slots):
    Dict, Sig, amp, T, L = tod
    U = len(codes)
    det = [set() for _ in range(U)]
    for t in range(T):
        y = sum(h[u] * amp * Sig[codes[u][0][t]] for u in range(U)) \
            + sig * noise_slots[t]
        resid = y.copy(); support = []
        for _ in range(U):
            corr = np.abs(Sig @ resid.conj())
            if support:
                corr[support] = -1
            kk = int(corr.argmax()); support.append(kk)
            Ah = (Sig[support].T * amp).astype(complex)
            coef, *_ = np.linalg.lstsq(Ah, y, rcond=None)
            resid = y - Ah @ coef
        sset = set(support)
        for u in range(U):
            if codes[u][0][t] in sset:
                det[u].add(codes[u][0][t])
    recs = []
    for u in range(U):
        idx, coef = codes[u]
        keep = [i for i, tid in enumerate(idx) if tid in det[u]]
        recs.append(sum(coef[i] * Dict[idx[i]] for i in keep)
                    if keep else None)
    return recs


# ------------------------------------------------------------------
def main():
    A, B, betas = load_pairs()
    npairs = len(A)
    model = train_attention(float(betas.mean()))
    tod = todma_prepare()
    codes = [(omp_code(tod[0], A[i], tod[3]),
              omp_code(tod[0], B[i], tod[3])) for i in range(npairs)]
    print("[todma] sparse codes prepared")

    keys = ("edma", "oma", "genie", "att", "att_x", "todma")
    res = {k: np.zeros(len(SNRS)) for k in keys}
    cnt = {k: np.zeros(len(SNRS)) for k in keys}
    t0 = time.time()
    for i in range(npairs):
        e1, e2, bi = A[i], B[i], float(betas[i])
        gi = 1.0 - bi**2
        c1, c2 = codes[i]
        for f in range(NFADE):
            U1, U2 = haar(D), haar(D)
            M1 = U1
            M2 = bi * U1 + math.sqrt(gi) * U2
            h = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) \
                / math.sqrt(2)
            h1, h2 = h
            r0 = h1 * (M1 @ e1) + h2 * (M2 @ e2)
            n = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) \
                / math.sqrt(2)
            n2 = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) \
                / math.sqrt(2)
            nslots = [(rng.standard_normal(tod[4])
                       + 1j * rng.standard_normal(tod[4])) / math.sqrt(2)
                      for _ in range(tod[3])]
            # attention scheme transmits with ITS OWN trained masks
            r0a = h1 * (model[0] @ e1) + h2 * (model[1] @ e2)
            for k, s in enumerate(SNRS):
                sig = 10 ** (-s / 20.0)
                r = r0 + sig * n
                t1 = M1.T @ r / h1; t2 = M2.T @ r / h2
                g1 = (t1 - bi * (h2 / h1) * t2) / gi
                g2 = (t2 - bi * (h1 / h2) * t1) / gi
                res["edma"][k] += 0.5 * (cosine(g1, e1) + cosine(g2, e2))
                o1 = e1 + math.sqrt(2) * sig * n / h1
                o2 = e2 + math.sqrt(2) * sig * n2 / h2
                res["oma"][k] += 0.5 * (cosine(o1, e1) + cosine(o2, e2))
                ge1 = M1.T @ (r - h2 * (M2 @ e2)) / h1
                ge2 = M2.T @ (r - h1 * (M1 @ e1)) / h2
                res["genie"][k] += 0.5 * (cosine(ge1, e1) + cosine(ge2, e2))
                a1, a2 = att_apply(model, r0a + sig * n, h1, h2)
                res["att"][k] += 0.5 * (cosine(a1, e1) + cosine(a2, e2))
                res["att_x"][k] += 0.5 * (cosine(a1, e2) + cosine(a2, e1))
                for kk in ("edma", "oma", "genie", "att", "att_x"):
                    cnt[kk][k] += 1
                if f < 40:            # ToDMA heavier: 40 fading draws
                    recs = todma_run(tod, (c1, c2), (h1, h2), sig, nslots)
                    got = [cosine(recs[j], (e1, e2)[j])
                           for j in range(2) if recs[j] is not None]
                    if got:
                        res["todma"][k] += float(np.mean(got))
                    cnt["todma"][k] += 1
        print(f"  pair {i+1}/{npairs} done ({time.time()-t0:.0f}s)",
              flush=True)
    for k in keys:
        res[k] /= np.maximum(cnt[k], 1)

    fig, ax = plt.subplots()
    ax.plot(SNRS, res["edma"], "o-", color="C3", label="EDMA (closed form)")
    ax.plot(SNRS, res["att"], "s--", color="C0",
            label="Attention-based (retrained)")
    ax.plot(SNRS, res["todma"], "d-.", color="C4", label="ToDMA-adapted")
    ax.plot(SNRS, res["oma"], "v:", color="C1", label="OMA")
    ax.plot(SNRS, res["genie"], "-", color="gray", lw=1.0,
            label="Genie-aided SIC bound")
    ax.set_xlabel("Per-block SNR $\\rho$ [dB]")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(SNRS[0], SNRS[-1]); ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left")
    fig.subplots_adjust(**AXES_RECT)
    fig.savefig(FIG / "fig_bertvit_merged.pdf")
    plt.close(fig)
    print(f"[OK] wrote {FIG/'fig_bertvit_merged.pdf'}")

    with open(DATA / "bertvit_merged.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["snr_db"] + list(keys))
        for k, s in enumerate(SNRS):
            w.writerow([s] + [res[key][k] for key in keys])
    print(f"[OK] wrote {DATA/'bertvit_merged.csv'}")
    for k, s in enumerate(SNRS):
        print(f"  {s:4.0f} dB  EDMA {res['edma'][k]:.3f}  "
              f"ATT {res['att'][k]:.3f} (x {res['att_x'][k]:.3f})  "
              f"ToDMA {res['todma'][k]:.3f}  OMA {res['oma'][k]:.3f}  "
              f"genie {res['genie'][k]:.3f}")


if __name__ == "__main__":
    main()
