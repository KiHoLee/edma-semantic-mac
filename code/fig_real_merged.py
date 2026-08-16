"""
Real-data comparison on cached BERT (text) + ViT (image) pairs.
===================================================================
Evaluates the schemes on the cached real embedding pairs
(16 pairs, d = 768, measured mean affinity ~0.028) under the
manuscript's complex block-Rayleigh channel:

  r = h1 M1 e1 + h2 M2 e2 + n,  n ~ CN(0, sigma^2 I),  h_u ~ CN(0,1),
  per-block energy E_b = 1, rho = 1/sigma^2 (per-block SNR).

Schemes (v2 design: independent Haar masks per user):
  1. EDMA         : affinity-aware Wiener demultiplexer with the
                    per-pair measured beta_i.
  2. OMA          : equivalent-bandwidth model, noise std x sqrt(2).
  3. Genie SIC    : perfect removal of the other user's waveform.
  4. ToDMA-adapted: OMP sparse coding of the real embedding (T = 16
                    atoms, V = 1024), T slots x L = 48 signatures,
                    per-slot OMP detection on the complex observation,
                    genie association, true coefficients granted.

The hybrid (EDMA + refinement stage) curve is produced separately by
refine_matched.py (torch) and merged by replot_merged.py.

EDMA/OMA/genie run in torch (CUDA when available, batched over the
SNR grid); the ToDMA detector runs in numpy on the CPU. Run under
WSL for GPU acceleration. Outputs: data/bertvit_merged.csv.
NFADE fading realisations per pair; ToDMA uses the first 40.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "fig"

SEED = 2026
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

D = 768
SNRS = np.arange(0.0, 31.0, 2.5)
NFADE = 100                      # fading realisations per pair
NFADE_TOD = 40                   # ToDMA heavier: first 40 draws


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


def haar_t(n, gen):
    G = torch.randn(n, D, D, generator=gen, device=DEV)
    Q, R = torch.linalg.qr(G)
    return Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)


def aware_batch(t, Q, beta, c, nvar):
    """Batched affinity-aware Wiener demux. t: (b,D) cfloat, Q: (D,D),
    c: complex scalar, nvar: (b,) real."""
    b = t.shape[0]
    g = 1.0 - beta * beta
    rho = g * abs(c)**2 / D + nvar                       # (b,)
    Qc = Q.to(torch.cfloat)
    A = torch.eye(D, device=DEV, dtype=torch.cfloat) + beta * c * Qc
    S = (A @ A.mH / D).unsqueeze(0) \
        + rho.view(b, 1, 1) * torch.eye(D, device=DEV,
                                        dtype=torch.cfloat)
    x = torch.linalg.solve(S, t.unsqueeze(-1))
    return (A.mH.unsqueeze(0) @ x).squeeze(-1) / D


def abscos(a, b):
    """a: (b,D) cfloat, b: (D,) float -> (b,) abs cosine."""
    num = (a * b.to(torch.cfloat).conj()).sum(1).abs()
    return (num / (a.norm(dim=1) * b.norm())).cpu().numpy()


# ------------------------------------------------------------------
# ToDMA-adapted on real embeddings (complex channel, numpy)
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
    tod = todma_prepare()
    codes = [(omp_code(tod[0], A[i], tod[3]),
              omp_code(tod[0], B[i], tod[3])) for i in range(npairs)]
    print(f"[todma] sparse codes prepared; device = {DEV}")

    gen = torch.Generator(device=DEV).manual_seed(SEED)
    nb = len(SNRS)
    sigs_t = torch.tensor(10 ** (-SNRS / 20.0), device=DEV,
                          dtype=torch.float32)
    keys = ("edma", "oma", "genie", "todma")
    res = {k: np.zeros(nb) for k in keys}
    cnt = {k: np.zeros(nb) for k in keys}
    t0 = time.time()
    for i in range(npairs):
        bi = float(betas[i])
        e1 = torch.tensor(A[i], dtype=torch.float32, device=DEV)
        e2 = torch.tensor(B[i], dtype=torch.float32, device=DEV)
        c1c, c2c = codes[i]
        for f in range(NFADE):
            M = haar_t(2, gen)
            M1, M2 = M[0], M[1]
            Q = M1.T @ M2
            h = (torch.randn(2, generator=gen, device=DEV)
                 + 1j * torch.randn(2, generator=gen, device=DEV)) \
                / math.sqrt(2)
            n = (torch.randn(D, generator=gen, device=DEV)
                 + 1j * torch.randn(D, generator=gen, device=DEV)) \
                / math.sqrt(2)
            n2 = (torch.randn(D, generator=gen, device=DEV)
                  + 1j * torch.randn(D, generator=gen, device=DEV)) \
                / math.sqrt(2)
            r0 = h[0] * (M1 @ e1).to(torch.cfloat) \
                + h[1] * (M2 @ e2).to(torch.cfloat)
            r = r0.unsqueeze(0) + sigs_t.view(-1, 1) * n.unsqueeze(0)
            t1 = (M1.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[0]
            t2 = (M2.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[1]
            c1 = (h[1] / h[0]).item()
            c2 = (h[0] / h[1]).item()
            v1 = sigs_t**2 / h[0].abs()**2
            v2 = sigs_t**2 / h[1].abs()**2
            g1 = aware_batch(t1, Q, bi, c1, v1)
            g2 = aware_batch(t2, Q.T, bi, c2, v2)
            res["edma"] += 0.5 * (abscos(g1, e1) + abscos(g2, e2))
            o1 = e1.to(torch.cfloat).unsqueeze(0) \
                + math.sqrt(2) * sigs_t.view(-1, 1) * n.unsqueeze(0) / h[0]
            o2 = e2.to(torch.cfloat).unsqueeze(0) \
                + math.sqrt(2) * sigs_t.view(-1, 1) * n2.unsqueeze(0) / h[1]
            res["oma"] += 0.5 * (abscos(o1, e1) + abscos(o2, e2))
            ge1 = (M1.T.to(torch.cfloat)
                   @ (r - h[1] * (M2 @ e2).to(torch.cfloat)).unsqueeze(-1)
                   ).squeeze(-1) / h[0]
            ge2 = (M2.T.to(torch.cfloat)
                   @ (r - h[0] * (M1 @ e1).to(torch.cfloat)).unsqueeze(-1)
                   ).squeeze(-1) / h[1]
            res["genie"] += 0.5 * (abscos(ge1, e1) + abscos(ge2, e2))
            for kk in ("edma", "oma", "genie"):
                cnt[kk] += 1
            if f < NFADE_TOD:
                hnp = (complex(h[0].item()), complex(h[1].item()))
                nslots = [(rng.standard_normal(tod[4])
                           + 1j * rng.standard_normal(tod[4]))
                          / math.sqrt(2) for _ in range(tod[3])]
                e1n, e2n = A[i], B[i]
                for k, s in enumerate(SNRS):
                    sig = 10 ** (-s / 20.0)
                    recs = todma_run(tod, (c1c, c2c), hnp, sig, nslots)
                    got = [cosine(recs[j], (e1n, e2n)[j])
                           for j in range(2) if recs[j] is not None]
                    if got:
                        res["todma"][k] += float(np.mean(got))
                    cnt["todma"][k] += 1
        print(f"  pair {i+1}/{npairs} done ({time.time()-t0:.0f}s)",
              flush=True)
    for k in keys:
        res[k] /= np.maximum(cnt[k], 1)

    with open(DATA / "bertvit_merged.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["snr_db"] + list(keys))
        for k, s in enumerate(SNRS):
            w.writerow([s] + [res[key][k] for key in keys])
    print(f"[OK] wrote {DATA/'bertvit_merged.csv'}")
    for k, s in enumerate(SNRS):
        print(f"  {s:4.1f} dB  EDMA {res['edma'][k]:.3f}  "
              f"ToDMA {res['todma'][k]:.3f}  OMA {res['oma'][k]:.3f}  "
              f"genie {res['genie'][k]:.3f}")


if __name__ == "__main__":
    main()
