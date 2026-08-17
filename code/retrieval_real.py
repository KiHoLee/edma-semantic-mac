"""
Task-level validation: nearest-neighbor retrieval with recovered
embeddings on the real BERT/ViT pairs.
===================================================================
For every recovered embedding, rank the 16 clean database embeddings
of the SAME modality by absolute cosine and score top-1 retrieval of
the transmitted item (chance level 1/16). This measures whether the
recovery preserves semantic identity, the operational question behind
the cosine metric of the manuscript.

Schemes: EDMA (aware Wiener), hybrid (EDMA + stored refinement gate
from data/refine_gates.npz), OMA (equivalent-bandwidth model),
genie-aided SIC bound, ToDMA-adapted (first 40 fading draws).

Same channel, energy, and seed conventions as fig_real_merged.py.
Writes data/retrieval_real.csv. Run under WSL (torch, CUDA).
"""
from __future__ import annotations
import csv
import math
import time
import numpy as np
import torch

from fig_real_merged import (load_pairs, haar_t, aware_batch, todma_prepare,
                             omp_code, todma_run, SNRS, NFADE, NFADE_TOD,
                             D, DATA, DEV, SEED)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def refine_apply_single(P1, z):
    """z: (b, D) real torch tensor; P1: (D, D) gate."""
    return D * torch.softmax((z @ P1.T) / math.sqrt(D), dim=1) * z


def top1(rec, db, idx):
    """rec: (b, D) cfloat; db: (n, D) float; returns (b,) 0/1 hits."""
    sims = (rec @ db.T.to(rec.dtype).conj()).abs()          # (b, n)
    sims = sims / (rec.norm(dim=1, keepdim=True)
                   * db.norm(dim=1).unsqueeze(0))
    return (sims.argmax(dim=1) == idx).float().cpu().numpy()


def main():
    A, B, betas = load_pairs()
    npairs = len(A)
    gates = np.load(DATA / "refine_gates.npz")
    P1 = torch.tensor(gates["P1"], dtype=torch.float32, device=DEV)
    tod = todma_prepare()
    codes = [(omp_code(tod[0], A[i], tod[3]),
              omp_code(tod[0], B[i], tod[3])) for i in range(npairs)]
    At = torch.tensor(A, dtype=torch.float32, device=DEV)
    Bt = torch.tensor(B, dtype=torch.float32, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    nb = len(SNRS)
    sigs = torch.tensor(10 ** (-SNRS / 20.0), dtype=torch.float32,
                        device=DEV)
    keys = ("edma", "hybrid", "oma", "genie", "todma")
    acc = {k: np.zeros(nb) for k in keys}
    cnt = {k: np.zeros(nb) for k in keys}
    t0 = time.time()
    for i in range(npairs):
        bi = float(betas[i])
        e1, e2 = At[i], Bt[i]
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
            r = r0.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
            t1 = (M1.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[0]
            t2 = (M2.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[1]
            c1 = (h[1] / h[0]).item()
            c2 = (h[0] / h[1]).item()
            v1 = sigs**2 / h[0].abs()**2
            v2 = sigs**2 / h[1].abs()**2
            g1 = aware_batch(t1, Q, bi, c1, v1)
            g2 = aware_batch(t2, Q.T, bi, c2, v2)
            acc["edma"] += 0.5 * (top1(g1, At, i) + top1(g2, Bt, i))
            hy1 = refine_apply_single(P1, g1.real.float()).to(torch.cfloat)
            hy2 = refine_apply_single(P1, g2.real.float()).to(torch.cfloat)
            acc["hybrid"] += 0.5 * (top1(hy1, At, i) + top1(hy2, Bt, i))
            o1 = e1.to(torch.cfloat).unsqueeze(0) \
                + math.sqrt(2) * sigs.view(-1, 1) * n.unsqueeze(0) / h[0]
            o2 = e2.to(torch.cfloat).unsqueeze(0) \
                + math.sqrt(2) * sigs.view(-1, 1) * n2.unsqueeze(0) / h[1]
            acc["oma"] += 0.5 * (top1(o1, At, i) + top1(o2, Bt, i))
            ge1 = (M1.T.to(torch.cfloat)
                   @ (r - h[1] * (M2 @ e2).to(torch.cfloat)).unsqueeze(-1)
                   ).squeeze(-1) / h[0]
            ge2 = (M2.T.to(torch.cfloat)
                   @ (r - h[0] * (M1 @ e1).to(torch.cfloat)).unsqueeze(-1)
                   ).squeeze(-1) / h[1]
            acc["genie"] += 0.5 * (top1(ge1, At, i) + top1(ge2, Bt, i))
            for kk in ("edma", "hybrid", "oma", "genie"):
                cnt[kk] += 1
            if f < NFADE_TOD:
                hnp = (complex(h[0].item()), complex(h[1].item()))
                nslots = [(rng.standard_normal(tod[4])
                           + 1j * rng.standard_normal(tod[4]))
                          / math.sqrt(2) for _ in range(tod[3])]
                for k, s in enumerate(SNRS):
                    sig = 10 ** (-s / 20.0)
                    recs = todma_run(tod, (c1c, c2c), hnp, sig, nslots)
                    hit = 0.0
                    for j, (rec, db, ii) in enumerate(
                            ((recs[0], A, i), (recs[1], B, i))):
                        if rec is None:
                            continue           # failed detection: no hit
                        sims = np.abs(db @ rec) / (
                            np.linalg.norm(db, axis=1)
                            * np.linalg.norm(rec))
                        hit += 0.5 * float(int(np.argmax(sims)) == ii)
                    acc["todma"][k] += hit
                    cnt["todma"][k] += 1
        print(f"  pair {i+1}/{npairs} done ({time.time()-t0:.0f}s)",
              flush=True)
    for k in keys:
        acc[k] /= np.maximum(cnt[k], 1)

    with open(DATA / "retrieval_real.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db"] + list(keys))
        for k, s in enumerate(SNRS):
            w.writerow([s] + [acc[key][k] for key in keys])
    print(f"[OK] wrote {DATA/'retrieval_real.csv'}")
    for k, s in enumerate(SNRS):
        print(f"  {s:4.1f} dB  EDMA {acc['edma'][k]:.3f}  "
              f"hybrid {acc['hybrid'][k]:.3f}  ToDMA {acc['todma'][k]:.3f} "
              f" OMA {acc['oma'][k]:.3f}  genie {acc['genie'][k]:.3f}")


if __name__ == "__main__":
    main()
