"""
Capacity-matched EDMA refinement (parameter budget equal to the
attention scheme: 4 d^2 = 2.36M at d = 768).
================================================================
Four-head averaged gated refinement applied to the closed-form
demultiplexer output:

    out = (1/4) sum_k D softmax(Q_k x / sqrt(D)) .* x,

with Q_1..Q_4 in R^{D x D} (4 d^2 parameters, exactly the
attention scheme's budget). The single-gate 0.59M refiner is the
special case of four identical heads, so the family contains it
by construction. Same training recipe: demux outputs from
parametric pairs at beta = 0.028, Haar pool 32, Rayleigh
channels, complex noise, training SNR uniform in [5, 25] dB,
Adam 5e-4 with gradient clipping, batch 48, 200 epochs.

Evaluation on the real BERT/ViT pairs with fresh Haar masks and
200 fading draws per pair. Appends column `edma_ref2` to
data/bertvit_merged.csv and prints all-curve numbers.
"""
from __future__ import annotations
import csv
import math
import time
import numpy as np
import torch

from fig_real_merged import load_pairs, cosine, SNRS, NFADE, D, DATA

SEED = 2026
rng = np.random.default_rng(SEED + 31)
torch.manual_seed(SEED + 31)
BETA0 = 0.028
G0 = 1.0 - BETA0**2


def haar_t(gen):
    Q, R = torch.linalg.qr(torch.randn(D, D, generator=gen))
    return Q * torch.sign(torch.diagonal(R))


def train_refiner2(epochs=220, steps=20, batch=48, lr=5e-4,
                   l2=0.5, l3=0.5, pool=32):
    """Stage 1 trains a single gate (the proven 0.59M recipe); stage 2
    warm-starts four heads from it plus small perturbations and
    fine-tunes at a reduced learning rate, so the capacity-matched
    family starts at the single-gate solution it contains."""
    print(f"=== training capacity-matched refinement (4-head gate, "
          f"4d^2 = {4*D*D/1e6:.2f}M params, warm-started) ===",
          flush=True)
    gen = torch.Generator().manual_seed(SEED + 31)
    masks = []
    for _ in range(pool):
        U1, U2 = haar_t(gen), haar_t(gen)
        masks.append((U1.numpy(), (BETA0 * U1
                                   + math.sqrt(G0) * U2).numpy()))
    Q0 = torch.nn.Parameter(torch.randn(D, D, generator=gen)
                            / math.sqrt(D))
    params = [Q0]
    opt = torch.optim.Adam(params, lr=lr)
    stage2_at = 120          # epochs of single-gate pre-training

    def forward(x):
        outs = [D * torch.softmax((x @ Qk.T) / math.sqrt(D), dim=1) * x
                for Qk in params]
        return sum(outs) / len(params)

    t0 = time.time()
    for ep in range(epochs):
        if ep == stage2_at:
            base = params[0].detach()
            params = [torch.nn.Parameter(
                base.clone() + 0.02 * torch.randn(D, D, generator=gen)
                / math.sqrt(D)) for _ in range(4)]
            opt = torch.optim.Adam(params, lr=2e-4)
            print(f"  [warm start] 4 heads initialised from the trained "
                  f"gate at epoch {ep}", flush=True)
        for _ in range(steps):
            xs, ts = [], []
            for _ in range(batch):
                e1 = torch.nn.functional.normalize(
                    torch.randn(D, generator=gen), dim=0).numpy()
                w = torch.randn(D, generator=gen).numpy()
                w = w - (w @ e1) * e1
                w = w / np.linalg.norm(w)
                e2 = BETA0 * e1 + math.sqrt(G0) * w
                M1, M2 = masks[int(torch.randint(pool, (1,),
                                                 generator=gen))]
                snr = float(5.0 + 20.0 * torch.rand(1, generator=gen))
                sig = 10 ** (-snr / 20.0)
                h = (torch.randn(2, generator=gen).numpy()
                     + 1j * torch.randn(2, generator=gen).numpy()) \
                    / math.sqrt(2)
                nc = (torch.randn(D, generator=gen).numpy()
                      + 1j * torch.randn(D, generator=gen).numpy()) \
                    / math.sqrt(2)
                rc = h[0] * (M1 @ e1) + h[1] * (M2 @ e2) + sig * nc
                t1 = M1.T @ rc / h[0]
                t2 = M2.T @ rc / h[1]
                g1 = (t1 - BETA0 * (h[1] / h[0]) * t2) / G0
                xs.append(torch.tensor(np.real(g1), dtype=torch.float32))
                ts.append(torch.tensor(e1, dtype=torch.float32))
            x = torch.stack(xs); t = torch.stack(ts)
            out = forward(x)
            mse = ((out - t)**2).mean()
            cs = torch.nn.functional.cosine_similarity(out, t, dim=1).mean()
            loss = l2 * mse + l3 * (1.0 - cs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}: loss {float(loss.detach()):.4f} "
                  f"(cos {float(cs.detach()):.3f})", flush=True)
    print(f"  trained in {time.time()-t0:.0f}s")
    return [p.detach().numpy() for p in params]


def refine2(P, g):
    x = np.real(g)

    def gate(Q, v):
        sc = (Q @ v) / math.sqrt(D)
        sc = sc - sc.max()
        w = np.exp(sc); w /= w.sum()
        return D * w * v

    return sum(gate(Qk, x) for Qk in P) / 4.0


def main():
    A, B, betas = load_pairs()
    P = train_refiner2()
    ref = np.zeros(len(SNRS)); cnt = 0
    t0 = time.time()
    for i in range(len(A)):
        e1, e2, bi = A[i], B[i], float(betas[i])
        gi = 1.0 - bi**2
        for f in range(NFADE):
            G1 = rng.standard_normal((D, D))
            Qh, Rh = np.linalg.qr(G1)
            U1 = Qh * np.sign(np.diag(Rh))
            G2 = rng.standard_normal((D, D))
            Qh, Rh = np.linalg.qr(G2)
            U2 = Qh * np.sign(np.diag(Rh))
            M1 = U1
            M2 = bi * U1 + math.sqrt(gi) * U2
            h = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) \
                / math.sqrt(2)
            h1, h2 = h
            r0 = h1 * (M1 @ e1) + h2 * (M2 @ e2)
            n = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) \
                / math.sqrt(2)
            for k, s in enumerate(SNRS):
                sig = 10 ** (-s / 20.0)
                r = r0 + sig * n
                t1 = M1.T @ r / h1; t2 = M2.T @ r / h2
                g1 = (t1 - bi * (h2 / h1) * t2) / gi
                g2 = (t2 - bi * (h1 / h2) * t1) / gi
                ref[k] += 0.5 * (cosine(refine2(P, g1), e1)
                                 + cosine(refine2(P, g2), e2))
            cnt += 1
        print(f"  pair {i+1}/{len(A)} done ({time.time()-t0:.0f}s)",
              flush=True)
    ref /= cnt

    rows = list(csv.DictReader(open(DATA / "bertvit_merged.csv")))
    names = list(rows[0].keys())
    if "edma_ref2" not in names:
        names.append("edma_ref2")
    for k, r in enumerate(rows):
        r["edma_ref2"] = f"{ref[k]}"
    with open(DATA / "bertvit_merged.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader(); w.writerows(rows)
    print("[OK] appended edma_ref2 to bertvit_merged.csv")
    for k, r in enumerate(rows):
        print(f"  {float(r['snr_db']):4.0f} dB  "
              f"EDMA {float(r['edma']):.3f}  "
              f"ref(0.59M) {float(r['edma_ref']):.3f}  "
              f"ref2(2.36M) {ref[k]:.3f}  "
              f"ATT(2.36M) {float(r['att']):.3f}  "
              f"genie {float(r['genie']):.3f}")


if __name__ == "__main__":
    main()
