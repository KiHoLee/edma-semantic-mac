"""
Refinement stage for the v2 (affinity-aware Wiener) EDMA receiver.
================================================================
Trains the single-gate refinement operator

    out = D softmax(W z / sqrt(D)) .* z,   z = Re(e_hat),

(0.59M parameters at d = 768) on aware-demultiplexer outputs, then
warm-starts a capacity-check variant with four heads (4 d^2 = 2.36M)
and fine-tunes it, so the family contains the single gate by
construction. Training data: parametric pairs at beta = 0.028, a
fixed pool of 32 independent Haar mask pairs, Rayleigh channels,
complex noise, training SNR uniform in [5, 25] dB, Adam 5e-4 with
gradient clipping, batch 48, 220 epochs (stage 2 from epoch 120 at
lr 2e-4).

Evaluation on the real BERT/ViT pairs with fresh Haar masks and
NFADE fading draws per pair. Appends columns `edma_ref` (single
gate) and `edma_ref2` (four heads) to data/bertvit_merged.csv.
Requires torch (run under WSL with CUDA if available).
"""
from __future__ import annotations
import csv
import math
import time
import numpy as np
import torch

from fig_real_merged import load_pairs, SNRS, NFADE, D, DATA

SEED = 2026
torch.manual_seed(SEED + 31)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BETA0 = 0.028
G0 = 1.0 - BETA0**2
print(f"[refine] device = {DEV}")


def haar_t(n, gen):
    G = torch.randn(n, D, D, generator=gen, device=DEV)
    Q, R = torch.linalg.qr(G)
    return Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)


def aware_t(t1, Q, beta, c1, nvar):
    """Batched affinity-aware Wiener demux in torch (complex)."""
    b = t1.shape[0]
    g = 1.0 - beta * beta
    rho = g * (c1.abs()**2) / D + nvar          # (b,)
    A = torch.eye(D, device=DEV, dtype=torch.cfloat).expand(b, D, D) \
        + beta * c1.view(b, 1, 1) * Q.to(torch.cfloat)
    S = A @ A.mH / D + rho.view(b, 1, 1) \
        * torch.eye(D, device=DEV, dtype=torch.cfloat)
    x = torch.linalg.solve(S, t1.unsqueeze(-1))
    return (A.mH @ x).squeeze(-1) / D


def train_batch(masks, Qs, gen, batch):
    """Generate one training batch of aware-demux outputs (user 1)."""
    e1 = torch.nn.functional.normalize(
        torch.randn(batch, D, generator=gen, device=DEV), dim=1)
    w = torch.randn(batch, D, generator=gen, device=DEV)
    w = w - (w * e1).sum(1, keepdim=True) * e1
    w = torch.nn.functional.normalize(w, dim=1)
    e2 = BETA0 * e1 + math.sqrt(G0) * w
    sel = torch.randint(len(masks), (batch,), generator=gen, device=DEV)
    M1 = masks[0][sel]; M2 = masks[1][sel]; Q = Qs[sel]
    snr = 5.0 + 20.0 * torch.rand(batch, generator=gen, device=DEV)
    sig = 10 ** (-snr / 20.0)
    h = (torch.randn(batch, 2, generator=gen, device=DEV)
         + 1j * torch.randn(batch, 2, generator=gen, device=DEV)) \
        / math.sqrt(2)
    n = (torch.randn(batch, D, generator=gen, device=DEV)
         + 1j * torch.randn(batch, D, generator=gen, device=DEV)) \
        / math.sqrt(2)
    r = h[:, :1] * (M1 @ e1.unsqueeze(-1)).squeeze(-1).to(torch.cfloat) \
        + h[:, 1:2] * (M2 @ e2.unsqueeze(-1)).squeeze(-1).to(torch.cfloat) \
        + sig.view(-1, 1) * n
    t1 = (M1.transpose(-1, -2).to(torch.cfloat)
          @ r.unsqueeze(-1)).squeeze(-1) / h[:, :1]
    c1 = h[:, 1] / h[:, 0]
    nvar = sig**2 / h[:, 0].abs()**2
    g1 = aware_t(t1, Q, BETA0, c1, nvar)
    return g1.real.float(), e1


def train_refiners(epochs=220, steps=20, batch=48, lr=5e-4,
                   l2=0.5, l3=0.5, pool=32, stage2_at=120):
    print(f"=== training refinement (single gate {D*D/1e6:.2f}M, "
          f"then 4-head warm start {4*D*D/1e6:.2f}M) ===", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(SEED + 31)
    U1 = haar_t(pool, gen); U2 = haar_t(pool, gen)
    masks = (U1, U2)
    Qs = U1.transpose(-1, -2) @ U2
    params = [torch.nn.Parameter(
        torch.randn(D, D, generator=gen, device=DEV) / math.sqrt(D))]
    opt = torch.optim.Adam(params, lr=lr)
    P_single = None

    def forward(x, ps):
        outs = [D * torch.softmax((x @ Qk.T) / math.sqrt(D), dim=1) * x
                for Qk in ps]
        return sum(outs) / len(ps)

    t0 = time.time()
    for ep in range(epochs):
        if ep == stage2_at:
            P_single = params[0].detach().clone()
            base = params[0].detach()
            params = [torch.nn.Parameter(
                base.clone() + 0.02 * torch.randn(D, D, generator=gen,
                                                  device=DEV)
                / math.sqrt(D)) for _ in range(4)]
            opt = torch.optim.Adam(params, lr=2e-4)
            print(f"  [warm start] 4 heads at epoch {ep}", flush=True)
        for _ in range(steps):
            with torch.no_grad():
                x, tgt = train_batch(masks, Qs, gen, batch)
            out = forward(x, params)
            mse = ((out - tgt)**2).mean()
            cs = torch.nn.functional.cosine_similarity(out, tgt, dim=1).mean()
            loss = l2 * mse + l3 * (1.0 - cs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (ep + 1) % 40 == 0:
            print(f"  epoch {ep+1}: loss {float(loss.detach()):.4f} "
                  f"(cos {float(cs.detach()):.3f})", flush=True)
    print(f"  trained in {time.time()-t0:.0f}s")
    return P_single, [p.detach() for p in params]


def refine_apply(ps, z):
    """z: (b, D) real torch tensor; ps: list of gates."""
    outs = [D * torch.softmax((z @ Qk.T) / math.sqrt(D), dim=1) * z
            for Qk in ps]
    return sum(outs) / len(ps)


def main():
    A, B, betas = load_pairs()
    P1, P4 = train_refiners()
    gen = torch.Generator(device=DEV).manual_seed(SEED + 77)
    ref1 = np.zeros(len(SNRS)); ref4 = np.zeros(len(SNRS)); cnt = 0
    t0 = time.time()
    At = torch.tensor(A, dtype=torch.float32, device=DEV)
    Bt = torch.tensor(B, dtype=torch.float32, device=DEV)
    for i in range(len(A)):
        bi = float(betas[i])
        e1 = At[i]; e2 = Bt[i]
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
            r0 = h[0] * (M1 @ e1).to(torch.cfloat) \
                + h[1] * (M2 @ e2).to(torch.cfloat)
            sigs = torch.tensor(10 ** (-SNRS / 20.0), device=DEV,
                                dtype=torch.float32)
            nb = len(SNRS)
            r = r0.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
            t1 = (M1.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[0]
            t2 = (M2.T.to(torch.cfloat) @ r.unsqueeze(-1)).squeeze(-1) / h[1]
            c1 = (h[1] / h[0]).expand(nb)
            c2 = (h[0] / h[1]).expand(nb)
            v1 = (sigs**2 / h[0].abs()**2)
            v2 = (sigs**2 / h[1].abs()**2)
            g1 = aware_t(t1, Q.expand(nb, D, D), bi, c1, v1).real.float()
            g2 = aware_t(t2, Q.T.expand(nb, D, D), bi, c2, v2).real.float()
            with torch.no_grad():
                for P, acc in ((([P1]), ref1), ((P4), ref4)):
                    o1 = refine_apply(P, g1)
                    o2 = refine_apply(P, g2)
                    cs1 = torch.nn.functional.cosine_similarity(
                        o1, e1.unsqueeze(0), dim=1).abs()
                    cs2 = torch.nn.functional.cosine_similarity(
                        o2, e2.unsqueeze(0), dim=1).abs()
                    acc += (0.5 * (cs1 + cs2)).cpu().numpy()
            cnt += 1
        print(f"  pair {i+1}/{len(A)} done ({time.time()-t0:.0f}s)",
              flush=True)
    ref1 /= cnt; ref4 /= cnt

    rows = list(csv.DictReader(open(DATA / "bertvit_merged.csv")))
    names = list(rows[0].keys())
    for col in ("edma_ref", "edma_ref2"):
        if col not in names:
            names.append(col)
    for k, r in enumerate(rows):
        r["edma_ref"] = f"{ref1[k]}"
        r["edma_ref2"] = f"{ref4[k]}"
    with open(DATA / "bertvit_merged.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader(); w.writerows(rows)
    print("[OK] appended edma_ref / edma_ref2 to bertvit_merged.csv")
    for k, r in enumerate(rows):
        print(f"  {float(r['snr_db']):4.1f} dB  "
              f"EDMA {float(r['edma']):.3f}  ref {ref1[k]:.3f}  "
              f"ref2 {ref4[k]:.3f}  genie {float(r['genie']):.3f}")


if __name__ == "__main__":
    main()
