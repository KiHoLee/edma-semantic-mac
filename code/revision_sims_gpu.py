"""
GPU-accelerated Monte Carlo experiments (torch backend).
========================================================
Computes the CSV artifacts of experiments E2, E3, E4, E5, E7c, E8
of revision_sims.py with identical models and conventions, using
torch (CUDA when available) for the dense linear algebra. All
random draws come from the numpy generator with the documented seed
2026, so the sample stream is platform-independent; torch only
accelerates QR, matrix products, and linear solves in float32 /
complex64 precision. Figures are rendered separately by
replot_all.py, which reads only data/.

Run under WSL:  python3 revision_sims_gpu.py E2 E3 E4 E5 E7c E8
"""
from __future__ import annotations
import csv
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data"

SEED = 2026
rng = np.random.default_rng(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[gpu] device = {DEV}")


def write_csv(name, header, rows):
    p = CSV_DIR / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[OK] wrote {p}")


def haar_g(d):
    """Haar orthogonal on the GPU from a numpy Gaussian draw."""
    G = torch.tensor(rng.standard_normal((d, d)), dtype=torch.float32,
                     device=DEV)
    Q, R = torch.linalg.qr(G)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def embed_pair(d, beta):
    e1 = rng.standard_normal(d)
    e1 /= np.linalg.norm(e1)
    w = rng.standard_normal(d)
    w -= (w @ e1) * e1
    w /= np.linalg.norm(w)
    e2 = beta * e1 + math.sqrt(1.0 - beta**2) * w
    return (torch.tensor(e1, dtype=torch.float32, device=DEV),
            torch.tensor(e2, dtype=torch.float32, device=DEV))


def rayleigh2():
    h = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) / math.sqrt(2)
    return complex(h[0]), complex(h[1])


def cnoise_g(d):
    n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
    return torch.tensor(n, dtype=torch.complex64, device=DEV)


def aware_g(t, Q, beta, c, nvar):
    """Batched aware Wiener demux. t: (b,d) cfloat, Q: (d,d) float,
    c: python complex, nvar: (b,) tensor."""
    d = Q.shape[0]
    b = t.shape[0]
    g = 1.0 - beta * beta
    rho = g * abs(c)**2 / d + nvar
    A = torch.eye(d, device=DEV, dtype=torch.complex64) \
        + beta * c * Q.to(torch.complex64)
    S = (A @ A.mH / d).unsqueeze(0) \
        + rho.view(b, 1, 1) * torch.eye(d, device=DEV,
                                        dtype=torch.complex64)
    x = torch.linalg.solve(S, t.unsqueeze(-1))
    return (A.mH.unsqueeze(0) @ x).squeeze(-1) / d


def abscos(a, e):
    num = (a * e.to(a.dtype).conj()).sum(-1).abs()
    return (num / (a.norm(dim=-1) * e.norm())).cpu().numpy()


# ------------------------------------------------------------------
def E2_sic(beta=0.311, d=512, ntr=400):
    print("\n=== E2: receiver comparison under block-Rayleigh fading ===")
    snr_db = np.arange(0, 31, 2.5)
    sigs = torch.tensor(10 ** (-snr_db / 20.0), dtype=torch.float32,
                        device=DEV)
    nb = len(snr_db)
    res = {k: np.zeros(nb) for k in ("edma", "blind", "oma", "genie", "sic")}
    t0 = time.time()
    for tr in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = haar_g(d), haar_g(d)
        Q = M1.T @ M2
        h1, h2 = rayleigh2()
        n = cnoise_g(d); n2 = cnoise_g(d)
        r0 = h1 * (M1 @ e1).to(torch.complex64) \
            + h2 * (M2 @ e2).to(torch.complex64)
        r = r0.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
        t1 = (M1.T.to(torch.complex64) @ r.unsqueeze(-1)).squeeze(-1) / h1
        t2 = (M2.T.to(torch.complex64) @ r.unsqueeze(-1)).squeeze(-1) / h2
        c1, c2 = h2 / h1, h1 / h2
        v1 = sigs**2 / abs(h1)**2
        v2 = sigs**2 / abs(h2)**2
        g1 = aware_g(t1, Q, beta, c1, v1)
        g2 = aware_g(t2, Q.T, beta, c2, v2)
        res["edma"] += 0.5 * (abscos(g1, e1) + abscos(g2, e2))
        res["blind"] += 0.5 * (abscos(t1, e1) + abscos(t2, e2))
        o1 = e1.to(torch.complex64).unsqueeze(0) \
            + math.sqrt(2) * sigs.view(-1, 1) * n.unsqueeze(0) / h1
        o2 = e2.to(torch.complex64).unsqueeze(0) \
            + math.sqrt(2) * sigs.view(-1, 1) * n2.unsqueeze(0) / h2
        res["oma"] += 0.5 * (abscos(o1, e1) + abscos(o2, e2))
        ge1 = (M1.T.to(torch.complex64)
               @ (r - h2 * (M2 @ e2).to(torch.complex64)).unsqueeze(-1)
               ).squeeze(-1) / h1
        ge2 = (M2.T.to(torch.complex64)
               @ (r - h1 * (M1 @ e1).to(torch.complex64)).unsqueeze(-1)
               ).squeeze(-1) / h2
        res["genie"] += 0.5 * (abscos(ge1, e1) + abscos(ge2, e2))
        # realizable decision-directed SIC: the stronger user is detected
        # with the same aware Wiener stage (a scalar-scaled matched-filter
        # decision would re-modulate to a multiple of r itself, because
        # M_s M_s^T = I, and cancel nothing), its re-modulated estimate is
        # subtracted, and the weaker user is read from the residual.
        if abs(h1) >= abs(h2):
            hs, hw, Ms, Mw, es, ew = h1, h2, M1, M2, e1, e2
            Qsw, csw, vsw = Q, c1, v1
        else:
            hs, hw, Ms, Mw, es, ew = h2, h1, M2, M1, e2, e1
            Qsw, csw, vsw = Q.T, c2, v2
        t_s = (Ms.T.to(torch.complex64) @ r.unsqueeze(-1)).squeeze(-1) / hs
        dec = aware_g(t_s, Qsw, beta, csw, vsw)
        r_res = r - hs * (Ms.to(torch.complex64)
                          @ dec.unsqueeze(-1)).squeeze(-1)
        d_w = (Mw.T.to(torch.complex64) @ r_res.unsqueeze(-1)).squeeze(-1) / hw
        res["sic"] += 0.5 * (abscos(dec, es) + abscos(d_w, ew))
        if (tr + 1) % 100 == 0:
            print(f"  {tr+1}/{ntr} ({time.time()-t0:.0f}s)", flush=True)
    for k in res:
        res[k] /= ntr
    rows = [[s] + [res[k][i] for k in ("edma", "blind", "oma", "genie", "sic")]
            for i, s in enumerate(snr_db)]
    write_csv("sic_comparison",
              ["snr_db", "edma", "blind", "oma", "genie", "sic"], rows)
    i20 = list(snr_db).index(20)
    print(f"  at 20 dB: EDMA {res['edma'][i20]:.3f}, blind "
          f"{res['blind'][i20]:.3f}, SIC {res['sic'][i20]:.3f}, "
          f"genie {res['genie'][i20]:.3f}, OMA {res['oma'][i20]:.3f}")


# ------------------------------------------------------------------
def E3_unconditional(beta=0.311, d=512, ntr=1500):
    print("\n=== E3: Rayleigh unconditional MSE of the aware receiver ===")
    rows = []
    for s in (10, 20):
        sig = 10 ** (-s / 20.0)
        mses, bl = [], []
        for _ in range(ntr):
            e1, e2 = embed_pair(d, beta)
            M1, M2 = haar_g(d), haar_g(d)
            Q = M1.T @ M2
            h1, h2 = rayleigh2()
            n = cnoise_g(d)
            r = h1 * (M1 @ e1).to(torch.complex64) \
                + h2 * (M2 @ e2).to(torch.complex64) + sig * n
            t1 = (M1.T.to(torch.complex64) @ r) / h1
            c1 = h2 / h1
            nv = torch.tensor([sig**2 / abs(h1)**2], device=DEV)
            g1 = aware_g(t1.unsqueeze(0), Q, beta, c1, nv)[0]
            mses.append(float((g1 - e1.to(torch.complex64)).norm()**2))
            lam = (1.0 / d) / (1.0 / d + abs(c1)**2 / d + sig**2 / abs(h1)**2)
            b1 = lam * t1
            bl.append(float((b1 - e1.to(torch.complex64)).norm()**2))
        rows.append([s, float(np.mean(mses)), float(np.median(mses)),
                     float(np.mean(bl)), float(np.median(bl))])
        print(f"  {s} dB: aware mean {rows[-1][1]:.4f} "
              f"(median {rows[-1][2]:.4f}) | blind mean {rows[-1][3]:.4f} "
              f"(median {rows[-1][4]:.4f})")
    write_csv("rayleigh_mse", ["snr_db", "aware_mean", "aware_median",
                               "blind_mean", "blind_median"], rows)


# ------------------------------------------------------------------
def E4_csi(beta=0.311, d=512, snr=30.0, ntr=400):
    print("\n=== E4: imperfect CSI robustness (EDMA vs realizable SIC) ===")
    sh2 = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    sig = 10 ** (-snr / 20.0)
    res_e = np.zeros(len(sh2)); res_s = np.zeros(len(sh2))
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = haar_g(d), haar_g(d)
        Q = M1.T @ M2
        h1, h2 = rayleigh2()
        n = cnoise_g(d)
        r = h1 * (M1 @ e1).to(torch.complex64) \
            + h2 * (M2 @ e2).to(torch.complex64) + sig * n
        eps1, eps2 = rayleigh2()
        for j, v in enumerate(sh2):
            hh1 = h1 + math.sqrt(v) * eps1
            hh2 = h2 + math.sqrt(v) * eps2
            t1 = (M1.T.to(torch.complex64) @ r) / hh1
            t2 = (M2.T.to(torch.complex64) @ r) / hh2
            nv1 = torch.tensor([sig**2 / abs(hh1)**2], device=DEV)
            nv2 = torch.tensor([sig**2 / abs(hh2)**2], device=DEV)
            g1 = aware_g(t1.unsqueeze(0), Q, beta, hh2 / hh1, nv1)[0]
            g2 = aware_g(t2.unsqueeze(0), Q.T, beta, hh1 / hh2, nv2)[0]
            res_e[j] += 0.5 * (float(abscos(g1.unsqueeze(0), e1)[0])
                               + float(abscos(g2.unsqueeze(0), e2)[0]))
            if abs(hh1) >= abs(hh2):
                hs, hw, Ms, Mw, es, ew = hh1, hh2, M1, M2, e1, e2
                Qsw, csw = Q, hh2 / hh1
            else:
                hs, hw, Ms, Mw, es, ew = hh2, hh1, M2, M1, e2, e1
                Qsw, csw = Q.T, hh1 / hh2
            t_s = (Ms.T.to(torch.complex64) @ r) / hs
            nvs = torch.tensor([sig**2 / abs(hs)**2], device=DEV)
            dec = aware_g(t_s.unsqueeze(0), Qsw, beta, csw, nvs)[0]
            r_res = r - hs * (Ms.to(torch.complex64) @ dec)
            d_w = (Mw.T.to(torch.complex64) @ r_res) / hw
            res_s[j] += 0.5 * (float(abscos(dec.unsqueeze(0), es)[0])
                               + float(abscos(d_w.unsqueeze(0), ew)[0]))
    res_e /= ntr; res_s /= ntr
    print(f"  EDMA:  {res_e[0]:.4f} -> {res_e[-1]:.4f} "
          f"(delta {100*(res_e[0]-res_e[-1]):.2f} points)")
    print(f"  SIC :  {res_s[0]:.4f} -> {res_s[-1]:.4f} "
          f"(delta {100*(res_s[0]-res_s[-1]):.2f} points)")
    rows = [[v, res_e[j], res_s[j]] for j, v in enumerate(sh2)]
    write_csv("csi_error", ["sigma_h2", "edma", "sic"], rows)


# ------------------------------------------------------------------
def E5_maskfam(beta=0.311, d=512, ntr=200):
    print("\n=== E5: Walsh-Hadamard diagonal variant vs Haar ===")
    snr_db = np.arange(0, 41, 5)
    sigs = torch.tensor(10 ** (-snr_db / 20.0), dtype=torch.float32,
                        device=DEV)
    nb = len(snr_db)
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    Ht = torch.tensor(H / math.sqrt(d), dtype=torch.float32, device=DEV)
    g = 1.0 - beta**2
    res = {"haar": np.zeros(nb), "wh": np.zeros(nb)}
    exact = np.zeros(nb)
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = haar_g(d), haar_g(d)
        Q = M1.T @ M2
        D1 = torch.tensor(np.sign(rng.standard_normal(d)),
                          dtype=torch.float32, device=DEV)
        D2 = torch.tensor(np.sign(rng.standard_normal(d)),
                          dtype=torch.float32, device=DEV)
        W1, W2 = Ht * D1.unsqueeze(0), Ht * D2.unsqueeze(0)
        q = D1 * D2
        n = cnoise_g(d)
        r0h = (M1 @ e1 + M2 @ e2).to(torch.complex64)
        r0w = (W1 @ e1 + W2 @ e2).to(torch.complex64)
        rh = r0h.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
        rw = r0w.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
        t1 = (M1.T.to(torch.complex64) @ rh.unsqueeze(-1)).squeeze(-1)
        g1 = aware_g(t1, Q, beta, 1.0, sigs**2)
        res["haar"] += abscos(g1, e1)
        tw = (W1.T.to(torch.complex64) @ rw.unsqueeze(-1)).squeeze(-1)
        a = 1.0 + beta * q                             # (d,)
        rho = g / d + sigs**2                          # (nb,)
        wdiag = a.unsqueeze(0) / (a.unsqueeze(0)**2 / d
                                  + rho.view(-1, 1))   # (nb,d)
        w1 = (wdiag.to(torch.complex64) / d) * tw
        res["wh"] += abscos(w1, e1)
        exact += ((g + d * sigs.view(-1, 1)**2)
                  / (a.unsqueeze(0)**2 + g + d * sigs.view(-1, 1)**2)
                  ).mean(1).cpu().numpy()
    for k in res:
        res[k] /= ntr
    exact /= ntr
    print(f"  max |WH - Haar| cosine dev: "
          f"{100*np.max(np.abs(res['wh']-res['haar'])):.2f} points; "
          f"40 dB WH {res['wh'][-1]:.4f} vs Haar {res['haar'][-1]:.4f}")
    rows = [[s, res["haar"][i], res["wh"][i], exact[i]]
            for i, s in enumerate(snr_db)]
    write_csv("mask_family_rev", ["snr_db", "haar", "wh", "wh_exact_mse"],
              rows)


# ------------------------------------------------------------------
def E7_multiuser(beta=0.311, d=512, ntr=100):
    print("\n=== E7c: multi-user scaling (joint Wiener) ===")
    snr_db = np.arange(0, 31, 2.5)
    sigs = torch.tensor(10 ** (-snr_db / 20.0), dtype=torch.float32,
                        device=DEV)
    nb = len(snr_db)
    rows = []
    for U in (2, 3, 4):
        B = (1 - beta) * np.eye(U) + beta * np.ones((U, U))
        A = np.linalg.cholesky(B)
        Bt = torch.tensor(B, dtype=torch.float32, device=DEV)
        err = np.zeros(nb)
        t0 = time.time()
        for _ in range(ntr):
            F = rng.standard_normal((d, U))
            Fq, _ = np.linalg.qr(F)
            E = (Fq @ A.T).T
            Et = torch.tensor(E, dtype=torch.float32, device=DEV)
            masks = [haar_g(d) for _ in range(U)]
            n = cnoise_g(d)
            r0 = sum(masks[u] @ Et[u] for u in range(U)).to(torch.complex64)
            Qs = [masks[0].T @ masks[v] for v in range(U)]
            Ret = sum(Bt[0, v] * Qs[v].T for v in range(U)) / d
            S0 = sum(Bt[v, w] * (Qs[v] @ Qs[w].T)
                     for v in range(U) for w in range(U)) / d
            r = r0.unsqueeze(0) + sigs.view(-1, 1) * n.unsqueeze(0)
            t = (masks[0].T.to(torch.complex64)
                 @ r.unsqueeze(-1)).squeeze(-1)
            S = S0.to(torch.complex64).unsqueeze(0) \
                + (sigs**2).view(-1, 1, 1) \
                * torch.eye(d, device=DEV, dtype=torch.complex64)
            x = torch.linalg.solve(S, t.unsqueeze(-1))
            eh = (Ret.to(torch.complex64).unsqueeze(0) @ x).squeeze(-1)
            err += ((eh - Et[0].to(torch.complex64)).norm(dim=1)**2
                    ).cpu().numpy()
        err /= ntr
        T_mc = U * np.log2(1.0 / err)
        r0v = (U - 1) + d / 10 ** (snr_db / 10.0)
        T_bl = U * np.log2(1.0 + 1.0 / r0v)
        T_oma = U * np.log2(1 + 10 ** (snr_db / 10.0) / (U * d))
        for i, s in enumerate(snr_db):
            rows.append([U, s, T_mc[i], T_bl[i], T_oma[i], err[i]])
        i20 = list(snr_db).index(20.0)
        print(f"  U={U}: at 20 dB EDMA {T_mc[i20]:.3f} vs blind "
              f"{T_bl[i20]:.3f} vs OMA {T_oma[i20]:.3f} "
              f"(gain {T_mc[i20]/T_oma[i20]:.2f}x), floor MSE {err[-1]:.4f}"
              f"  [{time.time()-t0:.0f}s]")
    write_csv("multiuser_corrected",
              ["U", "snr_db", "edma_mc", "blind", "oma", "mse_mc"], rows)


# ------------------------------------------------------------------
def E8_mismatch(beta=0.311, d=512, snr=20.0, ntr=200):
    print("\n=== E8: affinity mismatch of the aware receiver ===")
    sig = 10 ** (-snr / 20.0)
    bhs = [b for b in (beta - 0.1, beta - 0.06, beta, beta + 0.06,
                       beta + 0.1, beta + 2 ** -8, 0.0) if b >= 0]
    accs = np.zeros(len(bhs)); msea = np.zeros(len(bhs))
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        M1, M2 = haar_g(d), haar_g(d)
        Q = M1.T @ M2
        n = cnoise_g(d)
        r = (M1 @ e1 + M2 @ e2).to(torch.complex64) + sig * n
        t1 = (M1.T.to(torch.complex64) @ r)
        nv = torch.tensor([sig**2], device=DEV)
        for j, bh in enumerate(bhs):
            g1 = aware_g(t1.unsqueeze(0), Q, bh, 1.0, nv)[0]
            accs[j] += float(abscos(g1.unsqueeze(0), e1)[0])
            msea[j] += float((g1 - e1.to(torch.complex64)).norm()**2)
    accs /= ntr; msea /= ntr
    rows = []
    for j, bh in enumerate(bhs):
        tag = ("quant b=7" if abs(bh - beta - 2**-8) < 1e-12 else
               ("blind" if bh == 0.0 else f"delta={bh-beta:+.2f}"))
        print(f"  beta_hat={bh:.4f} ({tag}): cosine {accs[j]:.4f}, "
              f"MSE {msea[j]:.4f}")
        rows.append([bh, accs[j], msea[j]])
    write_csv("mismatch", ["beta_hat", "cosine", "mse"], rows)




# ------------------------------------------------------------------
def E9_ceiling(d=512, snr=60.0, ntr=200):
    print(chr(10) + '=== E9: cosine-ceiling verification (h=1) ===')
    sig = 10 ** (-snr / 20.0)
    rows = []
    for beta in (0.311, 0.8):
        g = 1.0 - beta**2
        acc_a = 0.0; acc_b = 0.0
        for _ in range(ntr):
            e1, e2 = embed_pair(d, beta)
            M1, M2 = haar_g(d), haar_g(d)
            Q = M1.T @ M2
            n = cnoise_g(d)
            r = (M1 @ e1 + M2 @ e2).to(torch.complex64) + sig * n
            t1 = (M1.T.to(torch.complex64) @ r)
            nv = torch.tensor([sig**2], device=DEV)
            g1 = aware_g(t1.unsqueeze(0), Q, beta, 1.0, nv)[0]
            acc_a += float(abscos(g1.unsqueeze(0), e1)[0])
            acc_b += float(abscos(t1.unsqueeze(0), e1)[0])
        acc_a /= ntr; acc_b /= ntr
        import math as _m
        pred_a = _m.sqrt(1.0 - _m.sqrt(g) / 2.0)
        pred_b = _m.sqrt(0.5)
        print(f'  beta={beta}: aware MC {acc_a:.4f} pred {pred_a:.4f} | '
              f'blind MC {acc_b:.4f} pred {pred_b:.4f}')
        rows.append([beta, acc_a, pred_a, acc_b, pred_b])
    write_csv('cosine_ceiling', ['beta', 'aware_mc', 'aware_pred',
                                 'blind_mc', 'blind_pred'], rows)


# ------------------------------------------------------------------
def E10_whpad(beta=0.311, d=768, dpad=1024, ntr=200):
    """Zero-padded WH at d=768 (padded to 1024) vs dense Haar at 768.

    The embedding (768) is zero-padded to 1024, masked by H_1024 D_u,
    and the exact per-coordinate Wiener uses the true prior (signal
    variance 1/768 on the active support, zero on the padding, so the
    padded coordinates are discarded). Reference: Haar masks at the
    native d=768 with the standard aware demultiplexer. Same per-block
    energy E_b = 1 and the same noise PSD; the padded block occupies
    dpad channel uses, a bandwidth cost of dpad/d."""
    print(chr(10) + '=== E10: zero-padded WH (768->1024) vs native Haar 768 ===')
    snr_db = np.arange(0, 41, 5)
    sigs = torch.tensor(10 ** (-snr_db / 20.0), dtype=torch.float32,
                        device=DEV)
    nb = len(snr_db)
    H = np.array([[1.0]])
    while H.shape[0] < dpad:
        H = np.block([[H, H], [H, -H]])
    Ht = torch.tensor(H / math.sqrt(dpad), dtype=torch.float32, device=DEV)
    g = 1.0 - beta**2
    res = {'haar': np.zeros(nb), 'whpad': np.zeros(nb)}
    for _ in range(ntr):
        e1, e2 = embed_pair(d, beta)
        # ---- native Haar at 768 ----
        M1, M2 = haar_g(d), haar_g(d)
        Q = M1.T @ M2
        n = cnoise_g(d)
        r = (M1 @ e1 + M2 @ e2).to(torch.complex64).unsqueeze(0) \
            + sigs.view(-1, 1) * n.unsqueeze(0)
        t1 = (M1.T.to(torch.complex64) @ r.unsqueeze(-1)).squeeze(-1)
        g1 = aware_g(t1, Q, beta, 1.0, sigs**2)
        res['haar'] += abscos(g1, e1)
        # ---- zero-padded WH at 1024 ----
        z = torch.zeros(dpad - d, device=DEV)
        e1p = torch.cat([e1, z]); e2p = torch.cat([e2, z])
        D1 = torch.tensor(np.sign(rng.standard_normal(dpad)),
                          dtype=torch.float32, device=DEV)
        D2 = torch.tensor(np.sign(rng.standard_normal(dpad)),
                          dtype=torch.float32, device=DEV)
        W1, W2 = Ht * D1.unsqueeze(0), Ht * D2.unsqueeze(0)
        npad = cnoise_g(dpad)
        rp = (W1 @ e1p + W2 @ e2p).to(torch.complex64).unsqueeze(0) \
            + sigs.view(-1, 1) * npad.unsqueeze(0)
        tw = (W1.T.to(torch.complex64) @ rp.unsqueeze(-1)).squeeze(-1)
        q = (D1 * D2)[:d]                       # active coordinates only
        a = 1.0 + beta * q                      # (d,)
        s_var = 1.0 / d                         # true signal variance
        v = g / d + (sigs**2).view(-1, 1)       # interference + noise
        gains = (s_var * a.unsqueeze(0)) / (a.unsqueeze(0)**2 * s_var + v)
        w1 = gains.to(torch.complex64) * tw[:, :d]
        res['whpad'] += abscos(w1, e1)
    for k in res:
        res[k] /= ntr
    rows = [[s, res['haar'][i], res['whpad'][i]]
            for i, s in enumerate(snr_db)]
    write_csv('wh_padding', ['snr_db', 'haar768', 'whpad1024'], rows)
    dev = res['haar'] - res['whpad']
    print(f'  cosine delta (haar - whpad): max {dev.max():.4f}, '
          f'at 20 dB {dev[list(snr_db).index(20)]:.4f}, '
          f'at 40 dB {dev[-1]:.4f}')
    print(f'  bandwidth cost: {dpad}/{d} = {dpad/d:.3f}x uses '
          f'(per-use rate factor {d/dpad:.3f})')


if __name__ == "__main__":
    todo = set(sys.argv[1:])
    ALL = {"E2": E2_sic, "E3": E3_unconditional, "E4": E4_csi,
           "E5": E5_maskfam, "E7c": E7_multiuser, "E8": E8_mismatch,
           "E9": E9_ceiling, "E10": E10_whpad}
    for name, fn in ALL.items():
        if not todo or name in todo:
            fn()
    print("\nAll requested GPU simulations complete.")
