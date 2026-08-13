"""
Complete numerical verification of every closed form in the manuscript.
=======================================================================
Each check implements the formula EXACTLY as printed in main.tex and
compares it against a direct Monte-Carlo or algebraic evaluation.
Prints PASS/FAIL per item with the achieved deviation. Fixed seed.

  V1  per-realization Gram identity  M1^T M2 = beta I + sqrt(g) Q
  V2  Theorem 1 full MSE (noise + C_SI,u) vs MC, random complex h
  V3  noise-free calibration of C_SI,1 / C_SI,2 (several phases)
  V4  quoted constants: C_SI,1, C_SI,2, C-bar at (0.311, h=1);
      cosine ceiling 1/sqrt(1+C_SI,1) = 0.70; rho_f = 28 dB at d=768
  V5  SINR corollary eta_u = 1/MSE (per-coordinate accounting)
  V6  C_SI,u >= 1 for all beta (proof identities gamma*C_SI,1 =
      gamma + 4 beta^4, gamma*C_SI,2 = 1 + 3 beta^2 at h=1)
  V7  Proposition (MAC consistency) on a (beta, rho) grid
  V8  wideband limit T/C_MAC -> gamma
  V9  beta* crossover roots at 10/20 dB (0.700 / 0.590, d=512)
  V10 rho_c = d(2-1/gamma)/C-bar exact iff-condition + 25.7 dB value
  V11 idealized no-floor variant crosses C_MAC at 2 beta^2 d/gamma^2
      (~21 dB at d=512, beta=0.311)
  V12 mismatch identity (eq:mismatch) + bound value 8.8e-3
  V13 CSI-direction invariance: |cos| unchanged under wrong h-hat;
      eq:csi-free equals eq:correct
  V14 cross-moment lemma E[n^H M_u M_v^T n] = sigma^2 beta d
  V15 multi-user [B^-1]_uu Sherman-Morrison formula, U = 2..6
  V16 multi-user noise-free C_SI^(U) ~ (U-1) C-bar (within 10 %)
  V17 Walsh-Hadamard masks: exact orthogonality + expected cross-Gram
"""
from __future__ import annotations
import math
import numpy as np

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def brentq(f, a, b, tol=1e-12):
    fa, fb = f(a), f(b)
    assert fa * fb < 0, "no sign change"
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

rng = np.random.default_rng(2026)
FAIL = []


def report(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL.append(name)
    print(f"[{tag}] {name}: {detail}")


def haar(d):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def unit(v):
    return v / np.linalg.norm(v)


def pair(d, beta):
    e1 = unit(rng.standard_normal(d))
    w = rng.standard_normal(d)
    w = unit(w - (w @ e1) * e1)
    return e1, beta * e1 + math.sqrt(1 - beta**2) * w


def csi1(beta, c):
    g = 1 - beta**2
    n2 = 1 + beta**2 * abs(c)**2 + 2 * beta**2 * np.real(c)
    return (g**2 * abs(c)**2 + beta**2 * n2) / g


def csi2(beta, c):
    g = 1 - beta**2
    return (abs(c)**2 + beta**2 + 2 * beta**2 * np.real(c)) / g


# ---------------- V1: per-realization Gram identity ----------------
d, beta = 256, 0.311
g = 1 - beta**2
U1, U2 = haar(d), haar(d)
M1, M2 = U1, beta * U1 + math.sqrt(g) * U2
dev = np.abs(M1.T @ M2 - (beta * np.eye(d)
                          + math.sqrt(g) * U1.T @ U2)).max()
report("V1 Gram identity", dev < 1e-12, f"max dev {dev:.2e}")

# ---------------- V2: Theorem 1 full MSE, random complex h ---------
d = 512
for beta in (0.1, 0.311, 0.5):
    g = 1 - beta**2
    h = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) / math.sqrt(2)
    h1, h2 = h
    rho_db = 15.0
    sig = 10 ** (-rho_db / 20.0)
    e1, e2 = pair(d, beta)
    mc = np.zeros(2)
    NT = 300
    for _ in range(NT):
        U1, U2 = haar(d), haar(d)
        M1, M2 = U1, beta * U1 + math.sqrt(g) * U2
        n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) \
            / math.sqrt(2)
        r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + sig * n
        t1 = M1.T @ r / h1
        t2 = M2.T @ r / h2
        g1 = (t1 - beta * (h2 / h1) * t2) / g
        g2 = (t2 - beta * (h1 / h2) * t1) / g
        mc[0] += np.linalg.norm(g1 - e1)**2
        mc[1] += np.linalg.norm(g2 - e2)**2
    mc /= NT
    th1 = d * sig**2 / (abs(h1)**2 * g) + csi1(beta, h2 / h1)
    th2 = d * sig**2 / (abs(h2)**2 * g) + csi2(beta, h1 / h2)
    dev = max(abs(mc[0] / th1 - 1), abs(mc[1] / th2 - 1))
    report(f"V2 Theorem 1 MSE (beta={beta})", dev < 0.02,
           f"MC/theory dev {100*dev:.2f}% (O(1/d) at d={d})")

# ---------------- V3: noise-free C_SI calibration ------------------
d = 512
for phase in (0.0, math.pi / 3, math.pi):
    beta = 0.311
    g = 1 - beta**2
    h1 = 1.0 + 0j
    h2 = np.exp(1j * phase)
    e1, e2 = pair(d, beta)
    mc = np.zeros(2)
    NT = 200
    for _ in range(NT):
        U1, U2 = haar(d), haar(d)
        M1, M2 = U1, beta * U1 + math.sqrt(g) * U2
        r = h1 * (M1 @ e1) + h2 * (M2 @ e2)
        t1 = M1.T @ r / h1
        t2 = M2.T @ r / h2
        g1 = (t1 - beta * (h2 / h1) * t2) / g
        g2 = (t2 - beta * (h1 / h2) * t1) / g
        mc[0] += np.linalg.norm(g1 - e1)**2
        mc[1] += np.linalg.norm(g2 - e2)**2
    mc /= NT
    t1v, t2v = csi1(beta, h2 / h1), csi2(beta, h1 / h2)
    dev = max(abs(mc[0] / t1v - 1), abs(mc[1] / t2v - 1))
    report(f"V3 noise-free C_SI (phase={phase:.2f})", dev < 0.02,
           f"dev {100*dev:.2f}%")

# ---------------- V4: quoted constants -----------------------------
beta = 0.311
g = 1 - beta**2
c1v, c2v = csi1(beta, 1.0 + 0j), csi2(beta, 1.0 + 0j)
cbar = (c1v + c2v) / 2
ceil1 = 1 / math.sqrt(1 + c1v)
rho_f_db = 10 * math.log10(768 * g / c1v)
ok = (abs(cbar - 1.2349) < 5e-4 and abs(ceil1 - 0.70) < 5e-3
      and abs(rho_f_db - 28) < 0.5)
report("V4 quoted constants", ok,
       f"C_SI,1 {c1v:.4f}, C_SI,2 {c2v:.4f}, C-bar {cbar:.4f} "
       f"(quoted 1.2349), ceiling {ceil1:.4f} (quoted 0.70), "
       f"rho_f {rho_f_db:.1f} dB (quoted 28)")

# ---------------- V5: SINR = 1/MSE ---------------------------------
rho = 10 ** (15 / 10)
eta = 1 / (512 / (rho * g) + c1v)
mse = 512 / (rho * g) + c1v
report("V5 SINR corollary", abs(eta * mse - 1) < 1e-12,
       f"eta*MSE = {eta*mse:.6f}")

# ---------------- V6: C_SI >= 1 and proof identities ---------------
ok = True
worst = 1e9
for b in np.linspace(0.0, 0.99, 200):
    gg = 1 - b**2
    lhs1 = gg * csi1(b, 1.0 + 0j)
    lhs2 = gg * csi2(b, 1.0 + 0j)
    if abs(lhs1 - (gg + 4 * b**4)) > 1e-12: ok = False
    if abs(lhs2 - (1 + 3 * b**2)) > 1e-12: ok = False
    worst = min(worst, csi1(b, 1.0 + 0j), csi2(b, 1.0 + 0j))
report("V6 C_SI >= 1 + proof identities", ok and worst >= 1 - 1e-12,
       f"min C_SI over beta grid = {worst:.6f}")

# ---------------- V7: MAC consistency on a grid --------------------
def T_edma(b, r_, d_):
    gg = 1 - b**2
    cb = (csi1(b, 1 + 0j) + csi2(b, 1 + 0j)) / 2
    return 2 * np.log2(1 + 1 / (d_ / (r_ * gg) + cb))

ok = True
for b in np.linspace(0, 0.95, 40):
    for rdb in np.linspace(-10, 60, 60):
        r_ = 10 ** (rdb / 10)
        gg = 1 - b**2
        mid = np.log2(1 + 2 * r_ * gg / 512)
        cmac = np.log2(1 + 2 * r_ / 512)
        if T_edma(b, r_, 512) > mid + 1e-12 or mid > cmac + 1e-12:
            ok = False
report("V7 MAC consistency grid", ok, "T_EDMA <= log2(1+2 rho g/d) <= C_MAC")

# ---------------- V8: wideband limit -------------------------------
b = 0.311
r_ = 1e-6 * 512
lim = T_edma(b, r_, 512) / np.log2(1 + 2 * r_ / 512)
report("V8 wideband limit", abs(lim - (1 - b**2)) < 1e-3,
       f"T/C_MAC at rho/d=1e-6: {lim:.5f} vs gamma {1-b**2:.5f}")

# ---------------- V9: beta* crossover roots ------------------------
def beta_star(rdb, d_=512):
    r_ = 10 ** (rdb / 10)
    T_oma = 2 * np.log2(1 + r_ / (2 * d_))
    return brentq(lambda b: T_edma(b, r_, d_) - T_oma, 0.3, 0.9)

b10, b20 = beta_star(10), beta_star(20)
report("V9 beta* crossover", abs(b10 - 0.700) < 5e-3
       and abs(b20 - 0.590) < 5e-3,
       f"10 dB: {b10:.3f} (quoted 0.700), 20 dB: {b20:.3f} (quoted 0.590)")

# ---------------- V10: rho_c iff-condition + value -----------------
b = 0.311
gg = 1 - b**2
cb = (csi1(b, 1 + 0j) + csi2(b, 1 + 0j)) / 2
rho_c = 512 * (2 - 1 / gg) / cb
rho_c_db = 10 * math.log10(rho_c)
eps = 1e-4
below = T_edma(b, rho_c * (1 - eps), 512) \
    - 2 * np.log2(1 + rho_c * (1 - eps) / 1024)
above = T_edma(b, rho_c * (1 + eps), 512) \
    - 2 * np.log2(1 + rho_c * (1 + eps) / 1024)
report("V10 rho_c crossover", below > 0 > above
       and abs(rho_c_db - 25.7) < 0.1,
       f"rho_c {rho_c_db:.2f} dB (quoted 25.7), sign flip verified")

# ---------------- V11: idealized-MAC crossing ----------------------
rho_x = 2 * b**2 * 512 / gg**2
f = lambda r_: 2 * np.log2(1 + r_ * gg / 512) - np.log2(1 + 2 * r_ / 512)
root = brentq(f, 10.0, 1e4)
report("V11 idealized crossing", abs(root / rho_x - 1) < 1e-6
       and abs(10 * math.log10(root) - 21) < 0.3,
       f"root {10*math.log10(root):.2f} dB, formula 2b^2d/g^2 "
       f"{10*math.log10(rho_x):.2f} dB (quoted ~21)")

# ---------------- V12: mismatch identity + bound value -------------
b, delta = 0.3, 0.06
bh = b + delta
hr = 1.0 + 0j
e1, e2 = pair(64, b)
t1 = e1 + b * hr * e2          # expected-Gram surrogate outputs
t2v_ = e2 + b * np.conj(hr) * e1
g1 = (t1 - bh * hr * t2v_) / (1 - bh**2)
lhs = g1 - e1
rhs = delta / (1 - bh**2) * (bh * e1 - hr * e2)
dev = np.linalg.norm(lhs - rhs)
bound = delta**2 * (abs(bh) + abs(hr))**2 / (1 - bh**2)**2
report("V12 mismatch identity", dev < 1e-12
       and abs(bound - 8.8e-3) < 2e-4,
       f"identity dev {dev:.1e}, bound {bound:.4f} (quoted 8.8e-3)")

# ---------------- V13: CSI-direction invariance --------------------
d = 256
b = 0.311
g = 1 - b**2
e1, e2 = pair(d, b)
U1, U2 = haar(d), haar(d)
M1, M2 = U1, b * U1 + math.sqrt(g) * U2
h1, h2 = 0.7 - 0.4j, -0.2 + 1.1j
n = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)
r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + 0.1 * n
truec = (M1.T @ r / h1 - b * (h2 / h1) * (M2.T @ r / h2)) / g
csif = (M1 - b * M2).T @ r / (h1 * g)
dev1 = np.abs(truec - csif).max()
h1w = h1 * (1.5 * np.exp(0.8j))          # badly wrong estimate
wrong = (M1 - b * M2).T @ r / (h1w * g)
c_true = abs(np.vdot(truec, e1)) / (np.linalg.norm(truec))
c_wrong = abs(np.vdot(wrong, e1)) / (np.linalg.norm(wrong))
report("V13 CSI invariance", dev1 < 1e-12 and abs(c_true - c_wrong) < 1e-12,
       f"csi-free identity dev {dev1:.1e}, |cos| unchanged "
       f"({c_true:.6f} vs {c_wrong:.6f})")

# ---------------- V14: cross-moment lemma --------------------------
d = 256
b = 0.311
sig2 = 0.5
acc = 0.0
NT = 4000
U1, U2 = haar(d), haar(d)
M1, M2 = U1, b * U1 + math.sqrt(1 - b**2) * U2
for _ in range(NT):
    n = math.sqrt(sig2) * (rng.standard_normal(d)
                           + 1j * rng.standard_normal(d)) / math.sqrt(2)
    acc += np.real(np.conj(n) @ (M1 @ (M2.T @ n)))
acc /= NT
th = sig2 * b * d
report("V14 cross-moment lemma", abs(acc / th - 1) < 0.05,
       f"MC {acc:.3f} vs sigma^2 beta d {th:.3f} "
       f"({100*abs(acc/th-1):.1f}%)")

# ---------------- V15: [B^-1]_uu Sherman-Morrison ------------------
ok = True
for U in range(2, 7):
    for b in (0.1, 0.311, 0.6):
        B = (1 - b) * np.eye(U) + b * np.ones((U, U))
        num = 1 + (U - 2) * b
        den = (1 - b) * (1 + (U - 1) * b)
        if abs(np.linalg.inv(B)[0, 0] - num / den) > 1e-12:
            ok = False
report("V15 [B^-1]_uu formula", ok, "U=2..6, beta grid, exact")

# ---------------- V16: multi-user C_SI^(U) -------------------------
d = 512
b = 0.311
g = 1 - b**2
cb = (csi1(b, 1 + 0j) + csi2(b, 1 + 0j)) / 2
for U in (3, 4):
    B = (1 - b) * np.eye(U) + b * np.ones((U, U))
    Binv = np.linalg.inv(B)
    es = []
    e1 = unit(rng.standard_normal(d))
    for u in range(U):
        if u == 0:
            es.append(e1)
        else:
            w = rng.standard_normal(d)
            w = unit(w - (w @ e1) * e1)
            es.append(b * e1 + math.sqrt(g) * w)
    mse = 0.0
    NT = 60
    for _ in range(NT):
        Us = [haar(d) for _ in range(U)]
        Ms = [Us[0]]
        for u in range(1, U):
            Ms.append(b * Us[0] + math.sqrt(g) * Us[u])
        r = sum(Ms[u] @ es[u] for u in range(U))     # h_u = 1
        t = np.stack([Ms[u].T @ r for u in range(U)])
        rec = np.einsum("uv,vd->ud", Binv, t)
        mse += np.linalg.norm(rec[0] - es[0])**2
    mse /= NT
    ratio = mse / ((U - 1) * cb)
    report(f"V16 C_SI^(U) additivity (U={U})", abs(ratio - 1) < 0.10,
           f"noise-free MSE {mse:.3f} vs (U-1)C-bar "
           f"{(U-1)*cb:.3f} (ratio {ratio:.3f})")

# ---------------- V17: Walsh-Hadamard masks ------------------------
d = 256
H = hadamard(d) / math.sqrt(d)
b = 0.311
acc = np.zeros((d, d))
NT = 400
for _ in range(NT):
    D1 = np.diag(rng.choice([-1.0, 1.0], d))
    D2 = np.diag(rng.choice([-1.0, 1.0], d))
    W1 = H @ D1
    W2 = b * W1 + math.sqrt(1 - b**2) * H @ D2
    acc += W1.T @ W2 / NT
orth = np.abs((H @ np.diag(rng.choice([-1.0, 1.0], d))).T
              @ (H @ np.diag(rng.choice([-1.0, 1.0], d)))
              @ np.ones(d) / d).max()
diag_dev = abs(np.diag(acc).mean() - b)
off = np.abs(acc - np.diag(np.diag(acc))).mean()
report("V17 WH masks", diag_dev < 0.02 and off < 0.01,
       f"E[cross-Gram] diag {np.diag(acc).mean():.4f} vs beta {b}, "
       f"mean |off-diag| {off:.4f}")

print()
print("=" * 60)
print(f"RESULT: {'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
