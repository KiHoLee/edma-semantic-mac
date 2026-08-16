"""
Monte Carlo verification of every closed-form claim (v2 design).
================================================================
Independent Haar masks + affinity-aware Wiener demultiplexer.
Checks (d = 256 for speed; deviations shrink as O(1/d)):

  V1  Theorem 1 MSE formula vs MC at several (beta, SNR), h = 1
  V2  Theorem 1 under random channel phases, both users
  V3  floors: aware sqrt(g)/2 vs blind 1/2, and the value ratio
  V4  cosine ceiling sqrt(1 - MSE) (Corollary: cosine)
  V5  blind receiver == matched filter in cosine (scalar shrinkage)
  V6  monotonicity of the MSE in beta (Proposition)
  V7  full-cooperation bound T <= log2(1+4 rho/d), equality at beta=1
  V8  MAC condition gamma^2 (2+k) >= 2 beta^2 k^2 boundary
  V9  Walsh-Hadamard diagonal variant: exact finite-d closed form
  V10 mismatch stationarity: MSE(beta_hat) - MSE(beta) = O(delta^2)
  V11 correlated-mask alternative floor 1 + 4 beta^4 / gamma
      (Remark and Appendix), dominated by the aware receiver

Pure numpy, fixed seed, ~2 minutes on a laptop.
"""
from __future__ import annotations
import math
import numpy as np

rng = np.random.default_rng(2026)
D = 256


def haar(d):
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))


def unit(v):
    return v / np.linalg.norm(v)


def cosim(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_pair(d, beta):
    e1 = unit(rng.standard_normal(d))
    w = rng.standard_normal(d)
    w = unit(w - (w @ e1) * e1)
    return e1, beta * e1 + math.sqrt(1 - beta * beta) * w


def cnoise(d):
    return (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / math.sqrt(2)


def mse_theory(beta, c1, rho_e):
    a0 = 1.0 + beta**2 * abs(c1)**2 + rho_e
    return rho_e / math.sqrt(a0 * a0 - 4.0 * beta**2 * abs(c1)**2)


def aware(t1, Q, beta, c1, nvar, d):
    g = 1.0 - beta * beta
    rho = g * abs(c1)**2 / d + nvar
    S = beta * (c1 * Q + np.conj(c1) * Q.T) / d
    S[np.diag_indices(d)] += (1.0 + beta**2 * abs(c1)**2) / d + rho
    x = np.linalg.solve(S, t1)
    return (x + beta * np.conj(c1) * (Q.T @ x)) / d


def run_pair(beta, sig, h1=1.0 + 0j, h2=1.0 + 0j, d=D):
    e1, e2 = embed_pair(d, beta)
    M1, M2 = haar(d), haar(d)
    Q = M1.T @ M2
    r = h1 * (M1 @ e1) + h2 * (M2 @ e2) + sig * cnoise(d)
    t1 = M1.T @ r / h1
    return e1, e2, Q, t1, M2.T @ r / h2


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


allok = True

# ---------------------------------------------------------------- V1
devs = []
for beta in (0.0, 0.311, 0.6):
    for snr in (10.0, 20.0, 60.0):
        sig = 10 ** (-snr / 20.0)
        mc = 0.0
        NT = 40
        for _ in range(NT):
            e1, _, Q, t1, _ = run_pair(beta, sig)
            g1 = aware(t1, Q, beta, 1.0, sig * sig, D)
            mc += float(np.linalg.norm(g1 - e1) ** 2)
        mc /= NT
        th = mse_theory(beta, 1.0, (1 - beta**2) + D * sig * sig)
        devs.append(abs(mc / th - 1))
allok &= check("V1 Theorem 1 (h=1)", max(devs) < 0.03,
               f"max dev {100*max(devs):.2f}%")

# ---------------------------------------------------------------- V2
devs = []
sig = 10 ** (-20.0 / 20.0)
for beta in (0.311, 0.5):
    for _ in range(30):
        h1 = np.exp(1j * rng.uniform(0, 2 * np.pi))
        h2 = np.exp(1j * rng.uniform(0, 2 * np.pi))
        e1, e2, Q, t1, t2 = run_pair(beta, sig, h1, h2)
        c1, c2 = h2 / h1, h1 / h2
        g1 = aware(t1, Q, beta, c1, sig**2, D)
        g2 = aware(t2, Q.T, beta, c2, sig**2, D)
        g = 1 - beta**2
        th1 = mse_theory(beta, c1, g * abs(c1)**2 + D * sig**2)
        th2 = mse_theory(beta, c2, g * abs(c2)**2 + D * sig**2)
        devs.append(abs(np.linalg.norm(g1 - e1)**2 / th1 - 1))
        devs.append(abs(np.linalg.norm(g2 - e2)**2 / th2 - 1))
allok &= check("V2 Theorem 1 (random phases, both users)",
               float(np.mean(devs)) < 0.05,
               f"mean dev {100*float(np.mean(devs)):.2f}%")

# ---------------------------------------------------------------- V3
beta = 0.6
sig = 1e-3
mc_a, mc_b = 0.0, 0.0
NT = 40
for _ in range(NT):
    e1, _, Q, t1, _ = run_pair(beta, sig)
    g1 = aware(t1, Q, beta, 1.0, sig * sig, D)
    mc_a += float(np.linalg.norm(g1 - e1) ** 2)
    lam = (1.0 / D) / (2.0 / D + sig * sig)
    mc_b += float(np.linalg.norm(lam * t1 - e1) ** 2)
mc_a /= NT
mc_b /= NT
fa, fb = math.sqrt(1 - beta**2) / 2, 0.5
allok &= check("V3 floors sqrt(g)/2 vs 1/2",
               abs(mc_a - fa) < 0.02 and abs(mc_b - fb) < 0.02,
               f"aware {mc_a:.4f}~{fa:.4f}, blind {mc_b:.4f}~{fb:.4f}, "
               f"ratio {mc_b/mc_a:.3f}~{1/math.sqrt(1-beta**2):.3f}")

# ---------------------------------------------------------------- V4
acc = 0.0
for _ in range(NT):
    e1, _, Q, t1, _ = run_pair(beta, sig)
    acc += cosim(aware(t1, Q, beta, 1.0, sig * sig, D), e1)
acc /= NT
pred = math.sqrt(1 - fa)
allok &= check("V4 cosine ceiling sqrt(1-MSE)", abs(acc - pred) < 0.01,
               f"MC {acc:.4f} vs {pred:.4f}")

# ---------------------------------------------------------------- V5
e1, _, Q, t1, _ = run_pair(0.311, 0.1)
lam = 0.37  # any scalar
allok &= check("V5 blind == MF in cosine",
               abs(cosim(lam * t1, e1) - cosim(t1, e1)) < 1e-12)

# ---------------------------------------------------------------- V6
k = D / 100.0
vals = [mse_theory(b, 1.0, (1 - b * b) + k)
        for b in np.linspace(0, 0.99, 50)]
allok &= check("V6 monotonic decrease in beta",
               all(x > y for x, y in zip(vals, vals[1:])))

# ---------------------------------------------------------------- V7
ok7 = True
worst = 0.0
for rho in (1.0, 100.0, 1e4):
    kk = D / rho
    coop = math.log2(1 + 4 * rho / D)
    for b in np.linspace(0, 1.0, 41):
        m = mse_theory(b, 1.0, (1 - b * b) + kk)
        T = 2 * math.log2(1 / m)
        ok7 &= T <= coop + 1e-9
        worst = max(worst, T - coop)
    m1 = mse_theory(1.0, 1.0, kk)
    ok7 &= abs(2 * math.log2(1 / m1) - coop) < 1e-9
allok &= check("V7 full-cooperation bound, equality at beta=1", ok7,
               f"max T-coop {worst:.2e}")

# ---------------------------------------------------------------- V8
ok8 = True
for rho in (1.0, 10.0, 100.0, 1e3):
    kk = D / rho
    for b in (0.1, 0.311, 0.6, 0.9):
        g = 1 - b * b
        m = mse_theory(b, 1.0, g + kk)
        T = 2 * math.log2(1 / m)
        mac = math.log2(1 + 2 * rho / D)
        lhs = g * g * (2 + kk)
        rhs = 2 * b * b * kk * kk
        ok8 &= (T <= mac + 1e-9) == (lhs >= rhs - 1e-9)
allok &= check("V8 MAC-condition boundary", ok8)

# ---------------------------------------------------------------- V9
beta = 0.311
g = 1 - beta**2
sig = 10 ** (-20.0 / 20.0)
H = np.array([[1.0]])
while H.shape[0] < D:
    H = np.block([[H, H], [H, -H]])
H /= math.sqrt(D)
mc, th = 0.0, 0.0
for _ in range(30):
    e1, e2 = embed_pair(D, beta)
    D1 = np.sign(rng.standard_normal(D))
    D2 = np.sign(rng.standard_normal(D))
    W1, W2 = H * D1[None, :], H * D2[None, :]
    r = W1 @ e1 + W2 @ e2 + sig * cnoise(D)
    t1 = W1.T @ r
    q = D1 * D2
    a = 1.0 + beta * q
    rho = g / D + sig * sig
    w1 = (a / (a * a / D + rho)) * t1 / D
    mc += float(np.linalg.norm(w1 - e1) ** 2)
    th += float(np.mean((g + D * sig**2) / (a * a + g + D * sig**2)))
allok &= check("V9 WH exact finite-d closed form",
               abs(mc / th - 1) < 0.03, f"dev {100*abs(mc/th-1):.2f}%")

# ---------------------------------------------------------------- V10
beta = 0.3
sig = 10 ** (-20.0 / 20.0)
base, d1, d2 = 0.0, 0.0, 0.0
for _ in range(30):
    e1, _, Q, t1, _ = run_pair(beta, sig)
    for bh, tag in ((beta, "b"), (beta + 0.2, "1"), (beta + 0.4, "2")):
        g1 = aware(t1, Q, bh, 1.0, sig * sig, D)
        m = float(np.linalg.norm(g1 - e1) ** 2)
        if tag == "b":
            base += m
        elif tag == "1":
            d1 += m
        else:
            d2 += m
base /= 30; d1 /= 30; d2 /= 30
r_quad = (d2 - base) / max(d1 - base, 1e-12)
allok &= check("V10 quadratic mismatch (delta doubling ~ 4x)",
               2.5 < r_quad < 6.5,
               f"MSE(+0)={base:.4f} MSE(+0.2)={d1:.4f} "
               f"MSE(+0.4)={d2:.4f} ratio {r_quad:.2f}")

# ---------------------------------------------------------------- V11
beta = 0.311
g = 1 - beta**2
mc = 0.0
for _ in range(30):
    e1, e2 = embed_pair(D, beta)
    U1, U2 = haar(D), haar(D)
    M1, M2 = U1, beta * U1 + math.sqrt(g) * U2
    r = M1 @ e1 + M2 @ e2          # noise-free -> floor
    t1 = M1.T @ r
    t2 = M2.T @ r
    g1 = (t1 - beta * t2) / g
    mc += float(np.linalg.norm(g1 - e1) ** 2)
mc /= 30
th = 1 + 4 * beta**4 / g
allok &= check("V11 correlated-mask floor 1+4b^4/g",
               abs(mc / th - 1) < 0.05,
               f"MC {mc:.4f} vs {th:.4f}; aware floor "
               f"{math.sqrt(g)/2:.4f} (dominated)")

print("\nALL CHECKS PASSED" if allok else "\nSOME CHECKS FAILED")
