"""T1: diagonal network beta = u^2 - v^2 (predictions in FINDINGS.md sec. 40). numpy only."""
import numpy as np, sys
np.set_printoptions(linewidth=200, precision=4, suppress=True)
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
RICH = len(sys.argv) > 1
def uv_of_s(s, f, a=ALPHA):
    return a * (1 + 2 * (1 - f) * s) / (1 - 2 * f * s), a * (1 - 2 * (1 - f) * s) / (1 + 2 * f * s)
def beta_of_s(s, f, a=ALPHA):
    u, v = uv_of_s(s, f, a); return u * u - v * v
def dbeta_ds(s, f, a=ALPHA):
    u, v = uv_of_s(s, f, a)
    du = a * 2.0 / (1 - 2 * f * s) ** 2; dv = -a * 2.0 / (1 + 2 * f * s) ** 2
    return 2 * u * du - 2 * v * dv
# ---------------- D0: series check
print("D0  series check (alpha = 1): coefficient of s^3 in beta, and u v")
for f in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
    s = 1e-2; s2 = 2e-2
    c3 = ((beta_of_s(s2, f) - 8 * s2) / s2 ** 3 * 4 - (beta_of_s(s, f) - 8 * s) / s ** 3) / 3      # Richardson (error term is s^5)
    u, v = uv_of_s(0.05, f); uvp = (1 - 4 * (1 - f) ** 2 * 0.05 ** 2) / (1 - 4 * f * f * 0.05 ** 2)
    print("   f %+.1f  s^3 coefficient %9.3f  predicted 32 f(1+f) = %7.2f | u v at s=.05: %.6f predicted %.6f" % (f, c3, 32 * f * (1 + f), u * v, uvp))
# ---------------- alpha-point learner: solve X beta(X^T nu) = y
def alpha_point(X, y, f, iters=200):
    n = X.shape[0]; nu = np.zeros(n)
    for it in range(iters):
        s = X.T @ nu
        if f != 0 and np.max(np.abs(2 * f * s)) >= 0.999: return None
        r = X @ beta_of_s(s, f) - y
        if np.linalg.norm(r) < 1e-12 * max(1.0, np.linalg.norm(y)): return beta_of_s(s, f), s
        H = (X * dbeta_ds(s, f)) @ X.T
        step = np.linalg.solve(H, r); lam = 1.0
        while lam > 1e-6:
            s2 = X.T @ (nu - lam * step)
            if (f == 0 or np.max(np.abs(2 * f * s2)) < 0.999) and np.linalg.norm(X @ beta_of_s(s2, f) - y) < np.linalg.norm(r): break
            lam /= 2
        nu = nu - lam * step
    return None
def truth(kind, d, rng):
    b = np.zeros(d)
    if kind == "sparse": b[rng.choice(d, 8, replace=False)] = rng.choice([-1.0, 1.0], 8)
    elif kind == "dense": b = rng.standard_normal(d); b *= np.sqrt(8.0) / np.linalg.norm(b)
    else:
        b[rng.choice(d, 8, replace=False)] = rng.choice([-1.0, 1.0], 8); b *= np.sqrt(0.7)
        g = rng.standard_normal(d); b = b + g * np.sqrt(0.3 * 8.0) / np.linalg.norm(g)
    return b
FS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5] if RICH else [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
d = 400; SEEDS = 20
for kind, noise in ([("sparse", 0.0), ("sparse", 0.3), ("mixed", 0.3), ("dense", 0.0)] if RICH else [("sparse", 0.0), ("dense", 0.0), ("mixed", 0.1)]):
    print("\nD1-D3  truth = %s, noise = %g  (relative recovery error |beta - beta*|^2 / |beta*|^2, mean over %d seeds; '-' = no solution)" % (kind, noise, SEEDS))
    for n in [30, 60, 120]:
        E = np.full((SEEDS, len(FS)), np.nan); smax = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * n + seed); X = rng.standard_normal((n, d)) / np.sqrt(n); bs = truth(kind, d, rng)
            y = X @ bs; y = y + noise * np.linalg.norm(y) / np.sqrt(n) * rng.standard_normal(n)
            for j, f in enumerate(FS):
                out = alpha_point(X, y, f)
                if out is not None:
                    E[seed, j] = np.sum((out[0] - bs) ** 2) / np.sum(bs ** 2)
                    if f == 0: smax.append(np.max(np.abs(out[1])))
        ok = ~np.isnan(E).any(1); Eo = E[ok]
        mono_up = np.sum(np.all(np.diff(Eo[:, 2:], axis=1) < 0, axis=1)) if len(Eo) else 0; mono_dn = np.sum(np.all(np.diff(Eo[:, 2:], axis=1) > 0, axis=1)) if len(Eo) else 0
        best = np.array([FS[j] for j in np.argmin(Eo, 1)]) if len(Eo) else np.array([])
        print("   n %-4d mean error by f %s : %s | seeds solved %d | error falls f=0..1.5 in %d, rises in %d | worst f = -0.5 in %d | |E(-1)/E(0) - 1| median %.3f | best f: %s | max|s| at f=0 %.2f" % (
            n, FS, " ".join("%.4f" % x for x in np.nanmean(E, 0)), ok.sum(), mono_up, mono_dn, np.sum(np.argmax(Eo, 1) == 1) if len(Eo) else 0,
            np.median(np.abs(Eo[:, 0] / Eo[:, 2] - 1)) if len(Eo) else float("nan"),
            " ".join("%g:%d" % (f, np.sum(best == f)) for f in FS), np.mean(smax) if smax else float("nan")))
# ---------------- D4: iterated learners built from Gauss-Newton jumps, theta = (u, v)
def F(th, X, d): return X @ (th[:d] ** 2 - th[d:] ** 2)
def Jm(th, X, d): return np.hstack([2 * X * th[:d], -2 * X * th[d:]])
def gn_to(th, X, d, t, c=1.0, tol=1e-13, iters=4000):
    for _ in range(iters):
        r = t - F(th, X, d)
        if np.linalg.norm(r) < tol * max(1.0, np.linalg.norm(t)): return th
        J = Jm(th, X, d); th = th + J.T @ np.linalg.solve(J @ J.T, c * r)
    return th
def learner(name, X, d, t):
    th = np.full(2 * d, ALPHA)
    if name.startswith("DJ"): return gn_to(th, X, d, t, c=float(name[3:-1]))
    m = int(name[3:-1])
    for k in range(1, m + 1): th = gn_to(th, X, d, t * k / m)
    return th
print("\nD4  iterated learners (DJ = damped Gauss-Newton jumps, SL(m) = m sub-targets), sparse truth, n = 60, d = 400; targets scaled by tau")
NAMES = ["DJ(1.5)", "DJ(1)", "SL(2)", "SL(4)", "SL(8)"]
for seed in range(4):
    rng = np.random.default_rng(77 + seed); n = 60; X = rng.standard_normal((n, d)) / np.sqrt(n); bs = truth("sparse", d, rng); y = X @ bs
    prev = None
    for tau in [0.1, 0.3, 1.0]:
        res = {nm: learner(nm, X, d, tau * y) for nm in NAMES}
        b = {nm: res[nm][:d] ** 2 - res[nm][d:] ** 2 for nm in NAMES}; uvm = {nm: np.mean(res[nm][:d] * res[nm][d:]) - ALPHA ** 2 for nm in NAMES}
        D = {nm: np.linalg.norm(b[nm] - b["DJ(1)"]) for nm in NAMES}
        err = {nm: np.sum((b[nm] - tau * bs) ** 2) / np.sum((tau * bs) ** 2) for nm in NAMES}
        expo = "" if prev is None else " | exponent of |beta(SL8) - beta(DJ1)|: %.2f" % (np.log(D["SL(8)"] / prev[1]) / np.log(tau / prev[0]))
        print("   seed %d tau %-4g error: %s | |dbeta| vs DJ(1): %s | mean(uv) - alpha^2: %s%s" % (
            seed, tau, " ".join("%s %.5f" % (nm, err[nm]) for nm in NAMES), " ".join("%.2e" % D[nm] for nm in NAMES),
            " ".join("%+.2e" % uvm[nm] for nm in NAMES), expo))
        prev = (tau, D["SL(8)"])
