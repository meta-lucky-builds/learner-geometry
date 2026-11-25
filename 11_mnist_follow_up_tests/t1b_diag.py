"""Two-phase check in beta = u^2 - v^2: does a large-step first half change what the second half learns?  alpha = 0.3, sparse truth."""
import numpy as np
A = 0.3; d = 400
def F(th, X): return X @ (th[:d] ** 2 - th[d:] ** 2)
def Jm(th, X): return np.hstack([2 * X * th[:d], -2 * X * th[d:]])
def gn_to(th, X, t, c=1.0):
    for _ in range(6000):
        r = t - F(th, X)
        if np.linalg.norm(r) < 1e-12 * max(1.0, np.linalg.norm(t)): return th
        J = Jm(th, X); th = th + J.T @ np.linalg.solve(J @ J.T, c * r)
    return th
def smooth(th, X, t0, t1, m=16):
    for k in range(1, m + 1): th = gn_to(th, X, t0 + (t1 - t0) * k / m)
    return th
print("error |beta - beta*|^2/|beta*|^2;  first half of the move by learner in column, second half always smooth (16 sub-targets)")
print("seed  n   all-smooth   DJ(1)-then-smooth   DJ(1.5)-then-smooth   DJ(1.9)-then-smooth | mean(uv)/alpha^2 after first half: smooth DJ(1) DJ(1.5) DJ(1.9)")
wins = np.zeros(3)
for n in (60, 120):
    for seed in range(6):
        rng = np.random.default_rng(500 + 10 * n + seed); X = rng.standard_normal((n, d)) / np.sqrt(n)
        bs = np.zeros(d); bs[rng.choice(d, 8, replace=False)] = rng.choice([-1.0, 1.0], 8); y = X @ bs; th0 = np.full(2 * d, A)
        res = {}; uvs = {}
        for nm, c in [("smooth", None), ("DJ(1)", 1.0), ("DJ(1.5)", 1.5), ("DJ(1.9)", 1.9)]:
            h = smooth(th0, X, 0 * y, 0.5 * y) if c is None else gn_to(th0, X, 0.5 * y, c)
            uvs[nm] = np.mean(h[:d] * h[d:]) / A ** 2; th = smooth(h, X, 0.5 * y, y); b = th[:d] ** 2 - th[d:] ** 2
            res[nm] = np.sum((b - bs) ** 2) / np.sum(bs ** 2)
        wins += np.array([res["DJ(1)"] > res["smooth"], res["DJ(1.5)"] > res["smooth"], res["DJ(1.9)"] > res["smooth"]])
        print("%3d %4d   %.5f      %.5f            %.5f              %.5f        | %.5f %.5f %.5f %.5f" % (seed, n, res["smooth"], res["DJ(1)"], res["DJ(1.5)"], res["DJ(1.9)"], uvs["smooth"], uvs["DJ(1)"], uvs["DJ(1.5)"], uvs["DJ(1.9)"]))
print("large-step first half ends WORSE than all-smooth in: DJ(1) %d, DJ(1.5) %d, DJ(1.9) %d of 12" % tuple(wins))
