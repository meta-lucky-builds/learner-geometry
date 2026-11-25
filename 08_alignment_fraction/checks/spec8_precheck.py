"""Pre-check of SPEC8's measurement (vector rho with the projector at the start, even part, NP on the learner's own
fibre) in the solvable model F = u^2 - v^2, theta0 = (1,1). Values quoted in SPEC8 U5 must come out."""
import numpy as np
th0 = np.array([1.0, 1.0])
F = lambda th: th[0] ** 2 - th[1] ** 2
J = lambda th: np.array([[2 * th[0], -2 * th[1]]])
J0 = J(th0); K0 = (J0 @ J0.T)[0, 0]; F0 = F(th0); N = 1
V0 = lambda w: w - J0.T @ (J0 @ w) / K0
def jump(thb, t):
    Jb = J(thb); Kb = (Jb @ Jb.T)[0, 0]; a = 0.0
    for _ in range(500):
        th = thb + Jb[0] * a; r = t - F(th)
        if abs(r) <= 1e-13: break
        a += r / Kb
    return thb + Jb[0] * a
def np_point(FL, th):
    for _ in range(300):
        Jt = J(th); Kt = (Jt @ Jt.T)[0, 0]
        new = th0 + Jt[0] * (((FL - F(th)) + Jt[0] @ (th - th0)) / Kt)
        d = new - th; th = new
        Jt = J(th); stat = np.linalg.norm((th - th0) - Jt[0] * (Jt[0] @ (th - th0)) / (Jt @ Jt.T)[0, 0]) / max(np.linalg.norm(th - th0), 1e-300)
        if abs(F(th) - FL) <= 1e-12 and stat <= 1e-10: break
    return th
def SL(m, t):
    th = th0.copy()
    for k in range(1, m + 1): th = jump(th, F0 + k / m * (t - F0))
    return th
def DJ(c, t):
    th = th0.copy()
    for _ in range(2000):
        r = F(th) - t
        if abs(r) <= 1e-9 * abs(t - F0): break
        Jt = J(th); th = th - c * Jt[0] * r / (Jt @ Jt.T)[0, 0]
    return th
def GD(c, t, mu=0.0):
    th = th0.copy(); u = np.zeros(2); eta = c * N / K0
    for _ in range(3000000):
        r = F(th) - t
        if abs(r) <= 1e-8 * abs(t - F0) and np.linalg.norm(u) <= 1e-12: break
        u = mu * u - eta * J(th)[0] * r / N; th = th + u
    return th
def PROX(g, t):
    th = th0.copy()
    for _ in range(500):
        Jt = J(th); Kt = (Jt @ Jt.T)[0, 0] + g
        th = th0 + Jt[0] * (((t - F(th)) + Jt[0] @ (th - th0)) / Kt)
    return th
def rho(learner, s=0.02, even=True):
    vL, vN = [], []
    for sg in ((1, -1) if even else (1,)):
        thL = learner(F0 + sg * s); thN = np_point(F(thL), thL.copy())
        assert np.linalg.norm(thN - th0) <= np.linalg.norm(thL - th0) * (1 + 1e-9)
        vL.append(V0(thL - th0)); vN.append(V0(thN - th0))
    vL = np.mean(vL, 0); vN = np.mean(vN, 0)
    return (vL @ vN) / (vN @ vN), (vL @ vN) / (np.linalg.norm(vL) * np.linalg.norm(vN) + 1e-300)
fhb = lambda c, mu: 1 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))
cases = [("SL(2)", lambda t: SL(2, t), 0.25), ("SL(8)", lambda t: SL(8, t), 0.4375), ("SL(32)", lambda t: SL(32, t), 0.484375),
         ("TH", lambda t: jump(th0, t), 0.0), ("DJ(0.5)", lambda t: DJ(0.5, t), 1 / 3), ("DJ(1.0)", lambda t: DJ(1.0, t), 0.0), ("DJ(1.5)", lambda t: DJ(1.5, t), -1.0),
         ("DJ(1.9)", lambda t: DJ(1.9, t), -9.0), ("GD(0.5)", lambda t: GD(0.5, t), 1 / 3), ("GD(1.5)", lambda t: GD(1.5, t), -1.0),
         ("HB(0.05,.9)", lambda t: GD(0.05, t, 0.9), fhb(0.05, 0.9)), ("HB(0.0105,.9)", lambda t: GD(0.0105, t, 0.9), fhb(0.0105, 0.9)),
         ("PROX(1e-2 K0)", lambda t: PROX(1e-2 * K0, t), 1.0), ("PROX(1e-3 K0)", lambda t: PROX(1e-3 * K0, t), 1.0)]
print("%-15s %10s %10s %8s | single-sign rho at s=0.02, 0.01" % ("learner", "rho(even)", "predicted", "cos"))
for name, fn, pred in cases:
    r, c = rho(fn)
    r1, _ = rho(fn, 0.02, False); r2, _ = rho(fn, 0.01, False)
    print("%-15s %10.4f %10.4f %8.4f | %.4f %.4f" % (name, r, pred, c, r1, r2))
