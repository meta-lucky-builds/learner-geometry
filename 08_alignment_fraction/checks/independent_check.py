"""Independent re-implementation of the SPEC8 measurement (an independent code, torch.autograd Jacobian, numpy linear algebra)
compared with the delivered library on the SAME network and start."""
import sys, math
sys.path.insert(0, ".")
exec(open("config_cell.py", encoding="utf-8").read())
import part8lib as L
L.configure(CFG)
import torch, numpy as np
torch.set_default_dtype(torch.float64)
u = L._u6_setup()                      # only used to obtain the same X-based problem and start
prob, st, s = u["prob"], u["st"], u["s"]
X = prob.X; shapes = prob.shapes
th0 = st.th0.clone()
# ---- an independent forward and Jacobian (no library code) ----
def myF(th):
    o = 0; h = X
    for l, (wo, wi) in enumerate(shapes):
        W = th[o:o+wo*wi].reshape(wo, wi); o += wo*wi
        b = th[o:o+wo]; o += wo
        z = h @ W.T + b
        h = torch.tanh(z) if l < len(shapes)-1 else z
    return h.reshape(-1)
def myJ(th):
    return torch.autograd.functional.jacobian(myF, th).detach().numpy()
F0 = myF(th0).numpy(); J0 = myJ(th0); K0 = J0 @ J0.T
t0 = th0.numpy()
lam, Q = np.linalg.eigh(K0); lam = lam[::-1]; Q = Q[:, ::-1]
N = len(F0)
V0 = lambda w: w - J0.T @ np.linalg.solve(K0, J0 @ w)
f = lambda th: myF(torch.tensor(th)).numpy()
jac = lambda th: myJ(torch.tensor(th))
def jump(thb, t):
    Jb = jac(thb); Kb = Jb @ Jb.T; a = np.zeros(N)
    for _ in range(500):
        th = thb + Jb.T @ a; r = t - f(th)
        if np.abs(r).max() <= 1e-12: break
        a = a + np.linalg.solve(Kb, r)
    return thb + Jb.T @ a
def nearest(FL, th):
    for _ in range(500):
        Jt = jac(th); Kt = Jt @ Jt.T; w = th - t0
        stat = np.linalg.norm(w - Jt.T @ np.linalg.solve(Kt, Jt @ w)) / np.linalg.norm(w)
        if np.abs(f(th) - FL).max() <= 1e-11 and stat <= 1e-9: return th
        th = t0 + Jt.T @ np.linalg.solve(Kt, (FL - f(th)) + Jt @ w)
    raise RuntimeError("NP not converged")
def SL(m, t):
    th = t0.copy()
    for k in range(1, m+1): th = jump(th, F0 + k/m*(t - F0))
    return th
def DJ(c, t):
    th = t0.copy()
    for _ in range(3000):
        r = f(th) - t
        if np.linalg.norm(r) <= 1e-9*np.linalg.norm(t - F0): break
        Jt = jac(th); th = th - c * Jt.T @ np.linalg.solve(Jt @ Jt.T, r)
    return th
def GD(c, t, mu=0.0):
    th = torch.tensor(t0.copy()); uu = torch.zeros_like(th); eta = c * N / lam[0]; tt = torch.tensor(t)
    for _ in range(400000):
        thv = th.clone().requires_grad_(True)
        r = myF(thv) - tt
        loss = (r @ r) / (2 * N)
        g, = torch.autograd.grad(loss, thv)
        if float(torch.linalg.norm(r)) <= 1e-8*np.linalg.norm(t - F0) and float(torch.linalg.norm(uu)) <= 1e-12: break
        uu = mu * uu - eta * g; th = th + uu
    return th.numpy()
def rho(learner, d):
    vL, vN = [], []
    for sg in (1, -1):
        thL = learner(F0 + sg * s * d); thN = nearest(f(thL), thL.copy())
        vL.append(V0(thL - t0)); vN.append(V0(thN - t0))
    vL = np.mean(vL, 0); vN = np.mean(vN, 0)
    return (vL @ vN)/(vN @ vN), (vL @ vN)/(np.linalg.norm(vL)*np.linalg.norm(vN))
fhb = lambda c, mu: 1 - ((1-mu)**2*(1+mu) + 2*mu*c)/((1-mu)**2*(2*(1+mu) - c))
dirs = {"q2": Q[:, 1].copy(), "q5": Q[:, 4].copy(), "rand": u["rand"].numpy()}
ratio = {"q2": lam[1]/lam[0], "q5": lam[4]/lam[0]}
gs, gc = CFG["GD_STOP_REL"], CFG["GD_CAP"]
cases = [("SL(8)", lambda t: SL(8, t), L.L_SL(8), lambda r: 0.4375),
         ("DJ(1.5)", lambda t: DJ(1.5, t), L.L_DJ(1.5), lambda r: -1.0),
         ("DJ(0.25)", lambda t: DJ(0.25, t), L.L_DJ(0.25), lambda r: 0.75/1.75),
         ("GD(1.5)", lambda t: GD(1.5, t), L.L_GD(1.5, gs, gc), lambda r: (1-1.5*r)/(2-1.5*r) if r else None),
         ("HB(0.1,.9)", lambda t: GD(0.1, t, 0.9), L.L_HB(0.1, 0.9, gs, gc, CFG["HB_U_TOL_REL"]), lambda r: fhb(0.1*r, 0.9) if r else None)]
print("same network (8-32-32-1 tanh, n=16), s = %.4f; lam ratios q2 %.4f q5 %.4f" % (s, ratio["q2"], ratio["q5"]))
print("%-11s %-5s | %-22s | %-22s | %s" % ("learner", "dir", "MINE rho, cos", "LIBRARY rho, cos", "prediction"))
worst = 0.0
for name, mine, Ldef, pred in cases:
    for dn, d in dirs.items():
        if name.startswith(("GD", "HB")) and dn == "rand": continue
        r1, c1 = rho(note, d)
        ev, scs, _ = L.run_pair(prob, st, torch.tensor(d), s, Ldef)
        p = pred(ratio.get(dn))
        worst = max(worst, abs(r1 - ev["rho"]))
        print("%-11s %-5s | %+.6f  %+.6f   | %+.6f  %+.6f   | %s" % (name, dn, r1, c1, ev["rho"], ev["cos"], ("%+.4f" % p) if p is not None else "-"))
print("largest |mine - library| in rho: %.2e" % worst)
