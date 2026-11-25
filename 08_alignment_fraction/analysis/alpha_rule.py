"""The alpha-point rule. One jump theta = theta0 + J(theta_e)^T a with the Jacobian evaluated at
theta_e = theta0 + alpha (theta - theta0), and F(theta) = t.
alpha = 0: pre-point (the exact jump TH, Ito-like).  alpha = 1: post-point = stationarity condition of the nearest point.
alpha = 1/2: midpoint (Stratonovich-like).  PREDICTION: alignment fraction rho(alpha) = alpha, for any alpha (also outside [0,1]).
Measured with the reviewed library's measure() (NP on the learner's own fibre, projector at the start, even part)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "review8", "final"))
exec(open(os.path.join(HERE, "..", "review8", "final", "config_cell.py"), encoding="utf-8").read())
import part8lib as L
CFG["JUMP_TOL"] = 1e-13; CFG["NP_RES_TOL"] = 1e-12
L.configure(CFG)
import torch
torch.set_default_dtype(torch.float64)
L._load_mnist_raw()


def alpha_jump(prob, th0, t, alpha, iters=200):
    th = th0.clone()
    for it in range(iters):
        Je = prob.J(th0 + alpha * (th - th0))
        Le = L.chol(L.gram(Je))
        a = torch.zeros(prob.N, dtype=torch.float64)
        for _ in range(400):                                   # inner chord iteration with Je fixed
            cur = th0 + Je.T @ a
            r = t - prob.F(cur)
            if float(r.abs().max()) <= 1e-13:
                break
            a = a + L.csolve(Le, r)
        new = th0 + Je.T @ a
        if L.vnorm(new - th) <= 1e-13 * max(1.0, L.vnorm(new - th0)):
            return new, it + 1
        th = new
    return th, iters


def rho_alpha(prob, st, d, s, alpha):
    vL, vN = [], []
    for sg in (1, -1):
        th, its = alpha_jump(prob, st.th0, st.F0 + sg * s * d, alpha)
        sc, a, b = L.measure(prob, st, th)
        vL.append(a); vN.append(b)
    ev = L.even_stats(vL[0], vN[0], vL[1], vN[1])
    return ev["rho"], ev["cos"]


ALPHAS = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
prob, th0 = L.toy_problem(); st = L.Start(prob, th0)
print("solvable model u^2 - v^2, s = 0.02:   " + "  ".join("a=%+.2f: %+.4f" % (a, rho_alpha(prob, st, torch.ones(1), 0.02, a)[0]) for a in ALPHAS))
u = L._u6_setup()
for dn in ("q1", "rand"):
    for s in (u["s"], u["s"] / 2):
        out = [rho_alpha(u["prob"], u["st"], u[dn], s, a) for a in ALPHAS]
        print("tanh net 8-32-32-1, dir %-4s s=%.4f:  " % (dn, s) + "  ".join("a=%+.2f: %+.4f[%+.3f]" % (a, r, c) for a, (r, c) in zip(ALPHAS, out)), flush=True)
for act in ("tanh", "gelu"):
    CFG["WIDTH"] = 32; CFG["N_TRAIN"] = 8; L.configure(CFG)
    cb = dict(act=act, start="trained", seed=0)
    prob, th, Y, labels = L.combo_problem(cb)
    th, _ = L.fit_dj(prob, th, Y)
    st = L.Start(prob, th)
    dirs, dinfo, S0 = L.make_directions(st, 0, 8)
    for dn in ("eigA", "label"):
        out = [rho_alpha(prob, st, dirs[dn], 0.01 * S0, a) for a in ALPHAS]
        print("%s MLP 196-32-32-10 (n=8), dir %-5s s=0.01 S0:  " % (act, dn) + "  ".join("a=%+.2f: %+.4f[%+.3f]" % (a, r, c) for a, (r, c) in zip(ALPHAS, out)), flush=True)
