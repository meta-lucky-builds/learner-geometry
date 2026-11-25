"""Long-run holonomy: does the vertical drift per closed output loop die out, stay steady or grow when the SAME loop is
repeated thousands of times? (Robotics analogue: pseudoinverse control of redundant arms - joint drift per task-space loop,
Klein & Huang 1983; drift-free limit trajectories.)  Small tanh MLP of the project (tcr2.FastMLP, d=8, scalar output, n=16).
Transport: theta' = J^+ y' integrated by RK4 (nsub steps per edge) + Gauss-Newton corrector at each corner (no jitter term
to second order). Loop = rectangle spanned by two fixed unit output directions u, v, side eps*|y_A|. Control: 'retrace'
(out along u, v and back the same way: zero area)."""
import sys, math, time, torch
sys.path.insert(0, sys.argv[1])
import tcr, tcr2
torch.set_num_threads(2)
h = int(sys.argv[2]); seed = int(sys.argv[3]); eps = float(sys.argv[4]); NC = int(sys.argv[5]); kinds = sys.argv[6].split(",")
model = tcr2.FastMLP(h); fam = tcr.TaskFamily(seed); X = fam.X_A; yA = fam.teacher(X); Xp = fam.X_probe
th0, _ = tcr.gn_learn(model, model.init(seed), (X, yA))
def xi(th, e):
    J = model.jac(th, X); return J.T @ torch.linalg.solve(J @ J.T, e)
def edge(th, y_from, y_to, nsub=4):
    d = (y_to - y_from) / nsub
    for _ in range(nsub):
        k1 = xi(th, d); k2 = xi(th + 0.5 * k1, d); k3 = xi(th + 0.5 * k2, d); k4 = xi(th + k3, d)
        th = th + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    th, _ = tcr.gn_learn(model, th, (X, y_to))
    return th
def kstats(th):
    J = model.jac(th, X); lam = torch.linalg.eigvalsh(J @ J.T)
    return float(lam[0]), float(lam[-1]), float(lam.sum())
J0 = model.jac(th0, X); lam, Q = torch.linalg.eigh(J0 @ J0.T)
g = torch.Generator().manual_seed(77 + seed)
R = torch.linalg.qr(torch.randn(16, 2, generator=g, dtype=th0.dtype))[0]
pairs = {"weak": (Q[:, 0], Q[:, 1]), "strong": (Q[:, -1], Q[:, -2]), "rand": (R[:, 0], R[:, 1]), "mixed": (Q[:, -1], Q[:, 0])}
S = float(yA.norm()) * eps
fp0 = model.f(th0, Xp)
print("h %d seed %d P %d eps %.3g side %.4f |yA| %.3f  K: lam_min %.3e lam_max %.3e tr %.3e" % (h, seed, model.P, eps, S, float(yA.norm()), *kstats(th0)), flush=True)
marks = sorted(set([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000]) | {NC})
for kind in kinds:
    if kind.startswith("retrace_"):
        u, v = pairs[kind.split("_")[1]]; route = [yA + S * u, yA + S * u + S * v, yA + S * u, yA]
    else:
        u, v = pairs[kind]; route = [yA + S * u, yA + S * u + S * v, yA + S * v, yA]
    th = th0.clone(); prev = None; d1 = None; t0 = time.time()
    print("== %s: cycle  |d_k|  |d_k|/|d_1|  cos(d_k,d_k-1)  cos(d_k,d_1)  |th-th0|  lam_min  lam_max  trK  probe rms change  (sec)" % kind, flush=True)
    for c in range(1, NC + 1):
        tb = th; y = yA
        try:
            for yt in route:
                th = edge(th, y, yt); y = yt
        except Exception as e:
            print("   cycle %d FAILED: %s" % (c, str(e)[:100])); break
        d = th - tb
        if d1 is None: d1 = d.clone()
        if c in marks:
            ks = kstats(th)
            cs = float(d @ prev) / float(d.norm() * prev.norm()) if prev is not None else float("nan")
            print("   %6d  %.3e  %.3f  %+.4f  %+.4f  %.4f  %.3e  %.3e  %.3e  %.3e  (%.0f)" % (
                c, float(d.norm()), float(d.norm() / d1.norm()), cs, float(d @ d1) / float(d.norm() * d1.norm()), float((th - th0).norm()),
                ks[0], ks[1], ks[2], float((model.f(th, Xp) - fp0).pow(2).mean().sqrt()), time.time() - t0), flush=True)
        prev = d
