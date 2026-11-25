"""Prediction from g4_ends (FINDINGS.md sec. 25), written before this was run: the test-loss cost of a larger alignment fraction f
comes from the growth of the random function the network has at initialisation; everything that is learned moves toward
the true labels as f grows. So if the start function is removed (model g(x) = f(theta, x) - f(theta0, x), same Jacobian,
start outputs 0), a larger f should give a LOWER test loss at every move size, where the plain model gives a higher one at
small moves.

Teacher-student, synthetic inputs, real structure in the labels (teacher network of the project, tcr.TaskFamily.teacher).
Student: tcr2.FastMLP tanh, width h, scalar output. n training inputs, 2000 test inputs.
Learners: damped Gauss-Newton theta <- theta - c J^+ r with c = 1.5, 1, 0.5, 0.25  (f = (1-c)/(2-c) = -1, 0, 1/3, 0.4286),
all run to training residual 1e-10, so they end at the same training outputs.
Targets: plain model t = F0 + tau (y - F0); start-removed model t = tau y (outputs of g).
Reported: test loss of each learner minus that of c = 1 (f = 0), for both models, and the sign of the trend in f."""
import sys, math, torch
sys.path.insert(0, sys.argv[1])
import tcr, tcr2
torch.set_num_threads(4)
h = int(sys.argv[2]); n = int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
CS = [1.5, 1.0, 0.5, 0.25]
FS = [(1 - c) / (2 - c) for c in CS]
TAUS = [0.1, 0.25, 0.5, 1.0]


def learn(model, th0, X, t, c):
    th = th0.clone()
    s = float((t - model.f(th0, X)).norm())
    for it in range(400):
        r = model.f(th, X) - t
        if float(r.norm()) <= 1e-10 * max(s, 1e-300):
            return th, True
        J = model.jac(th, X)
        th = th - c * (J.T @ torch.linalg.solve(J @ J.T, r))
    return th, False


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, j in enumerate(o): r[j] = i
        return r
    ra, rb = rk(a), rk(b); m = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (m * (m * m - 1))


print("h %d n %d  learners c = %s  f = %s" % (h, n, CS, ["%.3f" % f for f in FS]))
print("per block: test loss minus that of f = 0, in units of 1e-6, for f = -1, 1/3, 0.43; Spearman(f, test loss) over the 4 learners")
tot = {"plain": {t: [] for t in TAUS}, "start removed": {t: [] for t in TAUS}}
for seed in seeds:
    fam = tcr.TaskFamily(seed)
    g = torch.Generator().manual_seed(4242 + seed)
    X = torch.randn(n, tcr.D_IN, generator=g, dtype=fam.X_A.dtype)
    Xt = torch.randn(2000, tcr.D_IN, generator=g, dtype=fam.X_A.dtype)
    y, yt = fam.teacher(X), fam.teacher(Xt)
    model = tcr2.FastMLP(h)
    th0 = model.init(seed)
    F0, F0t = model.f(th0, X), model.f(th0, Xt)
    for tau in TAUS:
        for name in ("plain", "start removed"):
            t = F0 + tau * (y - F0) if name == "plain" else F0 + tau * y
            losses, ok = [], True
            for c in CS:
                th, conv = learn(model, th0, X, t, c)
                ok = ok and conv
                out = model.f(th, Xt) - (0 if name == "plain" else F0t)
                losses.append(float(((out - yt) ** 2).mean()) / 2)
            if not ok:
                print("seed %d tau %g %s: a learner did not converge - block skipped" % (seed, tau, name)); continue
            sp = spearman(FS, losses)
            tot[name][tau].append((sp, losses[3] - losses[1], losses[0] - losses[1], losses[1]))
            print("seed %d tau %-5g %-14s loss(f=0) %.6f | %+9.1f %+9.1f %+9.1f | Spearman %+.2f" % (
                seed, tau, name, losses[1], 1e6 * (losses[0] - losses[1]), 1e6 * (losses[2] - losses[1]), 1e6 * (losses[3] - losses[1]), sp))
print("\nSUMMARY: blocks where test loss RISES with f (Spearman > 0) / blocks; blocks where loss(f=0.43) > loss(f=0); median relative difference")
for name in tot:
    for tau in TAUS:
        v = tot[name][tau]
        if not v: continue
        rel = sorted((a[1] / a[3]) for a in v)
        print("  %-14s tau %-5g rises %2d/%2d   loss(0.43) > loss(0): %2d/%2d   median (loss(0.43)-loss(0))/loss(0) = %+.2e" % (
            name, tau, sum(a[0] > 0 for a in v), len(v), sum(a[1] > 0 for a in v), len(v), rel[len(rel) // 2]))
