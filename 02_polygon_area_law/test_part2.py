"""SPEC2 section 4 tests. Run: python test_part2.py"""
import math
import sys

import numpy as np
import torch

import tcr
import tcr2
from tcr import LinearModel, TaskFamily, MLP, gn_learn, make_learn, run_sequences

torch.set_default_dtype(torch.float64)
SEED, EPS = 0, 0.01
NS = [3, 4, 6, 8, 16, 32, 64]
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


fam = TaskFamily(SEED)
lm = LinearModel(SEED)
pt = tcr2.PlaneTasks(fam, EPS, label_fn=lm.labels)
theta_A, _ = gn_learn(lm, lm.init(), pt.A)
learn = lambda th, task: gn_learn(lm, th, task)
_cache = {}


def poly(name):
    if name not in _cache:
        res, *_ = tcr2.polygon_run(lm, learn, theta_A, tcr2.get_poly(name), pt, fam.X_probe)
        _cache[name] = res[1]
    return _cache[name]


def test1():
    tasks = fam.tasks(EPS, gamma=math.pi / 2, label_fn=lm.labels)
    thA_old, _ = gn_learn(lm, lm.init(), tasks["A"])
    th, _ = run_sequences(make_learn(lm, tasks, "gn"), thA_old, ["loop+", "loop-"])
    a_loop = tcr.a_loop_vec(thA_old, th)
    a = poly("tri")["a"]
    err = torch.linalg.norm(a - a_loop).item()
    dA = torch.linalg.norm(theta_A - thA_old).item()
    report("1 tri == generic gamma=pi/2 loop of SPEC.md (< 1e-12)", err < 1e-12,
           f"||a_tri - a_loop||={err:.2e} (||a_loop||={torch.linalg.norm(a_loop).item():.4e}, "
           f"||theta_A(new) - theta_A(old)||={dA:.1e})")


def test2_3():
    a_tri = poly("tri")["a"]
    na_tri = torch.linalg.norm(a_tri).item()
    rels, coss, parts = [], [], []
    for N in NS:
        a = poly(f"circ{N}")["a"]
        ratio = torch.linalg.norm(a).item() / na_tri
        pred = N * math.sin(2 * math.pi / N)
        rel = abs(ratio - pred) / pred
        c = tcr.cosine(a, a_tri)
        rels.append(rel)
        coss.append(c)
        parts.append(f"N={N}: ratio={ratio:.5f} pred={pred:.5f} rel={rel:.2e} cos={c:.5f}")
    report("2 F1 on linear model (rel err < 0.01)", max(rels) < 0.01, "; ".join(parts[:]))
    report("3 F2 on linear model (cos > 0.99)", min(coss) > 0.99,
           "cos = [" + ", ".join(f"{c:.5f}" for c in coss) + f"] for N={NS}")


def test4():
    Ns = [16, 32, 64]
    ns = [torch.linalg.norm(poly(f"circ{N}")["s"]).item() for N in Ns]
    sl = float(np.polyfit(np.log(Ns), np.log(ns), 1)[0])
    report("4 F3 slope of log||s|| on log N over {16,32,64} in [-1.1,-0.9]", -1.1 <= sl <= -0.9,
           f"slope={sl:.4f}; ||s||={['%.5e' % v for v in ns]}; N||s||={['%.5f' % (N * v) for N, v in zip(Ns, ns)]}")


def test5():
    worst = 0.0
    parts = []
    for (T, n, d) in ((2, 4, 20), (25, 4, 400), (25, 4, 105)):
        rm = tcr2.RModel(0, T, n, d)
        th = rm.learn_exact(torch.zeros(d), 0)
        L = rm.losses(th)[0].item()
        worst = max(worst, L)
        parts.append(f"(T={T},n={n},d={d}) L_0 after 1 step={L:.2e}")
    ok1 = worst < 1e-20
    rm = tcr2.RModel(0, 2, 4, 20)
    E = 6
    _, _, thetas = tcr2.run_schedule(rm, "cyclic", E, 0, keep_thetas=True)
    I = torch.eye(rm.d)
    P = [I - torch.linalg.pinv(rm.X[t]) @ rm.X[t] for t in range(2)]
    M = P[1] @ P[0]
    err = 0.0
    v0 = torch.zeros(rm.d) - rm.theta_star
    for e in range(1, E + 1):
        closed = rm.theta_star + torch.linalg.matrix_power(M, e) @ v0
        err = max(err, torch.linalg.norm(thetas[2 * e - 1] - closed).item())
    ok2 = err < 1e-10
    report("5 part R: exact projection L_t < 1e-20; T=2 cyclic == (P1 P0)^e closed form (< 1e-10)",
           ok1 and ok2, "; ".join(parts) + f"; max_e ||theta_e - closed form||={err:.2e} (e=1..{E})")


def test6():
    T, E = 3, 3
    L = np.array([[0.0, 9.0, 5.0],
                  [8.0, 0.0, 3.0],
                  [4.0, 6.0, 0.0],
                  [0.0, 3.0, 2.0],
                  [2.0, 0.0, 8.0],
                  [6.0, 7.0, 0.0],
                  [0.0, 1.0, 4.0],
                  [5.0, 0.0, 1.0],
                  [4.0, 2.0, 0.0]])
    expect_W = {(2, 0): [0, 8, 4], (3, 0): [0, 2, 6],
                (2, 1): [0, 6, 3], (3, 1): [0, 7, 1],
                (2, 2): [0, 2, 8], (3, 2): [0, 4, 1]}
    expect_j = {(2, 0): 1, (3, 0): 2, (2, 1): 1, (3, 1): 1, (2, 2): 2, (3, 2): 1}
    expect_r = {(2, 0): 0.5, (3, 0): 1.0, (2, 1): 0.5, (3, 1): 1 / 7, (2, 2): 1.0, (3, 2): 0.25}
    okW = all(list(tcr2.window(L, T, e, s)) == expect_W[(e, s)] for (e, s) in expect_W)
    rho, js = tcr2.window_measures(L, T, E)
    okj = all(js[e - 2, s] == expect_j[(e, s)] for (e, s) in expect_j)
    okr = all(abs(rho[e - 2, s] - expect_r[(e, s)]) < 1e-15 for (e, s) in expect_r)
    pq = tcr2.prequential(L, tcr2.schedule_orders("cyclic", T, E, 0), T, E)
    okp = list(pq) == [4.0, 3.0, 8.0, 6.0, 1.0, 1.0]
    report("6 part R window bookkeeping (T=3, E=3 hand table)", okW and okj and okr and okp,
           f"windows ok={okW}, j* ok={okj} (got {js.tolist()}), rho ok={okr} "
           f"(got {[[round(x, 6) for x in r] for r in rho.tolist()]}), "
           f"prequential (extra) ok={okp} (got {pq.tolist()})")


def testX1():
    errs = []
    for h in (32, 512):
        mf, ms = tcr2.FastMLP(h), MLP(h)
        th = ms.init(SEED)
        g = torch.Generator().manual_seed(99)
        th = th + 0.3 * torch.randn(ms.P, generator=g)
        X = torch.cat([fam.X_A, fam.X_probe[:5]])
        J0, J1 = ms.jac(th, X), mf.jac(th, X)
        errs.append((h, (torch.linalg.norm(J1 - J0) / torch.linalg.norm(J0)).item()))
    print(f"[{'PASS' if all(e < 1e-12 for _, e in errs) else 'FAIL'}] X1 (extra, not in SPEC2) "
          f"FastMLP analytic Jacobian == tcr.MLP jacrev Jacobian (rel < 1e-12): "
          + "; ".join(f"h={h} rel={e:.2e}" for h, e in errs), flush=True)
    return all(e < 1e-12 for _, e in errs)


if __name__ == "__main__":
    test1()
    test2_3()
    test4()
    test5()
    test6()
    x1 = testX1()
    print(f"\n{sum(results)}/{len(results)} spec tests passed; extra X1 {'PASS' if x1 else 'FAIL'}")
    sys.exit(0 if all(results) and x1 else 1)
