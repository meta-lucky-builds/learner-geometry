"""SPEC section 7 tests. Run: python test_linear.py"""
import math
import sys

import numpy as np
import torch

import tcr
from tcr import (MLP, LinearModel, TaskFamily, gn_learn, gn_step, make_learn,
                 make_lin_learner, run_sequences, displacement_scalars, SEQUENCES)

torch.set_default_dtype(torch.float64)
EPS_GRID = [0.2, 0.1, 0.05, 0.025, 0.0125]
SEED = 0
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def slope(eps, vals):
    return float(np.polyfit(np.log(eps), np.log(vals), 1)[0])


fam = TaskFamily(SEED)
lm = LinearModel(SEED)
label = lm.labels


def lin_setup(eps, gamma=math.pi / 2, alpha=None):
    tasks = fam.tasks(eps, gamma=gamma, alpha=alpha, label_fn=label)
    theta_A, _ = gn_learn(lm, lm.init(), tasks["A"])
    return tasks, theta_A


# ---- Test 1: MLP Jacobian vs central finite differences -------------------
def test1():
    errs = []
    for h, ncols in ((32, None), (512, 300)):
        m = MLP(h)
        th = m.init(SEED)
        # random non-zero biases so their columns are exercised too
        g = torch.Generator().manual_seed(99)
        th = th + 0.3 * torch.randn(m.P, generator=g) * (th == 0)
        X = fam.X_A
        J = m.jac(th, X)
        if ncols is None:
            cols = torch.arange(m.P)
        else:
            cols = torch.randperm(m.P, generator=g)[:ncols]
            # always include the scalar bias b3 and some b1/b2 entries
            cols = torch.unique(torch.cat([cols, torch.tensor([m.P - 1, m.slices["b1"][0],
                                                               m.slices["b2"][0]])]))
        step = 1e-5
        Jfd = torch.empty(X.shape[0], len(cols))
        for j, c in enumerate(cols.tolist()):
            e = torch.zeros(m.P)
            e[c] = step
            Jfd[:, j] = (m.f(th + e, X) - m.f(th - e, X)) / (2 * step)
        rel = (torch.linalg.norm(J[:, cols] - Jfd) / torch.linalg.norm(J[:, cols])).item()
        errs.append((h, len(cols), rel))
        # extra: manual GD gradient vs J^T r / n
        y = fam.teacher(X)
        gman, r = m.grad_loss(th, X, y)
        gref = J.T @ r / X.shape[0]
        errs.append((h, "grad", (torch.linalg.norm(gman - gref) / torch.linalg.norm(gref)).item()))
        gnp = np.empty(m.P)
        m.make_np_grad(X, y)(th.numpy().copy(), gnp)
        errs.append((h, "npgrad", (torch.linalg.norm(torch.from_numpy(gnp) - gref)
                                   / torch.linalg.norm(gref)).item()))
    ok = all(e[2] < 1e-6 for e in errs)
    report("1 MLP Jacobian vs central FD (rel err < 1e-6)", ok,
           "; ".join(f"h={h} cols={c} rel={r:.2e}" if c not in ("grad", "npgrad") else
                     f"h={h} (extra) {c}-vs-J^Tr/n rel={r:.2e}" for h, c, r in errs))


# ---- Test 2: one GN step on linear model = closed-form projection ---------
def test2():
    X, y = fam.tasks(0.1, gamma=math.pi / 2, label_fn=label)["A"]
    th0 = lm.init()
    th1, r1, s, _ = gn_step(lm, th0, X, y)
    Phi = lm.phi(X)
    th_proj = th0 - torch.linalg.pinv(Phi) @ (Phi @ th0 - y)
    rinf = r1.abs().max().item()
    diff = torch.linalg.norm(th1 - th_proj).item()
    _, info = gn_learn(lm, th0, (X, y))
    ok = rinf < 1e-10 and diff < 1e-10 and s == 1.0
    report("2 one GN step = closed-form projection", ok,
           f"||r||_inf after 1 step={rinf:.2e}, ||theta_GN - theta_proj||={diff:.2e}, s={s}, "
           f"gn_learn iterations={info['iters']}")


# ---- Test 3: empty continuation ------------------------------------------
def test3():
    tasks, theta_A = lin_setup(0.1)
    learn = make_learn(lm, tasks, "gn")
    th, infos = run_sequences(learn, theta_A, ["A"], sequences={"A": ("A",)})
    nrm = torch.linalg.norm(th["A"] - theta_A).item()
    report("3 empty continuation (A) from theta_A", nrm < 1e-10,
           f"||delta||={nrm:.2e}, GN iterations={infos['A'][0]['iters']}")


# ---- Test 4: loop reversal = transpose on range(P_A) ----------------------
def test4():
    tasks, theta_A = lin_setup(0.1)
    I = torch.eye(lm.P)
    Pm = {}
    for k, (X, _) in tasks.items():
        Jk = lm.phi(X)
        Pm[k] = I - torch.linalg.pinv(Jk) @ Jk
    Lp = Pm["A"] @ Pm["C"] @ Pm["B"] @ Pm["A"]   # operator of loop+ (B,C,A) on range(P_A)
    Lm = Pm["A"] @ Pm["B"] @ Pm["C"] @ Pm["A"]   # operator of loop- (C,B,A) on range(P_A)
    err = torch.linalg.norm(Pm["A"] @ Pm["B"] @ Pm["C"] @ Pm["A"]
                            - (Pm["A"] @ Pm["C"] @ Pm["B"] @ Pm["A"]).T).item()
    # extra: operators obtained from the actual sequence runner + GN learner
    learn = make_learn(lm, tasks, "gn")
    base, _ = run_sequences(learn, theta_A, ["loop+", "loop-"])
    Ep = torch.empty(lm.P, lm.P)
    Em = torch.empty(lm.P, lm.P)
    for i in range(lm.P):
        v = Pm["A"][:, i]
        th, _ = run_sequences(learn, theta_A + v, ["loop+", "loop-"])
        Ep[:, i] = th["loop+"] - base["loop+"]
        Em[:, i] = th["loop-"] - base["loop-"]
    e_run_p = torch.linalg.norm(Ep - Lp).item()
    e_run_m = torch.linalg.norm(Em - Lm).item()
    e_run_T = torch.linalg.norm(Em - Ep.T).item()
    asym = torch.linalg.norm(Lp - Lm).item()
    report("4 loop reversal: ||P_A P_B P_C P_A - (P_A P_C P_B P_A)^T|| < 1e-10", err < 1e-10,
           f"err={err:.2e} (non-trivial: ||L+ - L-||={asym:.3e}); extra, from sequence runner: "
           f"||L+_run - L+||={e_run_p:.2e}, ||L-_run - L-||={e_run_m:.2e}, "
           f"||L-_run - (L+_run)^T||={e_run_T:.2e}")
    return e_run_p, e_run_m, e_run_T


# ---- Test 5: slopes ------------------------------------------------------
def test5():
    Na, Nst, Ncol = [], [], []
    for eps in EPS_GRID:
        tasks, theta_A = lin_setup(eps)
        th, _ = run_sequences(make_learn(lm, tasks, "gn"), theta_A,
                              ["loop+", "loop-", "star+", "star-"])
        sc, _, _ = displacement_scalars(lm, theta_A, th, fam.X_probe)
        Na.append(sc["N_a"])
        Nst.append(sc["N_star"])
        tasks_c = fam.tasks(eps, alpha=2.0, label_fn=label)
        thc, _ = run_sequences(make_learn(lm, tasks_c, "gn"), theta_A, ["loop+", "loop-"])
        scc, _, _ = displacement_scalars(lm, theta_A, thc, fam.X_probe)
        Ncol.append(scc["N_a"])
    pa, ps, pc = slope(EPS_GRID, Na), slope(EPS_GRID, Nst), slope(EPS_GRID, Ncol)
    ok = 1.6 <= pa <= 2.2 and 3.4 <= ps <= 4.3 and 2.5 <= pc <= 3.3
    fmt = lambda v: "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
    report("5 slopes (N_a in [1.6,2.2], N_star in [3.4,4.3], collinear a=2 in [2.5,3.3])", ok,
           f"slope N_a={pa:.3f}, N_star={ps:.3f}, N_a^col(a=2)={pc:.3f}; "
           f"N_a={fmt(Na)}; N_star={fmt(Nst)}; N_a^col={fmt(Ncol)}")


# ---- Test 6: C = B ---------------------------------------------------------
def test6():
    tasks, theta_A = lin_setup(0.1, gamma=0.0)
    same_X = torch.equal(tasks["B"][0], tasks["C"][0])
    th, _ = run_sequences(make_learn(lm, tasks, "gn"), theta_A,
                          ["loop+", "loop-", "star+", "star-"])
    sc, _, _ = displacement_scalars(lm, theta_A, th, fam.X_probe)
    ok = sc["N_a"] < 1e-10 and sc["N_star"] < 1e-10
    report("6 C = B (gamma = 0)", ok,
           f"N_a={sc['N_a']:.2e}, N_star={sc['N_star']:.2e}, X_C==X_B bitwise: {same_X}")


# ---- Test 7: Learn_lin reproduces GN on the linear model -------------------
def test7():
    tasks, theta_A = lin_setup(0.1)
    names = list(SEQUENCES)
    th_gn, _ = run_sequences(make_learn(lm, tasks, "gn"), theta_A, names)
    th_ln, _ = run_sequences(make_lin_learner(lm, theta_A, tasks), theta_A, names)
    err = max(torch.linalg.norm(th_gn[k] - th_ln[k]).item() for k in names)
    report("7 Learn_lin == GN learner on linear model (< 1e-9)", err < 1e-9,
           f"max_s ||theta_s^GN - theta_s^lin|| over all 7 sequences = {err:.2e}")


if __name__ == "__main__":
    for t in (test1, test2, test3, test4, test5, test6, test7):
        t()
    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)
