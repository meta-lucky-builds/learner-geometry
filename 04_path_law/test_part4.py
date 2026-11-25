"""SPEC4 section 7 tests (+ implementation checks X1-X3). Run: python test_part4.py"""
import tcr4  # first: fixes thread count before numpy / torch are used
import sys
import time

import numpy as np
import torch

import tcr
import tcr3
from tcr3 import nrm, cos, jk, proj_v

results = []
T0 = time.time()
print(f"threads: torch {torch.get_num_threads()}, OMP/MKL/OPENBLAS env = {tcr4.THREADS}", flush=True)


def report(name, ok, detail, counted=True):
    if counted:
        results.append(ok)
    tag = ("PASS" if ok else "FAIL") if counted else ("info" if ok else "info*")
    print(f"[{tag}] {name}: {detail}", flush=True)


_S = {}


def S(h, seed=0):
    if (h, seed) not in _S:
        _S[(h, seed)] = tcr4.setup(h, seed)
    return _S[(h, seed)]


_L = {}


def LAW(h, seed=0):
    if (h, seed) not in _L:
        s = S(h, seed)
        _L[(h, seed)] = tcr4.Law(s.model, s.theta_A, s.X)
    return _L[(h, seed)]


# ---------------------------------------------------------------------------
def test1():
    FA = np.array([0.0, 0.0])
    path = [np.array(p, dtype=float) for p in ((0, 0), (1, 0), (1, 1), (0, 0))]
    C = np.zeros((2, 2))
    for k in range(len(path) - 1):
        tcr4.c_increment(C, path[k + 1], path[k], FA)
    want = np.array([[-1.0, -1.0], [0.0, -1.0]])
    ok1 = np.array_equal(C, want)
    # same orientation in the a-priori GN simulation (targets = the path's vertices)
    Ca, _ = tcr4.apriori_C(np.eye(2), FA, [path[1], path[2], path[3]], "GN")
    ok2 = np.array_equal(Ca, want)
    report("1 C bookkeeping", ok1 and ok2,
           f"C = {C.tolist()} (want {want.tolist()}, exact: {ok1}); a-priori GN rule on the same "
           f"vertices = {Ca.tolist()} (exact: {ok2})")


def test2():
    s = S(32)
    eps = 0.025
    y = (s.yA + eps * s.u)
    rec = []

    def hook(k, th, dth):
        if k < 20:
            rec.append((th, dth))

    npm = tcr4.np_model(s.model, s.X)
    C = np.zeros((16, 16))
    # first learn of oab; stop after 20 steps (fixed_steps=20: the first 20 steps are identical
    # to those of the full learn, which has the same eta and update)
    tcr4.fo_learn_C(npm, s.model, s.theta_A, s.X, y.numpy(), s.F_A_np, C, "GD", 1.0,
                    fixed_steps=20, hook=hook)
    worst = 0.0
    for th, dth in rec:
        tht, dt = torch.from_numpy(th), torch.from_numpy(dth)
        J, K = jk(s.model, tht, s.X)
        worst = max(worst, nrm(proj_v(J, K, dt)) / nrm(dt))
    report("2 GD steps are horizontal", len(rec) == 20 and worst < 1e-10,
           f"max over first {len(rec)} steps of ||P_v(theta_k) dtheta_k||/||dtheta_k|| = "
           f"{worst:.3e} (< 1e-10; h=32 seed 0 GD(1) oab first learn, eps=0.025)")


def test3():
    s = S(32)
    g = torch.Generator().manual_seed(4)
    C1 = torch.randn(16, 16, generator=g).numpy()
    C2 = torch.randn(16, 16, generator=g).numpy()
    ld = lambda C: tcr4.law_direct(s.model, s.theta_A, s.X, C)  # noqa: E731
    lhs = ld(2 * C1 - 3 * C2)
    rhs = 2 * ld(C1) - 3 * ld(C2)
    rel = nrm(lhs - rhs) / nrm(lhs)
    L = LAW(32)
    relf = nrm(L(C1) - ld(C1)) / nrm(ld(C1))
    report("3 LAW is linear in C", rel < 1e-6,
           f"literal column-by-column LAW: ||LAW(2C1-3C2) - (2LAW(C1)-3LAW(C2))||/||LAW(2C1-3C2)|| = "
           f"{rel:.3e} (< 1e-6); [X3] precomputed-tensor LAW vs literal LAW on C1: rel {relf:.3e}")
    return relf


def test4():
    s = S(32)
    eps = 0.05
    u, v = s.u.numpy(), s.v.numpy()
    cu = tcr3.self_term(s.model, s.theta_A, s.X, s.u)
    buv = tcr3.bracket(s.model, s.theta_A, s.X, s.u, s.v)
    L = LAW(32)
    out = []
    ok = True
    for tag, fn in (("literal", lambda C: tcr4.law_direct(s.model, s.theta_A, s.X, C)),
                    ("tensor", L)):
        V1 = fn(-eps ** 2 * np.outer(u, u))
        r1 = nrm(V1 - (-eps ** 2) * cu) / nrm(eps ** 2 * cu)
        V2 = fn((eps ** 2 / 2) * (np.outer(v, u) - np.outer(u, v)))
        tgt = (eps ** 2 / 2) * buv
        rp = nrm(V2 - tgt) / nrm(tgt)
        rm = nrm(V2 + tgt) / nrm(tgt)
        kappa = 1 if rp < rm else -1
        rk = min(rp, rm)
        ok &= r1 < 1e-5 and rk < 1e-5
        out.append(f"[{tag}] rel(LAW(-eps^2 u u^T), -eps^2 c_u) = {r1:.3e}; "
                   f"rel(LAW((eps^2/2)(v u^T - u v^T)), +(eps^2/2) b_uv) = {rp:.3e}, "
                   f"vs -(eps^2/2) b_uv = {rm:.3e} -> kappa = {kappa:+d}; "
                   f"cos(V2, b_uv) = {cos(V2, buv):+.8f}")
        if tag == "literal":
            kap = kappa
    report("4 LAW reproduces SPEC3 formulas", ok,
           " | ".join(out) + f" (both < 1e-5; h=32 seed 0 eps=0.05). KAPPA = {kap:+d}")


def test5():
    s = S(32)
    eps = 0.025
    L = LAW(32)
    tg = tcr4.make_targets(s, eps)
    det, ok = [], True
    ends = {}
    for name in ("GN", "GD1", "MOM"):
        learn = tcr4.make_learn(s, name)
        out = tcr4.run_paths(s, learn, tg, ["oab", "oneway"])
        ends[name] = out
        o = out["oab"]
        Vm = L.Pv(o["theta"] - s.theta_A)
        cmp_ = tcr4.compare(Vm, L(o["C"]))
        ok &= cmp_["cos"] > 0.999 and o["status"] == "OK"
        det.append(f"oab {name}: cos={cmp_['cos']:.6f} ratio={cmp_['ratio']:.6f} "
                   f"[{o['status']}, steps {[i['steps'] for i in o['infos']]}]")
    D = L.Pv(ends["GD1"]["oneway"]["theta"] - ends["GN"]["oneway"]["theta"])
    cmp_ = tcr4.compare(D, L(ends["GD1"]["oneway"]["C"] - ends["GN"]["oneway"]["C"]))
    ok &= cmp_["cos"] > 0.999
    det.append(f"oneway GD1 vs GN: cos={cmp_['cos']:.6f} ratio={cmp_['ratio']:.6f}")
    report("5 pre-check h=32 seed 0 eps=0.025", ok, "; ".join(det) + " (each cos > 0.999)")


def s6_run(seed, eps=0.1, uv_teacher=False, gn_fixed=False):
    s = tcr3.Setup(None, seed, linear=True, uv_from_teacher_for_linear=uv_teacher)
    s.F_A = s.model.f(s.theta_A, s.X)
    s.F_A_np = s.F_A.numpy().copy()
    J, K = jk(s.model, s.theta_A, s.X)
    tg = tcr4.make_targets(s, eps)
    vals = {}
    for name in ("GN", "GD1"):
        if name == "GN":
            learn = tcr4.make_learn(s, "GN", fixed_steps=2000 if gn_fixed else None)
        else:
            learn = tcr4.make_learn(s, "GD1", fixed_steps=2000)
        out = tcr4.run_paths(s, learn, tg, ["oab", "tri+"])
        for p in ("oab", "tri+"):
            vals[(name, p)] = (nrm(proj_v(J, K, out[p]["theta"] - s.theta_A)), out[p]["status"],
                               max(i["res_inf"] for i in out[p]["infos"][-1:]))
    return vals


def test6():
    det, ok = [], True
    worst = 0.0
    for seed in range(5):
        vals = s6_run(seed)
        for k, (v, st, res) in vals.items():
            worst = max(worst, v)
            ok &= v < 1e-10 and st != "DIVERGED"
        det.append(f"seed {seed}: " + ", ".join(f"{k[0]}/{k[1]} {v[0]:.2e} (end res {v[2]:.1e})"
                                                 for k, v in vals.items()))
    report("6 S6 linear control (u,v scaled to the linear model's ||y_A||; GN = tcr.gn_learn; "
           "GD(1) exactly 2000 steps/learn; eps=0.1)", ok,
           f"max ||P_v(theta_end - theta_A)|| = {worst:.3e} (< 1e-10) | " + " | ".join(det))
    # supplementary readings (not counted)
    sup = []
    for uvt, gnf, tag in ((True, False, "u,v scaled to teacher ||y_A||"),
                          (False, True, "GN also run for exactly 2000 iterations/learn")):
        w = 0.0
        for seed in range(5):
            vals = s6_run(seed, uv_teacher=uvt, gn_fixed=gnf)
            w = max(w, max(v[0] for v in vals.values()))
        sup.append(f"[{tag}] max = {w:.3e}")
    report("6-SUPP S6 alternative readings", True, "; ".join(sup), counted=False)


def testX1():
    s = S(32)
    y = s.yA + 0.05 * s.u
    th_ref, inf_ref = tcr.gn_learn(s.model, s.theta_A, (s.X, y))
    th, C, inf = tcr4.gn_learn_C(s.model, s.theta_A, s.X, y, s.F_A_np, np.zeros((16, 16)))
    ok = torch.equal(th, th_ref) and inf["steps"] == inf_ref["iters"]
    report("X1 gn_learn_C == tcr.gn_learn (bitwise)", ok,
           f"identical theta: {torch.equal(th, th_ref)}; iterations {inf['steps']} vs {inf_ref['iters']}")


def testX2():
    det, ok = [], True
    for h in (32, 128):
        s = S(h)
        npm = tcr4.np_model(s.model, s.X)
        th = s.theta_A + 0.01 * torch.randn(s.model.P, generator=torch.Generator().manual_seed(9))
        y = s.yA + 0.1 * s.u
        F, r, g, acts = npm.fg(th.numpy(), y.numpy())
        J = s.model.jac(th, s.X)
        rt = torch.from_numpy(r)
        g_ref = J.T @ rt / 16
        eg = nrm(torch.from_numpy(g) - g_ref) / nrm(g_ref)
        ef = nrm(torch.from_numpy(F) - s.model.f(th, s.X)) / nrm(s.model.f(th, s.X))
        Kr = (J @ J.T) @ rt / 16
        ej = nrm(torch.from_numpy(npm.jvp(th.numpy(), g, acts)) - Kr) / nrm(Kr)
        ok &= eg < 1e-12 and ef < 1e-12 and ej < 1e-12
        det.append(f"h={h}: grad vs J^T r/n {eg:.2e}, F {ef:.2e}, J g vs K r/n {ej:.2e}")
    report("X2 numpy gradient / JVP vs exact Jacobian", ok, "; ".join(det) + " (< 1e-12)")


for t in (test1, test2, test3, test4, test5, test6, testX1, testX2):
    try:
        t()
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        report(t.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
print(f"{sum(results)}/{len(results)} passed in {time.time() - T0:.1f}s")
sys.exit(0 if all(results) else 1)
