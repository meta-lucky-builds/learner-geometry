"""SPEC3 section 7 tests. Run: python test_part3.py"""
import sys
import time

import torch

import tcr
import tcr3
from tcr3 import Setup, LabelPlane, bracket, self_term, xi, jk, nrm, cos, stiffness

torch.set_default_dtype(torch.float64)
results = []
T0 = time.time()


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


_S = {}


def S(h, seed=0):
    if (h, seed) not in _S:
        _S[(h, seed)] = Setup(h, seed)
    return _S[(h, seed)]


def gn(model):
    return lambda th, task: tcr.gn_learn(model, th, task)


def test1():
    s = S(32)
    x = xi(s.model, s.theta_A, s.X, s.u)
    J = s.model.jac(s.theta_A, s.X)
    err = (J @ x - s.u).abs().max().item()
    report("1 xi_e check", err < 1e-10, f"||J xi_u - u||_inf = {err:.3e} (< 1e-10); "
           f"||u||={nrm(s.u):.6f} ||v||={nrm(s.v):.6f} ||y_A||={nrm(s.yA):.6f} u.v={torch.dot(s.u, s.v).item():.2e}")


def test2():
    ok, det = True, []
    for h in (32, 128):
        s = S(h)
        b = bracket(s.model, s.theta_A, s.X, s.u, s.v)
        J = s.model.jac(s.theta_A, s.X)
        val = nrm(J @ b) / (torch.linalg.matrix_norm(J, ord=2).item() * nrm(b))
        ok &= val < 1e-6
        det.append(f"h={h}: {val:.3e} (||b_uv||={nrm(b):.4e})")
    report("2 bracket is vertical", ok, "; ".join(det) + " (< 1e-6)")


def test3():
    s = S(32)
    m, th, X, u = s.model, s.theta_A, s.X, s.u
    c = self_term(m, th, X, u)
    st = 1e-4 / nrm(c)
    dd = (stiffness(m, th + st * c, X, u) - stiffness(m, th - st * c, X, u)) / (2 * st)
    target = -nrm(c) ** 2
    rel = abs(dd - target) / abs(target)
    report("3 identity P_v grad phi_u = -c_u", rel < 1e-3,
           f"D phi_u . c_u = {dd:.8e}, -||c_u||^2 = {target:.8e}, rel err {rel:.3e} (< 1e-3)")


def test4():
    s = S(32)
    b4 = bracket(s.model, s.theta_A, s.X, s.u, s.v, step=1e-4)
    b5 = bracket(s.model, s.theta_A, s.X, s.u, s.v, step=1e-5)
    rel = nrm(b4 - b5) / nrm(b4)
    report("4 FD step sanity", rel < 1e-4, f"||b(1e-4) - b(1e-5)|| / ||b(1e-4)|| = {rel:.3e} (< 1e-4)")


def test5():
    det, ok = [], True
    for uvt in (True, False):
        s = Setup(None, 0, linear=True, uv_from_teacher_for_linear=uvt)
        tag = "u,v scaled to teacher ||y_A|| (primary)" if uvt else "u,v scaled to linear ||y_A|| (suppl.)"
        learn = gn(s.model)
        # L5
        res, _, _, _, _ = tcr3.poly_run(s, learn, s.theta_A, [(1, 0), (0, 1)], LabelPlane(s, 0.2))
        na = nrm(res[1]["a"])
        # K6
        dret = max(nrm(tcr3.out_and_back(s, learn, s.theta_A, s.u, e)[0] - s.theta_A)
                   for e in (0.2, 0.1, 0.05, 0.025, 0.0125))
        dsub = 0.0
        for L in (0.1, 0.05):
            t1, _ = tcr3.refine_path(s, learn, s.theta_A, s.u, L, 1)
            t4, _ = tcr3.refine_path(s, learn, s.theta_A, s.u, L, 4)
            dsub = max(dsub, nrm(t4 - t1))
        dk2 = nrm(tcr3.out_and_back(s, learn, s.theta_A, s.u, 0.1, m=8)[0] - s.theta_A)
        o = na < 1e-10 and dret < 1e-10 and dsub < 1e-10 and dk2 < 1e-10
        if uvt:
            ok &= o
        det.append(f"[{tag}] L5 ||a_tri||(eps=0.2)={na:.3e}; K6 max||d_ret||={dret:.3e}, "
                   f"max||d_sub(4)||={dsub:.3e}, ||d_K2(m=8)||={dk2:.3e}")
    report("5 linear controls L5, K6", ok, " | ".join(det) + " (all < 1e-10)")


def test6():
    s = S(32)
    eps = 0.025
    learn = gn(s.model)
    res, _, _, _, _ = tcr3.poly_run(s, learn, s.theta_A, [(1, 0), (0, 1)], LabelPlane(s, eps))
    a = res[1]["a"]
    b = bracket(s.model, s.theta_A, s.X, s.u, s.v)
    ca = cos(a, b)
    ra = nrm(a) / (eps ** 2 / 2 * nrm(b))
    th_ret, _, _ = tcr3.out_and_back(s, learn, s.theta_A, s.u, eps)
    d = th_ret - s.theta_A
    c = self_term(s.model, s.theta_A, s.X, s.u)
    cd = cos(d, c)
    rd = nrm(d) / (eps ** 2 * nrm(c))
    ok = ca > 0.99 and 0.97 <= ra <= 1.03 and cd < -0.99 and 0.95 <= rd <= 1.05
    report("6 pre-check h=32 seed 0 eps=0.025", ok,
           f"cos(a_tri,b_uv)={ca:.6f} ratio={ra:.6f}; cos(d_ret,c_u)={cd:.6f} ratio={rd:.6f}")


def test7():
    s = S(32)
    learn = gn(s.model)
    _, _, _, _, rmax = tcr3.poly_run(s, learn, s.theta_A, [(1, 0), (0, 1)], LabelPlane(s, 0.1))
    fired = False
    try:
        tcr3.end_check(s.model, s.theta_A + 1e-6 * xi(s.model, s.theta_A, s.X, s.u), s.A)
    except AssertionError:
        fired = True
    r0 = (s.model.f(s.theta_A, s.X) - s.yA).abs().max().item()
    report("7 end-of-sequence residual assert", rmax <= 1e-12 and fired and r0 <= 1e-12,
           f"tri eps=0.1 max end residual {rmax:.3e}; theta_A residual {r0:.3e}; "
           f"assert fires on a perturbed endpoint: {fired}")


for t in (test1, test2, test3, test4, test5, test6, test7):
    try:
        t()
    except Exception as e:  # noqa
        report(t.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
print(f"{sum(results)}/{len(results)} passed in {time.time() - T0:.1f}s")
sys.exit(0 if all(results) else 1)
