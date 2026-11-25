"""SPEC6 section 7: Test 0 and unit tests 1-6. Run: python test_part6.py"""
import tcr6  # first: fixes thread count (via tcr4) before numpy / torch are used
import math
import sys
import time

import numpy as np
import torch

import tcr
import tcr3
import tcr4
from tcr3 import nrm, cos

results = []
T0 = time.time()
print(f"threads: torch {torch.get_num_threads()}, OMP/MKL/OPENBLAS env = {tcr4.THREADS}", flush=True)


def report(name, ok, detail, counted=True):
    if counted:
        results.append(ok)
    tag = ("PASS" if ok else "FAIL") if counted else "info"
    print(f"[{tag}] {name}: {detail}", flush=True)


_S = {}


def S(h=32, seed=0, linear=False):
    k = (h, seed, linear)
    if k not in _S:
        _S[k] = tcr6.setup(h, seed, linear=linear)
    return _S[k]


# ---------------------------------------------------------------------------
def test0():
    s = S()
    L = tcr6.law(s)
    eps = 0.025
    tg = tcr4.make_targets(s, eps)
    out = tcr4.run_paths(s, tcr4.make_learn(s, "GN"), tg, ["oab"])["oab"]
    V = L.Pv(out["theta"] - s.theta_A)
    c1 = cos(V, L(out["C"]))
    ok1 = c1 > 0.999
    # (ii) random symmetric C; 5 random output-preserving directions
    g = torch.Generator().manual_seed(60)
    R = torch.randn(16, 16, generator=g)
    Cs = ((R + R.T) / 2).numpy()
    Ct = torch.from_numpy(Cs)
    VC = L(Cs)

    def psi(th):
        K = tcr6.K_of(s.model, th, s.X)
        return 0.5 * torch.trace(Ct @ torch.linalg.inv(K)).item()
    rels = []
    for _ in range(5):
        w = L.Pv(torch.randn(s.model.P, generator=g))
        lhs = tcr6.dderiv(psi, s.theta_A, w)
        rhs = -torch.dot(w, VC).item()
        rels.append(abs(lhs - rhs) / abs(rhs))
    ok2 = max(rels) < 1e-6
    # kappa as in SPEC4 test 4: LAW((eps^2/2)(v u^T - u v^T)) = kappa (eps^2/2) b_uv
    buv = tcr3.bracket(s.model, s.theta_A, s.X, s.u, s.v)
    u, v = s.u.numpy(), s.v.numpy()
    V2 = L(0.5 * (np.outer(v, u) - np.outer(u, v)))
    rp, rm = nrm(V2 - 0.5 * buv) / nrm(0.5 * buv), nrm(V2 + 0.5 * buv) / nrm(0.5 * buv)
    kap = 1 if rp < rm else -1
    report("0 re-check (h=32 seed 0)", ok1 and ok2,
           f"(i) GN out-and-back along u eps=0.025: cos(P_v V, LAW(C)) = {c1:.6f} (> 0.999); "
           f"(ii) d/dw [1/2 tr(C K^-1)] vs -w.LAW(C), 5 random vertical w (random symmetric C, "
           f"Generator(60)): rel errors {', '.join(f'{r:.2e}' for r in rels)} (< 1e-6)")
    report("0b kappa (SPEC4 test-4 convention)", True,
           f"LAW((v u^T - u v^T)/2) vs +b_uv/2: rel {rp:.2e}, vs -b_uv/2: rel {rm:.2e} -> kappa = {kap:+d}",
           counted=False)


def test1():
    s = S()
    L = tcr6.law(s)
    J = L.J
    Jn = torch.linalg.matrix_norm(J, ord=2).item()
    g = torch.Generator().manual_seed(61)
    worst = 0.0
    pairs = []
    for _ in range(10):
        i, j = (int(x) for x in torch.randint(0, 16, (2,), generator=g))
        T = tcr6.T_ij(L, i, j)
        val = nrm(J @ T) / (Jn * nrm(T))
        worst = max(worst, val)
        pairs.append((i, j))
    # also check the matrix form used elsewhere equals T_ij
    Tm = tcr6.T_matrix(L)
    i, j = pairs[0]
    dm = nrm(Tm[:, i * 16 + j] - tcr6.T_ij(L, i, j)) / nrm(tcr6.T_ij(L, i, j))
    report("1 T is vertical", worst < 1e-6,
           f"max over 10 random (i,j) {pairs} of ||J T_ij||/(||J||_2 ||T_ij||) = {worst:.3e} (< 1e-6); "
           f"[impl] T_matrix column vs T_ij rel diff {dm:.1e}")


def test2():
    s = S()
    eps = 0.025
    y = s.yA + eps * s.u
    th, Cm, info = tcr6.gd_C(s, y, 1.0)
    Ca, ainf = tcr4.apriori_C(s.K.numpy(), s.F_A_np, [y.numpy()], "GD1")
    rel = np.linalg.norm(Cm - Ca) / np.linalg.norm(Cm)
    report("2 a-priori vs measured C, GD(1) one-way along u, eps=0.025", rel < 0.05,
           f"||C_meas - C_apriori||_F/||C_meas||_F = {rel:.4e} (< 0.05); GD steps {info['steps']} "
           f"({info['status']}), a-priori steps {ainf['steps']}")


def test3():
    s = S()
    eps = 0.025
    q1 = s.Q[:, 0]
    y = s.yA + eps * q1
    th1, _, _ = tcr6.gd_C(s, y, 1.0, fixed_steps=1)
    r0 = s.F_A - y
    r1 = s.model.f(th1, s.X) - y
    ratio_q1 = abs(torch.dot(q1, r1).item()) / abs(torch.dot(q1, r0).item())
    th_gn, C_gn, _ = tcr6.gn_C(s, y)
    lam_gn = tcr6.lam1(s.model, th_gn, s.X)
    d = {}
    st = {}
    for c in (1.0, 1.5):
        th, C, info = tcr6.gd_C(s, y, c)
        d[c] = tcr6.lam1(s.model, th, s.X) - lam_gn
        st[c] = f"{info['status']}, {info['steps']} steps"
    ok = ratio_q1 < 1e-2 and abs(d[1.0]) * 20 <= abs(d[1.5])
    report("3 Test-5 bookkeeping (h=32 seed 0, eps=0.025)", ok,
           f"c=1 after one step |q1.r_1|/|q1.r_0| = {ratio_q1:.3e} (criterion chosen: < 1e-2); "
           f"Dlam1(c=1) = {d[1.0]:.4e} [{st[1.0]}], Dlam1(c=1.5) = {d[1.5]:.4e} [{st[1.5]}], "
           f"ratio |c=1.5|/|c=1| = {abs(d[1.5]) / max(abs(d[1.0]), 1e-300):.2f} (>= 20)")


def test4():
    s = S()
    KA = s.K.numpy()
    Pm, ndraw, minre = tcr6.draw_P(0, KA)
    ok = True
    det = []
    for gm in tcr6.GAMMAS8:
        q = tcr6.lyap_quantities(KA, Pm, gm)
        res, asym, emin, emax = tcr6.lyap_residual(q)
        good = res < 1e-10 and asym < 1e-12 and emin >= -1e-12 * emax
        ok &= good
        det.append(f"gamma={gm}: resid {res:.2e}, asym {asym:.1e}, eig [{emin:.2e}, {emax:.2e}], "
                   f"rho(I-A) {q['spec_rad_IA']:.6f}")
    report("4 Test-8 Lyapunov (h=32 seed 0)", ok,
           f"P accepted at draw {ndraw}; " + "; ".join(det) +
           " (residual ||Sig-(I-A)Sig(I-A)^T-BB^T||/||Sig|| < 1e-10, symmetric, PSD)")


def test5():
    s = S()
    yA = s.yA.numpy()
    p, q = s.u.numpy(), s.v.numpy()
    a, Pi = 0.05, 16
    om = 2 * math.pi / Pi
    worst = 0.0
    for k in range(8 * Pi):
        yp = tcr6.drive_target(yA, a, p, q, om, k, "ROT") - yA
        ym = tcr6.drive_target(yA, a, p, q, -om, k, "ROT") - yA
        # mirror: same p-component, opposite q-component
        pp, pq_ = np.dot(yp, p) / np.dot(p, p), np.dot(yp, q) / np.dot(q, q)
        mp, mq = np.dot(ym, p) / np.dot(p, p), np.dot(ym, q) / np.dot(q, q)
        worst = max(worst, abs(pp - mp), abs(pq_ + mq),
                    np.abs(ym - (mp * p + mq * q)).max(), np.abs(yp - (pp * p + pq_ * q)).max())
    eta_n = 0.5 / s.lam[0].item()
    KA = s.K.numpy()
    CL = tcr6.apriori_drive_C(KA, s.F_A_np, yA, eta_n, s.u, s.v, om, Pi, "LINE", [8])[8]
    CR = tcr6.apriori_drive_C(KA, s.F_A_np, yA, eta_n, s.u, s.v, om, Pi, "ROT", [8])[8]
    areaL = np.linalg.norm(c_area := (CL - CL.T) / 2)
    areaR = np.linalg.norm((CR - CR.T) / 2)
    ok1 = worst < 1e-15
    ok2 = areaL < 1e-12
    report("5a Test-9 drives: ROT+ / ROT- mirror images", ok1,
           f"max deviation from exact mirror (p-component equal, q-component opposite, no other "
           f"component) over 8 periods (Pi=16) = {worst:.1e}")
    report("5b Test-9 drives: LINE has zero a-priori area matrix (< 1e-12)", ok2,
           f"||C_area(LINE)||_F = {areaL:.4e} vs ||C_area(ROT+)||_F = {areaR:.4e} "
           f"(h=32 seed 0, (u,v), Pi=16, m=8, incl. GN return); ||C_area(LINE)||/||C_area(ROT+)|| = "
           f"{areaL / areaR:.4e}")
    del c_area


def test6():
    det, worst = [], {}
    for seed in range(5):
        s = S(None, seed, linear=True)
        eps = 0.025
        q1 = s.Q[:, 0]
        y = s.yA + eps * q1
        th_gn, _, _ = tcr6.gn_C(s, y)
        l_gn = tcr6.lam1(s.model, th_gn, s.X)
        dl = 0.0
        for c in (0.5, 1.5):
            th, _, _ = tcr6.gd_C(s, y, c)
            dl = max(dl, abs(tcr6.lam1(s.model, th, s.X) - l_gn))
        sv, wv, s2, w2 = tcr6.strong_weak_dirs(s)
        do = 0.0
        for p, q in ((sv, s2), (wv, w2), (sv, wv)):
            tp, tq = tcr6.order_pair(s, p, q, eps)
            do = max(do, nrm(tp - tq))
        npm = tcr4.np_model(s.model, s.X)
        eta = 0.5 * 16 / s.lam[0].item()
        Pi = 16
        dv = 0.0
        for om in (2 * math.pi / Pi, -2 * math.pi / Pi):
            for kind in ("ROT", "LINE"):
                o = tcr6.run_drive(s, npm, eta, s.u, s.v, om, Pi, kind, [8])
                dv = max(dv, nrm(o[8][0] - s.theta_A))
        worst[seed] = (dl, do, dv)
        det.append(f"seed {seed}: |Dlam1| {dl:.1e}, ||O_pq|| {do:.1e}, ||V|| {dv:.1e}")
    ok = all(max(v) < 1e-10 for v in worst.values())
    report("6 linear control (tcr.LinearModel, seeds 0-4)", ok,
           "; ".join(det) + " (Test-5 c in {0.5,1.5} eps=0.025; Test-7 three pairs eps=0.025; "
           "Test-9 ROT/LINE +-omega, (u,v), Pi=16, m=8; all < 1e-10)")


for t in (test0, test1, test2, test3, test4, test5, test6):
    try:
        t()
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        report(t.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
print(f"{sum(results)}/{len(results)} passed in {time.time() - T0:.1f}s")
sys.exit(0 if all(results) else 1)
