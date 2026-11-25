"""SPEC7 section 2: unit tests U1-U9 (plus implementation checks X1-X4). Run: python test_part7.py"""
import tcr7  # first: fixes thread count (via tcr4)
import sys
import time

import numpy as np
import torch

import tcr4
import tcr6
from tcr7 import Net, np_solve, gn_run, sl_run, gd_run, eta_c, Pred, identity_err, phi

results = []
T0 = time.time()
print(f"threads: torch {torch.get_num_threads()}, BLAS env = {tcr4.THREADS}", flush=True)


def report(name, ok, detail, counted=True):
    if counted:
        results.append(ok)
    tag = ("PASS" if ok else "FAIL") if counted else "info"
    print(f"[{tag}] {name}: {detail}", flush=True)


_S = {}


def S(h=32, seed=0, linear=False):
    k = (h, seed, linear)
    if k not in _S:
        s = tcr6.setup(h, seed, linear=linear)
        s.net = Net(s.model, s.X)
        _S[k] = s
    return _S[k]


def tdirs(seed, k=3):
    return tcr6.unit_dirs(seed, k)


# ---------------------------------------------------------------------------
def x_checks():
    s = S()
    th = s.theta_init.numpy().copy()
    J = s.net.jac(th)
    Jt = s.model.jac(s.theta_init, s.X).numpy()
    K = s.net.K(th)
    e1 = np.abs(J - Jt).max() / np.abs(Jt).max()
    e2 = np.abs(K - Jt @ Jt.T).max() / np.abs(K).max()
    report("X1 numpy J and closed-form K vs tcr2.FastMLP.jac", e1 < 1e-12 and e2 < 1e-12,
           f"rel J err {e1:.2e}, rel K err {e2:.2e}")
    # X2 gd_run vs tcr4.fo_learn_C
    y = s.yA + 0.05 * s.Q[:, 0]
    npm = tcr4.np_model(s.model, s.X)
    C0 = np.zeros((16, 16))
    th1, C1, i1 = tcr4.fo_learn_C(npm, s.model, s.theta_A, s.X, y.numpy(), s.F_A_np, C0, "GD", 1.5)
    eta = eta_c(s.model, s.theta_A, s.X, 1.5)
    th2, C2, Q2, dY2, i2 = gd_run(s.net, s.theta_A, y, eta)
    d = np.abs(th1.numpy() - th2).max()
    dc = np.abs(C1 - C2).max() / np.abs(C1).max()
    report("X2 gd_run reproduces tcr4.fo_learn_C (GD 1.5, y_A + 0.05 q1, h=32 s0)",
           i1["steps"] == i2["steps"] and d < 1e-12 and dc < 1e-12,
           f"steps {i1['steps']} vs {i2['steps']}, max|dtheta| {d:.2e}, rel C diff {dc:.2e}, eta {i1['eta']:.6g} vs {eta:.6g}")
    # X3 gn_run C vs tcr6.gn_C C
    thg, Cg, ig = tcr6.gn_C(s, y)
    thh, Ch, Qh, dYh, ih = gn_run(s.model, s.theta_A, s.X, y)
    d = (thg - thh).abs().max().item()
    dc = np.abs(Cg - Ch).max()
    report("X3 gn_run reproduces tcr6.gn_C (theta and C)", d == 0.0 and dc == 0.0,
           f"max|dtheta| {d:.2e}, max|dC| {dc:.2e}, iters {ig['steps']} vs {ih['steps']}")


def U1():
    s = S(32, 0, linear=True)
    thb = s.theta_A.numpy().copy()
    Fb = s.net.F(thb)
    J = s.net.jac(thb)
    errs = []
    for d in tdirs(7101):
        t = Fb + d
        th, info = np_solve(s.net, thb, t)
        ref = thb + J.T @ np.linalg.solve(J @ J.T, t - Fb)
        errs.append((float(np.abs(th - ref).max()) if th is not None else np.inf, info["status"], info["iters"]))
    ok = all(e[0] <= 1e-10 for e in errs)
    report("U1 NP on linear control = theta_b + J^T K^-1 (t - F)", ok,
           "max|diff| / status / iters: " + "; ".join(f"{e[0]:.2e} {e[1]} {e[2]}" for e in errs))


def U2():
    s = S()
    thb = s.theta_A.numpy().copy()
    rows = []
    ok = True
    for d in tdirs(7102):
        t = s.yA.numpy() + 0.3 * d
        th, info = np_solve(s.net, thb, t)
        thg, _, _, _, ig = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
        eta = eta_c(s.model, s.theta_A, s.X, 0.25)
        thd, _, _, _, idd = gd_run(s.net, s.theta_A, t, eta)
        dn = np.linalg.norm(th - thb)
        dg = np.linalg.norm(thg.numpy() - thb)
        dd = np.linalg.norm(thd - thb)
        o = (info["status"] == "CONVERGED" and info["stat"] <= 1e-8 and dn <= dg + 1e-12 and dn <= dd)
        ok &= o
        rows.append(f"{info['status']} it={info['iters']} res={info['res_inf']:.1e} stat={info['stat']:.1e} "
                    f"|NP|={dn:.6f} |GN|={dg:.6f} |GD.25|={dd:.6f} (GD {idd['status']} {idd['steps']} steps)")
    report("U2 NP on MLP: converged, stationary, nearer than GN and GD(0.25)", ok, " | ".join(rows))


def U3():
    s = S()
    thb = s.theta_A.numpy().copy()
    ok = True
    rows = []
    for d in tdirs(7102):
        diffs = []
        for sz in (0.02, 0.04, 0.08):
            t = s.yA.numpy() + sz * d
            th, _ = np_solve(s.net, thb, t)
            thg, *_ = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
            diffs.append(np.linalg.norm(th - thg.numpy()))
        p = np.polyfit(np.log([0.02, 0.04, 0.08]), np.log(diffs), 1)[0]
        ok &= p >= 2.5
        rows.append(f"p={p:.3f} (diffs {', '.join(f'{x:.2e}' for x in diffs)})")
    report("U3 ||NP - GN|| ~ s^p with p >= 2.5", ok, " | ".join(rows))


def U4():
    s = S()
    th = s.theta_init.numpy().copy()
    y = s.yA.numpy()
    t0 = time.time()
    sh = tcr7.sharp_H(s.net, th, y)
    t1 = time.time()
    H = tcr7.hessian_dense(s.net, th, y)
    asym = np.linalg.norm(H - H.T) / np.linalg.norm(H)
    top = np.linalg.eigvalsh((H + H.T) / 2)[-1]
    rel1 = abs(sh - top) / abs(top)
    tha = s.theta_A.numpy().copy()
    FA = s.net.F(tha)
    sh2 = tcr7.sharp_H(s.net, tha, FA)
    lam = s.net.lam1(tha)
    rel2 = abs(sh2 - lam / 16) / (lam / 16)
    report("U4 S_H: eigsh vs dense FD Hessian at theta_init (rel <= 1e-6); at theta_A, r=0: S_H = lam1/n (rel <= 1e-5)",
           rel1 <= 1e-6 and rel2 <= 1e-5,
           f"eigsh {sh:.10g} dense {top:.10g} rel {rel1:.2e} (dense asym {asym:.1e}, eigsh {t1 - t0:.2f}s); "
           f"theta_A: S_H {sh2:.10g} lam1/n {lam / 16:.10g} rel {rel2:.2e}")


def U5():
    s = S()
    d = tdirs(7105, 1)[0]
    y = s.yA.numpy() + 0.05 * d
    eta = eta_c(s.model, s.theta_A, s.X, 1.5)
    _, C, Q, dY, i1 = gd_run(s.net, s.theta_A, y, eta)
    e1 = identity_err(C, Q, dY)
    _, C2, Q2, dY2, i2 = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(y))
    e2 = identity_err(C2, Q2, dY2)
    report("U5 C + C^T = dY dY^T - Q (rel 1e-10), GD(1.5) and GN", e1 <= 1e-10 and e2 <= 1e-10,
           f"GD(1.5) rel {e1:.2e} ({i1['steps']} steps), GN rel {e2:.2e} ({i2['steps']} iters)")


def U6():
    s = S()
    q1 = s.Q[:, 0].numpy()
    P = Pred(s.model, s.theta_A, s.X, tcr6.law(s))
    G1 = P(np.outer(q1, q1))
    y = s.yA.numpy() + 0.025 * q1
    eta = eta_c(s.model, s.theta_A, s.X, 1.5)
    thd, *_ = gd_run(s.net, s.theta_A, y, eta)
    thg, *_ = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(y))
    dl = s.net.lam1(thd) - s.net.lam1(thg.numpy())
    ratio = dl / (0.025 ** 2 * phi(1.5) * G1)
    report("U6 Test-5 regression: dlam1(GD1.5 - GN) / (eps^2 phi(1.5) G1) in [0.9, 1.1]", 0.9 <= ratio <= 1.1,
           f"ratio {ratio:.4f} (dlam1 {dl:.4e}, G1 {G1:.4e})")


def U7():
    s = S()
    L = tcr6.law(s)
    thb = s.theta_A.numpy().copy()
    ok = True
    rows = []
    for d in tdirs(7107):
        t = s.yA.numpy() + 0.05 * d
        thn, _ = np_solve(s.net, thb, t)
        ths, C, Q, dY, info = sl_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
        D = ths.numpy() - thn
        V = L(0.5 * np.outer(dY, dY)).numpy()
        c = float(D @ V / (np.linalg.norm(D) * np.linalg.norm(V)))
        ra = np.linalg.norm(D) / np.linalg.norm(V)
        ok &= c >= 0.99 and 0.9 <= ra <= 1.1
        rows.append(f"cos {c:.5f} ratio {ra:.4f} (ratio/(1-1/64) {ra / (1 - 1 / 64):.4f})")
    report("U7 SL - NP vs LAW(1/2 dY dY^T): cos >= 0.99, ratio in [0.9, 1.1]", ok, " | ".join(rows))


SUPP = []


def U3S():
    """SUPPLEMENTARY (spec error in U3): NP and GN differ at SECOND order by the vertical vector
    LAW(dY dY^T): P_v(theta_A)(theta_NP - theta_GN) = LAW_thetaA(dY dY^T) + O(s^3)."""
    s = S()
    L = tcr6.law(s)
    thb = s.theta_A.numpy().copy()
    ok = True
    rows = []
    for d in tdirs(7102):
        rems, coss = [], []
        for sz in (0.02, 0.04, 0.08):
            t = s.yA.numpy() + sz * d
            th, _ = np_solve(s.net, thb, t)
            thg, Cg, Qg, dYg, _ = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
            V = L.Pv(torch.from_numpy(th - thg.numpy())).numpy()
            W = L(np.outer(dYg, dYg)).numpy()
            rems.append(np.linalg.norm(V - W))
            coss.append(float(V @ W / np.linalg.norm(V) / np.linalg.norm(W)))
        p = np.polyfit(np.log([0.02, 0.04, 0.08]), np.log(rems), 1)[0]
        ok &= p >= 2.5 and coss[0] >= 0.99
        rows.append(f"cos(s=.02) {coss[0]:.5f}, remainder slope p={p:.3f} (rem {', '.join(f'{x:.2e}' for x in rems)})")
    SUPP.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] U3-SUPP P_v(NP - GN) = LAW(dY dY^T) + O(s^3): " + " | ".join(rows), flush=True)


def U7S():
    """SUPPLEMENTARY (spec error in U7): theta_SL - theta_NP = LAW(C_SL - dY dY^T)
    ~ -LAW(1/2 (1 + 1/m) dY dY^T)."""
    s = S()
    L = tcr6.law(s)
    thb = s.theta_A.numpy().copy()
    ok = True
    rows = []
    for d in tdirs(7107):
        t = s.yA.numpy() + 0.05 * d
        thn, _ = np_solve(s.net, thb, t)
        ths, C, Q, dY, info = sl_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
        D = ths.numpy() - thn
        V = L(C - np.outer(dY, dY)).numpy()
        c = float(D @ V / (np.linalg.norm(D) * np.linalg.norm(V)))
        ra = np.linalg.norm(D) / np.linalg.norm(V)
        ok &= c >= 0.99 and 0.9 <= ra <= 1.1
        rows.append(f"cos {c:.5f} ratio {ra:.4f}")
    SUPP.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] U7-SUPP SL - NP vs LAW(C_SL - dY dY^T): " + " | ".join(rows), flush=True)


def U8():
    s = S()
    y = s.yA.numpy() + 0.3 * tdirs(7108, 1)[0]
    Fb = s.net.F(s.theta_init.numpy())
    ym = 2 * Fb - y
    r0 = Fb - y
    r0m = Fb - ym
    err = float(np.abs(r0m + r0).max())
    report("U8 twin target: r0(y-) = -r0(y+) (1e-14)", err <= 1e-14, f"max|r0- + r0+| = {err:.2e}")


def U9():
    s = S(32, 0, linear=True)
    thb = s.theta_A.numpy().copy()
    Fb = s.net.F(thb)
    lam0 = s.net.lam1(thb)
    P = Pred(s.model, s.theta_A, s.X, tcr6.law(s))
    mx_l, mx_p = 0.0, 0.0
    info = []
    for d in tdirs(7109):
        t = Fb + d
        ths = {}
        Cs = {}
        for c in (0.25, 1.9):
            eta = eta_c(s.model, s.theta_A, s.X, c)
            th, C, Q, dY, i = gd_run(s.net, s.theta_A, t, eta)
            ths[f"GD{c}"] = th
            Cs[f"GD{c}"] = C
            info.append(f"GD{c}:{i['status']}/{i['steps']}")
        th, C, Q, dY, i = gn_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
        ths["GN"], Cs["GN"] = th.numpy(), C
        th, i = np_solve(s.net, thb, t)
        ths["NP"] = th
        th, C, Q, dY, i = sl_run(s.model, s.theta_A, s.X, torch.from_numpy(t))
        ths["SL"], Cs["SL"] = th.numpy(), C
        for k, th in ths.items():
            mx_l = max(mx_l, abs(s.net.lam1(th) - lam0))
        for k, C in Cs.items():
            mx_p = max(mx_p, abs(P(C)))
    report("U9 linear control: all lam1 changes and pred(C) <= 1e-9", mx_l <= 1e-9 and mx_p <= 1e-9,
           f"max|dlam1| {mx_l:.2e}, max|pred| {mx_p:.2e}; runs {' '.join(info)}")


if __name__ == "__main__":
    for fn in (x_checks, U1, U2, U3, U3S, U4, U5, U6, U7, U7S, U8, U9):
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            report(fn.__name__, False, f"EXCEPTION {e!r}")
        print(f"    ({fn.__name__}: {time.time() - t0:.1f}s)", flush=True)
    n_ok = sum(results)
    print(f"SUMMARY: {n_ok}/{len(results)} literal tests passed; SUPPLEMENTARY {sum(SUPP)}/{len(SUPP)} "
          f"passed; wall {time.time() - T0:.1f}s", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)
