"""Independent cross-check of the delivered SPEC9 library. An independent forward pass (plain torch), an independent Jacobian
(torch.autograd), an independent solvers (numpy linear algebra). Only the network weights, inputs, directions and sizes are taken
from the library, so the same problem is solved by two independent code paths.
C1 learners: final weights of AJ, SLa, TJ, TM.   C2/C3 Part A + Part G case (vectors v_L, v_NP, G_L, G_NP, rho, gamma).
C4 Part R ReLU case with sign -1 (reference path, field, logged learners, predictions)."""
import sys, math, time
from common9 import load
L, CFG = load("tiny")
import torch, numpy as np
torch.set_default_dtype(torch.float64)
nrm = np.linalg.norm
which = sys.argv[1] if len(sys.argv) > 1 else "C1C2C4"


def make(act):
    sm = L._small_mlp(act)
    prob, st = sm["prob"], sm["st"]
    X, shapes = prob.X, prob.shapes

    def fwd(th, Xin):
        o = 0
        h = Xin
        for l, (wo, wi) in enumerate(shapes):
            W = th[o:o + wo * wi].reshape(wo, wi)
            o += wo * wi
            b = th[o:o + wo]
            o += wo
            z = h @ W.T + b
            if l < len(shapes) - 1:
                h = torch.tanh(z) if act == "tanh" else torch.relu(z)
            else:
                h = z
        assert o == th.numel()
        return h.reshape(-1)
    f = lambda th: fwd(torch.tensor(th), X).numpy()
    jac = lambda th: torch.autograd.functional.jacobian(lambda v: fwd(v, X), torch.tensor(th)).detach().numpy()
    t0 = st.th0.numpy().copy()
    F0 = f(t0)
    assert np.abs(F0 - st.F0.numpy()).max() <= 1e-12, "the forward pass disagrees with the library's F"
    J0 = jac(t0)
    assert np.abs(J0 - st.J0.numpy()).max() <= 1e-9, "the Jacobian disagrees with the library's J"
    K0 = J0 @ J0.T
    V0 = lambda w: w - J0.T @ np.linalg.solve(K0, J0 @ w)
    return dict(sm=sm, prob=prob, st=st, fwd=fwd, f=f, jac=jac, t0=t0, F0=F0, J0=J0, V0=V0, N=len(F0))


def jump_a(M, thb, t, Jb=None, tol=1e-12):
    """theta = thb + Jb^T a with F(theta) = t (chord iteration with K_b; Newton fallback)."""
    f = M["f"]
    Jb = M["jac"](thb) if Jb is None else Jb
    Kb = Jb @ Jb.T
    a = np.zeros(M["N"])
    for _ in range(400):
        r = t - f(thb + Jb.T @ a)
        if np.abs(r).max() <= tol:
            return thb + Jb.T @ a, a
        a = a + np.linalg.solve(Kb, r)
    for _ in range(60):
        th = thb + Jb.T @ a
        r = t - f(th)
        if np.abs(r).max() <= tol:
            return th, a
        a = a + np.linalg.solve(M["jac"](th) @ Jb.T, r)
    raise RuntimeError("the jump did not converge")


def alpha_jump(M, thb, t, alpha, log=None):
    th = thb.copy()
    for it in range(400):
        the = thb + alpha * (th - thb)
        Je = M["jac"](the)
        new, a = jump_a(M, thb, t, Jb=Je)
        done = nrm(new - th) <= 1e-12 * nrm(new - thb) + 1e-15
        th = new
        if done:
            break
    else:
        raise RuntimeError("the alpha jump did not converge")
    if log is not None:
        the = thb + alpha * (th - thb)
        log.append((M["f"](the), M["jac"](the) @ (th - thb)))
    return th


def SLa(M, t, m, alpha, log=None):
    th = M["t0"].copy()
    for k in range(1, m + 1):
        th = alpha_jump(M, th, M["F0"] + (k / m) * (t - M["F0"]), alpha, log)
    return th


def TJ(M, t, c, alpha):
    q = (1 - c * (1 - alpha)) / (1 + c * alpha)
    th = M["t0"].copy()
    for k in range(2000):
        r = M["f"](th) - t
        if nrm(r) <= 1e-9 * nrm(t - M["F0"]):
            return th
        th = alpha_jump(M, th, t + q * r, alpha)
    raise RuntimeError("the TJ did not converge")


def TM(M, t, s, c, alpha, lam1):
    """theta+ = theta - eta grad L(theta + alpha (theta+ - theta)), L = |F - t|^2/(2N), eta = c N/lam1.
    Inner equation solved by a plain dense Newton-type iteration on delta with the Gauss-Newton matrix."""
    N = M["N"]
    eta = c * N / lam1
    th = M["t0"].copy()
    d = (t - M["F0"]) / nrm(t - M["F0"])
    run = 0
    for k in range(40000):
        r = M["f"](th) - t
        run = run + 1 if abs(d @ r) <= 1e-8 * s else 0
        if run >= 150:
            return th, k
        dl = np.zeros_like(th)
        for it in range(80):
            the = th + alpha * dl
            Je = M["jac"](the)
            g = dl + (eta / N) * (Je.T @ (M["f"](the) - t))
            if nrm(g) <= 1e-13 * nrm(dl) + 4e-16 * (1 + c) * nrm(th):
                break
            a = eta * alpha / N
            dl = dl - (g - Je.T @ np.linalg.solve(Je @ Je.T + np.eye(N) / a, Je @ g))
        th = th + dl
    raise RuntimeError("the TM did not converge")


def nearest(M, FL, th):
    t0, f, jac = M["t0"], M["f"], M["jac"]
    for _ in range(600):
        Jt = jac(th)
        Kt = Jt @ Jt.T
        w = th - t0
        stat = nrm(w - Jt.T @ np.linalg.solve(Kt, Jt @ w)) / nrm(w)
        if np.abs(f(th) - FL).max() <= 1e-11 and stat <= 1e-9:
            return th
        th = t0 + Jt.T @ np.linalg.solve(Kt, (FL - f(th)) + Jt @ w)
    raise RuntimeError("the nearest point did not converge")


def rel(a, b):
    return nrm(a - b) / max(nrm(b), 1e-300)


if "C1" in which:
    M = make("tanh")
    sm, prob, st = M["sm"], M["prob"], M["st"]
    S0 = sm["S0"]
    s = 0.01 * S0
    print("C1 final weights, library vs the implementation (small tanh net, s = 0.01 S0, + sign).")
    print("   rel_v = |V0(th_lib - th_mine)| / |V0(th_lib - th0)|   rel_th = |th_lib - th_mine| / |th_lib - th0|")
    for dn in ("eigA", "label"):
        d = sm["dirs"][dn].numpy()
        t = M["F0"] + s * d
        tt = torch.tensor(t)
        for Ls, mine in ((L.L_AJ(-1.0), lambda: alpha_jump(M, M["t0"], t, -1.0)), (L.L_AJ(0.5), lambda: alpha_jump(M, M["t0"], t, 0.5)),
                         (L.L_AJ(2.0), lambda: alpha_jump(M, M["t0"], t, 2.0)), (L.L_SLa(8, 2.0), lambda: SLa(M, t, 8, 2.0)),
                         (L.L_SLa(4, -1.0), lambda: SLa(M, t, 4, -1.0)), (L.L_TJ(1.5, 1.0), lambda: TJ(M, t, 1.5, 1.0)),
                         (L.L_TJ(3.0, 0.25), lambda: TJ(M, t, 3.0, 0.25)), (L.L_TJ(0.5, 0.5), lambda: TJ(M, t, 0.5, 0.5))):
            t1 = time.time()
            res = L.run_learner9(prob, st, tt, s, Ls)
            a = res["theta"].numpy()
            b = mine()
            print("   %-6s %-13s rel_v %.2e  rel_th %.2e   lib status %s / supp %s  (%.0fs)" % (
                dn, L.learner_name9(Ls), rel(M["V0"](a - b) + M["V0"](b - M["t0"]), M["V0"](b - M["t0"])) if False else
                nrm(M["V0"](a - b)) / nrm(M["V0"](a - M["t0"])), rel(a, b) * nrm(b) / nrm(a - M["t0"]),
                res["status"], res.get("status_supp"), time.time() - t1), flush=True)
    d = sm["dirs"]["eigA"].numpy()
    t = M["F0"] + s * d
    for c, al in ((1.5, 1.0), (0.5, 0.5), (6.0, 1.0)):
        t1 = time.time()
        res = L.run_TM(prob, st, torch.tensor(t), s, c, al)
        a = res["theta"].numpy()
        b, k = TM(M, t, s, c, al, st.lam1)
        print("   eigA   TM(%g,%g)     rel_v %.2e  rel_th %.2e   lib status %s steps %d (mine %d) inner caps %d inner total %d  (%.0fs)" % (
            c, al, nrm(M["V0"](a - b)) / nrm(M["V0"](a - M["t0"])), nrm(a - b) / nrm(a - M["t0"]), res["status"], res["steps"], k,
            res["tm_inner_capped"], res["tm_inner_total"], time.time() - t1), flush=True)

if "C2" in which:
    M = make("tanh")
    sm, prob, st = M["sm"], M["prob"], M["st"]
    cb = dict(act="tanh", start="trained", seed=0, n=8)
    ctx = L.Ctx9(cb, prob=sm["prob"], th0=sm["th"], Y=sm["Y"])
    Xt = ctx.test()["X"]
    ft = lambda th: M["fwd"](torch.tensor(th), Xt).numpy()
    print("C2/C3 one Part A + Part G unit run through the library's case runner vs the implementation (held-out inputs: %d)." % Xt.shape[0])
    for Ls, mine_fn, fpred in ((L.L_AJ(2.0), lambda t: alpha_jump(M, M["t0"], t, 2.0), 2.0),
                               (L.L_TJ(1.5, 1.0), lambda t: TJ(M, t, 1.5, 1.0), 5 / 7),
                               (L.L_SLa(4, -1.0), lambda t: SLa(M, t, 4, -1.0), 0.125)):
        ev = {}
        for sl, fr in (("s1", CFG["S1_FRAC"]), ("s2", CFG["S2_FRAC"])):
            mine, lib = {}, {}
            for sg in (1, -1):
                case = L._case9(ctx.cname, "A", "A", Ls, "eigA", sl, fr, sg)
                rec, vecs = L.run_case9(ctx, case)
                s = fr * sm["S0"]
                assert abs(rec["s"] - s) <= 1e-15
                t = M["F0"] + sg * s * sm["dirs"]["eigA"].numpy()
                thL = mine_fn(t)
                FL = M["f"](thL)
                thTH, _ = jump_a(M, M["t0"], FL, Jb=M["J0"])
                thNP = nearest(M, FL, thL.copy())
                mv = dict(vL=M["V0"](thL - M["t0"]), vNP=M["V0"](thNP - M["t0"]), GL=ft(thL) - ft(thTH), GNP=ft(thNP) - ft(thTH))
                mine[sg], lib[sg] = mv, {k: v.numpy() for k, v in vecs.items()}
                if sl == "s2":
                    print("   %-12s %s sign %+d: rel diff vL %.1e vNP %.1e GL %.1e GNP %.1e | lib status %s np %s | lib rho %.6f mine %.6f" % (
                        L.learner_name9(Ls), sl, sg, rel(lib[sg]["vL"], mv["vL"]), rel(lib[sg]["vNP"], mv["vNP"]),
                        rel(lib[sg]["GL"], mv["GL"]), rel(lib[sg]["GNP"], mv["GNP"]), rec["learner_status"], rec["np_status"],
                        rec["rho"], float(mv["vL"] @ mv["vNP"] / (mv["vNP"] @ mv["vNP"]))), flush=True)
            for who, D in (("lib", lib), ("mine", mine)):
                for a_, b_, nm in (("vL", "vNP", "rho"), ("GL", "GNP", "gam")):
                    x = 0.5 * (D[1][a_] + D[-1][a_])
                    y = 0.5 * (D[1][b_] + D[-1][b_])
                    ev[(who, nm, sl)] = float(x @ y / (y @ y))
        for nm in ("rho", "gam"):
            print("   %-12s %s: even s1/s2 lib %.6f/%.6f mine %.6f/%.6f | extrapolated lib %.6f mine %.6f | predicted f %.6f" % (
                L.learner_name9(Ls), nm, ev[("lib", nm, "s1")], ev[("lib", nm, "s2")], ev[("mine", nm, "s1")], ev[("mine", nm, "s2")],
                (4 * ev[("lib", nm, "s2")] - ev[("lib", nm, "s1")]) / 3, (4 * ev[("mine", nm, "s2")] - ev[("mine", nm, "s1")]) / 3, fpred), flush=True)

if "C4" in which:
    M = make("relu")
    sm, prob, st = M["sm"], M["prob"], M["st"]
    cb = dict(act="relu", start="trained", seed=0, n=8)
    ctx = L.Ctx9(cb, prob=sm["prob"], th0=sm["th"], Y=sm["Y"])
    G = int(CFG["G_REF"])
    for frac, sg in ((1e-2, -1), (1e-2, 1), (1e-3, -1)):
        case = L._case9(ctx.cname, "R", "R", None, "eigA", "r%g" % frac, frac, sg)
        rec, vecs = L.run_case9(ctx, case)
        s = frac * sm["S0"]
        d = sg * sm["dirs"]["eigA"].numpy()
        F0, t0, V0 = M["F0"], M["t0"], M["V0"]
        t = F0 + s * d
        ref = [t0.copy()]
        for j in range(1, G + 1):
            ref.append(jump_a(M, ref[-1], F0 + (2.0 * j / G) * s * d)[0])

        def pos(tau):
            tc = min(max(tau, 0.0), 2.0)
            j = min(int(math.floor(tc * G / 2.0)), G - 1)
            return jump_a(M, ref[j], F0 + tc * s * d)[0]

        def W(tau, b):
            J = M["jac"](pos(tau))
            return V0(J.T @ np.linalg.solve(J @ J.T, b))
        vPP = W(1.0, s * d)
        n2 = vPP @ vPP
        om = [float(W(k / 16.0, s * d) @ vPP / n2) for k in range(17)]
        print("C4 relu eigA s/S0=%g sign %+d: ref %s, switches n_sw(1)=%s cells %s/%s" % (frac, sg, rec["ref_status"], rec["nsw_1"],
              rec["cells_switch_01"], rec["cells_switch_02"]))
        print("   omega profile on [0,1]: max |lib - mine| = %.2e ; rel diff v_PP %.2e ; lib omega %s" % (
            max(abs(a - b) for a, b in zip(rec["omega"][:17], om)), rel(vecs["vPP"].numpy(), vPP), " ".join("%.3f" % x for x in rec["omega"][:17])))

        def logged(kind, par):
            log = []
            th = t0.copy()
            if kind == "SL":
                for k in range(1, par + 1):
                    Jb = M["jac"](th)
                    new, a = jump_a(M, th, F0 + (k / par) * (t - F0), Jb=Jb)
                    log.append((M["f"](th), Jb @ (Jb.T @ a)))
                    th = new
            elif kind == "DJ":
                for k in range(3000):
                    Fk = M["f"](th)
                    r = Fk - t
                    if nrm(r) <= CFG["DJ_TOL_REL"] * nrm(t - F0):
                        break
                    Jt = M["jac"](th)
                    log.append((Fk, -par * r))
                    th = th - par * Jt.T @ np.linalg.solve(Jt @ Jt.T, r)
            elif kind == "AJ":
                th = alpha_jump(M, t0, t, par, log)
            return th, log
        for nm, kind, par in (("SL(8)", "SL", 8), ("SL(2)", "SL", 2), ("DJ(0.5)", "DJ", 0.5), ("AJ(1)", "AJ", 1.0), ("AJ(0.5)", "AJ", 0.5), ("DJ(1.5)", "DJ", 1.5)):
            thL, log = logged(kind, par)
            vL = V0(thL - t0)
            vp = np.zeros_like(vL)
            rom, taus = 0.0, []
            for Fe, b in log:
                tau = float(d @ (Fe - F0)) / s
                taus.append(tau)
                vp = vp + W(tau, b)
                rom += (float(d @ b) / s) * float(W(tau, s * d) @ vPP / n2)
            lr = rec["learners"][nm]
            mine = dict(rho_PP=float(vL @ vPP / n2), rho_field=float(vp @ vPP / n2), cosF=float(vL @ vp / (nrm(vL) * nrm(vp))),
                        nrF=nrm(vL) / nrm(vp), rho_omega=rom)
            print("   %-8s lib/mine: rho_PP %+.5f/%+.5f rho_field %+.5f/%+.5f cosF %.6f/%.6f nrF %.5f/%.5f rho_omega %+.5f/%+.5f | steps %s/%d tau range lib [%.3f, %.3f] mine [%.3f, %.3f] | rel diff vL %.1e vpred %.1e | f_smooth %s status %s" % (
                nm, lr["rho_PP"], mine["rho_PP"], lr["rho_field"], mine["rho_field"], lr["cosF"], mine["cosF"], lr["nrF"], mine["nrF"],
                lr["rho_omega"], mine["rho_omega"], lr["n_log"], len(log), lr["min_tau"], lr["max_tau"], min(taus), max(taus),
                rel(vecs["vL__" + nm].numpy(), vL), rel(vecs["vpred__" + nm].numpy(), vp), lr["f_smooth"], lr["learner_status"]), flush=True)
