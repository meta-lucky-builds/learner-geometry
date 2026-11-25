"""Pre-checks for SPEC9 (an independent reference implementation, small networks, CPU). Uses the reviewed part8lib building blocks.
P1 SLa(m, alpha): rho = 1/2 + (alpha - 1/2)/m.   P2 TJ(c, alpha): rho = (1+k)/(2+k), k = c(2 alpha - 1).
P3 TM(c, alpha) (theta-method in WEIGHT space) along kernel eigen-direction i: rho = (1+k_i)/(2+k_i), k_i = c_i(2 alpha - 1).
P4 post-point reference PP vs NP in smooth nets.  P5 reference-field prediction (single sign), incl. ReLU."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "review8", "final"))
exec(open(os.path.join(HERE, "..", "review8", "final", "config_cell.py"), encoding="utf-8").read())
import part8lib as L
CFG["JUMP_TOL"] = 1e-13; CFG["NP_RES_TOL"] = 1e-12; CFG["NP_CAP"] = 40
L.configure(CFG)
import torch
torch.set_default_dtype(torch.float64)
L._load_mnist_raw()
vn = L.vnorm


def alpha_jump(prob, thb, t, alpha, log=None, iters=300):
    """theta = thb + J(thb + alpha (theta - thb))^T a, F(theta) = t."""
    th = thb.clone(); ok = False
    for it in range(iters):
        the = thb + alpha * (th - thb)
        Je = prob.J(the); Ke = L.gram(Je); Le = L.chol(Ke)
        a = torch.zeros(prob.N)
        for _ in range(500):
            r = t - prob.F(thb + Je.T @ a)
            if float(r.abs().max()) <= 1e-13: break
            a = a + L.csolve(Le, r)
        new = thb + Je.T @ a
        if vn(new - th) <= 1e-12 * max(1e-30, vn(new - thb)) + 1e-15:
            th = new; ok = True; break
        th = new
    if log is not None:
        the = thb + alpha * (th - thb); Je = prob.J(the)
        a = L.csolve(L.chol(L.gram(Je)), Je @ (th - thb))
        log.append((prob.F(the).clone(), L.gram(Je) @ a))            # (F at evaluation point, b = K_e a)
    return th, ok


def run_SLa(prob, st, t, m, alpha, log=None):
    th = st.th0.clone(); ok = True
    for k in range(1, m + 1):
        th, o = alpha_jump(prob, th, st.F0 + (k / m) * (t - st.F0), alpha, log); ok &= o
    return th, ok


def run_TJ(prob, st, t, c, alpha, log=None, cap=600):
    q = (1 - c * (1 - alpha)) / (1 + c * alpha)
    th = st.th0.clone(); tgt = 1e-8 * vn(t - st.F0); ok = True
    for k in range(cap):
        r = prob.F(th) - t
        if vn(r) <= tgt: return th, ok
        th, o = alpha_jump(prob, th, t + q * r, alpha, log); ok &= o
    return th, False


def run_DJ_logged(prob, st, t, c, log, cap=600):
    th = st.th0.clone(); tgt = 1e-8 * vn(t - st.F0)
    for k in range(cap):
        F = prob.F(th); r = F - t
        if vn(r) <= tgt: break
        J = prob.J(th)
        log.append((F.clone(), -c * r))
        th = th - c * (J.T @ L.csolve(L.chol(L.gram(J)), r))
    return th


def run_TM(prob, st, t, s, c, alpha, cap=20000):
    """theta-method in weight space: th+ = th - eta grad L(th + alpha (th+ - th)), eta = c N / lam1. Direction-converged stop."""
    N = prob.N; eta = c * N / st.lam1; th = st.th0.clone()
    d = (t - st.F0) / vn(t - st.F0); run = 0
    for k in range(cap):
        r = prob.F(th) - t
        run = run + 1 if abs(float(d @ r)) <= 1e-7 * s else 0
        if run >= 100: return th, True
        if alpha == 0:
            th = th - eta * (prob.J(th).T @ r) / N; continue
        dl = torch.zeros_like(th); a = eta * alpha / N
        for it in range(60):
            the = th + alpha * dl; Je = prob.J(the); re = prob.F(the) - t
            g = dl + eta * (Je.T @ re) / N
            if vn(g) <= 1e-13 * max(vn(dl), 1e-30) + 1e-16: break
            Kt = L.gram(Je) + torch.eye(N) / a
            dl = dl - (g - Je.T @ torch.linalg.solve(Kt, Je @ g))
        th = th + dl
    return th, False


def rho_even(prob, st, d, s, fn):
    vL, vN, ok = [], [], True
    for sg in (1, -1):
        th, o = fn(st.F0 + sg * s * d); ok &= o
        sc, a, b = L.measure(prob, st, th); ok &= sc["np_status"] == "CONVERGED"
        vL.append(a); vN.append(b)
    ev = L.even_stats(vL[0], vN[0], vL[1], vN[1])
    return ev["rho"], ev["cos"], ok


def build(act, W=32, n=8, start="trained"):
    CFG["WIDTH"] = W; CFG["N_TRAIN"] = n; L.configure(CFG)
    prob, th, Y, labels = L.combo_problem(dict(act=act, start=start, seed=0))
    if start == "trained": th, _ = L.fit_dj(prob, th, Y)
    st = L.Start(prob, th); dirs, dinfo, S0 = L.make_directions(st, 0, n)
    return prob, st, dirs, dinfo, S0


which = sys.argv[1] if len(sys.argv) > 1 else "P123"
if "P1" in which:
    prob, st, dirs, dinfo, S0 = build("tanh")
    for dn in ("eigA", "label"):
        d = dirs[dn]; print("tanh MLP dir %s (s = 0.01 S0, even part)  rho [cos] vs prediction" % dn)
        for m, al in ((2, 0.5), (2, 1.0), (2, 2.0), (8, 0.5), (8, 1.0), (8, 2.0), (4, -1.0)):
            r, c, ok = rho_even(prob, st, d, 0.01 * S0, lambda t: run_SLa(prob, st, t, m, al))
            print("   P1 SLa(m=%d, alpha=%+.1f): %+.4f [%+.3f] pred %+.4f %s" % (m, al, r, c, 0.5 + (al - 0.5) / m, "" if ok else "NOT-OK"), flush=True)
        for c_, al in ((0.5, 0.0), (0.5, 0.5), (0.5, 1.0), (1.5, 0.5), (1.5, 1.0), (8.0, 1.0), (1.5, 0.0), (3.0, 0.25)):
            k = c_ * (2 * al - 1)
            r, c, ok = rho_even(prob, st, d, 0.01 * S0, lambda t: run_TJ(prob, st, t, c_, al))
            print("   P2 TJ(c=%.1f, alpha=%.2f): %+.4f [%+.3f] pred %+.4f %s" % (c_, al, r, c, (1 + k) / (2 + k), "" if ok else "NOT-OK"), flush=True)
    for dn in ("eigA", "eigB"):
        d = dirs[dn]; lr = dinfo[dn]["lam_ratio"]; s = 0.01 * S0
        for c_, al in ((0.5, 0.5), (0.5, 1.0), (1.5, 0.5), (1.5, 1.0), (1.5, 0.0), (6.0, 1.0)):
            k = c_ * lr * (2 * al - 1)
            r, c, ok = rho_even(prob, st, d, s, lambda t: run_TM(prob, st, t, s, c_, al))
            print("   P3 TM(c=%.1f, alpha=%.1f) dir %s (lam ratio %.3f): %+.4f [%+.3f] pred %+.4f %s" % (c_, al, dn, lr, r, c, (1 + k) / (2 + k), "" if ok else "NOT-OK"), flush=True)
if "P5" in which:
    act = sys.argv[2]; fracs = [float(x) for x in sys.argv[3].split(",")]
    prob, st, dirs, dinfo, S0 = build(act)
    G, TMAX = 256, 2.0

    def pattern(t):
        _, cache = L.mlp_forward(t, prob.X, prob.shapes, prob.act, keep=True)
        return torch.cat([(cache[i][2] > 0).reshape(-1) for i in range(len(cache) - 1)])
    for dn in ("eigA", "rand"):
        d0 = dirs[dn]
        for fr in fracs:
            for sg in (1, -1):
                s = fr * S0; d = sg * d0; t = st.F0 + s * d
                # reference transport on the line F0 + tau s d, tau = j TMAX / G
                ref = [st.th0.clone()]; th = st.th0.clone(); refok = True
                for j in range(1, G + 1):
                    jr = L.jump(prob, th, st.F0 + (j * TMAX / G) * s * d); th = jr["theta"]; refok &= jr["status"] == "CONVERGED"; ref.append(th)
                nsw = [int((pattern(x) != pattern(st.th0)).sum()) for x in (ref[G // 8], ref[G // 2], ref[G])]

                def field(tau):
                    j = min(max(int(round(tau * G / TMAX)), 0), G); J = prob.J(ref[j]); return J, L.chol(L.gram(J))
                J1, L1 = field(1.0)
                vPP = st.V0(J1.T @ L.csolve(L1, s * d))                           # post-point reference
                omega = []
                for j in range(0, G // 2 + 1, G // 16):
                    Jr = prob.J(ref[j]); w = st.V0(Jr.T @ L.csolve(L.chol(L.gram(Jr)), s * d)); omega.append(float(w @ vPP) / float(vPP @ vPP))
                out = []
                for nm, f, runner in (("SL(2)", .25, lambda lg: run_SLa(prob, st, t, 2, 0.0, lg)[0]), ("SL(8)", .4375, lambda lg: run_SLa(prob, st, t, 8, 0.0, lg)[0]),
                                      ("SL(32)", .484375, lambda lg: run_SLa(prob, st, t, 32, 0.0, lg)[0]), ("DJ(.5)", 1 / 3, lambda lg: run_DJ_logged(prob, st, t, 0.5, lg)),
                                      ("DJ(1)", 0.0, lambda lg: run_DJ_logged(prob, st, t, 1.0, lg)), ("DJ(1.5)", -1.0, lambda lg: run_DJ_logged(prob, st, t, 1.5, lg)),
                                      ("DJ(1.9)", -9.0, lambda lg: run_DJ_logged(prob, st, t, 1.9, lg))):
                    lg = []; thL = runner(lg); vL = st.V0(thL - st.th0)
                    vp = torch.zeros_like(vL); rho_om = 0.0
                    for Fe, b in lg:
                        tau = float(d @ (Fe - st.F0)) / s
                        Jr, Lr = field(tau); vp = vp + Jr.T @ L.csolve(Lr, b)
                    vp = st.V0(vp)
                    rpp = float(vL @ vPP) / float(vPP @ vPP); rpred = float(vp @ vPP) / float(vPP @ vPP)
                    out.append("%s: rho %+.3f pred %+.3f smooth %+.3f | cos %.4f nr %.3f" % (nm, rpp, rpred, f, L.cosine(vL, vp), vn(vL) / vn(vp)))
                print("%s %s s/S0=%g sign %+d | switches at tau=.25/1/2: %s | refok %s | omega(tau=0..1 step 1/8): %s" % (act, dn, fr, sg, nsw, refok, " ".join("%.2f" % x for x in omega)))
                for o in out: print("      " + o, flush=True)
if "P4" in which:
    prob, st, dirs, dinfo, S0 = build("tanh")
    for dn in ("eigA", "label"):
        for fr in (0.02, 0.01):
            d = dirs[dn]; s = fr * S0; t = st.F0 + s * d
            th = st.th0.clone()
            for j in range(1, 65): th = L.jump(prob, th, st.F0 + (j / 64) * s * d)["theta"]
            J1 = prob.J(th); vPP = st.V0(J1.T @ L.csolve(L.chol(L.gram(J1)), s * d))
            npr = L.nearest_point(prob, st.th0, t, st.th0); vNP = st.V0(npr["theta"] - st.th0)
            print("P4 tanh %s s=%.2f S0: cos(v_PP, v_NP) %.6f  |v_PP|/|v_NP| %.5f" % (dn, fr, L.cosine(vPP, vNP), vn(vPP) / vn(vNP)))
