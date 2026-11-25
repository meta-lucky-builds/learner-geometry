"""Pre-check for SPEC12 (predictions in FINDINGS.md sec. 45). Usage: python precheck12.py <combo> <threads> <stage>
stage: info | modes | adam | metric"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
cname, threads, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3]
L, CFG = S.load(B, "cpu", threads, OUT + "/pre12_out")
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx)
prob, st = combo["prob"], combo["st"]; N, P = prob.N, prob.P
s1, s2 = combo["sizes"]["s1"], combo["sizes"]["s2"]
EPS = 1e-8
def P_(*a): print(*a, flush=True)

def grad_at(th, t):
    F, cache = prob.fwd(th); r = F - t
    return prob.vjp(cache, r) / N, L.vnorm(r)

def run(kind, par, th_start, t, s, cap, stop, state=None):
    """One leg of a first-order learner. Returns (theta, steps, res/s, state). gamma = look-ahead in units of the last step."""
    th = th_start.clone(); u = torch.zeros_like(th); k = 0
    if kind == "ADAM":
        lr, b1, b2, gamma = par
        m, v, kk = (torch.zeros_like(th), torch.zeros_like(th), 0) if state is None else state
    elif kind == "RMSmid":
        lr, b2 = par; gamma = 0.0
        v, kk = torch.zeros_like(th), 0
    elif kind == "HB":
        eta, mu, gamma = par
    elif kind == "GDmid":
        eta, = par; gamma = 0.0
    while True:
        if gamma == 0.0:
            g, rn = grad_at(th, t)
        else:
            rn = L.vnorm(prob.F(th) - t)
            g, _ = grad_at(th + gamma * u, t)
        done = rn <= stop * s and (k == 0 or L.vnorm(u) <= stop * L.vnorm(th - th_start))
        if done or k >= cap or not math.isfinite(rn) or rn > 1e4 * s:
            break
        if kind == "HB":
            u = mu * u - eta * g
        elif kind == "GDmid":
            gh, _ = grad_at(th - 0.5 * eta * g, t); u = -eta * gh
        elif kind == "ADAM":
            kk += 1
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            u = -lr * (m / (1 - b1 ** kk)) / ((v / (1 - b2 ** kk)).sqrt() + EPS)
        elif kind == "RMSmid":
            kk += 1
            v = b2 * v + (1 - b2) * g * g; den = (v / (1 - b2 ** kk)).sqrt() + EPS
            gh, _ = grad_at(th - 0.5 * lr * g / den, t); u = -lr * gh / den
        th = th + u; k += 1
    out_state = (m, v, kk) if kind == "ADAM" else None
    return th, k, rn / s, out_state

def em_to(th, t):
    """Close a leg with the explicit-midpoint learner (time-symmetric: adds no asymmetry of its own)."""
    F = prob.F(th); J = prob.J(th)
    h = L.jump(prob, th, F + 0.5 * (t - F), Jb=J)
    r = L.jump_a(prob, th, t, Jb=prob.J(h["theta"]))
    assert h["status"] == "CONVERGED" and r["status"] == "CONVERGED", (h["status"], r["status"])
    return r["theta"]

def loop(kind, par, u, s, sg, cap, stop, keep=False):
    t = st.F0 + sg * s * u
    a, k1, r1, state = run(kind, par, st.th0, t, s, cap, stop)
    th1 = em_to(a, t); p1 = L.vnorm(th1 - a)
    b, k2, r2, _ = run(kind, par, th1, st.F0, s, cap, stop, state=state if keep else None)
    th2 = em_to(b, st.F0); p2 = L.vnorm(th2 - b)
    return dict(th1=th1, th2=th2, k=(k1, k2), res=(r1, r2), pol=(p1, p2), n1=L.vnorm(th1 - st.th0), state=state)

def refs(u, s):
    out = {}
    for sg in (1, -1):
        rec, v = S.reference(L, prob, st, u, s, sg); assert S.ref_usable(rec); out[sg] = v
    return out

def jump_loop(u, s, sg):
    t = st.F0 + sg * s * u
    j1 = L.jump(prob, st.th0, t, Jb=st.J0); j2 = L.jump(prob, j1["theta"], st.F0)
    return j1["theta"], j2["theta"]

if stage == "info":
    P_("%s P %d N %d lam1 %.4e lamN %.4e lamN/lam1 %.3e S0 %.4f s1 %.5f s2 %.5f" % (cname, P, N, st.lam1, st.lamN, st.lamN / st.lam1, combo["S0"], s1, s2))
    for nm in ("eigA", "eigB", "eigC", "eigD", "label", "rand"):
        d = ctx.dirs[nm]; qd = float(d @ L.csolve(st.L0, d))
        P_("  dir %-5s info %s  q*lam1 %.4e" % (nm, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ctx.dinfo.get(nm, {}).items()}, qd * st.lam1))
    t0 = time.time(); J = prob.J(st.th0); tJ = time.time() - t0
    t0 = time.time(); g, rn = grad_at(st.th0, st.F0 + s1 * ctx.dirs["label"]); tg = time.time() - t0
    t0 = time.time(); Lc = L.chol(L.gram(J)); tc = time.time() - t0
    P_("  time: Jacobian %.2fs gradient %.4fs gram+cholesky %.2fs" % (tJ, tg, tc))
    j = L.jump(prob, st.th0, st.F0 + s1 * ctx.dirs["label"], Jb=st.J0); dTH = L.vnorm(j["theta"] - st.th0)
    P_("  exact-jump distance on label at s1: %.5e ; sqrt(P) %.1f ; lr for kappa=1: %.3e ; am11 lr 1e-5 = kappa %.4f" % (dTH, math.sqrt(P), dTH / math.sqrt(P), 1e-5 * math.sqrt(P) / dTH))

if stage == "modes":
    u = ctx.dirs["eigA"]; lr_ = ctx.dinfo["eigA"]["lam_ratio"]; s = s1
    ref = refs(u, s); eta = lambda c: c * N / st.lam1
    hbf = lambda c, mu: 1 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))
    gdm = lambda c: 0.5 + c * c / (2 * (c * c - 2 * c + 4))
    CONF = [("GD(0.5)", "HB", (eta(0.5), 0.0, 0.0), 2 * hbf(0.5 * lr_, 0.0) - 1),
            ("GDmid(0.5)", "GDmid", (eta(0.5),), 2 * gdm(0.5 * lr_) - 1),
            ("GDmid(1)", "GDmid", (eta(1.0),), 2 * gdm(1.0 * lr_) - 1),
            ("HB(0.02,0.9)", "HB", (eta(0.02), 0.9, 0.0), 2 * hbf(0.02 * lr_, 0.9) - 1),
            ("HBLA(0.02,0.9,9.5)", "HB", (eta(0.02), 0.9, 9.5), -0.07708),
            ("GDLA(0.1,0.5)", "HB", (eta(0.1), 0.0, 0.5), -0.00529),
            ("GD(0.1)", "HB", (eta(0.1), 0.0, 0.0), 2 * hbf(0.1 * lr_, 0.0) - 1)]
    P_("eigA lam_ratio %.6f ; predictions use c x lam_ratio (look-ahead rows: toy values at lam_ratio 1)" % lr_)
    for name, kind, par, pred in CONF:
        t0 = time.time(); V = {}; info = []
        for sg in (1, -1):
            r = loop(kind, par, u, s, sg, 6000, 1e-6)
            V[sg] = st.V0(r["th2"] - st.th0)
            info.append("steps %d/%d res %.0e/%.0e close/n1 %.1e/%.1e" % (r["k"][0], r["k"][1], r["res"][0], r["res"][1], r["pol"][0] / r["n1"], r["pol"][1] / r["n1"]))
        es = L.even_stats(V[1], ref[1], V[-1], ref[-1])
        P_("%-20s loop rho %+.4f (pred %+.4f) cos %+.4f nr %.4f | %s (%.0fs)" % (name, es["rho"], pred, es["cos"], es["nr"], " ; ".join(info), time.time() - t0))

if stage == "adam":
    u = ctx.dirs["label"]; W = [combo["dirs"]["W%d" % j] for j in range(1, 9)]; probes = [u] + W
    K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, probes); del K0
    kap = lambda k, s, dTH: k * dTH / math.sqrt(P)
    base = {}
    for s, lbl in ((s1, "s1"), (s2, "s2")):
        ref = refs(u, s); dq = {}; dTH = None
        for sg in (1, -1):
            a, b = jump_loop(u, s, sg); dTH = L.vnorm(a - st.th0)
            dq[sg] = [x - y for x, y in zip(S.q(L, prob, b, probes), q0)]
        dqe = [0.5 * (dq[1][i] + dq[-1][i]) for i in range(9)]
        ev = S.even(ref[1], ref[-1]); Muu = 2 * float(ev @ ev) / s ** 2
        Gref = sum(-dqe[i] / q0[i] for i in range(9)) / 9          # change of mean(q0/q) to first order
        base[lbl] = dict(ref=ref, dqe=dqe, Muu=Muu, dTH=dTH, Gref=Gref, s=s)
        P_("%s exact-jump loop: dq_u/M_uu %+.4f ; mean relative inverse-cost change over 9 probes %+.3e ; jump distance %.4e" % (lbl, dqe[0] / Muu, Gref, dTH))
    CONF = [("ADAM", 0.03, 0.9, 0.0, False), ("ADAM", 0.1, 0.9, 0.0, False), ("ADAM", 0.3, 0.9, 0.0, False), ("RMS", 0.1, 0.0, 0.0, False),
            ("ADAMLA", 0.1, 0.9, 9.5, False), ("RMSLA", 0.1, 0.0, 0.5, False), ("ADAMK", 0.1, 0.9, 0.0, True), ("RMS", 0.3, 0.0, 0.0, False),
            ("ADAMLA", 0.03, 0.9, 9.5, False), ("ADAM", 1.0, 0.9, 0.0, False)]
    for name, kappa, b1, gamma, keep in CONF:
        row = {}
        for lbl in ("s1", "s2"):
            bs = base[lbl]; s = bs["s"]; lr = kap(kappa, s, bs["dTH"]); t0 = time.time(); V = {}; V1 = {}; dq = {}; info = []
            for sg in (1, -1):
                r = loop("ADAM", (lr, b1, 0.999, gamma), u, s, sg, 6000, 1e-3, keep=keep)
                V[sg] = st.V0(r["th2"] - st.th0); V1[sg] = st.V0(r["th1"] - st.th0)
                dq[sg] = [x - y for x, y in zip(S.q(L, prob, r["th2"], probes), q0)]
                info.append("k %d/%d res %.0e/%.0e close %.3f/%.3f n1/dTH %.2f" % (r["k"][0], r["k"][1], r["res"][0], r["res"][1], r["pol"][0] / bs["dTH"], r["pol"][1] / bs["dTH"], r["n1"] / bs["dTH"]))
            ev, od = 0.5 * (V[1] + V[-1]), 0.5 * (V[1] - V[-1]); ev1, od1 = 0.5 * (V1[1] + V1[-1]), 0.5 * (V1[1] - V1[-1])
            es = L.even_stats(V[1], bs["ref"][1], V[-1], bs["ref"][-1])
            dqe = [0.5 * (dq[1][i] + dq[-1][i]) for i in range(9)]; dqo = [0.5 * (dq[1][i] - dq[-1][i]) for i in range(9)]
            G = sum(-dqe[i] / q0[i] for i in range(9)) / 9; Go = sum(abs(dqo[i]) / q0[i] for i in range(9)) / 9
            row[lbl] = dict(ne=L.vnorm(ev), no=L.vnorm(od), ne1=L.vnorm(ev1), no1=L.vnorm(od1))
            P_("%-7s kappa %.2f g %.1f %s lr %.2e | loop rho %+.3f cos %+.3f | a_u %+.4f | gauge ratio (9 probes) %+.4f | odd gauge/|even gauge| %.2f | |even| %.3e |odd| %.3e ; single leg vertical |odd|/dTH %.3f |even|/|vNPe| %.2f | %s (%.0fs)" % (
                name, kappa, gamma, lbl, lr, es["rho"], es["cos"], dqe[0] / bs["Muu"], G / bs["Gref"], Go / max(abs(G), 1e-300), row[lbl]["ne"], row[lbl]["no"],
                row[lbl]["no1"] / bs["dTH"], row[lbl]["ne1"] / es["nvNPe"], " ; ".join(info), time.time() - t0))
        ex = lambda key: math.log2(row["s1"][key] / row["s2"][key]) if row["s2"][key] > 0 else float("nan")
        P_("   %-7s kappa %.2f g %.1f scaling exponents s2->s1: loop even %.2f loop odd %.2f ; single-leg even %.2f odd %.2f ; loop |odd|/|even| at s1 %.3f" % (
            name, kappa, gamma, ex("ne"), ex("no"), ex("ne1"), ex("no1"), row["s1"]["no"] / row["s1"]["ne"]))

if stage == "metric":
    u = ctx.dirs["label"]; s = s1
    j = L.jump(prob, st.th0, st.F0 + s * u, Jb=st.J0); dTH = L.vnorm(j["theta"] - st.th0); lr = 0.1 * dTH / math.sqrt(P)
    a, k1, r1, state = run("ADAM", (lr, 0.9, 0.999, 0.0), st.th0, st.F0 + s * u, s, 6000, 1e-3)
    m, v, kk = state; D = 1.0 / ((v / (1 - 0.999 ** kk)).sqrt() + EPS)
    P_("Adam leg: %d steps. D quantiles (min, 1%%, 50%%, 99%%, max): %s ; share of coordinates at the eps cap: %.4f" % (
        k1, ["%.2e" % float(torch.quantile(D[::7], q_)) for q_ in (0.0, 0.01, 0.5, 0.99, 1.0)], float((D > 0.5 / EPS).double().mean())))
    KD = L.gram(st.J0 * D.sqrt()); lam = torch.linalg.eigvalsh(KD)
    lam0 = st.lam
    P_("K_D: lam1 %.3e lamN %.3e cond %.3e ; K: cond %.3e ; normalised spectra (lam_k/lam_1) at k = 1, 10, 32, 100, 320:" % (float(lam[-1]), float(lam[0]), float(lam[-1] / lam[0]), st.lam1 / st.lamN))
    P_("   K_D: %s" % ["%.2e" % float(lam[-k] / lam[-1]) for k in (1, 10, 32, 100, 320)])
    P_("   K  : %s" % ["%.2e" % float(lam0[k - 1] / lam0[0]) for k in (1, 10, 32, 100, 320)])
    sq = D.sqrt()
    probD = L.Problem(F=lambda ph: prob.F(sq * ph), J=lambda ph: prob.J(sq * ph) * sq, N=N, P=P, C=prob.C,
                      fwd=lambda ph: prob.fwd(sq * ph), bwd=lambda cache, r: sq * prob.vjp(cache, r), name="D")
    t0 = time.time(); stD = L.Start(probD, st.th0 / sq)
    P_("Start in D-geometry built (%.0fs): |F0 diff| %.1e" % (time.time() - t0, L.vnorm(stD.F0 - st.F0)))
    out = {}
    for sg in (1, -1):
        t0 = time.time(); rec, vD = S.reference(L, probD, stD, u, s, sg)
        P_("  D-reference sign %+d: usable %s np_status %s (%.0fs)" % (sg, S.ref_usable(rec), rec.get("np_status"), time.time() - t0)); out[sg] = vD
    evD = S.even(out[1], out[-1])
    # exact-jump loop in the D-geometry: must give -1 x vNP_D (covariance control)
    V = {}
    for sg in (1, -1):
        t = stD.F0 + sg * s * u
        j1 = L.jump(probD, stD.th0, t, Jb=stD.J0); j2 = L.jump(probD, j1["theta"], stD.F0)
        V[sg] = stD.V0(j2["theta"] - stD.th0)
    es = L.even_stats(V[1], out[1], V[-1], out[-1])
    P_("covariance control, exact-jump loop in D-geometry: rho %+.4f (pred -1) cos %+.4f" % (es["rho"], es["cos"]))
    # real Adam loop, read in the D-geometry
    V = {}; VE = {}
    for sg in (1, -1):
        r = loop("ADAM", (lr, 0.9, 0.999, 0.0), u, s, sg, 6000, 1e-3)
        V[sg] = stD.V0((r["th2"] - st.th0) / sq); VE[sg] = st.V0(r["th2"] - st.th0)
    es = L.even_stats(V[1], out[1], V[-1], out[-1])
    P_("real ADAM(kappa 0.1) loop read in its own D-geometry: rho %+.4f cos %+.4f nr %.3f" % (es["rho"], es["cos"], es["nr"]))
    refE = refs(u, s); esE = L.even_stats(VE[1], refE[1], VE[-1], refE[-1])
    P_("same loop read in the ordinary geometry: rho %+.4f cos %+.4f nr %.3f" % (esE["rho"], esE["cos"], esE["nr"]))
P_("precheck12 %s done" % stage)
