"""Why does the scalar form (HR2) fail on ReLU? Measure every link of the chain on the case found in the cross-check:
small ReLU MLP, eigA, s = 1e-2 S0, sign -1, DJ(0.5). Also the smooth control (small tanh MLP, same setting)."""
import sys, math
from common9 import load
L, CFG = load("tiny")
import torch
vn = L.vnorm
for act in ("relu", "tanh"):
    sm = L._small_mlp(act)
    prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
    s = 1e-2 * S0
    d = -sm["dirs"]["eigA"]
    t = st.F0 + s * d
    ref = L.RefPath(prob, st, d, s, 64)
    vPP, om, omp = L.R_profile(ref)
    n2 = float(vPP @ vPP)
    lam, Q = st.lam, st.Q
    print("==== %s: lam1 %.3g lamN %.3g cond %.3g | switches on [0,1]: %d cells, n_sw(1) %d | omega(1/16) %.3f" % (
        act, float(lam[0]), float(lam[-1]), float(lam[0] / lam[-1]), sum(ref.cell_change[:32]), ref.nsw[32], om[1]))
    for c in (0.5,):
        log = []
        res = L.run_DJ_logged(prob, st, t, c, log, keep_points=True)
        vL = st.V0(res["theta"] - st.th0)
        print("  DJ(%g): rho_PP %.4f" % (c, float(vL @ vPP) / n2))
        tot_on = tot_off = 0.0
        print("   k   tau_k   beta_k  |b_perp|/|b|  on-line part  off-line part   |K^-1 b_perp|/|K^-1 b_on|   share of b_perp energy in the softest half of the spectrum   pattern differs from start (learner / reference at same tau)")
        p0 = L.preact_pattern(prob, st.th0)
        for k, it in enumerate(log[:6]):
            Fe, b = it["Fe"], it["b"]
            tau = float(d @ (Fe - st.F0)) / s
            beta = float(d @ b) / s
            b_on = beta * s * d
            b_off = b - b_on
            X, _ = ref.W(tau, torch.stack([b_on, b_off], 1))
            on = float(X[:, 0] @ vPP) / n2
            off = float(X[:, 1] @ vPP) / n2
            tot_on += on
            tot_off += off
            thr, _ = ref.pos(tau)
            Jr = prob.J(thr)
            Kr = L.gram(Jr)
            a_on = torch.linalg.solve(Kr, b_on)
            a_off = torch.linalg.solve(Kr, b_off)
            co = Q.T @ b_off
            soft = float((co[len(co) // 2:] ** 2).sum() / (co ** 2).sum()) if vn(b_off) > 0 else float("nan")
            nl = int((L.preact_pattern(prob, it["the"]) != p0).sum())
            nr = int((L.preact_pattern(prob, thr) != p0).sum())
            print("  %2d  %6.3f  %7.4f   %9.2e   %+10.5f   %+10.5f        %9.2f                        %6.3f                                   %d / %d" % (
                k, tau, beta, vn(b_off) / max(vn(b), 1e-300), on, off, vn(a_off) / max(vn(a_on), 1e-300), soft, nl, nr))
        # all steps
        on_all = off_all = 0.0
        for it in log:
            Fe, b = it["Fe"], it["b"]
            tau = float(d @ (Fe - st.F0)) / s
            beta = float(d @ b) / s
            X, _ = ref.W(tau, torch.stack([beta * s * d, b - beta * s * d], 1))
            on_all += float(X[:, 0] @ vPP) / n2
            off_all += float(X[:, 1] @ vPP) / n2
        print("  sum over all %d steps: on-line (= rho_omega) %.4f + off-line (= rho_perp) %.4f = rho_field %.4f" % (len(log), on_all, off_all, on_all + off_all))
        # the first step: how far off the line does the linearised step land, and is it the kink?
        th1 = log[1]["the"]
        F1 = log[1]["Fe"]
        lin = st.F0 + c * s * d
        e = F1 - lin
        print("  first step: linear prediction F0 + c s d; actual landing error |e|/(c s) = %.3e ; e along d: %.3e" % (vn(e) / (c * s), float(d @ e) / (c * s)))
