"""Pre-check of the LOOK-MATRIX LAW on a generic output direction (prediction PRE-5 in FINDINGS.md sec. 45).
Loop displacement of any learner whose weight change is a sum of gradients J(theta_j)^T c_j:
   d = LAW(C),  C = K0 sum_j c_j E_j^T,  E_j = F(look point j) - F0   (payment x look position).
Usage: python precheck12b.py <combo> <threads>"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
cname, threads = sys.argv[1], int(sys.argv[2])
L, CFG = S.load(B, "cpu", threads, OUT + "/pre12_out")
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx)
prob, st = combo["prob"], combo["st"]; N, P = prob.N, prob.P; s = combo["sizes"]["s1"]; u = ctx.dirs["label"]
def P_(*a): print(*a, flush=True)

def leg(th_start, T, eta, mu, gamma, cap, stop, Cacc):
    """Heavy-ball family with look-ahead; accumulates the look matrix (relative to the ORIGINAL start outputs st.F0) into Cacc."""
    th = th_start.clone(); uu = torch.zeros_like(th); k = 0
    A1 = torch.zeros(N, N, dtype=L.DT); T2 = torch.zeros(N, N, dtype=L.DT)
    while True:
        Fk = prob.F(th); rn = L.vnorm(Fk - T)
        if (rn <= stop * s and (k == 0 or L.vnorm(uu) <= stop * L.vnorm(th - th_start))) or k >= cap:
            break
        Fy, cache = prob.fwd(th + gamma * uu); r = Fy - T
        g = prob.vjp(cache, r) / N
        o = torch.outer(r, Fy - st.F0); A1 += o; T2 = mu * (T2 + o)
        uu = mu * uu - eta * g; th = th + uu; k += 1
    Cacc += -(eta / N) / (1 - mu) * (st.K0 @ (A1 - T2))
    # closing with the explicit-midpoint learner: one applied look at the half-way outputs, paying the whole remainder
    F = prob.F(th); J = prob.J(th)
    h = L.jump(prob, th, F + 0.5 * (T - F), Jb=J); r2 = L.jump_a(prob, th, T, Jb=prob.J(h["theta"]))
    assert h["status"] == "CONVERGED" and r2["status"] == "CONVERGED"
    Cacc += torch.outer(T - F, F + 0.5 * (T - F) - st.F0)
    return r2["theta"], k, rn / s, L.vnorm(r2["theta"] - th)

eta = lambda c: c * N / st.lam1
ref = {}
for sg in (1, -1):
    rec, v = S.reference(L, prob, st, u, s, sg); ref[sg] = v
vNPe = S.even(ref[1], ref[-1])
for name, c, mu, gamma, cap in (("HB(0.02,0.9)", 0.02, 0.9, 0.0, 6000), ("HBLA(0.02,0.9,9.5)", 0.02, 0.9, 9.5, 6000), ("GD(1)", 1.0, 0.0, 0.0, 6000)):
    t0 = time.time(); V = {}; Cs = {}; info = []
    for sg in (1, -1):
        Cacc = torch.zeros(N, N, dtype=L.DT); T = st.F0 + sg * s * u
        th1, k1, r1, p1 = leg(st.th0, T, eta(c), mu, gamma, cap, 1e-3, Cacc)
        th2, k2, r2, p2 = leg(th1, st.F0, eta(c), mu, gamma, cap, 1e-3, Cacc)
        V[sg] = st.V0(th2 - st.th0); Cs[sg] = Cacc; n1 = L.vnorm(th1 - st.th0)
        info.append("steps %d/%d res %.0e/%.0e close/n1 %.3f/%.3f" % (k1, k2, r1, r2, p1 / n1, p2 / n1))
    de = 0.5 * (V[1] + V[-1]); Ce = 0.5 * (Cs[1] + Cs[-1])
    sym = 0.5 * (Ce + Ce.T); asym = 0.5 * (Ce - Ce.T)
    t1 = time.time(); pred, li = L.law(prob, st, Ce); tl = time.time() - t1
    cos = float(de @ pred) / (L.vnorm(de) * L.vnorm(pred)); rho = float(de @ pred) / float(pred @ pred)
    cnp = float(de @ vNPe) / (L.vnorm(de) * L.vnorm(vNPe)); rnp = float(de @ vNPe) / float(vNPe @ vNPe)
    P_("%-20s measured vs LAW(look matrix): ratio %+.4f cos %+.4f | |d|/|vNPe| %.4f ; projection on vNPe: rho %+.4f cos %+.3f | u.C.u/s^2 %+.4f |sym| %.3e |antisym| %.3e | law terms %d kept_frac %.6f (%.0fs) | %s (%.0fs)" % (
        name, rho, cos, L.vnorm(de) / L.vnorm(vNPe), rnp, cnp, float(u @ Ce @ u) / s ** 2, L.vnorm(sym), L.vnorm(asym), li["kept"], li["kept_frac"], tl, " ; ".join(info), time.time() - t0))
P_("precheck12b done")
