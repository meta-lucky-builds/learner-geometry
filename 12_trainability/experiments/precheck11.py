"""Pre-check for SPEC11 (note, before the spec): out-and-back excursion, real MNIST n = 32.
Predictions: loop displacement v = V0(theta2 - theta0) = (2f - 1) vNP;  change of q_u = u^T K^-1 u:  dq = (1 - 2f) 2 |vNP|^2 / s^2;
identity: derivative of q_u along vNP_hat = -2 |vNP| / s^2;  cross: dq_w = (1 - 2f) 2 <vNP_w, vNP_u> / s^2."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=3)
want, dname, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
LS = [("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(2)", L.L_SL(2), 0.25), ("SL(8)", L.L_SL(8), 0.4375), ("AJ(0.5)", L.L_AJ(0.5), 0.5), ("AJ(1)", L.L_AJ(1.0), 1.0)]
def q_of(prob, th, vecs):
    K = L.gram(prob.J(th)); Lc = L.chol(K); return [float(w @ L.csolve(Lc, w)) for w in vecs]
for cb in [c for c in L.combos("A") if L.cname9(c) == want]:
    ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; u = ctx.dirs[dname]; s = frac * ctx.S[dname]
    wname = "rand" if dname != "rand" else "label"; w = ctx.dirs[wname]
    q0 = q_of(prob, st.th0, [u, w]); ref = {}
    for sg in (1, -1):                                   # references: nearest-point shift of a single leg along u and along w
        for nm, d_ in ((dname, u), (wname, w)):
            thT = L.jump(prob, st.th0, st.F0 + sg * s * d_, Jb=st.J0)["theta"]
            sc, vL, vNP, thNP, thTH = L.measure_ext(prob, st, thT); ref[(nm, sg)] = vNP
            print("reference %s sign %+d: np %s |vNP| %.4e |vNP|/|dNP| %.3e rho(TH) %.2e" % (nm, sg, sc["np_status"], sc["nvNP"], sc["vNP_over_dNP"], sc["rho"]), flush=True)
    vu = 0.5 * (ref[(dname, 1)] + ref[(dname, -1)]); vw = 0.5 * (ref[(wname, 1)] + ref[(wname, -1)])
    nu = L.vnorm(vu); print("even |vNP_u| %.4e  cos(vNP_u, vNP_w) %+.4f  cos(vNP_u(+), vNP_u(-)) %+.5f" % (nu, float(vu @ vw) / (nu * L.vnorm(vw)), L.cosine(ref[(dname, 1)], ref[(dname, -1)])))
    # identity check by central difference along vNP_hat
    vh = vu / nu; eps = 0.5 * nu
    qp = q_of(prob, st.th0 + eps * vh, [u])[0]; qm = q_of(prob, st.th0 - eps * vh, [u])[0]
    print("identity: dq_u/d(vNP_hat) measured %.6e   predicted -2|vNP|/s^2 = %.6e   ratio %.4f" % ((qp - qm) / (2 * eps), -2 * nu / s ** 2, (qp - qm) / (2 * eps) / (-2 * nu / s ** 2)), flush=True)
    for name, Ld, f in LS:
        t0 = time.time(); V, DQ, stat = {}, {}, []
        for sg in (1, -1):
            r1 = L.run_learner9(prob, st, st.F0 + sg * s * u, s, Ld); st1 = L.Start(prob, r1["theta"])
            r2 = L.run_learner9(prob, st1, st.F0, s, Ld); th2 = r2["theta"]; d = th2 - st.th0
            V[sg] = st.V0(d); q2 = q_of(prob, th2, [u, w]); DQ[sg] = (q2[0] - q0[0], q2[1] - q0[1])
            stat.append("%s/%s close %.1e horiz %.1e" % (r1["status"], r2["status"], L.vnorm(prob.F(th2) - st.F0) / s, L.vnorm(d - V[sg]) / max(L.vnorm(d), 1e-300)))
            del st1; L.free_memory()
        es = L.even_stats(V[1], ref[(dname, 1)], V[-1], ref[(dname, -1)])
        dq = 0.5 * (DQ[1][0] + DQ[-1][0]); dqw = 0.5 * (DQ[1][1] + DQ[-1][1])
        print("%-8s f %.4f | loop: rho %+.4f (pred %+.4f) cos %+.5f nr %.4f | dq_u/(2|vNP|^2/s^2) %+.4f (pred %+.4f) | dq_w/(2<vNP_w,vNP_u>/s^2) %+.4f (pred %+.4f) | %s (%.0fs)" % (
            name, f, es["rho"], 2 * f - 1, es["cos"], es["nr"], dq / (2 * nu * nu / s ** 2), 1 - 2 * f, dqw / (2 * float(vu @ vw) / s ** 2), 1 - 2 * f, " ; ".join(stat), time.time() - t0), flush=True)
print("precheck done")
