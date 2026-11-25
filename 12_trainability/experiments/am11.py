"""a-meter for ordinary optimisers (prediction in FINDINGS.md sec. 43)."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/am11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; u = combo["dirs"]["E1"]; s = combo["sizes"]["s1"]; N = prob.N
ref = {}
for sg in (1, -1):
    rec, v = S.reference(L, prob, st, u, s, sg); ref[sg] = v
K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, [u])[0]; del K0
ev = S.even(ref[1], ref[-1]); Muu = 2 * float(ev @ ev) / s ** 2
def grad(th, t):
    F, cache = prob.fwd(th); r = F - t; return prob.vjp(cache, r) / N, L.vnorm(r)
def run(kind, par, th0, t, cap, stop=1e-3):
    th = th0.clone(); k = 0; m = torch.zeros_like(th); v = torch.zeros_like(th); b1, b2, eps = 0.9, 0.999, 1e-8
    def adam_step(g, m, v, k):
        m2 = b1 * m + (1 - b1) * g; v2 = b2 * v + (1 - b2) * g * g
        return -par * (m2 / (1 - b1 ** (k + 1))) / ((v2 / (1 - b2 ** (k + 1))).sqrt() + eps), m2, v2
    while True:
        g, rn = grad(th, t)
        if rn <= stop * s or k >= cap or not math.isfinite(rn): break
        if kind == "GD": th = th - par * g
        elif kind == "GDmid":
            gh, _ = grad(th - 0.5 * par * g, t); th = th - par * gh
        elif kind == "ADAM":
            d, m, v = adam_step(g, m, v, k); th = th + d
        elif kind == "ADAMmid":
            d1, _, _ = adam_step(g, m, v, k); gh, _ = grad(th + 0.5 * d1, t); d, m, v = adam_step(gh, m, v, k); th = th + d
        k += 1
    return th, k, rn / s
eta = lambda c: c * N / st.lam1
CONF = [("GD", eta(1.0), "GD(1)", 20000), ("GDmid", eta(1.0), "GDmid(1)", 20000), ("GD", eta(1.8), "GD(1.8)", 20000), ("GDmid", eta(1.8), "GDmid(1.8)", 20000),
        ("ADAM", 1e-5, "ADAM(1e-5)", 6000), ("ADAMmid", 1e-5, "ADAMmid(1e-5)", 6000), ("ADAM", 1e-4, "ADAM(1e-4)", 6000), ("ADAMmid", 1e-4, "ADAMmid(1e-4)", 6000)]
for kind, par, name, cap in CONF:
    t0 = time.time(); V = {}; V1 = {}; DQ = {}; info = []
    for sg in (1, -1):
        th1, k1, r1 = run(kind, par, st.th0, st.F0 + sg * s * u, cap); th2, k2, r2 = run(kind, par, th1, st.F0, cap)
        V[sg] = st.V0(th2 - st.th0); V1[sg] = st.V0(th1 - st.th0); DQ[sg] = S.q(L, prob, th2, [u])[0] - q0; info.append("steps %d/%d res %.1e/%.1e" % (k1, k2, r1, r2))
    es = L.even_stats(V[1], ref[1], V[-1], ref[-1]); e1 = L.even_stats(V1[1], ref[1], V1[-1], ref[-1])
    print("%s %-14s single-leg fraction %+.4f (cos %+.3f) | loop rho %+.4f cos %+.4f -> time asymmetry a = %+.4f | own cost dq_u/M_uu %+.4f | %s (%.0fs)" % (
        cname, name, e1["rho"], e1["cos"], es["rho"], es["cos"], -es["rho"], 0.5 * (DQ[1] + DQ[-1]) / Muu, " ; ".join(info), time.time() - t0), flush=True)
print("am11 done", flush=True)
