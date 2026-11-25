"""Explicit predicted-end-point learner EP(alpha): loop law on a fresh network, then healing of the worn gelu network."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, os, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
mode, cname, thr = sys.argv[1], sys.argv[2], int(sys.argv[3])
L, CFG = S.load(B, "cpu", thr, OUT + "/ep11_out")
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; sd = combo["sd"]; s = combo["sizes"]["s1"]
def ep(prob, st_, t, alpha):
    p = L.jump(prob, st_.th0, t, Jb=st_.J0)
    r = L.jump_a(prob, st_.th0, t, Jb=prob.J(st_.th0 + alpha * (p["theta"] - st_.th0))); return r["theta"], p["status"] == "CONVERGED" and r["status"] == "CONVERGED"
if mode == "loop":
    u = combo["dirs"]["E1"]; ref = {}
    for sg in (1, -1):
        rec, v = S.reference(L, prob, st, u, s, sg); ref[sg] = v
    for alpha in [float(x) for x in sys.argv[4].split(",")]:
        t0 = time.time(); V = {}; V1 = {}; ok = True
        for sg in (1, -1):
            th1, a = ep(prob, st, st.F0 + sg * s * u, alpha); st1 = L.Start(prob, th1); th2, b = ep(prob, st1, st.F0, alpha); del st1; ok &= a and b
            V[sg] = st.V0(th2 - st.th0); V1[sg] = st.V0(th1 - st.th0)
        es = L.even_stats(V[1], ref[1], V[-1], ref[-1]); e1 = L.even_stats(V1[1], ref[1], V1[-1], ref[-1])
        print("%s EP(%g) converged %s | single-leg fraction %+.4f (predicted %g) | loop rho %+.5f (predicted %+g) cos %+.5f (%.0fs)" % (cname, alpha, ok, e1["rho"], alpha, es["rho"], 2 * alpha - 1, es["cos"], time.time() - t0), flush=True)
else:
    alpha, npairs, src = float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    Wv = [combo["dirs"][x] for x in S.W_NAMES]
    K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, Wv); k0 = S.ksum(K0); del K0
    STATE = OUT + "/ep11_state_%s_%g.pt" % (cname, alpha)
    if os.path.exists(STATE): stt = torch.load(STATE, map_location="cpu"); th = stt["th"].to(L.DT); m0 = int(stt["m"]); print("resuming after pair", m0, flush=True)
    else: th = torch.load(src, map_location="cpu")["th"].to(L.DT); m0 = 0
    def report(m, th, status, sec):
        Jm = prob.J(th); K = L.gram(Jm); k = S.ksum(K); q = S.q_from_kernel(L, K, Wv)
        Z = torch.linalg.solve_triangular(torch.linalg.cholesky(K), Jm, upper=False); fp = (Z * Z).sum(0); lam = torch.linalg.eigvalsh(K)
        print("EP(%g) healing pair %2d %-8s lamN/lamN0 %.4f  four smallest %s  mean q0/q %.4f  trK %.4f lam1 %.4f A %.4f dist %.4f leverage sum of squares %.3f (%.0fs)" % (
            alpha, m, status, k["lamN"] / k0["lamN"], " ".join("%.4f" % (float(x) / k0["lamN"]) for x in lam[:4]), sum(a / b for a, b in zip(q0, q)) / 8, k["trK"] / k0["trK"], k["lam1"] / k0["lam1"], k["A"],
            L.vnorm(th - st.th0), float((fp * fp).sum()), sec), flush=True)
    if m0 == 0: report(0, th, "worn", 0)
    for m in range(m0 + 1, npairs + 1):
        E = S.label_dir(L, st, 8000 + 100 * sd + 20 + m) if m % 2 else S.gauss_dir(L, st, 9000 + 100 * sd + 20 + m)
        t0 = time.time(); status = "OK"
        for sg in (1, -1):
            try:
                stc = L.Start(prob, th); th1, a = ep(prob, stc, st.F0 + sg * s * E, alpha); st1 = L.Start(prob, th1); th2, b = ep(prob, st1, st.F0, alpha); del stc, st1
                cl = L.vnorm(prob.F(th2) - st.F0) / s
                if not (a and b) or not (cl <= 1e-6): status = "FAILED conv %s/%s close %.1e" % (a, b, cl); break
                th = th2
            except Exception as e:
                status = "EXC:" + type(e).__name__; break
            L.free_memory()
        report(m, th, status, time.time() - t0)
        if status != "OK": break
        torch.save(dict(th=th.detach().cpu(), m=m), STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
print("ep11 done", flush=True)
