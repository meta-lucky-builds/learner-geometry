"""Two-gradient explicit learners with f = 1/2 (prediction in FINDINGS.md sec. 43). Uses the SPEC11 bundle code for references and q."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/em11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; u = combo["dirs"]["E1"]
def em(prob, st_, t):
    h = L.jump(prob, st_.th0, st_.F0 + 0.5 * (t - st_.F0), Jb=st_.J0)
    r = L.jump_a(prob, st_.th0, t, Jb=prob.J(h["theta"])); return r["theta"], h["status"] + "/" + r["status"]
def heun(prob, st_, t):
    p = L.jump(prob, st_.th0, t, Jb=st_.J0)
    r = L.jump_a(prob, st_.th0, t, Jb=0.5 * (st_.J0 + prob.J(p["theta"]))); return r["theta"], p["status"] + "/" + r["status"]
def sl2(prob, st_, t):
    r = L.run_learner9(prob, st_, t, L.vnorm(t - st_.F0), L.L_SL(2)); return r["theta"], r["status"]
K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, [u])[0]; del K0
for size in ("s1", "s2"):
    s = combo["sizes"][size]; ref = {}
    for sg in (1, -1):
        rec, v = S.reference(L, prob, st, u, s, sg); ref[sg] = v; assert S.ref_usable(rec), rec
    ev = S.even(ref[1], ref[-1]); Muu = 2 * float(ev @ ev) / s ** 2
    for name, fn in (("SL(2)", sl2), ("EM", em), ("HEUN", heun)):
        t0 = time.time(); V, V1, DQ, stt = {}, {}, {}, []
        for sg in (1, -1):
            th1, a = fn(prob, st, st.F0 + sg * s * u); st1 = L.Start(prob, th1)
            th2, b = fn(prob, st1, st.F0); del st1
            V[sg] = st.V0(th2 - st.th0); V1[sg] = st.V0(th1 - st.th0); DQ[sg] = S.q(L, prob, th2, [u])[0] - q0
            stt.append("%s|%s close %.1e" % (a, b, L.vnorm(prob.F(th2) - st.F0) / s)); L.free_memory()
        es = L.even_stats(V[1], ref[1], V[-1], ref[-1]); e1 = L.even_stats(V1[1], ref[1], V1[-1], ref[-1])
        print("%s %s %-6s single-leg fraction %+.4f | loop: rho %+.5f  |v|/|vNP| %.5f | own cost dq_u/M_uu %+.5f | %s (%.0fs)" % (
            cname, size, name, e1["rho"], es["rho"], es["nr"], 0.5 * (DQ[1] + DQ[-1]) / Muu, " ; ".join(stt), time.time() - t0), flush=True)
print("em11 done", flush=True)
