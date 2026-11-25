"""Healing test (prediction in FINDINGS.md sec. 43): implicit excursions on the worn gelu network."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, os, json, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", 7, OUT + "/heal11_out")
cname, src, npairs = sys.argv[1], sys.argv[2], int(sys.argv[3])
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; sd = combo["sd"]; s = combo["sizes"]["s1"]
Wv = [combo["dirs"][x] for x in S.W_NAMES]; Ld = L.L_AJ(1.0)
K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, Wv); k0 = S.ksum(K0); del K0
STATE = OUT + "/heal11_state_%s.pt" % cname
if os.path.exists(STATE):
    stt = torch.load(STATE, map_location="cpu"); th = stt["th"].to(L.DT); m0 = int(stt["m"]); print("resuming after healing pair", m0, flush=True)
else:
    stt = torch.load(src, map_location="cpu"); th = stt["th"].to(L.DT); m0 = 0; print("starting from worn state after pair", stt["m"], flush=True)
def report(m, th, status, sec, info=""):
    Jm = prob.J(th); K = L.gram(Jm); k = S.ksum(K); q = S.q_from_kernel(L, K, Wv)
    Z = torch.linalg.solve_triangular(torch.linalg.cholesky(K), Jm, upper=False); fp = (Z * Z).sum(0); info = ("leverage sum of squares %.3f " % float((fp * fp).sum())) + info; del K, Jm, Z
    print("healing pair %2d status %-12s lamN/lamN0 %.4f  mean q0/q %.4f  trK %.4f lam1 %.4f A %.4f |theta-theta0| %.4f %s (%.0fs)" % (
        m, status, k["lamN"] / k0["lamN"], sum(a / b for a, b in zip(q0, q)) / 8, k["trK"] / k0["trK"], k["lam1"] / k0["lam1"], k["A"], L.vnorm(th - st.th0), info, sec), flush=True)
if m0 == 0: report(0, th, "worn", 0)
for m in range(m0 + 1, npairs + 1):
    E = S.label_dir(L, st, 8000 + 100 * sd + 20 + m) if m % 2 else S.gauss_dir(L, st, 9000 + 100 * sd + 20 + m)
    t0 = time.time(); status = "OK"; info = ""
    for sigma in S.SIGNS:
        try:
            stc = L.Start(prob, th); leg = S.two_legs(L, prob, stc, st.F0, Ld, E, s, sigma, ""); del stc
            if leg["rec"]["status"] != "OK": status = leg["rec"]["status"]; info = str((leg["rec"]["leg1_status"], leg["rec"]["leg2_status"], leg["rec"]["close"])); break
            th = leg["th2"]
        except Exception as e:
            status = "EXC:" + type(e).__name__; info = repr(e)[:100]; break
        L.free_memory()
    report(m, th, status, time.time() - t0, info)
    if status != "OK": break
    torch.save(dict(th=th.detach().cpu(), m=m), STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
print("heal11 done", flush=True)
