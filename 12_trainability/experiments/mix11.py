"""Mixed-learner loops (prediction in FINDINGS.md sec. 43): out with learner A, back with learner B."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/mix11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; u = combo["dirs"]["E1"]; s = combo["sizes"]["s1"]
ref = {}
for sg in (1, -1):
    rec, v = S.reference(L, prob, st, u, s, sg); ref[sg] = v
PAIRS = [(("SL", 1), ("AJ", 1.0), 0.0), (("DJ", 1.0), ("AJ", 1.0), 0.0), (("AJ", 1.0), ("DJ", 1.0), 0.0), (("SL", 2), ("AJ", 1.0), 0.25), (("AJ", 1.0), ("SL", 8), 0.4375)]
for a, b, pred in PAIRS:
    La, Lb = S.make_learner(L, a), S.make_learner(L, b); t0 = time.time(); V = {}; stt = []
    for sg in (1, -1):
        r1 = L.run_learner9(prob, st, st.F0 + sg * s * u, s, La); st1 = L.Start(prob, r1["theta"])
        r2 = L.run_learner9(prob, st1, st.F0, s, Lb); del st1
        V[sg] = st.V0(r2["theta"] - st.th0); stt.append("%s/%s close %.1e" % (r1["status"], r2["status"], L.vnorm(prob.F(r2["theta"]) - st.F0) / s)); L.free_memory()
    es = L.even_stats(V[1], ref[1], V[-1], ref[-1])
    print("%s out %-6s back %-6s predicted %+.4f measured rho_loop %+.5f cos %+.4f |v|/|vNP| %.5f | %s (%.0fs)" % (
        cname, L.learner_name9(La), L.learner_name9(Lb), pred, es["rho"], es["cos"], es["nr"], " ; ".join(stt), time.time() - t0), flush=True)
print("mix11 done", flush=True)
