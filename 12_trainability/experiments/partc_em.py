"""SPEC11 Part C protocol with the explicit-midpoint learner EMS(m) (prediction in FINDINGS.md sec. 43)."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, json, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/partc_em_out")
cname = sys.argv[1]; ms = [int(x) for x in sys.argv[3].split(",")]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]
Wv = [combo["dirs"][x] for x in S.W_NAMES]
def em(prob, st_, t):
    h = L.jump(prob, st_.th0, st_.F0 + 0.5 * (t - st_.F0), Jb=st_.J0)
    r = L.jump_a(prob, st_.th0, t, Jb=prob.J(h["theta"])); return r["theta"], (h["status"], r["status"])
def ems(prob, th, t, m):
    stc = L.Start(prob, th); F0 = stc.F0; stat = []
    for k in range(1, m + 1):
        th, s_ = em(prob, stc, F0 + (k / m) * (t - F0)); stat.extend(s_)
        if k < m: stc = L.Start(prob, th)
    return th, all(x == "CONVERGED" for x in stat)
out = open(OUT + "/partc_em_%s.jsonl" % cname, "a")
for m in ms:
    th = st.th0; Qs = []; ok = True; t0 = time.time(); extra = ""
    for task in range(1, 7):
        Y = S.task_target(L, combo, task); s_m = L.vnorm(Y - prob.F(th))
        try:
            th, conv = ems(prob, th, Y, m)
        except Exception as e:
            ok = False; extra = "EXC:" + type(e).__name__ + " at task %d" % task; break
        fit = L.vnorm(prob.F(th) - Y) / s_m
        if not conv or not (fit <= 1e-6): ok = False; extra = "not converged / fit %.1e at task %d" % (fit, task); break
        K = S.kernel(L, prob, th); q = S.q_from_kernel(L, K, Wv); k = S.ksum(K); del K
        Qs.append(sum(math.log(x) for x in q) / 8); last = dict(A=k["A"], lamN=k["lamN"], lam1=k["lam1"], trK=k["trK"], dist=L.vnorm(th - st.th0)); L.free_memory()
    rec = dict(combo=cname, m=m, ok=ok, Q=Qs, note=extra, sec=time.time() - t0)
    if ok:
        stf = L.Start(prob, th); rf = L.run_learner9(prob, stf, combo["Y"], L.vnorm(combo["Y"] - stf.F0), L.L_SL(8)); td = combo["test"]()
        loss, acc = L.test_loss_acc(L.test_outputs(prob, td["X"], rf["theta"]), td); rec.update(last, final_loss=loss, final_acc=acc, final_status=rf["status"])
    out.write(json.dumps(rec) + "\n"); out.flush()
    print("%s EMS(%d) [cost of SL(%d)] ok %s %s | Q per task: %s | %s (%.0fs)" % (cname, m, 2 * m, ok, extra, " ".join("%.3f" % x for x in Qs),
          ("A6 %.4f lamN6 %.4f lam1 %.0f trK %.0f dist %.3f | final test loss %.5f acc %.4f" % (rec["A"], rec["lamN"], rec["lam1"], rec["trK"], rec["dist"], rec["final_loss"], rec["final_acc"])) if ok else "", rec["sec"]), flush=True)
print("partc_em done", flush=True)
