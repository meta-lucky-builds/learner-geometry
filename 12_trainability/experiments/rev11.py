"""Reversible learning: learn 3 tasks with exact jumps, undo with the implicit jump given stored outputs only."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/rev11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; s = 0.03 * combo["S0"]
targets = [st.F0 + s * S.label_dir(L, st, 8000 + 30 + k) for k in (1, 2, 3)]
th = st.th0; stored = []; path = [th]
for t in targets:
    stored.append(prob.F(th).clone()); stc = L.Start(prob, th); r = L.jump(prob, th, t, Jb=stc.J0); assert r["status"] == "CONVERGED"; th = r["theta"]; path.append(th)
th3 = th; d3 = L.vnorm(th3 - st.th0); print("%s after 3 tasks: |theta - theta0| = %.5e" % (cname, d3), flush=True)
for name, Ld in (("implicit jump AJ(1) (time-reverse)", L.L_AJ(1.0)), ("exact jump SL(1) (same learner)", L.L_SL(1))):
    th = th3; t0 = time.time(); ok = True
    for k in (2, 1, 0):
        stc = L.Start(prob, th); r = L.run_learner9(prob, stc, stored[k], L.vnorm(stored[k] - stc.F0), Ld); ok &= r["status"] == "CONVERGED"; th = r["theta"]
        print("   undo task %d with %s: status %s, distance to the weights before that task / d3 = %.3e" % (k + 1, name.split()[0] + " " + name.split()[1], r["status"], L.vnorm(th - path[k]) / d3), flush=True)
    print("%s undo with %s: converged %s | |theta_end - theta0| / |theta_3 - theta0| = %.3e | output error %.1e (%.0fs)" % (cname, name, ok, L.vnorm(th - st.th0) / d3, L.vnorm(prob.F(th) - st.F0) / s, time.time() - t0), flush=True)
print("rev11 done", flush=True)
