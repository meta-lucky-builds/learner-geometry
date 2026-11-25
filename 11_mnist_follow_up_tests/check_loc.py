import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, glob, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load()
cb = [c for c in L.combos("G4") if c["act"] == "gelu" and c["seed"] == 0 and c["n"] == 32][0]
t0 = time.time(); ctx = L.Ctx9(cb, loc.RD); print("Ctx9 (start, one Jacobian, eigen) %.0f s" % (time.time() - t0), flush=True)
F0t, l0, a0 = ctx.test0(); print("test loss at theta0 %.6f acc %.4f  (rig: 0.051824, 0.0499)" % (l0, a0), flush=True)
t0 = time.time(); J = ctx.prob.J(ctx.st.th0); print("one Jacobian %.1f s, shape %s" % (time.time() - t0, tuple(J.shape)), flush=True); del J
tau = 0.1; st = ctx.st; t = st.F0 + tau * ctx.Y; s = L.vnorm(t - st.F0)
t0 = time.time(); res = L.run_learner9(ctx.prob, st, t, s, L.L_DJ(1.0)); td = ctx.test()
out = L.test_outputs(ctx.prob, td["X"], res["theta"]) - F0t; lz, az = L.test_loss_acc(out, td)
rec = json.load(open(DATA + "/keep/spec10_out/records_2.json"))
e = [e for b in rec["combos"]["gelu-init-s0"]["blocks"] if abs(b["tau"] - 0.1) < 1e-9 for e in b["ZERO"]["ends"] if e["learner"] == "DJ(1)"][0]
print("ZERO DJ(1) tau 0.1: local loss %.10f acc %.4f | rig %.10f %.4f | rel diff %.1e  (%.0f s)" % (lz, az, e["loss"], e["acc"], abs(lz - e["loss"]) / e["loss"], time.time() - t0))
