from common9 import load
L, CFG = load("quick")
import torch
ok, msg = L.t_V5()
print("V5 as delivered (quick config):", "PASS" if ok else "FAIL"); print("   ", msg[-520:])
ok7, msg7 = L.t_V7(); print("V7:", "PASS" if ok7 else "FAIL", msg7[:140])
# exercise the AJ-FAILED branch: emulate what the GPU saw (AJ-type learners fail on the ReLU net at 1e-2 S0)
orig = L.run_logged
sm = L._small_mlp("relu")
calls = {"n": 0}
def fake(prob, st, t, Ls, log, keep_points=False):
    big = float(L.vnorm(t - st.F0)) > 5e-3 * sm["S0"]
    if prob is sm["prob"] and Ls["kind"] in ("AJ", "SLa") and big:
        calls["n"] += 1
        r = orig(prob, st, t, Ls, log, keep_points)
        if Ls["kind"] == "AJ":
            del log[:]                                   # as on the GPU: 0 steps logged
        else:
            del log[3:]                                  # as on the GPU: failed at sub-jump 3 -> 3 steps logged... keep 3
        r = dict(r, status="AJ-FAILED", status_supp="AJ-FAILED", aj_why="forced in the review (emulating the GPU)")
        return r
    return orig(prob, st, t, Ls, log, keep_points)
L.run_logged = fake
ok2, msg2 = L.t_V5()
print("V5 with AJ-type learners forced to AJ-FAILED on the ReLU net at 1e-2 S0 (%d forced):" % calls["n"], "PASS" if ok2 else "FAIL")
i = msg2.find("ReLU 196-32-32-10 eigA AJ(0.5)"); print("   ", msg2[i:i + 620])
# and a forced WRONG status must fail
def fake2(prob, st, t, Ls, log, keep_points=False):
    r = orig(prob, st, t, Ls, log, keep_points)
    if prob is sm["prob"] and Ls["kind"] == "AJ" and float(L.vnorm(t - st.F0)) > 5e-3 * sm["S0"]:
        r = dict(r, status="AJ-INCONSISTENT")
    return r
L.run_logged = fake2
ok3, _ = L.t_V5(); print("V5 with a forced AJ-INCONSISTENT status must FAIL:", "FAIL (correct)" if not ok3 else "PASS (WRONG)")
# and a corrupted log must fail
def fake3(prob, st, t, Ls, log, keep_points=False):
    r = orig(prob, st, t, Ls, log, keep_points)
    if prob is sm["prob"] and Ls["kind"] == "DJ":
        log[1]["b"] = log[1]["b"] * 1.001
    return r
L.run_logged = fake3
ok4, _ = L.t_V5(); print("V5 with one logged step scaled by 1.001 must FAIL:", "FAIL (correct)" if not ok4 else "PASS (WRONG)")
