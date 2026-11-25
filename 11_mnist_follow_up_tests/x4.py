import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=2)
def kr(prob, th, Xt, tgt, eps=1e-3):
    """Kernel regression prediction with the tangent kernel at th for training target change tgt: J_x(th) J(th)^T K(th)^-1 tgt (JVP by central differences)."""
    J = prob.J(th); a = torch.linalg.solve(J @ J.T, tgt); v = J.T @ a
    return (L.test_outputs(prob, Xt, th + eps * v) - L.test_outputs(prob, Xt, th - eps * v)) / (2 * eps)
for cb in [c for c in L.combos("G4") if c["n"] == 32 and (c["act"], c["seed"]) in (("tanh", 1), ("gelu", 1))]:
    cn = L.cname9(cb); ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; td = ctx.test(); F0t, _, _ = ctx.test0()
    for tau in (0.03, 0.1):
        tgt = tau * ctx.Y; t = st.F0 + tgt; s = L.vnorm(tgt)
        gTH = L.test_outputs(prob, td["X"], L.run_learner9(prob, st, t, s, L.L_DJ(1.0))["theta"]) - F0t
        rNP = L.run_learner9(prob, st, t, s, L.L_AJ(1.0)); gNP = L.test_outputs(prob, td["X"], rNP["theta"]) - F0t
        thH = L.run_learner9(prob, st, st.F0 + 0.5 * tgt, 0.5 * s, L.L_DJ(1.0))["theta"]
        thE = L.run_learner9(prob, st, t, s, L.L_SL(32))["theta"]
        K0, Km, Ke = kr(prob, st.th0, td["X"], tgt), kr(prob, thH, td["X"], tgt), kr(prob, thE, td["X"], tgt)
        mid = 0.5 * (gTH + gNP); G = L.vnorm(gNP - gTH)
        gFL = L.test_outputs(prob, td["X"], thE) - F0t       # SL(32): f = 0.4844, close to flow
        print("%-14s tau %-5g AJ %s |G| %.3e | |KR_mid - mid|/|G| %.4f  |KR_0 - mid|/|G| %.3f  |KR_end - g_NP|/|G| %.3f | |SL(32) - KR_mid|/|G| %.4f (expected ~0.016 + O(tau))" % (
            cn, tau, rNP["status"], G, L.vnorm(Km - mid) / G, L.vnorm(K0 - mid) / G, L.vnorm(Ke - gNP) / G, L.vnorm(gFL - Km) / G), flush=True)
