"""Independent cross-check of spec10_run.py (written by the reviewer, shares no code with it).
Own learners (damped Gauss-Newton loop, own exact sub-jumps), own targets, own loss formula; from the library only the
network forward/Jacobian (prob.F, prob.J), the data and the stored start weights. Compares with review10/runs/main records:
ZERO losses of DJ(1), DJ(0.25), DJ(1.5), SL(4) and PLAIN losses of DJ(1), DJ(1.5), and the differences the rules HZ1-HZ4 use.
Run after review10/test_spec10.py:   python review10/crosscheck10.py"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PRJ, "review9", "v3"))
from common9 import load
rd = os.path.join(HERE, "tiny_run10")
L, CFG = load("tiny", rd)
torch.set_num_threads(4)
C = CFG["N_CLASSES"]
rec = json.load(open(os.path.join(HERE, "runs", "main", "spec10_out", "records_0.json")))
cb = L.combos("G4")[0]
cn = L.cname9(cb)
prob, _, Y, _ = L.combo_problem9(cb)
th0 = torch.load(L.start_paths(rd, cn)[0], map_location="cpu").to(L.DT)
td = L.get_test_data(cb["seed"], int(cb["n"]))
Yt = td["Y"]
F0 = prob.F(th0)
F0t = L.test_outputs(prob, td["X"], th0)


def nrm(x):
    return float(torch.linalg.vector_norm(x))


def damped(t, c, tol=1e-11, cap=4000):
    """theta <- theta - c J(theta)^T (J J^T)^-1 (F(theta) - t), Jacobian at the current point."""
    th = th0.clone()
    s = nrm(t - F0)
    for _ in range(cap):
        r = prob.F(th) - t
        if nrm(r) <= tol * s:
            return th
        J = prob.J(th)
        th = th - c * (J.T @ torch.linalg.solve(J @ J.T, r))
    raise RuntimeError("own damped loop did not converge")


def exact_jump(thb, t, tol=1e-12, cap=200):
    """theta = thb + J(thb)^T a with F(theta) = t (Newton on a)."""
    Jb = prob.J(thb)
    a = torch.zeros(Jb.shape[0], dtype=thb.dtype)
    for _ in range(cap):
        th = thb + Jb.T @ a
        r = t - prob.F(th)
        if float(r.abs().max()) <= tol:
            return th
        a = a + torch.linalg.solve(prob.J(th) @ Jb.T, r)
    raise RuntimeError("own jump did not converge")


def straight(t, m):
    th = th0.clone()
    for k in range(1, m + 1):
        th = exact_jump(th, F0 + (k / m) * (t - F0))
    return th


def loss(o):
    return float(((o - Yt) ** 2).sum()) / (2.0 * o.numel())


OWN = {"DJ(1)": lambda t: damped(t, 1.0), "DJ(0.25)": lambda t: damped(t, 0.25), "DJ(1.5)": lambda t: damped(t, 1.5),
       "SL(4)": lambda t: straight(t, 4)}
worst_abs, worst_diff = 0.0, 0.0
print("own value vs spec10_run record (relative difference); tiny run, %s" % cn)
for blk in rec["combos"][cn]["blocks"]:
    tau = blk["tau"]
    mine = {"ZERO": {}, "PLAIN": {}}
    for ln, fn in OWN.items():
        thZ = fn(F0 + tau * Y)                                          # ZERO target
        assert nrm((prob.F(thZ) - F0) - tau * Y) <= 1e-9 * nrm(tau * Y)
        mine["ZERO"][ln] = loss(L.test_outputs(prob, td["X"], thZ) - F0t)   # ZERO output = F_test - F0t
    for ln in ("DJ(1)", "DJ(1.5)"):
        thP = OWN[ln](F0 + tau * (Y - F0))                              # PLAIN target
        mine["PLAIN"][ln] = loss(L.test_outputs(prob, td["X"], thP))
    for kind in ("PLAIN", "ZERO"):
        theirs = {e["learner"]: e["loss"] for e in blk[kind]["ends"]}
        for ln, v in mine[kind].items():
            d = abs(v - theirs[ln]) / abs(theirs[ln])
            worst_abs = max(worst_abs, d)
            print("  tau %-5g %-5s %-9s own %.10f  record %.10f  rel %.1e" % (tau, kind, ln, v, theirs[ln], d))
    tz = {e["learner"]: e["loss"] for e in blk["ZERO"]["ends"]}
    for a, b in (("DJ(0.25)", "DJ(1)"), ("DJ(1.5)", "DJ(1)"), ("SL(4)", "DJ(1)")):
        dm, dr = mine["ZERO"][a] - mine["ZERO"][b], tz[a] - tz[b]
        d = abs(dm - dr) / abs(dr)
        worst_diff = max(worst_diff, d)
        print("  tau %-5g ZERO loss[%s] - loss[%s]: own %+.6e  record %+.6e  rel %.1e  same sign %s" % (tau, a, b, dm, dr, d, (dm > 0) == (dr > 0)))
print("worst relative difference: single losses %.2e, rule differences %.2e" % (worst_abs, worst_diff))
print("CROSS-CHECK " + ("PASS" if worst_abs <= 1e-5 and worst_diff <= 2e-2 else "FAIL"))
