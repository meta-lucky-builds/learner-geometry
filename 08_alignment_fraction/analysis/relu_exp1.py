"""ReLU experiment 1: alignment fractions versus move size, with the number of ReLU (unit, input) pairs that switch.
Uses the reviewed library (review8/final/part8lib.py). Small network so that the CPU can do a size ladder.
Hypothesis: inside one linear region (no switches) F is a polynomial, so the law must hold exactly;
deviations should appear once the paths of the different learners cross kinks."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "review8", "final"))
exec(open(os.path.join(HERE, "..", "review8", "final", "config_cell.py"), encoding="utf-8").read())
import part8lib as L
CFG["JUMP_TOL"] = 1e-13; CFG["NP_RES_TOL"] = 1e-13; CFG["NP_STAT_TOL"] = 1e-9; CFG["NP_CAP"] = int(os.environ.get("NPCAP", 60)); CFG["DJ_TOL_REL"] = 1e-9
L.configure(CFG)
import torch
torch.set_default_dtype(torch.float64)
act = sys.argv[1] if len(sys.argv) > 1 else "relu"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 32
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
start = sys.argv[4] if len(sys.argv) > 4 else "trained"
CFG["WIDTH"] = W; CFG["N_TRAIN"] = n; L.configure(CFG)
L._load_mnist_raw()
cb = dict(act=act, start=start, seed=0)
prob, th, Y, labels = L.combo_problem(cb)
if start == "trained":
    th, fi = L.fit_dj(prob, th, Y)
st = L.Start(prob, th)
dirs, dinfo, S0 = L.make_directions(st, 0, n)


def pattern(t):
    _, cache = L.mlp_forward(t, prob.X, prob.shapes, prob.act, keep=True)
    return torch.cat([(c[2] > 0).reshape(-1) for c in cache[1:]] if False else [(cache[i][2] > 0).reshape(-1) for i in range(len(cache) - 1)])


p0 = pattern(st.th0)
print("%s W=%d n=%d %s: N=%d P=%d  (unit,input) pairs=%d  active %.3f  cond %.2e  S0 %.3f  |theta0| %.2f" % (
    act, W, n, start, prob.N, prob.P, p0.numel(), float(p0.double().mean()), st.lam1 / st.lamN, S0, L.vnorm(st.th0)))
LS = [("SL(2)", L.L_SL(2), 0.25), ("SL(8)", L.L_SL(8), 0.4375), ("SL(32)", L.L_SL(32), 0.484375), ("DJ(0.5)", L.L_DJ(0.5), 1 / 3),
      ("DJ(1)", L.L_DJ(1.0), 0.0), ("DJ(1.5)", L.L_DJ(1.5), -1.0), ("DJ(1.9)", L.L_DJ(1.9), -9.0)]
fracs = [float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else "2e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5".split(","))]
for dn in os.environ.get("DIRS", "eigA,eigC,rand").split(","):
    d = dirs[dn]
    print("\n direction %s   columns per learner: rho_even(vs NP) [cos] | flips(+ run, endpoint vs start) | NP status(+/-)" % dn)
    print("   s/S0    " + "  ".join("%-27s" % ("%s f=%+.3f" % (nm, f)) for nm, _, f in LS) + "  | rho' relative to SL(32) (pred f/0.484): " + " ".join(nm for nm, _, _ in LS[:-0 or None]))
    for fr in fracs:
        s = fr * S0
        row, vecs = [], {}
        for nm, Ld, f in LS:
            ev, scs, vs = L.run_pair(prob, st, d, s, Ld)
            thp = None
            res = L.run_learner(prob, st, st.F0 + s * d, s, Ld)
            fl = int((pattern(res["theta"]) != p0).sum())
            stt = "".join("C" if x.get("np_status") == "CONVERGED" else "F" for x in scs)
            row.append("%+8.4f [%+.3f] %4d %s" % (ev["rho"], ev["cos"], fl, stt))
            if "vL" in vs[0] and "vL" in vs[1]:
                vecs[nm] = 0.5 * (vs[0]["vL"] + vs[1]["vL"])
        ref = vecs.get("SL(32)")
        rel = []
        for nm, _, f in LS:
            if ref is not None and nm in vecs:
                rel.append("%+.3f(%+.3f)" % (float(vecs[nm] @ ref) / float(ref @ ref), f / 0.484375))
        print("   %-7.0e " % fr + "  ".join("%-27s" % x for x in row) + "  | " + " ".join(rel), flush=True)
