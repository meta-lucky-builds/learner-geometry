"""X7 / E1: how many functions carry the learner-dependence at third order? Doubled network (as x3b), test outputs of six learners saved.
usage: python x7.py <combo name> <tau>"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=7)
C = CFG["N_CLASSES"]; R2 = math.sqrt(2.0)
want, tau = sys.argv[1], float(sys.argv[2])
SPECS = [("DJ(1)", L.L_DJ(1.0)), ("DJ(0.5)", L.L_DJ(0.5)), ("DJ(1.5)", L.L_DJ(1.5)), ("SL(2)", L.L_SL(2)), ("SL(4)", L.L_SL(4)), ("SL(8)", L.L_SL(8))]
for cb in [c for c in L.combos("G4") if str(c["n"]) == "32" and L.cname9(c) == want]:
    cn = L.cname9(cb); base, _, Y, _ = L.combo_problem9(cb)
    th0 = torch.load(L.start_paths(loc.RD, cn)[0], map_location="cpu").to(L.DT); P = th0.numel(); td = L.get_test_data(cb["seed"], int(cb["n"]))
    prob = L.Problem(F=lambda th: (base._F(th[:P]) - base._F(th[P:])) / R2,
                     J=lambda th: torch.cat([base._J(th[:P]), -base._J(th[P:])], 1) / R2, N=base.N, P=2 * P, C=C, name=cn + "-doubled")
    outf = lambda th: (L.test_outputs(base, td["X"], th[:P]) - L.test_outputs(base, td["X"], th[P:])) / R2
    st = L.Start(prob, torch.cat([th0, th0])); t = st.F0 + tau * Y; s = L.vnorm(t - st.F0); outs = {}
    for name, Ld in SPECS:
        t0 = time.time(); res = L.run_learner9(prob, st, t, s, Ld); outs[name] = outf(res["theta"]).clone()
        print("%s tau %g %-8s %-9s train res %.1e (%.0fs)" % (cn, tau, name, res["status"], L.vnorm(prob.F(res["theta"]) - t) / s, time.time() - t0), flush=True)
        L.free_memory()
    torch.save(dict(combo=cn, tau=tau, outs=outs, Y=td["Y"]), OUT + "/x7_%s_%g.pt" % (cn, tau))
print("x7 done", flush=True)
