"""X8 / test 1: third-order three-function law with more learners and starts. Doubled network as x3b/x7.
One file per (combo, tau, learner) so a killed run loses at most one learner.  usage: python x8.py combo1,combo2 tau1,tau2"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, os, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=7)
C = CFG["N_CLASSES"]; R2 = math.sqrt(2.0)
OUT = OUT + "/x8_out"; os.makedirs(OUT, exist_ok=True)
want = sys.argv[1].split(","); taus = [float(x) for x in sys.argv[2].split(",")]
SPECS = [("DJ(1)", L.L_DJ(1.0)), ("SL(2)", L.L_SL(2)), ("SL(3)", L.L_SL(3)), ("SL(4)", L.L_SL(4)), ("SL(8)", L.L_SL(8)),
         ("DJ(0.75)", L.L_DJ(0.75)), ("DJ(1.25)", L.L_DJ(1.25)), ("DJ(0.5)", L.L_DJ(0.5)), ("DJ(1.5)", L.L_DJ(1.5)),
         ("AJ(0.5)", L.L_AJ(0.5)), ("AJ(1)", L.L_AJ(1.0)), ("SLa(2,1)", L.L_SLa(2, 1.0)), ("SLa(4,2)", L.L_SLa(4, 2.0)), ("TJ(1.5,1)", L.L_TJ(1.5, 1.0))]
fname = lambda cn, tau, nm: "%s/%s_%g_%s.pt" % (OUT, cn, tau, nm.replace("(", "_").replace(")", "").replace(",", "_"))
for cb in [c for c in L.combos("G4") if str(c["n"]) == "32" and L.cname9(c) in want]:
    cn = L.cname9(cb); base, _, Y, _ = L.combo_problem9(cb)
    th0 = torch.load(L.start_paths(loc.RD, cn)[0], map_location="cpu").to(L.DT); P = th0.numel(); td = L.get_test_data(cb["seed"], int(cb["n"]))
    prob = L.Problem(F=lambda th: (base._F(th[:P]) - base._F(th[P:])) / R2,
                     J=lambda th: torch.cat([base._J(th[:P]), -base._J(th[P:])], 1) / R2, N=base.N, P=2 * P, C=C, name=cn + "-doubled")
    outf = lambda th: (L.test_outputs(base, td["X"], th[:P]) - L.test_outputs(base, td["X"], th[P:])) / R2
    st = L.Start(prob, torch.cat([th0, th0]))
    for tau in taus:
        old = OUT + "/x7_%s_%g.pt" % (cn, tau)
        if os.path.exists(old):                                           # reuse the six learners of the first run
            for nm, o in torch.load(old, map_location="cpu", weights_only=False)["outs"].items():
                if not os.path.exists(fname(cn, tau, nm)): torch.save(dict(out=o, status="CONVERGED", res=float("nan"), reused=True), fname(cn, tau, nm))
        t = st.F0 + tau * Y; s = L.vnorm(t - st.F0)
        for name, Ld in SPECS:
            fn = fname(cn, tau, name)
            if os.path.exists(fn): continue
            t0 = time.time()
            try:
                res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]; r = L.vnorm(prob.F(th) - t) / s
                torch.save(dict(out=outf(th).clone(), status=res["status"], res=r), fn)
                print("%s tau %g %-10s %-10s train res %.1e (%.0fs)" % (cn, tau, name, res["status"], r, time.time() - t0), flush=True)
            except Exception as e:
                print("%s tau %g %-10s ERROR %s (%.0fs)" % (cn, tau, name, str(e)[:160], time.time() - t0), flush=True)
            L.free_memory()
    del st; L.free_memory()
print("x8 done", flush=True)
