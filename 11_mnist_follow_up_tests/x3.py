import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=5)
C = CFG["N_CLASSES"]; R2 = math.sqrt(2.0)
which = sys.argv[1]
TAUS = [0.01, 0.03, 0.1, 0.3, 1.0]
SPECS = [("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(2)", L.L_SL(2), 0.25), ("SL(4)", L.L_SL(4), 0.375), ("SL(8)", L.L_SL(8), 0.4375)]
NEG = ("DJ(1.5)", L.L_DJ(1.5), -1.0)
out = open(OUT + "/x3_%s.jsonl" % which, "a")
for cb in [c for c in L.combos("G4") if str(c["n"]) == which]:
    cn = L.cname9(cb)
    base, _, Y, _ = L.combo_problem9(cb)
    th0 = torch.load(L.start_paths(loc.RD, cn)[0], map_location="cpu").to(L.DT)
    P = th0.numel(); td = L.get_test_data(cb["seed"], int(cb["n"])); Yt = td["Y"]; N = Yt.numel()
    F0t = L.test_outputs(base, td["X"], th0)
    probD = L.Problem(F=lambda th: (base._F(th[:P]) - base._F(th[P:])) / R2,
                      J=lambda th: torch.cat([base._J(th[:P]), -base._J(th[P:])], 1) / R2, N=base.N, P=2 * P, C=C, name=cn + "-doubled")
    models = {
        "doubled": (probD, torch.cat([th0, th0]), lambda th: (L.test_outputs(base, td["X"], th[:P]) - L.test_outputs(base, td["X"], th[P:])) / R2),
        "subtract": (base, th0, lambda th: L.test_outputs(base, td["X"], th) - F0t),
    }
    for mname, (prob, thS, outf) in models.items():
        st = L.Start(prob, thS)
        assert float(outf(thS).abs().max()) == 0.0
        if mname == "doubled":
            assert float(st.F0.abs().max()) == 0.0
        for tau in TAUS:
            t = st.F0 + tau * Y; s = L.vnorm(t - st.F0); ends = {}
            for name, Ld, f in SPECS + ([NEG] if tau in (0.1, 1.0) else []):
                t0 = time.time()
                try:
                    res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]
                    o = outf(th); loss, acc = L.test_loss_acc(o, td)
                    ends[name] = dict(o=o, loss=loss, acc=acc, status=res["status"], res=L.vnorm(prob.F(th) - t) / s, f=f, sec=time.time() - t0)
                except Exception as e:
                    print("%s %s tau %g %s ERROR %s" % (cn, mname, tau, name, str(e)[:100]), flush=True)
                L.free_memory()
            if "DJ(1)" not in ends: continue
            lo = ends["DJ(1)"]
            for name, e in ends.items():
                D = e["o"] - lo["o"]
                r = dict(combo=cn, model=mname, tau=tau, learner=name, f=e["f"], status=e["status"], train_res=e["res"], loss=e["loss"], acc=e["acc"],
                         dloss=e["loss"] - lo["loss"], Dnorm=L.vnorm(D) / math.sqrt(N), lin=float((lo["o"] - Yt) @ D) / N, quad=float(D @ D) / (2 * N),
                         out_rms=L.vnorm(lo["o"]) / math.sqrt(N), sec=e["sec"])
                out.write(json.dumps(r) + "\n"); out.flush()
                print("%-14s %-8s tau %-5g %-8s f %+.3f %-9s res %.0e loss %.8f acc %.4f | loss-lo %+.3e |D|rms %.3e (%.0fs)" % (
                    cn, mname, tau, name, e["f"], e["status"], e["res"], e["loss"], e["acc"], r["dloss"], r["Dnorm"], e["sec"]), flush=True)
        del st; L.free_memory()
print("x3 done", flush=True)
