"""Doubled network F_D = (F(theta_a) - F(theta_b))/sqrt2, theta_a = theta_b = stored start. Target tau*Y. Records raw loss, loss after the
best one-number rescaling, accuracy, and the split of D = out(L) - out(DJ(1)) into a rescaling of the learned function S = out(DJ(1)) and the
rest:  dloss = const + dS + drest + quad  (as SPEC10 sec. 6 without the start-function part, which is absent here)."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=7)
C = CFG["N_CLASSES"]; R2 = math.sqrt(2.0)
which = sys.argv[1]
TAUS = [0.03, 0.1, 0.3, 1.0]
SPECS = [("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(4)", L.L_SL(4), 0.375), ("SL(8)", L.L_SL(8), 0.4375), ("DJ(1.5)", L.L_DJ(1.5), -1.0)]
def cen(o):
    M = o.reshape(-1, C); return M - M.mean(0, keepdim=True)
def star(o, Yt):
    M = o.reshape(-1, C); v = cen(o); a = float((Yt.reshape(-1, C) * v).sum() / (v * v).sum())
    rs = a * v + (M - v) - Yt.reshape(-1, C); return a, float((rs * rs).sum()) / (2.0 * o.numel())
out = open(OUT + "/x3b_%s.jsonl" % which, "a")
for cb in [c for c in L.combos("G4") if str(c["n"]) == which]:
    cn = L.cname9(cb)
    base, _, Y, _ = L.combo_problem9(cb)
    th0 = torch.load(L.start_paths(loc.RD, cn)[0], map_location="cpu").to(L.DT)
    P = th0.numel(); td = L.get_test_data(cb["seed"], int(cb["n"])); Yt = td["Y"]; N = Yt.numel(); Ym = Yt.reshape(-1, C)
    prob = L.Problem(F=lambda th: (base._F(th[:P]) - base._F(th[P:])) / R2,
                     J=lambda th: torch.cat([base._J(th[:P]), -base._J(th[P:])], 1) / R2, N=base.N, P=2 * P, C=C, name=cn + "-doubled")
    outf = lambda th: (L.test_outputs(base, td["X"], th[:P]) - L.test_outputs(base, td["X"], th[P:])) / R2
    st = L.Start(prob, torch.cat([th0, th0]))
    assert float(st.F0.abs().max()) == 0.0
    for tau in TAUS:
        t = st.F0 + tau * Y; s = L.vnorm(t - st.F0); ends = {}
        for name, Ld, f in SPECS:
            t0 = time.time()
            try:
                res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]; o = outf(th)
                loss, acc = L.test_loss_acc(o, td); a, ls = star(o, Yt)
                ends[name] = dict(o=o, loss=loss, acc=acc, a=a, ls=ls, status=res["status"], res=L.vnorm(prob.F(th) - t) / s, f=f, sec=time.time() - t0)
            except Exception as e:
                print("%s tau %g %s ERROR %s" % (cn, tau, name, str(e)[:100]), flush=True)
            L.free_memory()
        lo = ends.get("DJ(1)")
        if lo is None: continue
        Sv = cen(lo["o"]); r = lo["o"].reshape(-1, C) - Ym; gS = float((r * Sv).sum()) / N
        for name, e in ends.items():
            D = (e["o"] - lo["o"]).reshape(-1, C); Dv = cen(e["o"] - lo["o"]); Dc = D - Dv
            eS = float((Dv * Sv).sum() / (Sv * Sv).sum()); rest = Dv - eS * Sv
            rec = dict(combo=cn, tau=tau, learner=name, f=e["f"], status=e["status"], train_res=e["res"], loss=e["loss"], loss_star=e["ls"], a_star=e["a"],
                       acc=e["acc"], dloss=e["loss"] - lo["loss"], dstar=e["ls"] - lo["ls"], dacc=e["acc"] - lo["acc"], Drms=float((D * D).mean().sqrt()),
                       const=float((r * Dc).sum()) / N, dS=eS * gS, drest=float((r * rest).sum()) / N, quad=float((D * D).sum()) / (2 * N), eS=eS, gS=gS)
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("%-13s tau %-4g %-8s f %+.3f %-9s loss %.7f a* %.3f acc %.4f | dloss %+.2e dloss* %+.2e dacc %+.4f | dS %+.2e drest %+.2e quad %+.2e eS %+.4f |D| %.2e (%.0fs)" % (
                cn, tau, name, e["f"], e["status"], e["loss"], e["a"], e["acc"], rec["dloss"], rec["dstar"], rec["dacc"], rec["dS"], rec["drest"], rec["quad"], eS, rec["Drms"], e["sec"]), flush=True)
    del st; L.free_memory()
print("x3b done", flush=True)
