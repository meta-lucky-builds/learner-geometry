import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load()
C = CFG["N_CLASSES"]
which = sys.argv[1]            # "32" or "128"
parts = sys.argv[2].split(",") # a, b
def numbers(o, td):
    Yt = td["Y"]; N = o.numel()
    loss, acc = L.test_loss_acc(o, td)
    ov = o.reshape(-1, C); m = ov.mean(0, keepdim=True); v = ov - m
    a = float((Yt.reshape(-1, C) * v).sum() / (v * v).sum())
    rs = a * v + m - Yt.reshape(-1, C)
    return loss, acc, a, float((rs * rs).sum()) / (2.0 * N)
done = set()
for _l in open(OUT + "/x1_%s.jsonl" % which):
    if _l.strip():
        _r = json.loads(_l); done.add((_r["combo"], _r["tau"], _r["learner"]))
SKIP = ("tanh-init-s0-n128", "tanh-init-s1-n128")   # finished or solver-failed earlier; not repeated
out = open(OUT + "/x1_%s.jsonl" % which, "a")
for cb in [c for c in L.combos("G4") if str(c["n"]) == which]:
    cn = L.cname9(cb)
    if cn in SKIP: continue
    ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; td = ctx.test(); F0t, _, _ = ctx.test0()
    jobs = []
    for tau in [float(x) for x in sys.argv[3].split(",")]:
        for al in [float(x) for x in sys.argv[4].split(",")]:
            jobs.append((tau, "AJ(%g)" % al, L.L_AJ(al), al))
    for tau, name, Ld, f in jobs:
        if (cn, tau, name) in done: continue
        t = st.F0 + tau * ctx.Y; s = L.vnorm(t - st.F0); t0 = time.time()
        try:
            res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]
            tr = L.vnorm(prob.F(th) - t) / s
            if not bool(torch.isfinite(th).all()): raise RuntimeError("non-finite")
            loss, acc, a, ls = numbers(L.test_outputs(prob, td["X"], th) - F0t, td)
            lam1, trK = L._lam1_tr(L.gram(prob.J(th)))
            r = dict(combo=cn, tau=tau, learner=name, f=f, status=res["status"], train_res=tr, loss=loss, acc=acc, a_star=a, loss_star=ls,
                     dist=L.vnorm(th - st.th0), trK=trK, lam1=lam1, sec=time.time() - t0)
        except Exception as e:
            r = dict(combo=cn, tau=tau, learner=name, f=f, status="ERROR", error=str(e)[:200], sec=time.time() - t0)
        out.write(json.dumps(r) + "\n"); out.flush()
        print("%-18s tau %-5g %-12s f %+.4f %-10s res %.1e loss %.7f loss* %.7f acc %.4f a* %.3f trK %.4g dist %.3f (%.0fs)" % (
            cn, tau, name, f, r["status"], r.get("train_res", float("nan")), r.get("loss", float("nan")), r.get("loss_star", float("nan")),
            r.get("acc", float("nan")), r.get("a_star", float("nan")), r.get("trK", float("nan")), r.get("dist", float("nan")), r["sec"]), flush=True)
        L.free_memory()
    del ctx, prob, st; L.free_memory()
print("x1 done", flush=True)
