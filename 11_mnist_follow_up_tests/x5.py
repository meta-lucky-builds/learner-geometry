"""X5: is alignment specialisation?  (predictions: FINDINGS.md sec. 39)
Per (start, tau) block, ZERO target, five learners: class scatter of last-hidden features (train and test), probe with k labels,
swap of readout and hidden layers between learners.  Before that: readout-flip symmetry check on two starts."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=7)
C = CFG["N_CLASSES"]
SPECS = [("DJ(1.5)", L.L_DJ(1.5), -1.0), ("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(2)", L.L_SL(2), 0.25), ("SL(4)", L.L_SL(4), 0.375), ("SL(8)", L.L_SL(8), 0.4375)]
KS = [(30, 20), (100, 20), (300, 10), (1000, 5), (5000, 1)]          # (labels per readout, number of disjoint label sets from test images 0-4999)
LAMS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
def hid(prob, X, th):
    with torch.no_grad():
        _, cache = L.mlp_forward(th, X, prob.shapes, prob.act, keep=True)
    return cache[-1][0]
def scatter(H, labels):
    mu = H.mean(0); sw = 0.0; sb = 0.0; n = H.shape[0]
    for c in range(C):
        m = labels == c; k = int(m.sum())
        if k == 0: continue
        mc = H[m].mean(0); sw += float(((H[m] - mc) ** 2).sum()); sb += k * float(((mc - mu) ** 2).sum())
    return sw / n, sb / n
def probes(H, Yoh, labels):
    A = torch.cat([H, torch.ones(H.shape[0], 1, dtype=H.dtype)], 1); Ate, Yte, lte = A[5000:], Yoh[5000:], labels[5000:]
    I = torch.eye(A.shape[1], dtype=H.dtype); res = {}
    for k, reps in KS:
        mse = {lam: 0.0 for lam in LAMS}; acc = {lam: 0.0 for lam in LAMS}
        for r in range(reps):
            Ak, Yk = A[r * k:(r + 1) * k], Yoh[r * k:(r + 1) * k]
            G = Ak.T @ Ak; AY = Ak.T @ Yk; md = float(torch.diagonal(G).mean())
            for lam in LAMS:
                P = Ate @ torch.linalg.solve(G + lam * md * I, AY)
                mse[lam] += float(((P - Yte) ** 2).mean()) / reps; acc[lam] += float((P.argmax(1) == lte).double().mean()) / reps
        res[str(k)] = {"mse": [mse[l] for l in LAMS], "acc": [acc[l] for l in LAMS]}
    return res
cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()))
out = open(OUT + "/x5_32.jsonl", "a")
combos = [c for c in L.combos("G4") if str(c["n"]) == "32" and L.cname9(c).split("-")[0] in sys.argv[1:]]
flip_done = set()
for cb in combos:
    cn = L.cname9(cb); ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; td = ctx.test(); F0t, _, _ = ctx.test0()
    Yoh = td["Y"].reshape(-1, C); labels = td["labels"]; trlab = ctx.Y.reshape(-1, C).argmax(1)
    wo, wi = prob.shapes[-1]; nr = wo * wi + wo; P = st.th0.numel()
    # ---------- readout-flip symmetry check (first tanh and first gelu start only)
    act = cn.split("-")[0]
    if act not in flip_done:
        flip_done.add(act)
        S = torch.ones(P, dtype=st.th0.dtype); S[-nr:] = -1.0
        stf = L.Start(prob, st.th0 * S)
        print("%-14s FLIP start check: max|F0' + F0| = %.2e (max|F0| %.2e)" % (cn, float((stf.F0 + st.F0).abs().max()), float(st.F0.abs().max())), flush=True)
        for tau in (0.03, 0.1):
            D = {}
            for tag, s_ in (("orig", st), ("flip", stf)):
                t = s_.F0 + tau * ctx.Y; s = L.vnorm(t - s_.F0)
                a = L.run_learner9(prob, s_, t, s, L.L_DJ(1.0)); b = L.run_learner9(prob, s_, t, s, L.L_SL(8))
                D[tag] = (b["theta"] - a["theta"], L.test_outputs(prob, td["X"], b["theta"]) - L.test_outputs(prob, td["X"], a["theta"]), a["status"], b["status"])
                L.free_memory()
            d0, o0 = D["orig"][0], D["orig"][1]; d1, o1 = D["flip"][0], D["flip"][1]
            rec = dict(kind="flip", combo=cn, tau=tau, cos_hidden=cos(d0[:-nr], d1[:-nr]), cos_readout=cos(d0[-nr:], d1[-nr:]), cos_out=cos(o0, o1),
                       ratio_hidden=float(d1[:-nr].norm() / d0[:-nr].norm()), ratio_readout=float(d1[-nr:].norm() / d0[-nr:].norm()),
                       ratio_out=float(o1.norm() / o0.norm()), share_readout=float(d0[-nr:].norm() ** 2 / d0.norm() ** 2), status=[D["orig"][2], D["orig"][3], D["flip"][2], D["flip"][3]])
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("%-14s FLIP tau %-5g cos hidden %+.4f  cos readout %+.4f  cos test-output %+.4f | size ratios %.3f %.3f %.3f | readout share of |D|^2 %.3f %s" % (
                cn, tau, rec["cos_hidden"], rec["cos_readout"], rec["cos_out"], rec["ratio_hidden"], rec["ratio_readout"], rec["ratio_out"], rec["share_readout"], rec["status"]), flush=True)
        del stf; L.free_memory()
    # ---------- main blocks
    for tau in (0.5, 1.0):
        t = st.F0 + tau * ctx.Y; s = L.vnorm(t - st.F0); ends = {}
        for name, Ld, f in SPECS:
            t0 = time.time()
            res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]
            H = hid(prob, td["X"], th); Htr = hid(prob, prob.X, th)
            sw, sb = scatter(H, labels); swt, sbt = scatter(Htr, trlab)
            o = L.test_outputs(prob, td["X"], th) - F0t; loss, acc = L.test_loss_acc(o, td)
            rec = dict(kind="main", combo=cn, tau=tau, learner=name, f=f, status=res["status"], train_res=L.vnorm(prob.F(th) - t) / s, loss=loss, acc=acc,
                       sw_test=sw, sb_test=sb, sw_train=swt, sb_train=sbt, probe=probes(H, Yoh, labels), sec=time.time() - t0)
            ends[name] = th
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("%-14s tau %-4g %-8s f %+.4f %-9s loss %.6f acc %.4f | scatter within/between: train %.4f test %.4f | probe mse k30 %.5f k5000 %.5f (%.0fs)" % (
                cn, tau, name, f, res["status"], loss, acc, swt / sbt, sw / sb, min(rec["probe"]["30"]["mse"]), min(rec["probe"]["5000"]["mse"]), rec["sec"]), flush=True)
            L.free_memory()
        M = {}
        for ra in ends:                 # readout taken from ra, hidden layers from hb
            for hb in ends:
                th = ends[hb].clone(); th[-nr:] = ends[ra][-nr:]
                l_, a_ = L.test_loss_acc(L.test_outputs(prob, td["X"], th) - F0t, td); M[ra + "|" + hb] = [l_, a_]
        out.write(json.dumps(dict(kind="swap", combo=cn, tau=tau, M=M)) + "\n"); out.flush()
        lo, hi = "DJ(1)", "SL(8)"; b0 = M[lo + "|" + lo][0]
        print("%-14s tau %-4g SWAP DJ(1)->SL(8): total %+.3e  features only %+.3e  readout only %+.3e  interaction %+.3e" % (
            cn, tau, M[hi + "|" + hi][0] - b0, M[lo + "|" + hi][0] - b0, M[hi + "|" + lo][0] - b0,
            M[hi + "|" + hi][0] - M[lo + "|" + hi][0] - M[hi + "|" + lo][0] + b0), flush=True)
    del ctx, prob, st; L.free_memory()
print("x5 done", flush=True)
