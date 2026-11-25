import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=2)
C = CFG["N_CLASSES"]
which = sys.argv[1]
SPECS = [("DJ(1.5)", L.L_DJ(1.5), -1.0), ("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(4)", L.L_SL(4), 0.375), ("DJ(0.25)", L.L_DJ(0.25), 0.4286)]
def feats(prob, X, th):
    with torch.no_grad():
        _, cache = L.mlp_forward(th, X, prob.shapes, prob.act, keep=True)
    return cache[-1][0], cache[0][3]          # last hidden activations, first hidden activations
def probe(H, Yoh, labels):
    n = H.shape[0] // 2
    A = torch.cat([H, torch.ones(H.shape[0], 1, dtype=H.dtype)], 1)
    G = A[:n].T @ A[:n]
    lam = 1e-6 * float(torch.diagonal(G).mean())
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=H.dtype), A[:n].T @ Yoh[:n])
    P = A[n:] @ W
    acc = float((P.argmax(1) == labels[n:]).double().mean()); mse = float(((P - Yoh[n:]) ** 2).mean())
    Hc = H[n:] - H[n:].mean(0); Yc = Yoh[n:] - Yoh[n:].mean(0)
    cka = float((Yc.T @ Hc).pow(2).sum() / ((Hc.T @ Hc).norm() * (Yc.T @ Yc).norm()))
    return acc, mse, cka
out = open(OUT + "/x2_%s.jsonl" % which, "a")
for cb in [c for c in L.combos("G4") if str(c["n"]) == which]:
    cn = L.cname9(cb); ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; td = ctx.test()
    Yoh = td["Y"].reshape(-1, C); labels = td["labels"]
    h2, h1 = feats(prob, td["X"], st.th0); b2, b1 = probe(h2, Yoh, labels), probe(h1, Yoh, labels)
    print("%-18s start: probe acc last hidden %.4f (mse %.5f, cka %.4f) | first hidden %.4f" % (cn, b2[0], b2[1], b2[2], b1[0]), flush=True)
    for tau in (0.5, 1.0):
        for kind in ("ZERO", "PLAIN"):
            t = st.F0 + tau * (ctx.Y if kind == "ZERO" else (ctx.Y - st.F0)); s = L.vnorm(t - st.F0)
            for name, Ld, f in SPECS:
                t0 = time.time()
                res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]
                h2, h1 = feats(prob, td["X"], th); p2, p1 = probe(h2, Yoh, labels), probe(h1, Yoh, labels)
                r = dict(combo=cn, tau=tau, kind=kind, learner=name, f=f, status=res["status"], train_res=L.vnorm(prob.F(th) - t) / s,
                         acc2=p2[0], mse2=p2[1], cka2=p2[2], acc1=p1[0], mse1=p1[1], cka1=p1[2], sec=time.time() - t0)
                out.write(json.dumps(r) + "\n"); out.flush()
                print("%-18s tau %-4g %-5s %-9s f %+.3f %-10s | last hidden: acc %.4f mse %.5f cka %.4f | first hidden: acc %.4f cka %.4f (%.0fs)" % (
                    cn, tau, kind, name, f, r["status"], p2[0], p2[1], p2[2], p1[0], p1[2], r["sec"]), flush=True)
                L.free_memory()
    del ctx, prob, st; L.free_memory()
print("x2 done", flush=True)
