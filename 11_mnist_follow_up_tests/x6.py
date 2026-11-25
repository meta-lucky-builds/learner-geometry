"""X6: does larger f concentrate the learned function on fewer units? (predictions M1, M1b, M3, M2 in FINDINGS.md sec. 40)"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=7)
C = CFG["N_CLASSES"]
SPECS = [("DJ(1.5)", L.L_DJ(1.5), -1.0), ("DJ(1)", L.L_DJ(1.0), 0.0), ("SL(2)", L.L_SL(2), 0.25), ("SL(4)", L.L_SL(4), 0.375), ("SL(8)", L.L_SL(8), 0.4375)]
def hid(prob, X, th):
    with torch.no_grad():
        _, cache = L.mlp_forward(th, X, prob.shapes, prob.act, keep=True)
    return cache[-1][0]
pr = lambda e: float(e.sum() ** 2 / (e * e).sum())
out = open(OUT + "/x6_32.jsonl", "a")
for cb in [c for c in L.combos("G4") if str(c["n"]) == "32" and L.cname9(c).split("-")[0] in sys.argv[1:]]:
    cn = L.cname9(cb); ctx = L.Ctx9(cb, loc.RD); prob, st = ctx.prob, ctx.st; td = ctx.test(); F0t, _, _ = ctx.test0()
    (w3o, w3i), (w2o, w2i) = prob.shapes[-1], prob.shapes[-2]; n3 = w3o * w3i + w3o; n2 = w2o * w2i + w2o; P = st.th0.numel()
    def parts(th):
        W3 = th[P - n3:P - w3o].view(w3o, w3i); blk = th[P - n3 - n2:P - n3]; W2 = blk[:w2o * w2i].view(w2o, w2i); b2 = blk[w2o * w2i:]
        return W3, (W2 * W2).sum(1) + b2 * b2
    W30, in0 = parts(st.th0); H0 = hid(prob, td["X"], st.th0)
    for tau in (0.5, 1.0):
        t = st.F0 + tau * ctx.Y; s = L.vnorm(t - st.F0)
        for name, Ld, f in SPECS:
            t0 = time.time(); res = L.run_learner9(prob, st, t, s, Ld); th = res["theta"]; W3, inn = parts(th); H = hid(prob, td["X"], th)
            # energy of unit k's contribution to the learned output change on test images: mean_x | W3[:,k] h_k(x) - W30[:,k] h0_k(x) |^2
            E = (W3 * W3).sum(0) * (H * H).mean(0) - 2 * (W3 * W30).sum(0) * (H * H0).mean(0) + (W30 * W30).sum(0) * (H0 * H0).mean(0)
            dr = ((W3 - W30) ** 2).sum(0); dh = ((H - H0) ** 2).mean(0); rn = (W3 * W3).sum(0)
            o = L.test_outputs(prob, td["X"], th) - F0t; loss, acc = L.test_loss_acc(o, td)
            rec = dict(combo=cn, tau=tau, learner=name, f=f, status=res["status"], loss=loss, acc=acc, pr_contrib=pr(E), pr_readout=pr(dr), pr_feat=pr(dh),
                       top10_contrib=float(E.sort(descending=True)[0][:10].sum() / E.sum()), balance=float((rn - inn).sum()), rn_sum=float(rn.sum()), in_sum=float(inn.sum()),
                       rn=[float(x) for x in rn], rn0=[float(x) for x in (W30 * W30).sum(0)], inn=[float(x) for x in inn], in0=[float(x) for x in in0], sec=time.time() - t0)
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("%-13s tau %-4g %-8s f %+.4f %-9s loss %.6f | participation ratio (of %d units): contribution %.2f readout change %.2f feature change %.2f | top-10 share %.4f | balance %.4f (%.0fs)" % (
                cn, tau, name, f, res["status"], loss, w3i, rec["pr_contrib"], rec["pr_readout"], rec["pr_feat"], rec["top10_contrib"], rec["balance"], rec["sec"]), flush=True)
            L.free_memory()
    del ctx, prob, st; L.free_memory()
print("x6 done", flush=True)
