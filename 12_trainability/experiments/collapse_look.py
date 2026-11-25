"""What does the network look like where the smallest kernel eigenvalue has been driven down? (state saved by collapse11.py)"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, os, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", 3, OUT + "/collapse11_out")
cname, state = sys.argv[1], sys.argv[2]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); prob, st = ctx.prob, ctx.st
stt = torch.load(state, map_location="cpu"); th = stt["th"].to(L.DT); print("state after pair", stt["m"])
td = ctx.test()
def look(tag, t):
    with torch.no_grad():
        out, cache = L.mlp_forward(t, prob.X, prob.shapes, prob.act, keep=True)
        outT, cacheT = L.mlp_forward(t, td["X"][:2000], prob.shapes, prob.act, keep=True)
    K = L.gram(prob.J(t)); lam, Q = torch.linalg.eigh(K)
    line = "%-6s |theta| %.4f lamN %.5f lam1 %.1f trK %.1f |" % (tag, L.vnorm(t), float(lam[0]), float(lam[-1]), float(lam.sum()))
    for li, (c, cT) in enumerate(zip(cache[:-1], cacheT[:-1])):
        H = c[3]; HT = cT[3]; sv = torch.linalg.svdvals(H - H.mean(0)); p = sv / sv.sum(); erank = float(torch.exp(-(p * torch.log(p + 1e-300)).sum()))
        std = HT.std(0); dead = float((std < 1e-3 * std.max()).double().mean()); z = cT[2]
        Hn = (HT - HT.mean(0)) / (HT.std(0) + 1e-12); Cm = (Hn.T @ Hn) / HT.shape[0]; Cm.fill_diagonal_(0); clone = float((Cm.abs().max(0)[0] > 0.98).double().mean())
        line += " layer %d: feature erank(train) %.2f, min/median unit std %.3e/%.3e, units with std < 0.1%% of max %.3f, units with a |corr| > 0.98 twin %.3f, mean |pre-activation| %.3f |" % (
            li + 1, erank, float(std.min()), float(std.median()), dead, clone, float(z.abs().mean()))
    Lc = torch.linalg.cholesky(K); Z = torch.linalg.solve_triangular(Lc, prob.J(t), upper=False); fp = (Z * Z).sum(0)
    line += " parameter leverage: sum %.2f, sum of squares %.4f, participation %.0f of %d, max %.4f |" % (float(fp.sum()), float((fp * fp).sum()), float(fp.sum() ** 2 / (fp * fp).sum()), fp.numel(), float(fp.max()))
    print(line)
    return lam, Q
l0, Q0 = look("start", st.th0); l1, Q1 = look("end", th)
print("bottom 8 eigenvalues start:", " ".join("%.4f" % float(x) for x in l0[:8]))
print("bottom 8 eigenvalues end  :", " ".join("%.4f" % float(x) for x in l1[:8]))
ov = (Q1[:, :4].T @ Q0).pow(2)            # overlap of the 4 weakest end eigenvectors with start eigenvectors
print("weakest end eigenvectors: share of weight on the 16 weakest start eigenvectors:", " ".join("%.3f" % float(ov[i, :16].sum()) for i in range(4)))
d = th - st.th0; o = 0
for li, (wo, wi) in enumerate(prob.shapes):
    n = wo * wi + wo; print("  layer %d: |dtheta| %.4f  |theta0| %.4f  ratio %.4f" % (li + 1, L.vnorm(d[o:o + n]), L.vnorm(st.th0[o:o + n]), L.vnorm(d[o:o + n]) / L.vnorm(st.th0[o:o + n]))); o += n
