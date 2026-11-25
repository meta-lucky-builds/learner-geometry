import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; the = sum(ys)/n; sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-the) for x, y in zip(xs, ys))
    b = sxy/sxx; a = the-b*mx; ss = sum((y-the)**2 for y in ys); sr = sum((y-a-b*x)**2 for x, y in zip(xs, ys)); return b, a, (1-sr/ss if ss > 0 else float("nan"))
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    for k, i in enumerate(o): r[i] = k
    return r
def spear(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma = (n-1)/2
    return sum((x-ma)*(y-ma) for x, y in zip(ra, rb)) / sum((x-ma)**2 for x in ra)
blocks = collections.OrderedDict()
for r in rows: blocks.setdefault((r["combo"], r["tau"]), {})[r["learner"]] = r
cnt = collections.Counter()
for b, g in blocks.items():
    ls = sorted(g.values(), key=lambda r: r["f"]); fs = [r["f"] for r in ls]; pos = [r for r in ls if r["f"] >= 0]; fp = [r["f"] for r in pos]
    bal0 = sum(ls[0]["rn0"]) - sum(ls[0]["in0"])
    sb, ab, r2b = fit(fs, [r["balance"] for r in ls]); fzero = (bal0 - ab)/sb
    s1, _, _ = fit(fs, [r["pr_contrib"] for r in ls]); s1p, _, _ = fit(fp, [r["pr_contrib"] for r in pos])
    s2, _, _ = fit(fs, [r["pr_readout"] for r in ls]); s3, _, _ = fit(fs, [r["pr_feat"] for r in ls])
    U = len(ls[0]["rn"]); slope_rn = [fit(fs, [r["rn"][k] for r in ls])[0] for k in range(U)]; slope_in = [fit(fs, [r["inn"][k] for r in ls])[0] for k in range(U)]
    sp = spear(ls[0]["rn0"], slope_rn); sp_in = spear(ls[0]["in0"], slope_in)
    drn = sum(r["rn_sum"] for r in ls[-1:]) - sum(ls[0]["rn0"]); din = ls[-1]["in_sum"] - sum(ls[0]["in0"])
    srn, _, _ = fit(fs, [r["rn_sum"] for r in ls]); sin_, _, _ = fit(fs, [r["in_sum"] for r in ls])
    print("%-13s tau %-4g | PR contribution %s slope %+.2f (f>=0: %+.2f) | PR readout slope %+.2f | PR feature slope %+.2f | balance: start %.4f, values %s, slope %+.4f (R2 %.3f), equals start at f = %+.3f | slope of readout^2 %+.4f incoming^2 %+.4f | Spearman(start readout norm, its f-slope) %+.3f (incoming %+.3f)" % (
        b[0], b[1], " ".join("%.1f" % r["pr_contrib"] for r in ls), s1, s1p, s2, s3, bal0, " ".join("%.4f" % r["balance"] for r in ls), sb, r2b, fzero, srn, sin_, sp, sp_in))
    cnt["M1 PR contribution falls (all f)"] += s1 < 0; cnt["   same, f >= 0 only"] += s1p < 0; cnt["M1b PR readout change falls"] += s2 < 0
    cnt["M3 Spearman > 0"] += sp > 0; cnt["balance rises with f"] += sb > 0; cnt["balance = start at f in [0.4, 0.6]"] += 0.4 <= fzero <= 0.6
print(); [print("  %-40s %d / %d" % (k, v, len(blocks))) for k, v in cnt.items()]
