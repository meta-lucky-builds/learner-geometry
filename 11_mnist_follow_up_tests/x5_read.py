import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
LAMS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]; KS = ["30", "100", "300", "1000", "5000"]
def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; the = sum(ys)/n; sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-the) for x, y in zip(xs, ys))
    b = sxy/sxx; a = the-b*mx; ss = sum((y-the)**2 for y in ys); sr = sum((y-a-b*x)**2 for x, y in zip(xs, ys))
    return b, (1-sr/ss if ss > 0 else float("nan"))
print("FLIP CHECK")
for r in rows:
    if r["kind"] == "flip":
        print("  %-14s tau %-5g cos hidden %+.4f readout %+.4f test-output %+.4f | size ratios %.3f %.3f %.3f | readout share %.3f" % (
            r["combo"], r["tau"], r["cos_hidden"], r["cos_readout"], r["cos_out"], r["ratio_hidden"], r["ratio_readout"], r["ratio_out"], r["share_readout"]))
main = collections.OrderedDict()
for r in rows:
    if r["kind"] == "main": main.setdefault((r["combo"], r["tau"]), {})[r["learner"]] = r
cnt = collections.Counter(); nb = 0
print("\nCLASS SCATTER within/between (last hidden) and its slope in f;  own test loss slope")
for b, g in main.items():
    if len(g) < 5: print(b, "incomplete"); continue
    nb += 1; ls = sorted(g.values(), key=lambda r: r["f"]); fs = [r["f"] for r in ls]
    tr = [r["sw_train"]/r["sb_train"] for r in ls]; te = [r["sw_test"]/r["sb_test"] for r in ls]
    s_tr, r_tr = fit(fs, tr); s_te, r_te = fit(fs, te); s_l, r_l = fit(fs, [r["loss"] for r in ls]); s_a, _ = fit(fs, [r["acc"] for r in ls])
    sbt, _ = fit(fs, [r["sb_test"] for r in ls]); swt, _ = fit(fs, [r["sw_test"] for r in ls])
    print("  %-13s tau %-4g train %s slope %+.4f (R2 %.2f) | test %s slope %+.5f (R2 %.2f) [between %+.3e within %+.3e] | own loss slope %+.2e (R2 %.2f) acc slope %+.4f" % (
        b[0], b[1], " ".join("%.4f" % v for v in tr), s_tr, r_tr, " ".join("%.4f" % v for v in te), s_te, r_te, sbt, swt, s_l, r_l, s_a))
    cnt["N1 train ratio falls"] += s_tr < 0; cnt["N2 test ratio falls (%s)" % b[0].split("-")[0]] += s_te < 0; cnt["own loss falls with f"] += s_l < 0
print("\nPROBE with k labels: ridge chosen on the f=0 learner; slope of probe error in f (x1e5), R2; acc slope (points per unit f)")
for b, g in main.items():
    if len(g) < 5: continue
    ls = sorted(g.values(), key=lambda r: r["f"]); fs = [r["f"] for r in ls]; z = g["DJ(1)"]; line = "  %-13s tau %-4g" % b; sl = {}
    for k in KS:
        j = min(range(len(LAMS)), key=lambda i: z["probe"][k]["mse"][i])
        s, r2 = fit(fs, [r["probe"][k]["mse"][j]*1e5 for r in ls]); sa, _ = fit(fs, [r["probe"][k]["acc"][j]*100 for r in ls]); sl[k] = s
        line += " | k%-4s lam %-5g mse %.4f slope %+7.1f (%.2f) acc %+.2f" % (k, LAMS[j], z["probe"][k]["mse"][j], s, r2, sa)
        cnt["probe slope < 0 at k=%s" % k] += s < 0
    print(line)
    cnt["K3 slope(30) < slope(5000)"] += sl["30"] < sl["5000"]
    # relative slopes
print("\n  same, slope as % of the probe error per unit f:")
for b, g in main.items():
    if len(g) < 5: continue
    ls = sorted(g.values(), key=lambda r: r["f"]); fs = [r["f"] for r in ls]; z = g["DJ(1)"]; line = "  %-13s tau %-4g" % b
    for k in KS:
        j = min(range(len(LAMS)), key=lambda i: z["probe"][k]["mse"][i]); s, _ = fit(fs, [r["probe"][k]["mse"][j] for r in ls])
        line += " | k%-4s %+6.2f%%" % (k, 100*s/z["probe"][k]["mse"][j])
    print(line)
print("\nSWAP (readout|hidden), change of own test loss against DJ(1)|DJ(1)")
for r in rows:
    if r["kind"] != "swap": continue
    M = r["M"]; b0 = M["DJ(1)|DJ(1)"][0]
    for hi in ["SL(8)", "DJ(1.5)"]:
        tot = M[hi+"|"+hi][0]-b0; feat = M["DJ(1)|"+hi][0]-b0; rd = M[hi+"|DJ(1)"][0]-b0; inter = tot-feat-rd
        print("  %-13s tau %-4g -> %-8s total %+.3e features %+.3e readout %+.3e interaction %+.3e (%.0f%% of |total|) | acc: total %+.4f feat %+.4f readout %+.4f" % (
            r["combo"], r["tau"], hi, tot, feat, rd, inter, 100*abs(inter)/abs(tot) if tot else float("nan"),
            M[hi+"|"+hi][1]-M["DJ(1)|DJ(1)"][1], M["DJ(1)|"+hi][1]-M["DJ(1)|DJ(1)"][1], M[hi+"|DJ(1)"][1]-M["DJ(1)|DJ(1)"][1]))
        if hi == "SL(8)":
            cnt["W1 interaction < 20% of total"] += abs(inter) < 0.2*abs(tot); cnt["W2a features part has sign of total"] += feat*tot > 0
            cnt["W2b |features| > |readout|"] += abs(feat) > abs(rd); cnt["features part lowers own loss (SL8)"] += feat < 0; cnt["readout part lowers own loss (SL8)"] += rd < 0
print("\nCOUNTS over %d blocks" % nb)
for k, v in cnt.items(): print("  %-40s %d" % (k, v))
