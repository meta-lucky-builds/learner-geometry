import json, sys, re, math
rows = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        r = json.loads(line); rows[(r["combo"], r["tau"], r["kind"], r["learner"])] = r
start = {}
for line in open(sys.argv[2]):
    m = re.match(r"(\S+)\s+start: probe acc last hidden ([\d.]+) \(mse ([\d.]+), cka ([\d.]+)\) \| first hidden ([\d.]+)", line)
    if m: start[m.group(1)] = dict(acc2=float(m.group(2)), mse2=float(m.group(3)), cka2=float(m.group(4)), acc1=float(m.group(5)))
Ls = ["DJ(1.5)", "DJ(1)", "SL(4)", "DJ(0.25)"]; fs = [-1.0, 0.0, 0.375, 0.4286]
def fit(y):
    n = len(fs); mx = sum(fs)/n; the = sum(y)/n
    sxx = sum((x-mx)**2 for x in fs); sxy = sum((x-mx)*(v-the) for x, v in zip(fs, y))
    b = sxy/sxx; a = the - b*mx; ss = sum((v-the)**2 for v in y); sr = sum((v-a-b*x)**2 for x, v in zip(fs, y))
    return b, (1 - sr/ss if ss > 0 else float("nan"))
print("LAST HIDDEN, squared error of the refitted readout (x1e5).  'train' = f0 minus start (effect of training itself); slope = per unit f; R2 of straight line in f")
print("%-13s %-4s %-5s  start   f0     train | f-1    f.375  f.43  (minus f0) | slope   R2   | slope/|train| | first hidden: train  slope  R2 | cka2 slope R2 | cka1 slope R2" % ("start","tau","kind"))
for b in sorted({k[:3] for k in rows}):
    g = [rows[b + (L,)] for L in Ls]; s = start[b[0]]
    m2 = [r["mse2"]*1e5 for r in g]; m1 = [r["mse1"]*1e5 for r in g]
    sl2, r2 = fit(m2); sl1, r1 = fit(m1)
    c2, cr2 = fit([r["cka2"] for r in g]); c1, cr1 = fit([r["cka1"] for r in g])
    tr2 = m2[1] - s["mse2"]*1e5
    print("%-13s %-4s %-5s %6.0f %6.0f %+6.1f | %+6.1f %+6.1f %+6.1f | %+6.1f %.3f | %5.2f | %+6.1f %.3f | %+.4f %.2f | %+.4f %.2f" % (
        b[0], b[1], b[2], s["mse2"]*1e5, m2[1], tr2, m2[0]-m2[1], m2[2]-m2[1], m2[3]-m2[1], sl2, r2, abs(sl2/tr2) if tr2 else float("nan"),
        sl1, r1, c2, cr2, c1, cr1))
print()
print("ACCURACY of the refitted readout, last hidden (points = %), relative to start;  paired noise ~ +-0.15 points")
for b in sorted({k[:3] for k in rows}):
    g = [rows[b + (L,)] for L in Ls]; s = start[b[0]]
    print("%-13s %-4s %-5s start %.2f | f-1 %+.2f  f0 %+.2f  f.375 %+.2f  f.43 %+.2f || first hidden start %.2f | f-1 %+.2f f0 %+.2f f.375 %+.2f f.43 %+.2f" % (
        b[0], b[1], b[2], s["acc2"]*100, *[(r["acc2"]-s["acc2"])*100 for r in g], s["acc1"]*100, *[(r["acc1"]-s["acc1"])*100 for r in g]))
print()
print("ZERO vs PLAIN: slope of last-hidden squared error in f (x1e5)")
for cn in sorted({k[0] for k in rows}):
    for tau in sorted({k[1] for k in rows}):
        z = fit([rows[(cn, tau, "ZERO", L)]["mse2"]*1e5 for L in Ls])[0]; p = fit([rows[(cn, tau, "PLAIN", L)]["mse2"]*1e5 for L in Ls])[0]
        print("  %-13s tau %-4s ZERO %+6.1f  PLAIN %+6.1f  ratio %.2f" % (cn, tau, z, p, p/z))
print("tau scaling of the slope (tau 1 / tau .5):")
for cn in sorted({k[0] for k in rows}):
    for kind in ["ZERO", "PLAIN"]:
        a = fit([rows[(cn, 0.5, kind, L)]["mse2"]*1e5 for L in Ls])[0]; c = fit([rows[(cn, 1.0, kind, L)]["mse2"]*1e5 for L in Ls])[0]
        print("  %-13s %-5s %.2f  (exponent %.2f)" % (cn, kind, c/a, math.log(c/a)/math.log(2)))
