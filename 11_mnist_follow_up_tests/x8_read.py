"""Reader for X8 (test 1). Weights (A2, A3, B3) of every learner from its step sizes and evaluation points; rank and fits."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import torch, glob, os, sys, re
OUT = OUT + "/x8_out"
def steps(name):
    m = re.match(r"(\w+)\(([^)]*)\)", name); kind = m.group(1); a = [float(x) for x in m.group(2).split(",")]
    if kind == "SL": return [(1.0 / a[0], 0.0)] * int(a[0])
    if kind == "SLa": return [(1.0 / a[0], a[1])] * int(a[0])
    if kind == "AJ": return [(1.0, a[0])]
    if kind == "DJ":
        c = a[0]; out = []; k = 0
        while abs(c * (1 - c) ** k) > 1e-15 and k < 5000: out.append((c * (1 - c) ** k, 0.0)); k += 1
        return out
    if kind == "TJ":
        c, al = a; r = (1 - (1 - al) * c) / (1 + al * c); out = []; k = 0
        while abs((1 - r) * r ** k) > 1e-15 and k < 5000: out.append(((1 - r) * r ** k, al)); k += 1
        return out
    raise ValueError(name)
def weights(name):
    c = q = A3 = B3 = 0.0                      # c = sum of earlier steps, q = A2 of earlier steps
    for d, al in steps(name):
        e = c + al * d; A3 += d * (q + al * d * e); B3 += 0.5 * d * e * e; q += d * e; c += d
    return [q, A3, B3]
EXPL = ["SL(2)", "SL(3)", "SL(4)", "SL(8)", "DJ(0.5)", "DJ(0.75)", "DJ(1.25)", "DJ(1.5)"]
IMPL = ["AJ(0.5)", "AJ(1)", "SLa(2,1)", "SLa(4,2)", "TJ(1.5,1)"]
key = lambda nm: nm.replace("(", "_").replace(")", "").replace(",", "_")
print("weights (A2, A3, B3):  " + "  ".join("%s %.4f/%.4f/%.4f" % ((n,) + tuple(weights(n))) for n in EXPL + IMPL))
blocks = sorted({"_".join(os.path.basename(f).split("_")[:2]) for f in glob.glob(OUT + "/*.pt")})
for blk in blocks:
    ld = lambda nm: (torch.load("%s/%s_%s.pt" % (OUT, blk, key(nm)), map_location="cpu", weights_only=False) if os.path.exists("%s/%s_%s.pt" % (OUT, blk, key(nm))) else None)
    ref = ld("DJ(1)")
    if ref is None: print(blk, "no reference"); continue
    have = {}
    for nm in EXPL + IMPL:
        x = ld(nm)
        if x is not None and x["status"] == "CONVERGED": have[nm] = (x["out"] - ref["out"]).reshape(-1)
        elif x is not None: print("   %s %s excluded: %s" % (blk, nm, x["status"]))
    print("\n%s : %d explicit + %d implicit learners usable" % (blk, sum(n in have for n in EXPL), sum(n in have for n in IMPL)))
    for label, names in (("EXPLICIT", [n for n in EXPL if n in have]), ("ALL", [n for n in EXPL + IMPL if n in have])):
        if len(names) < 4: continue
        D = torch.stack([have[n] for n in names]); nr = D.norm(dim=1); tot = float((D * D).sum())
        sv = torch.linalg.svdvals(D); svn = torch.linalg.svdvals(D / nr[:, None])
        print(" %-8s (%d) singular values / largest: %s" % (label, len(names), " ".join("%.4f" % float(x / sv[0]) for x in sv[:7])))
        print("          normalised vectors:         %s" % " ".join("%.4f" % float(x / svn[0]) for x in svn[:7]))
        W = torch.tensor([weights(n) for n in names], dtype=D.dtype)
        for mn, cols in (("A2", W[:, :1]), ("A2, A3", W[:, :2]), ("A2, B3", W[:, [0, 2]]), ("A2, A3, B3", W)):
            R = D - cols @ torch.linalg.lstsq(cols, D).solution
            worst = max(range(len(names)), key=lambda i: float((R[i] * R[i]).sum() / (D[i] * D[i]).sum()))
            print("          functions weighted by %-11s residual %.5f of sum|D|^2 ; worst learner %s %.4f" % (mn, float((R * R).sum()) / tot, names[worst], float((R[worst] * R[worst]).sum() / (D[worst] * D[worst]).sum())))
        # model-free: best rank-k approximation residuals
        s2 = (sv ** 2); print("          best possible with k free functions: " + " ".join("k=%d %.5f" % (k, float(s2[k:].sum() / s2.sum())) for k in (1, 2, 3, 4)))
    if "AJ(1)" in have and "SL(8)" in have:
        print("          |D| rms: " + " ".join("%s %.2e" % (n, float(have[n].norm()) / have[n].numel() ** 0.5) for n in have))
