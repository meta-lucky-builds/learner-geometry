import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import torch, glob, json, collections
recs = {}
for fn in glob.glob(DATA + "/keep/spec10_out/records_*.json"):
    for r in json.load(open(fn)) if True else []:
        pass
cnt = collections.Counter(); nblk = collections.Counter()
LAMS = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
shown = False
for fn in sorted(glob.glob(DATA + "/keep/spec10_out/bundle_*.pt")):
    b = torch.load(fn, map_location="cpu", weights_only=False)
    for cn, v in b["combos"].items():
        lab = v["labels"].long(); Y = torch.zeros(lab.numel(), 10, dtype=v["F0"].dtype); Y[torch.arange(lab.numel()), lab] = 1.0
        ends = v["ends"]; taus = sorted({k.split("|")[2] for k in ends if k.startswith("large|ZERO|")}, key=lambda s: float(s[3:]))
        for tk in taus:
            g = lambda nm: ends.get("large|ZERO|%s|%s" % (tk, nm))
            a, c = g("DJ(1)"), g("SL(4)")
            F0 = v["F0"]
            if a is None or c is None: continue
            if not shown:
                shown = True
                for off, nm in ((0, "as stored"), (1, "minus F0")):
                    o = a - off * v["F0"]; print("format check %s %s DJ(1) %s: loss %.7f acc %.4f" % (cn, tk, nm, float(((o - Y) ** 2).sum()) / (2 * o.numel()), float((o.argmax(1) == lab).double().mean())))
            def la(o): return float(((o - Y) ** 2).sum()) / (2 * o.numel()), float((o.argmax(1) == lab).double().mean())
            res = [la(a - F0 + lam * (c - a)) for lam in LAMS]
            extra = {nm: la(g(nm) - F0) for nm in ["DJ(0.25)", "DJ(1.5)", "TJ(1.5,1)", "SLa(4,2)", "AJ(1)"] if g(nm) is not None}
            best = min(range(len(LAMS)), key=lambda i: res[i][0]); n = "n128" if "n128" in cn else "n32"; tau = float(tk[3:])
            print("%-18s %-7s loss at lam %s: %s | acc: %s | best lam %g | real learners: %s" % (cn, tk, LAMS, " ".join("%.5f" % r[0] for r in res), " ".join("%.4f" % r[1] for r in res), LAMS[best],
                  " ".join("%s %.5f/%.4f" % (k, x[0], x[1]) for k, x in extra.items())))
            if tau >= 0.5:
                nblk[n] += 1; cnt[n, "best lam > 1"] += LAMS[best] > 1; cnt[n, "acc(lam 2) >= acc(lam 1)"] += res[3][1] >= res[1][1]
                cnt[n, "loss(lam 2) < loss(lam 1)"] += res[3][0] < res[1][0]; cnt[n, "acc(lam 3) >= acc(lam 1)"] += res[4][1] >= res[1][1]
for (n, k), val in sorted(cnt.items()): print("%s  %-28s %d / %d" % (n, k, val, nblk[n]))
