"""Alignment-fraction check on SPEC7 Part A records (read-only on results7.json / results7_mats).
For points on the same fibre: R_L = (lam1(L) - lam1(GN)) / (lam1(NP) - lam1(GN)).
Prediction: NP = 1 by definition, GN = 0, SL(m=64) = 0.5*(1-1/64) = 0.492,
GD(c) along eigen-direction q_i: f = (1-c_i)/(2-c_i), c_i = c*lam_i/lam_1.
Closed form for the whole path matrix of GD (linearised): C_ij = r_i r_j * c_j(1-c_i)/(c_i+c_j-c_i c_j) in the eigenbasis."""
import json, sys, os
import numpy as np
from collections import defaultdict
base = sys.argv[1]
R = json.load(open(os.path.join(base, "results7.json")))["records"]
A = [r for r in R if r.get("part") == "A" and r.get("status") == "OK"]
print("Part A records:", [(r["h"], r["seed"]) for r in A])
rows = defaultdict(list)
randrows = defaultdict(list)
for rec in A:
    lam = np.array(rec["lamA"]); l1 = lam[0]
    runs = defaultdict(dict)
    for x in rec["runs"]:
        runs[(x["dir"], x["s"], x["sign"])][x["learner"]] = x
    for (d, s, sg), L in runs.items():
        if (d, s, -sg) not in runs or sg < 0:
            continue
        M = runs[(d, s, -sg)]
        if not all(k in L and k in M for k in ("NP", "GN")):
            continue
        def ev(name, key="lam1"):
            if name not in L or name not in M:
                return None
            return 0.5 * (L[name][key] + M[name][key])
        for key in ("lam1", "trK"):
            den = ev("NP", key) - ev("GN", key)
            for name in L:
                if name in ("NP", "GN") or ev(name, key) is None:
                    continue
                ratio = (ev(name, key) - ev("GN", key)) / den
                info = rec["dirs"][d]
                if info["eig_index"] is not None:
                    if name.startswith("GD"):
                        c = float(name.replace("GD", "").strip("()_ ")) if any(ch.isdigit() for ch in name) else None
                        ci = c * info["lam_i"] / l1
                        pred = (1 - ci) / (2 - ci)
                    else:
                        pred = 0.5 * (1 - 1 / 64)
                    rows[(key, rec["h"], name, info["eig_index"], s)].append((ratio, pred))
                else:
                    randrows[(key, rec["h"], name, s)].append(ratio)
names = sorted({k[2] for k in rows})
print("learner names:", names)
for key in ("lam1", "trK"):
    print("\n== %s: ratio R_L = (L - GN)/(NP - GN), even part over +-; median over seeds [min,max]; prediction" % key)
    for h in sorted({k[1] for k in rows}):
        for s in (0.025, 0.1):
            print("  h=%d s=%.3f" % (h, s))
            for name in names:
                line = "    %-9s" % name
                for i in (1, 2, 4, 7, 10, 13, 16):
                    v = rows.get((key, h, name, i, s))
                    if not v:
                        continue
                    a = np.array(v)
                    line += " q%-2d %.3f (pred %.3f)" % (i, np.median(a[:, 0]), a[0, 1])
                print(line)
            for name in names:
                v = randrows.get((key, h, name, s))
                if v:
                    print("    %-9s random dirs: median %.3f  [%.3f, %.3f]  n=%d" % (name, np.median(v), min(v), max(v), len(v)))
# closed form of the full path matrix, random directions
print("\n== closed form C_ij = r_i r_j c_j(1-c_i)/(c_i+c_j-c_i c_j) vs measured C (random directions, s=0.025)")
for rec in A[:2]:
    mf = os.path.join(base, "results7_mats", rec["matfile"])
    if not os.path.exists(mf):
        continue
    Z = np.load(mf, allow_pickle=True)
    keys = list(Z.keys())
    if rec is A[0]:
        print("  npz keys sample:", keys[:6], "... total", len(keys))
    Q = None
    for k in keys:
        if k.lower() in ("q", "qa", "evecs", "q_a"):
            Q = Z[k]
    lam = np.array(rec["lamA"])
    print("  eigvec key found:", Q is not None)
    if Q is None:
        continue
    for x in rec["runs"]:
        if x["dir"].startswith("rand") and x["s"] == 0.025 and x["sign"] == 1 and x["learner"].startswith("GD") and x["dir"] in ("rand1", "rand2"):
            ck = [k for k in keys if x["dir"] in k and x["learner"] in k and "0.025" in k and "C" in k]
            if not ck:
                continue
            C = Z[ck[0]]
            c = float("".join(ch for ch in x["learner"] if ch.isdigit() or ch == "."))
            ci = c * lam / lam[0]
            r = Q.T @ (np.array(x["dY"]))
            pred = np.outer(r, r) * (ci[None, :] * (1 - ci[:, None])) / (ci[:, None] + ci[None, :] - ci[:, None] * ci[None, :])
            Ce = Q.T @ C @ Q
            print("  h=%d seed=%d %s %s: rel err %.3e  (key %s)" % (rec["h"], rec["seed"], x["dir"], x["learner"], np.linalg.norm(Ce - pred) / np.linalg.norm(Ce), ck[0]))
