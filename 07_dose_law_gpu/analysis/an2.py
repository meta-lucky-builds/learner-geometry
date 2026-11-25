import json, sys, math
import numpy as np
R = json.load(open(sys.argv[1]))["conditions"]


def rk(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def sp(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return float("nan")
    return float(np.corrcoef(rk(a[m]), rk(b[m]))[0, 1])


def dz(t, i):
    d = t["dead_frac"]
    return d[i] if len(d) > i else 0.0


relu = [c for c in R.values() if c["model"] == "relu"]


def feats(c):
    t = c["tasks"]
    return dict(A=c["summary"]["A_final"], D=c["summary"]["D_cum"],
                Dbot=sum(x["D_band"]["BOT"] for x in t[1:]), Dtop=sum(x["D_band"]["TOP"] for x in t[1:]),
                W=t[-1]["W_move_cum"], Draw=t[-1]["D_raw_cum"], dead1=dz(t[-1], 0), dead2=dz(t[-1], 1),
                er=t[-1]["eff_rank"] / t[0]["eff_rank"], lmax=t[-1]["lam_max"] / t[0]["lam_max"],
                tr=t[-1]["trK"] / t[0]["trK"], p10=t[-1]["lam_p10"] / t[0]["lam_p10"],
                p50=t[-1]["lam_p50"] / t[0]["lam_p50"], logpsi=math.log(max(t[-1]["psi"], 1e-300)),
                maxdF=max((x["first_step_dF"] or 0.0) for x in t[1:]), maxdth=max((x["first_step_dtheta"] or 0.0) for x in t[1:]),
                steps=c["summary"]["steps_total"], alpha=c["summary"]["alpha_median"])


F = [feats(c) for c in relu]
print("(i) ReLU, condition level (n=%d): Spearman of A_final with each feature; and of log psi" % len(F))
for k in ["D", "Dbot", "Dtop", "W", "Draw", "dead1", "dead2", "er", "lmax", "tr", "p10", "p50", "maxdF", "maxdth", "steps", "alpha"]:
    print("   %-7s: vs A %+.3f   vs log psi %+.3f" % (k, sp([f["A"] for f in F], [f[k] for f in F]), sp([f["logpsi"] for f in F], [f[k] for f in F])))
print("   within learner (all N, eps, seeds): Spearman(A,D) | (A,dead1) | (A,dead2) | (A,er) | (A,p10) | (A,lmax) | n")
for L in ["GD", "MOM", "ADAM", "ADAM-R", "GN"]:
    f = [feats(c) for c in relu if c["learner"] == L or (L == "GD" and c["learner"] in ("GD-half", "GD-quarter"))]
    A = [x["A"] for x in f]
    print("   %-7s: %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %d" % (L, sp(A, [x["D"] for x in f]), sp(A, [x["dead1"] for x in f]), sp(A, [x["dead2"] for x in f]), sp(A, [x["er"] for x in f]), sp(A, [x["p10"] for x in f]), sp(A, [x["lmax"] for x in f]), len(f)))

print("\n(ii) GN (relu): pooled over all tasks, N, eps, seeds")
pts = []
for c in relu:
    if c["learner"] != "GN":
        continue
    t = c["tasks"]
    a0 = t[0]["A_raw"]
    for x in t[1:]:
        pts.append((x["D_K_cum"], x["A_raw"] - a0, x["m"], c["N"], c["eps"], x["eff_rank"] / t[0]["eff_rank"], dz(x, 0), dz(x, 1), x["lam_max"] / t[0]["lam_max"], x["W_move_cum"]))
P = np.array([[p[0], p[1], p[2], p[5], p[6], p[7], p[8], p[9]] for p in pts])
print("   n=%d  Spearman(A,D_cum)=%.3f  (A,task)=%.3f  (A,effrank)=%.3f (A,dead1)=%.3f (A,dead2)=%.3f (A,lmax)=%.3f (A,Wmove)=%.3f" % (len(P), sp(P[:, 1], P[:, 0]), sp(P[:, 1], P[:, 2]), sp(P[:, 1], P[:, 3]), sp(P[:, 1], P[:, 4]), sp(P[:, 1], P[:, 5]), sp(P[:, 1], P[:, 6]), sp(P[:, 1], P[:, 7])))
m = (P[:, 0] > 0) & (P[:, 1] > 0)
b = np.polyfit(np.log(P[m, 0]), P[m, 1], 1)
print("   fit A = %.3f ln D + %.3f ; residual sd %.3f" % (b[0], b[1], np.std(P[m, 1] - np.polyval(b, np.log(P[m, 0])))))
print("   binned by D_cum: median A (count)")
edges = [0, 1, 2, 4, 8, 16, 32, 64, 200]
sels = [("R N=1", lambda p: p[4] is None and p[3] == 1), ("R N=4", lambda p: p[4] is None and p[3] == 4), ("R N=16", lambda p: p[4] is None and p[3] == 16),
        ("E .05", lambda p: p[4] == 0.05), ("E .1", lambda p: p[4] == 0.1), ("E .2", lambda p: p[4] == 0.2), ("E .4", lambda p: p[4] == 0.4)]
for lo, hi in zip(edges[:-1], edges[1:]):
    row = "   D in [%d,%d):" % (lo, hi)
    for nm, sel in sels:
        v = [p[1] for p in pts if sel(p) and lo <= p[0] < hi]
        row += (" %s:%.2f(%d)" % (nm, np.median(v), len(v))) if v else (" %s:--" % nm)
    print(row)

print("\n(iii) GD family N=1 sequence R: per seed")
for L in ["GD", "GD-half", "GD-quarter"]:
    for c in sorted([c for c in relu if c["learner"] == L and c["N"] == 1 and c["seq"] == "R"], key=lambda c: c["seed"]):
        t = c["tasks"]
        print("   %-10s seed %d lr=%s: lmax final/init %.3f (init %.1f; max over tasks %.3f; lr*ratio %.3f) trK %.2f psi %.3f A %.3f" % (L, c["seed"], c["lr"], t[-1]["lam_max"] / t[0]["lam_max"], t[0]["lam_max"], max(x["lam_max"] for x in t) / t[0]["lam_max"], c["lr"] * t[-1]["lam_max"] / t[0]["lam_max"], t[-1]["trK"] / t[0]["trK"], t[-1]["psi"] / t[0]["psi"], c["summary"]["A_final"]))
print("   GD by N (seed 0): lmax ratio per task")
for N in [1, 4, 32]:
    c = [c for c in relu if c["learner"] == "GD" and c["N"] == N and c["seq"] == "R" and c["seed"] == 0][0]
    t = c["tasks"]
    print("     N=%d:" % N, " ".join("%.2f" % (x["lam_max"] / t[0]["lam_max"]) for x in t))

print("\n(iv) time courses, seed 0, sequence R, N=1")
for L in ["GD", "GN", "MOM", "ADAM"]:
    c = [c for c in relu if c["learner"] == L and c["N"] == 1 and c["seq"] == "R" and c["seed"] == 0][0]
    t = c["tasks"]
    print("  ", L)
    for nm, fn in [("A", lambda x: x["A_raw"] - t[0]["A_raw"]), ("dead1", lambda x: dz(x, 0)), ("dead2", lambda x: dz(x, 1)), ("lmax", lambda x: x["lam_max"] / t[0]["lam_max"]),
                   ("p10", lambda x: x["lam_p10"] / t[0]["lam_p10"]), ("effr", lambda x: x["eff_rank"] / t[0]["eff_rank"]), ("D_K", lambda x: x["D_K"]), ("psi", lambda x: x["psi"] / t[0]["psi"]), ("steps", lambda x: x["steps"])]:
        print("     %-6s" % nm, " ".join("%.3g" % fn(x) for x in t))

print("\n(viii) per-task increments (relu, all runs)")
inc = []
for c in relu:
    t = c["tasks"]
    for a, bb in zip(t[:-1], t[1:]):
        inc.append((bb["A_raw"] - a["A_raw"], dz(bb, 0) + dz(bb, 1) - dz(a, 0) - dz(a, 1), bb["D_K"], bb["D_band"]["BOT"], (bb["first_step_dF"] or 0.0), (bb["first_step_dtheta"] or 0.0), bb["W_move"], bb["steps"], c["learner"]))
I = np.array([x[:8] for x in inc], float)
for j, nm in enumerate(["D_K", "D_bot", "first dF", "first dtheta", "W_move", "steps"], start=2):
    print("   %-12s: dA %+.3f   d(dead) %+.3f" % (nm, sp(I[:, 0], I[:, j]), sp(I[:, 1], I[:, j])))
print("   dA vs d(dead): %.3f" % sp(I[:, 0], I[:, 1]))
for L in ["GD", "MOM", "ADAM", "GN"]:
    mm = np.array([x[8] == L for x in inc])
    print("   %s: dA vs D_K %+.3f; dA vs d(dead) %+.3f; d(dead) vs D_K %+.3f; d(dead) vs first dtheta %+.3f; n=%d" % (L, sp(I[mm, 0], I[mm, 2]), sp(I[mm, 0], I[mm, 1]), sp(I[mm, 1], I[mm, 2]), sp(I[mm, 1], I[mm, 5]), mm.sum()))

print("\n(ix) tanh arm per seed")
for c in sorted([c for c in R.values() if c["model"] == "tanh"], key=lambda c: (c["learner"], c["N"], c["seed"])):
    t = c["tasks"]
    print("   %-5s N=%d s%d: A %+.3f p10 %.2f p50 %.2f lmax %.2f tr %.2f effr %.2f psi %.3f sat %s dead %s steps %d/%d" % (c["learner"], c["N"], c["seed"], c["summary"]["A_final"], t[-1]["lam_p10"] / t[0]["lam_p10"], t[-1]["lam_p50"] / t[0]["lam_p50"], t[-1]["lam_max"] / t[0]["lam_max"], t[-1]["trK"] / t[0]["trK"], t[-1]["eff_rank"] / t[0]["eff_rank"], t[-1]["psi"] / t[0]["psi"], [round(v, 3) for v in t[-1]["sat_frac"]], [round(v, 3) for v in t[-1]["dead_frac"]], t[-1]["steps"], t[1]["steps"]))

print("\n(x) meta keys:", list(json.load(open(sys.argv[1]))["meta"].keys())[:40])
