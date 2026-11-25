import json, sys, math
import numpy as np
from collections import defaultdict
R = json.load(open(sys.argv[1]))["conditions"]


def rk(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rk(a[m]), rk(b[m]))[0, 1]) if m.sum() > 3 else float("nan")


def band(t, which):
    return sum(t["D_band"][b] for b in which)


def dz(t, i):
    d = t["dead_frac"]; return d[i] if len(d) > i else 0.0


C = list(R.values())
print("(1) H3 redone in the small-damage regime: sequence E, damage after task m vs eps (median over seeds); slope of log A_m on log eps")
for L in ["GN", "GD", "ADAM"]:
    for m in [1, 2, 4, 20]:
        xs, ys, ds = [], [], []
        for eps in [0.05, 0.1, 0.2, 0.4]:
            v = [c for c in C if c["model"] == "relu" and c["seq"] == "E" and c["learner"] == L and c["eps"] == eps]
            a = np.median([c["tasks"][m]["A_raw"] - c["tasks"][0]["A_raw"] for c in v]); d = np.median([c["tasks"][m]["D_K_cum"] for c in v])
            xs.append(eps); ys.append(a); ds.append(d)
        ys = np.array(ys); ds = np.array(ds)
        sl = np.polyfit(np.log(xs), np.log(ys), 1)[0] if (ys > 0).all() else float("nan")
        sd = np.polyfit(np.log(xs), np.log(ds), 1)[0]
        print("   %-4s m=%-2d A=%s  D_cum=%s  slope A %.2f  slope D %.2f" % (L, m, np.round(ys, 4), np.round(ds, 4), sl, sd))

print("\n(2) FIRST-TASK test (same start, one task): damage after task 1 vs dose of task 1, across ALL relu conditions; per seed Spearman")
for nm, dfn in [("D_K total", lambda t: t["D_K"]), ("D_bot", lambda t: band(t, ["BOT"])), ("D_mid+bot", lambda t: band(t, ["MID", "BOT"])), ("D_top", lambda t: band(t, ["TOP"])), ("D_raw", lambda t: t["D_raw"]), ("W_move", lambda t: t["W_move"]), ("first dtheta", lambda t: t["first_step_dtheta"] or 0), ("d dead1", None), ("d dead1+2", None)]:
    line = "   %-12s" % nm
    for scope, sel in [("all learners", lambda c: True), ("GD+GN only", lambda c: c["learner"] in ("GD", "GN", "GD-half", "GD-quarter")), ("no Adam", lambda c: not c["learner"].startswith("ADAM"))]:
        vals = []
        for s in [0, 1, 2]:
            v = [c for c in C if c["model"] == "relu" and c["seed"] == s and sel(c)]
            a = [c["tasks"][1]["A_raw"] - c["tasks"][0]["A_raw"] for c in v]
            if nm == "d dead1":
                d = [dz(c["tasks"][1], 0) - dz(c["tasks"][0], 0) for c in v]
            elif nm == "d dead1+2":
                d = [dz(c["tasks"][1], 0) - dz(c["tasks"][0], 0) + dz(c["tasks"][1], 1) - dz(c["tasks"][0], 1) for c in v]
            else:
                d = [dfn(c["tasks"][1]) for c in v]
            vals.append(sp(a, d))
        line += " | %s: %s (n=%d/seed)" % (scope, np.round(vals, 2), len(v))
    print(line)
print("   first-task table, seed 0 (sorted by D_bot): learner N eps | D_K D_top D_bot | A_1 | d dead1 d dead2 | lmax ratio")
v = [c for c in C if c["model"] == "relu" and c["seed"] == 0]
for c in sorted(v, key=lambda c: band(c["tasks"][1], ["BOT"])):
    t0, t1 = c["tasks"][0], c["tasks"][1]
    print("     %-10s N=%-2d eps=%-4s | %8.4f %8.4f %8.4f | %+.3f | %+.3f %+.3f | %.2f" % (c["learner"], c["N"], c["eps"], t1["D_K"], band(t1, ["TOP"]), band(t1, ["BOT"]), t1["A_raw"] - t0["A_raw"], dz(t1, 0) - dz(t0, 0), dz(t1, 1) - dz(t0, 1), t1["lam_max"] / t0["lam_max"]))

print("\n(3) H5 redone (N=1, five learners, final): Spearman per seed with different dose definitions; relu then tanh (all 6 tanh conditions)")
for nm, dfn in [("D_K total", lambda c: c["summary"]["D_cum"]), ("D_bot", lambda c: sum(band(t, ["BOT"]) for t in c["tasks"][1:])), ("D_mid+bot", lambda c: sum(band(t, ["MID", "BOT"]) for t in c["tasks"][1:])),
                ("D_bot pre-collapse", lambda c: sum(band(t, ["BOT"]) for t in c["tasks"][1:] if t["lam_p10"] / c["tasks"][0]["lam_p10"] > 0.05))]:
    r1 = []
    for s in [0, 1, 2]:
        v = [c for c in C if c["model"] == "relu" and c["seq"] == "R" and c["N"] == 1 and c["seed"] == s and c["learner"] in ("GD", "MOM", "ADAM", "ADAM-R", "GN")]
        r1.append(sp([c["summary"]["A_final"] for c in v], [dfn(c) for c in v]))
    r2 = []
    for s in [0, 1, 2]:
        v = [c for c in C if c["model"] == "tanh" and c["seed"] == s]
        r2.append(sp([c["summary"]["A_final"] for c in v], [dfn(c) for c in v]))
    v = [c for c in C if c["model"] == "tanh"]
    print("   %-18s relu per seed %s | tanh per seed %s | tanh pooled 18 runs %.3f (vs log psi %.3f)" % (nm, np.round(r1, 2), np.round(r2, 2), sp([c["summary"]["A_final"] for c in v], [dfn(c) for c in v]), sp([math.log(c["tasks"][-1]["psi"]) for c in v], [dfn(c) for c in v])))

print("\n(4) H6 redone (pooled task records, relu R): Spearman(A_m, cumulative dose) under variants; compare with task index")
def pooled(sel, dfn, precollapse=False):
    A, D, M = [], [], []
    for c in C:
        if c["model"] != "relu" or c["seq"] != "R" or not sel(c):
            continue
        t = c["tasks"]; cum = 0.0
        for x in t[1:]:
            cum += dfn(x)
            if precollapse and x["lam_p10"] / t[0]["lam_p10"] < 0.05:
                break
            A.append(x["A_raw"] - t[0]["A_raw"]); D.append(cum); M.append(x["m"])
    return sp(A, D), sp(A, M), len(A)
for scope, sel in [("all 5 learners (as H6)", lambda c: c["learner"] in ("GD", "MOM", "ADAM", "ADAM-R", "GN")), ("GD+GN", lambda c: c["learner"] in ("GD", "GN")), ("GD+GN+MOM", lambda c: c["learner"] in ("GD", "GN", "MOM")), ("GN only", lambda c: c["learner"] == "GN"), ("GD only", lambda c: c["learner"] == "GD"), ("MOM only", lambda c: c["learner"] == "MOM"), ("ADAM only", lambda c: c["learner"] == "ADAM")]:
    line = "   %-24s" % scope
    for nm, dfn in [("D_K", lambda x: x["D_K"]), ("D_bot", lambda x: band(x, ["BOT"])), ("D_mid+bot", lambda x: band(x, ["MID", "BOT"]))]:
        a, b, n = pooled(sel, dfn)
        a2, b2, n2 = pooled(sel, dfn, True)
        line += " | %s %.3f (pre-collapse %.3f)" % (nm, a, a2)
    line += " | task index %.3f n=%d" % (b, n)
    print(line)

print("\n(5) own-optimizer trainability: steps needed for task 20 / task 1 (median over seeds), relu R; and unfit micro-targets")
g = defaultdict(list)
for c in C:
    if c["model"] == "relu" and c["seq"] == "R":
        t = c["tasks"]; g[(c["learner"], c["N"])].append((np.mean([x["steps"] for x in t[-3:]]) / max(np.mean([x["steps"] for x in t[1:4]]), 1), c["summary"]["unfit_total"], t[-1]["psi"] / t[0]["psi"]))
for k in sorted(g):
    a = np.array(g[k]); print("   %-10s N=%-2d steps(last3)/steps(first3) %.2f  unfit %d  psi ratio %.2f" % (k[0], k[1], np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2])))

print("\n(6) within-task kernel drift (why alpha != 2): GD relu R N=1 seed 0: lam_max and p50 ratio task to task; alpha per task")
c = [c for c in C if c["model"] == "relu" and c["seq"] == "R" and c["N"] == 1 and c["seed"] == 0 and c["learner"] == "GD"][0]
print("   alpha per task:", " ".join("%.2f" % (x["alpha"] or float("nan")) for x in c["tasks"]))
for L in ["GD", "GD-half", "GD-quarter"]:
    v = [c for c in C if c["model"] == "relu" and c["seq"] == "R" and c["N"] == 1 and c["learner"] == L]
    print("   %-10s alpha at task 0 (from init, per seed): %s ; median tasks 1-20: %s" % (L, [round(c["tasks"][0]["alpha"], 2) for c in v], [round(c["summary"]["alpha_median"], 2) for c in v]))
v = [c for c in C if c["model"] == "linear" and c["learner"] == "GD"]
print("   linear control GD alpha median:", [round(c["summary"]["alpha_median"], 3) for c in v])
