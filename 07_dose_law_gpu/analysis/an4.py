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


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X)); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - res.var() / y.var(), b


relu = [c for c in R.values() if c["model"] == "relu"]
rows = []
for c in relu:
    t = c["tasks"]; m = np.arange(len(t))
    s1 = np.array([max(1 - x["dead_frac"][0], 1e-4) for x in t]); s2 = np.array([max(1 - x["dead_frac"][1], 1e-4) for x in t])
    y1 = -np.log(s1 / s1[0]); y2 = -np.log(s2 / s2[0])
    h1 = np.polyfit(m, y1, 1); lin1 = 1 - np.var(y1 - np.polyval(h1, m)) / max(np.var(y1), 1e-12)
    # exponential-survival vs linear-death fit quality
    d1 = 1 - s1; l1 = np.polyfit(m, d1, 1); linD = 1 - np.var(d1 - np.polyval(l1, m)) / max(np.var(d1), 1e-12)
    fs = [x["first_step_dtheta"] or 0.0 for x in t[1:]]
    rows.append(dict(L=c["learner"], N=c["N"], seq=c["seq"], eps=c["eps"], seed=c["seed"], h1=y1[-1] / 20, h2=y2[-1] / 20, R2exp=lin1, R2lin=linD,
                     Dtask=c["summary"]["D_cum"] / 20, Dbot=sum(x["D_band"]["BOT"] for x in t[1:]) / 20, fs=float(np.mean(fs)), fsmax=float(np.max(fs)),
                     W=t[-1]["W_move_cum"] / 20, dF=float(np.mean([x["first_step_dF"] or 0.0 for x in t[1:]])), A=c["summary"]["A_final"],
                     logpsi=math.log(t[-1]["psi"] / t[0]["psi"]), lmax=math.log(t[-1]["lam_max"] / t[0]["lam_max"]), er=math.log(t[-1]["eff_rank"] / t[0]["eff_rank"]),
                     steps=c["summary"]["steps_total"] / 20, lr=c["lr"], y1=y1[-1], y2=y2[-1], dead1=1 - s1[-1], dead2=1 - s2[-1]))
print("(1) unit survival in layer 1: hazard per task h1 = -ln(S_final/S_0)/20; R2 of exponential-survival fit vs linear-death fit (median over seeds)")
g = defaultdict(list)
for r in rows:
    g[(r["L"], r["seq"], r["eps"], r["N"])].append(r)
for k in sorted(g, key=lambda x: (x[0], x[1], str(x[2]), x[3])):
    v = g[k]; md = lambda f: float(np.median([x[f] for x in v]))
    print("   %-10s %s eps=%-4s N=%-2d h1=%.4f h2=%.4f R2exp=%.3f R2lin=%.3f | Dtask=%.3f Dbot=%.3f first|dtheta|=%.3f W/task=%.2f steps/task=%.0f" % (k[0], k[1], k[2], k[3], md("h1"), md("h2"), md("R2exp"), md("R2lin"), md("Dtask"), md("Dbot"), md("fs"), md("W"), md("steps")))
print("\n(2) which per-task variable orders the layer-1 hazard?  Spearman over all ReLU runs (n=%d), then within learner" % len(rows))
for f in ["Dtask", "Dbot", "fs", "fsmax", "W", "dF", "steps"]:
    line = "   %-6s all %+.3f" % (f, sp([r["h1"] for r in rows], [r[f] for r in rows]))
    for L in ["GD", "MOM", "ADAM", "ADAM-R", "GN"]:
        v = [r for r in rows if r["L"] == L]
        line += " | %s %+.3f" % (L, sp([r["h1"] for r in v], [r[f] for r in v]))
    print(line)
print("   log-log slope of h1 on Dtask within learner (R sequence only, by N) and (E sequence, by eps):")
for L in ["GN", "MOM", "ADAM", "GD"]:
    for seq in ["R", "E"]:
        v = [r for r in rows if r["L"] == L and r["seq"] == seq and r["h1"] > 1e-4]
        if len(v) > 5:
            b = np.polyfit(np.log([r["Dtask"] for r in v]), np.log([r["h1"] for r in v]), 1)
            b2 = np.polyfit(np.log([r["fs"] + 1e-9 for r in v]), np.log([r["h1"] for r in v]), 1) if all(r["fs"] > 0 for r in v) else [float("nan")]
            print("     %-5s %s: slope vs dose %.2f ; slope vs first-step size %.2f ; n=%d" % (L, seq, b[0], b2[0], len(v)))
print("\n(3) GN, sequence R: A_final vs N (median over seeds): fits a + b/N and a + c ln N")
N = np.array([1, 2, 4, 8, 16.0]); A = np.array([np.median([r["A"] for r in rows if r["L"] == "GN" and r["seq"] == "R" and r["N"] == n]) for n in N])
ra, ba = r2([1 / N], A); rb, bb = r2([np.log(N)], A)
print("   A =", np.round(A, 3), "| a+b/N: a=%.3f b=%.3f R2=%.3f | a+c lnN: a=%.3f c=%.3f R2=%.3f" % (ba[0], ba[1], ra, bb[0], bb[1], rb))
for L in ["GD", "MOM"]:
    Nn = np.array([1, 2, 4, 8, 16, 32.0]); Aa = np.array([np.median([r["A"] for r in rows if r["L"] == L and r["seq"] == "R" and r["N"] == n]) for n in Nn])
    ra, ba = r2([1 / Nn], Aa); rb, bb = r2([np.log(Nn)], Aa); rc, bc = r2([Nn ** -0.5], Aa)
    print("   %s A =" % L, np.round(Aa, 3), "| a+b/N R2=%.3f (a=%.2f b=%.2f) | a+c lnN R2=%.3f | a+b/sqrtN R2=%.3f" % (ra, ba[0], ba[1], rb, rc))
print("\n(4) two-channel model for ReLU, all 129 runs: A_final regressed on y1 = -ln(surviving frac layer 1), y2 (layer 2), ln(eff-rank ratio), ln(lam_max ratio)")
y = np.array([r["A"] for r in rows])
for nm, cols in [("dose only (ln D)", ["lnD"]), ("y1", ["y1"]), ("y1+y2", ["y1", "y2"]), ("y1+y2+er", ["y1", "y2", "er"]), ("y1+y2+er+lmax", ["y1", "y2", "er", "lmax"]), ("er+lmax", ["er", "lmax"])]:
    for r in rows:
        r["lnD"] = math.log(r["Dtask"])
    rr, b = r2([np.array([r[c] for r in rows]) for c in cols], y)
    print("   %-18s R2=%.3f coef=%s" % (nm, rr, np.round(b, 3)))
print("   same for log(psi_final/psi_0) (real trainability):")
y = np.array([r["logpsi"] for r in rows])
for nm, cols in [("dose only (ln D)", ["lnD"]), ("y1", ["y1"]), ("y1+y2", ["y1", "y2"]), ("y1+y2+er", ["y1", "y2", "er"]), ("y1+y2+er+lmax", ["y1", "y2", "er", "lmax"]), ("er+lmax", ["er", "lmax"]), ("lmax", ["lmax"])]:
    rr, b = r2([np.array([r[c] for r in rows]) for c in cols], y)
    print("   %-18s R2=%.3f coef=%s" % (nm, rr, np.round(b, 3)))
print("\n(5) tanh, all 18 runs: A and log psi vs ln(lam_max ratio), ln(eff rank ratio), ln(p10 ratio)")
tr = []
for c in R.values():
    if c["model"] == "tanh":
        t = c["tasks"]
        tr.append((c["summary"]["A_final"], math.log(t[-1]["psi"] / t[0]["psi"]), math.log(t[-1]["lam_max"] / t[0]["lam_max"]), math.log(t[-1]["eff_rank"] / t[0]["eff_rank"]), math.log(t[-1]["lam_p10"] / t[0]["lam_p10"]), math.log(c["summary"]["D_cum"])))
T = np.array(tr)
for j, nm in [(2, "ln lmax"), (3, "ln effrank"), (4, "ln p10"), (5, "ln dose")]:
    print("   %-10s: Spearman with A %+.3f, with log psi %+.3f" % (nm, sp(T[:, 0], T[:, j]), sp(T[:, 1], T[:, j])))
print("\n(6) psi vs sharpness across the GD family + GN (relu R N=1): ln psi ratio and ln lmax ratio per run")
for r in sorted(rows, key=lambda r: (r["L"], r["seed"])):
    if r["seq"] == "R" and r["N"] == 1 and r["L"] in ("GD", "GD-half", "GD-quarter", "GN"):
        print("   %-10s s%d ln lmax %+.2f ln psi %+.2f  A %.2f dead1 %.2f dead2 %.2f" % (r["L"], r["seed"], r["lmax"], r["logpsi"], r["A"], r["dead1"], r["dead2"]))
