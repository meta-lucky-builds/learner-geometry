import json, sys
import numpy as np
from collections import defaultdict
R = json.load(open(sys.argv[1]))["conditions"]
NK = 640.0
print("(a) edge check: eta*lam_max/(2*640) at task 0, median over tasks 1..20, final  [GD family, all conditions]")
g = defaultdict(list)
for c in R.values():
    if c["learner"] in ("GD", "GD-half", "GD-quarter", "MOM") and c["model"] != "linear":
        t = c["tasks"]
        v = [c["lr"] * x["lam_max"] / (2 * NK) for x in t]
        g[(c["model"], c["seq"], c["eps"], c["learner"], c["N"])].append((v[0], float(np.median(v[1:])), v[-1], min(v[1:]), max(v[1:])))
for k in sorted(g, key=lambda x: (x[0], x[3], x[1], str(x[2]), x[4])):
    a = np.array(g[k])
    print("   %-5s %-10s %s eps=%-4s N=%-2d: task0 %s | median %s | final %s | min %.3f max %.3f" % (k[0], k[3], k[1], k[2], k[4], np.round(a[:, 0], 3), np.round(a[:, 1], 3), np.round(a[:, 2], 3), a[:, 3].min(), a[:, 4].max()))
print("\n(b) tanh ADAM N=8 seed 0 time course; tanh GN N=1 seed 0")
for L, N in [("ADAM", 8), ("ADAM", 1), ("GN", 1), ("GD", 1)]:
    c = [c for c in R.values() if c["model"] == "tanh" and c["learner"] == L and c["N"] == N and c["seed"] == 0][0]
    t = c["tasks"]
    print("  ", L, "N=%d" % N)
    for nm, fn in [("A", lambda x: x["A_raw"] - t[0]["A_raw"]), ("p10", lambda x: x["lam_p10"] / t[0]["lam_p10"]), ("p50", lambda x: x["lam_p50"] / t[0]["lam_p50"]), ("p90", lambda x: x["lam_p90"] / t[0]["lam_p90"]), ("lmax", lambda x: x["lam_max"] / t[0]["lam_max"]), ("psi", lambda x: x["psi"] / t[0]["psi"]), ("steps", lambda x: x["steps"])]:
        print("     %-6s" % nm, " ".join("%.3g" % fn(x) for x in t))
print("\n(c) weight norm proxy: W_move_cum final and D_raw_cum final, lam_max ratio, trK ratio (relu R, seed 0)")
for L in ["GD", "GN", "MOM", "ADAM"]:
    for N in [1, 4, 16]:
        cc = [c for c in R.values() if c["model"] == "relu" and c["learner"] == L and c["N"] == N and c["seq"] == "R" and c["seed"] == 0]
        if not cc:
            continue
        t = cc[0]["tasks"]
        print("   %-5s N=%-2d W %.1f Draw %.1f D %.1f lmax %.2f tr %.2f" % (L, N, t[-1]["W_move_cum"], t[-1]["D_raw_cum"], t[-1]["D_K_cum"], t[-1]["lam_max"] / t[0]["lam_max"], t[-1]["trK"] / t[0]["trK"]))
print("\n(d) ADAM relu R N=1: task at which p10 ratio first < 1e-3, dead1 just before, per seed; same for MOM, ADAM-R, all N")
for c in sorted(R.values(), key=lambda c: (c["learner"], c["N"], c["seed"])):
    if c["model"] != "relu" or c["seq"] != "R" or c["learner"] not in ("ADAM", "ADAM-R", "MOM"):
        continue
    t = c["tasks"]
    hit = [x["m"] for x in t if x["lam_p10"] / t[0]["lam_p10"] < 1e-3]
    d1 = [x["dead_frac"][0] for x in t]
    d2 = [x["dead_frac"][1] for x in t]
    if hit:
        m = hit[0]
        print("   %-6s N=%-2d s%d: collapse at task %d; dead1 before/at %.3f/%.3f dead2 %.3f/%.3f; final dead1 %.3f" % (c["learner"], c["N"], c["seed"], m, d1[m - 1], d1[m], d2[m - 1], d2[m], d1[-1]))
    else:
        print("   %-6s N=%-2d s%d: no collapse; final dead1 %.3f dead2 %.3f min p10 ratio %.3g" % (c["learner"], c["N"], c["seed"], d1[-1], d2[-1], min(x["lam_p10"] / t[0]["lam_p10"] for x in t)))
