"""Weight-norm random walk vs kernel growth in the GPU run (read-only on results_part5.json)."""
import json, sys, math
import numpy as np
from collections import defaultdict
R = json.load(open(sys.argv[1]))["conditions"]
C = list(R.values())
print("keys in a task record with 'norm':", [k for k in C[0]["tasks"][0] if "norm" in k or "theta" in k])


def get(model, L, N, seq="R", eps=None):
    return sorted([c for c in C if c["model"] == model and c["learner"] == L and c["N"] == N and c["seq"] == seq and c["eps"] == eps], key=lambda c: c["seed"])


print("\n(1) per learner (relu, R): ||theta|| ratio final/initial, (ratio)^4, trK ratio, lam_max ratio, p50 ratio  [median over seeds]")
for L, Ns in [("GD", [1, 4, 32]), ("GD-half", [1]), ("GD-quarter", [1]), ("GN", [1, 2, 4, 8, 16]), ("MOM", [1, 8, 32]), ("ADAM", [1, 8, 32])]:
    for N in Ns:
        v = get("relu", L, N)
        if not v:
            continue
        a = np.array([[c["tasks"][-1]["theta_norm"] / c["tasks"][0]["theta_norm"], c["tasks"][-1]["trK"] / c["tasks"][0]["trK"], c["tasks"][-1]["lam_max"] / c["tasks"][0]["lam_max"], c["tasks"][-1]["lam_p50"] / c["tasks"][0]["lam_p50"], c["tasks"][0]["theta_norm"]] for c in v])
        m = np.median(a, axis=0)
        print("   %-10s N=%-2d norm x%.3f (^4 = %.2f)  trK x%.2f  lam_max x%.2f  p50 x%.2f   ||theta_0||=%.2f" % (L, N, m[0], m[0] ** 4, m[1], m[2], m[3], m[4]))
print("\n(2) GN N=1 seed 0 time course: ||theta||^2, increment per task, first-step ||dtheta||^2, trK ratio vs (norm ratio)^4, lam_max ratio")
c = get("relu", "GN", 1)[0]
t = c["tasks"]
n2 = np.array([x["theta_norm"] ** 2 for x in t])
print("   ||theta||^2   :", " ".join("%.1f" % v for v in n2))
print("   d||theta||^2  :", " ".join("%.2f" % v for v in np.diff(n2)))
print("   first dth^2   :", " ".join("%.2f" % ((x["first_step_dtheta"] or 0) ** 2) for x in t[1:]))
print("   W_move^2/task :", " ".join("%.2f" % (x["W_move"] ** 2) for x in t[1:]))
print("   trK ratio     :", " ".join("%.2f" % (x["trK"] / t[0]["trK"]) for x in t))
print("   norm ratio^4  :", " ".join("%.2f" % ((x["theta_norm"] / t[0]["theta_norm"]) ** 4) for x in t))
print("   lam_max ratio :", " ".join("%.2f" % (x["lam_max"] / t[0]["lam_max"]) for x in t))
print("\n(3) growth exponents: OLS slope of log(quantity ratio) on log(m) for m=2..20 (median over seeds)")
for L, N in [("GN", 1), ("GN", 4), ("GN", 16), ("GD", 1), ("GD-quarter", 1), ("MOM", 8), ("ADAM", 1)]:
    out = []
    for c in get("relu", L, N):
        t = c["tasks"]; m = np.arange(2, 21)
        row = []
        for f in (lambda x: x["theta_norm"] ** 2 - t[0]["theta_norm"] ** 2, lambda x: x["trK"] / t[0]["trK"], lambda x: x["lam_max"] / t[0]["lam_max"]):
            y = np.array([f(t[i]) for i in m])
            row.append(np.polyfit(np.log(m), np.log(np.clip(y, 1e-9, None)), 1)[0])
        out.append(row)
    o = np.median(np.array(out), axis=0)
    print("   %-10s N=%-2d slope of [||theta||^2 - ||theta_0||^2] %.2f | trK %.2f | lam_max %.2f" % (L, N, o[0], o[1], o[2]))
print("\n(4) across ALL relu runs: Spearman of ln(trK ratio) with ln(norm ratio); OLS slope")
x = np.array([math.log(c["tasks"][-1]["theta_norm"] / c["tasks"][0]["theta_norm"]) for c in C if c["model"] == "relu"])
y = np.array([math.log(c["tasks"][-1]["trK"] / c["tasks"][0]["trK"]) for c in C if c["model"] == "relu"])
L = np.array([c["learner"] for c in C if c["model"] == "relu"])
for name in ["GD", "GN", "MOM", "ADAM", "ADAM-R"]:
    mm = L == name
    b = np.polyfit(x[mm], y[mm], 1)
    print("   %-7s n=%d slope %.2f intercept %.2f corr %.3f" % (name, mm.sum(), b[0], b[1], np.corrcoef(x[mm], y[mm])[0, 1]))
print("\n(5) tanh: norm ratio and p10 ratio")
for c in sorted([c for c in C if c["model"] == "tanh" and c["seed"] == 0], key=lambda c: (c["learner"], c["N"])):
    t = c["tasks"]
    print("   %-5s N=%d norm x%.3f trK x%.2f p10 x%.2f lam_max x%.2f" % (c["learner"], c["N"], t[-1]["theta_norm"] / t[0]["theta_norm"], t[-1]["trK"] / t[0]["trK"], t[-1]["lam_p10"] / t[0]["lam_p10"], t[-1]["lam_max"] / t[0]["lam_max"]))
