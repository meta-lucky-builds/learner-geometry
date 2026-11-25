import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import json, math
U = DATA + "/spec11_rig/spec11_out/units/"
F = {"DJ(1.5)": -1.0, "DJ(1)": 0.0, "SL(2)": .25, "SL(8)": .4375, "AJ(0.5)": .5, "AJ(1)": 1.0}
W = ["W%d" % i for i in range(1, 9)]
def fit0(xs, ys):                     # y = 1 + b x
    b = sum(x * (y - 1) for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    ss = sum((y - 1) ** 2 for y in ys); sr = sum((y - 1 - b * x) ** 2 for x, y in zip(xs, ys)); return b, 1 - sr / ss
for c in ("tanh-init-s0", "gelu-init-s0"):
    print("\n===", c, " (t = (1-2f) x number of excursions)")
    P = {k: ([], []) for k in ("lamN", "invq", "trK", "lam1", "q")}
    for ln, f in F.items():
        r = json.load(open(U + "B__%s__%s.json" % (c, ln))); k0 = r["ksum0"]; q0 = r["q0"]; g = 1 - 2 * f
        lam = [cp["ksum"]["lamN"] / k0["lamN"] for cp in r["checkpoints"]]; tr = [cp["ksum"]["trK"] / k0["trK"] for cp in r["checkpoints"]]
        l1 = [cp["ksum"]["lam1"] / k0["lam1"] for cp in r["checkpoints"]]
        iq = [sum(q0[x] / cp["q"][x] for x in W) / 8 for cp in r["checkpoints"]]; qq = [sum(cp["q"][x] / q0[x] for x in W) / 8 for cp in r["checkpoints"]]
        ts = [g * 2 * (m + 1) for m in range(len(lam))]
        print("  %-8s lamN/lamN0: %s | T_N from each point: %s" % (ln, " ".join("%.4f" % x for x in lam), " ".join(("%.1f" % (t / (1 - x)) if abs(1 - x) > 1e-9 else "-") for t, x in zip(ts, lam))))
        print("  %-8s mean q0/q:  %s | T* from each point: %s | trK/trK0 last %.4f lam1/lam1_0 last %.4f" % ("", " ".join("%.4f" % x for x in iq), " ".join(("%.1f" % (t / (1 - x)) if abs(1 - x) > 1e-9 else "-") for t, x in zip(ts, iq)), tr[-1], l1[-1]))
        if g != 0:
            for key, arr in (("lamN", lam), ("invq", iq), ("trK", tr), ("lam1", l1), ("q", qq)):
                P[key][0].extend(ts); P[key][1].extend(arr)
    for key in ("lamN", "invq", "q", "trK", "lam1"):
        b, r2 = fit0(*P[key]); print("  pooled straight line through (0,1) for %-5s: slope %+.5f per unit t (zero crossing at t = %.1f), R^2 %.4f  [30 points, 5 learners]" % (key, b, -1 / b, r2))
