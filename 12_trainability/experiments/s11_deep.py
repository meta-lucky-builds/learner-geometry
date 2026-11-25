import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import json, glob, os, math, sys
U = DATA + "/spec11_rig/spec11_out/units/"
def rd(n):
    return json.load(open(U + n + ".json"))
W = ["W%d" % i for i in range(1, 9)]; E = ["E%d" % i for i in range(1, 7)]
F = {"DJ(1.5)": -1.0, "DJ(1)": 0.0, "DJ(0.5)": 1/3, "SL(2)": .25, "SL(4)": .375, "SL(8)": .4375, "AJ(0.5)": .5, "AJ(1)": 1.0, "AJ(1.5)": 1.5, "SLa(2,1)": .75, "SLa(8,1)": .5625}
def eig_sym(G):
    import numpy as np
    w, v = np.linalg.eigh(np.array(G)); return w[::-1], v[:, ::-1]
for c in ("tanh-init-s0", "gelu-init-s0"):
    g = rd("G__" + c); names = g["names_s1"]; G = g["G_s1"]; n = len(names)
    w, v = eig_sym(G); cosm = [[G[i][j] / math.sqrt(G[i][i] * G[j][j]) for j in range(n)] for i in range(n)]
    off = [cosm[i][j] for i in range(n) for j in range(i)]
    wc, vc = eig_sym(cosm)
    print("\n=== %s: Gram of the 14 nearest-point shifts: off-diagonal cosines min %.3f mean %.3f max %.3f; eigenvalue shares of the cosine matrix: %s; top eigenvector entries min/max %.3f/%.3f" % (
        c, min(off), sum(off) / len(off), max(off), " ".join("%.3f" % (x / n) for x in wc[:4]), min(abs(vc[:, 0])), max(abs(vc[:, 0]))))
    print("   |vNPe|^2 x 2/s1^2 = self stiffening M_uu: " + " ".join("%s %.3e" % (names[i], 2 * G[i][i] / g["s1"] ** 2) for i in range(n)))
    # ---- Part B: mean relative cost vs effective time (1-2f) * (number of excursions)
    print("   PART B: mean over 8 probes of q_w(m)/q_w(0) - 1 ; effective time t = (1-2f) x 2m")
    rows = {}
    for ln in ("DJ(1.5)", "DJ(1)", "SL(2)", "SL(8)", "AJ(0.5)", "AJ(1)"):
        r = rd("B__%s__%s" % (c, ln)); q0 = r["q0"]; vals = []
        for cp in r["checkpoints"]:
            vals.append(sum(cp["q"][x] / q0[x] - 1 for x in W) / 8)
        rows[ln] = vals; k = [cp["ksum"] for cp in r["checkpoints"]]; k0 = r["ksum0"]
        print("     %-8s f %+.4f: " % (ln, F[ln]) + " ".join("%+.3e" % x for x in vals) + " | per unit t at m=1: %+.3e, m=6: %+.3e | A: %.4f -> %.4f lamN %.4f -> %.4f trK %.0f -> %.0f lam1 %.1f -> %.1f" % (
            (vals[0] / ((1 - 2 * F[ln]) * 2) if F[ln] != .5 else float("nan")), (vals[5] / ((1 - 2 * F[ln]) * 12) if F[ln] != .5 else float("nan")), k0["A"], k[-1]["A"], k0["lamN"], k[-1]["lamN"], k0["trK"], k[-1]["trK"], k0["lam1"], k[-1]["lam1"]))
    # collapse: list (t, value) sorted
    pts = sorted([((1 - 2 * F[ln]) * 2 * (m + 1), rows[ln][m], ln, m + 1) for ln in rows for m in range(6) if F[ln] != .5])
    print("     collapse table (t, mean relative cost change, learner, m): " + "  ".join("%+.2f:%+.4f(%s,%d)" % p for p in pts))
    # exponential test: ln(1 + x) / t
    print("     growth rate ln(1+x)/t: " + "  ".join("%s[%s]" % (ln, " ".join("%.5f" % (math.log(1 + rows[ln][m]) / ((1 - 2 * F[ln]) * 2 * (m + 1))) for m in range(6))) for ln in rows if F[ln] != .5))
    # ---- Part C
    print("   PART C: per learner Q0 (start), Q per task, final")
    for ln in ("DJ(1.5)", "DJ(1)", "SL(2)", "SL(4)", "SL(8)", "SLa(8,1)"):
        r = rd("C__%s__%s" % (c, ln)); Q0 = sum(math.log(r["start"]["q"][x]) for x in W) / 8; k0 = r["start"]["ksum"]
        Q = [sum(math.log(cp["q"][x]) for x in W) / 8 for cp in r["tasks"]]; fin = r.get("final", {})
        sm = [cp["s_m"] for cp in r["tasks"]]; dist = [cp["dist"] for cp in r["tasks"]]
        print("     %-9s f %+.4f 1-2f %+.4f | Q0 %.4f A0 %.4f lamN0 %.4f | Q: %s | A6 %.4f lamN6 %.4f lam1_6 %.0f trK6 %.0f dist6 %.3f | s_m %s | final loss %.5f acc %.4f" % (
            ln, F[ln], 1 - 2 * F[ln], Q0, k0["A"], k0["lamN"], " ".join("%.3f" % x for x in Q), r["tasks"][-1]["ksum"]["A"], r["tasks"][-1]["ksum"]["lamN"], r["tasks"][-1]["ksum"]["lam1"],
            r["tasks"][-1]["ksum"]["trK"], dist[-1], " ".join("%.2f" % x for x in sm), fin.get("test_loss", float("nan")), fin.get("test_acc", float("nan"))))
    # ---- Part A kernel changes per unit (1-2f), direction E1, s1
    print("   PART A kernel changes at s1 per unit (1-2f): dtrK, dlam1, dlamN, dA")
    for u in ("E1", "E2"):
        out = []
        for ln in ("DJ(1.5)", "DJ(1)", "DJ(0.5)", "SL(2)", "SL(8)", "AJ(1)", "AJ(1.5)", "SLa(2,1)"):
            r = rd("A__%s__%s__s1__%s" % (c, u, ln)); d = r["dksum"]; g_ = 1 - 2 * F[ln]
            out.append("%s %+.3f/%+.4f/%+.5f/%+.2e" % (ln, d["trK"] / g_, d["lam1"] / g_, d["lamN"] / g_, d["A"] / g_))
        print("     %s: " % u + "  ".join(out))
