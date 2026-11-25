"""SPEC10 report: reads every records_*.json written by spec10_run.py and judges them. CPU only, runs no learner.

Prints, and writes into the same directory as spec10_report.txt and spec10_results.json:
  1. the validations V1-V7 (SPEC10 sec. 7), with a banner INVALID if any fails, any worker file is missing or incomplete
     (the number of workers M is read from the records), or the test hook SPEC10_BREAK was set in any worker;
  2. the verdict table HZ1-HZ4 (sec. 8) with counts per block, per combination and per tau;
  3. the stated expectations E1-E8 (sec. 9), report only;
  4. the per-block tables.
Every number comes from the records; thresholds are those of SPEC10 (constants below) and CFG as stored by the worker.

usage:  python spec10_report.py [dir]      (dir = the spec10_out folder; default ./spec10_out)
        part9lib is imported only for its spearman(); it is looked up in PART9LIB_DIR, else next to this script.
"""
import os, sys, json, glob, math

sys.path.insert(0, os.environ.get("PART9LIB_DIR", os.path.dirname(os.path.abspath(__file__))))
import part9lib as L

D = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "spec10_out"))

# ------------------------------------------------------------------ SPEC10 constants (section in brackets)
V1_TOL, V2_TOL, V4_TOL, V5_TOL, V6_TOL, V7_TOL = 1e-9, 1e-9, 1e-7, 1e-7, 1e-12, 1e-12      # sec. 6, 7
HZ_FRAC = {"HZ1": 0.75, "HZ2": 0.75, "HZ3": 0.875, "HZ4": 0.75}                           # sec. 8 (24/32, 28/32)
E2_FRAC, E3_FRAC, E3_MIN_LEARNERS, E4_FRAC, E5_FRAC = 0.875, 0.70, 6, 0.625, 0.75          # sec. 9 (28/32, 70%, 6, 20/32, 24/32)
REF, Q25, Q15, SL4 = "DJ(1)", "DJ(0.25)", "DJ(1.5)", "SL(4)"
BAD_STATUS = ("DIVERGED", "AJ-FAILED", "AJ-INCONSISTENT", "ERROR")

LINES = []


def p(s=""):
    print(s, flush=True)
    LINES.append(s)


def fin(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def rel(a, b):
    if not fin(a, b):
        return float("nan")
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else float("inf"))


def f(x, w=10, prec=3, sign=True):
    if not fin(x):
        return ("%" + str(w) + "s") % "-"
    return ("%" + ("+" if sign else "") + str(w) + "." + str(prec) + "e") % x


def thr(frac, n):
    return int(math.ceil(frac * n - 1e-12))


# ------------------------------------------------------------------ load
files = sorted(glob.glob(os.path.join(D, "records_*.json")))
if not files:
    p("spec10_report: no records_*.json in %s" % D)
    sys.exit(1)
recs = [L.read_json(fp) for fp in files]
CFG = recs[0]["cfg"]
C = CFG["N_CLASSES"]
RES_MAX = CFG["G4_RES_MAX"]
M = recs[0]["n_workers"]
invalid = []
if any(r["n_workers"] != M for r in recs):
    invalid.append("records disagree on the number of workers")
ks = sorted(r["worker"] for r in recs)
miss_w = [k for k in range(M) if k not in ks]
if miss_w:
    invalid.append("worker file(s) missing: %s" % ["records_%d.json" % k for k in miss_w])
inc = [r["worker"] for r in recs if not r.get("complete")]
if inc:
    invalid.append("worker(s) not complete: %s" % inc)
brk = sorted({r.get("BREAK", "") for r in recs if r.get("BREAK")})
if brk:
    invalid.append("TEST HOOK SPEC10_BREAK=%s was set" % ",".join(brk))
assigned = [c for r in recs for c in r["combos_assigned"]]
if not miss_w and len(assigned) != len(CFG["G4_COMBOS"]):
    invalid.append("%d of %d G4 combinations assigned to the workers" % (len(assigned), len(CFG["G4_COMBOS"])))
combos = {}
for r in recs:
    combos.update(r["combos"])
missing_c = [c for c in assigned if c not in combos]
if missing_c:
    invalid.append("combination(s) without records: %s" % missing_c)
lin = [r["linear"] for r in recs if r["worker"] == 0 and r.get("linear")]
lin = lin[0] if lin else None
TAUS = sorted(CFG["G4_TAUS"])
names = [c for c in sorted(combos, key=lambda c: (-combos[c]["n"], assigned.index(c) if c in assigned else 0))]
n_expected_blocks = len(CFG["G4_COMBOS"]) * len(TAUS)


def small_usable(e):
    return (e.get("learner_status") == "CONVERGED" and e.get("np_status") == "CONVERGED" and e.get("th_status") == "CONVERGED"
            and fin(e.get("rho"), e.get("loss_NP"), e.get("loss_TH"), e.get("acc_NP"), e.get("acc_TH")))


def usable(e):
    return e.get("learner_status") not in BAD_STATUS and fin(e.get("train_res")) and e["train_res"] <= RES_MAX


def v4_ok(sp):
    terms = (sp["const"], sp["dN"], sp["dS"], sp["drest"], sp["quad"])
    return fin(sum(terms), sp["dl"]) and abs(sum(terms) - sp["dl"]) <= V4_TOL * max([abs(x) for x in terms] + [abs(sp["dl"])])


def blocks():
    for cn in names:
        for b in combos[cn]["blocks"]:
            yield cn, b


# ------------------------------------------------------------------ 1. validations
val = {}
p("SPEC10 report: %s   (%d worker file(s) of M = %d; mode %s; data_source %s; devices %s)" % (
    D, len(recs), M, recs[0]["mode"], sorted({r["data_source"] for r in recs}), sorted({r["device"] for r in recs})))
p("")
# V1
v1 = [(cn, b["tau"], e["learner"], e.get("v1_rel_loss"), e.get("v1_rel_dist")) for cn, b in blocks() for e in b["PLAIN"]["ends"]]
v1_vals = [max(x[3], x[4]) if fin(x[3], x[4]) else float("nan") for x in v1]
v1_exp = len(names) * len(TAUS) * 2
v1_max = max(v1_vals) if v1_vals and all(fin(x) for x in v1_vals) else float("nan")
val["V1"] = dict(passed=bool(len(v1) == v1_exp and v1_exp > 0 and fin(v1_max) and v1_max <= V1_TOL), max=v1_max, n=len(v1),
                 n_expected=v1_exp, rows=v1)
# V2
v2 = [(cn, x["sign"], x["rel_loss_NP"], x["rel_loss_TH"], x["rel_rho"]) for cn in names for x in combos[cn]["small"]["V2"]]
v2_vals = [max(x[2:]) if fin(*x[2:]) else float("nan") for x in v2]
v2_max = max(v2_vals) if v2_vals and all(fin(x) for x in v2_vals) else (float("nan") if v2_vals else 0.0)
val["V2"] = dict(passed=bool(fin(v2_max) and v2_max <= V2_TOL), max=v2_max, n=len(v2), rows=v2)
# V3
tr = {"PLAIN": [], "ZERO": []}
zr = []
for cn in names:
    for kind in ("PLAIN", "ZERO"):
        for e in combos[cn]["small"][kind]["signs"].values():
            if small_usable(e):
                tr[kind].append(e.get("train_res", float("nan")))
                if kind == "ZERO":
                    zr.append(e.get("zero_res", float("nan")))
    for _, b in [(cn, b) for b in combos[cn]["blocks"]]:
        for kind in ("PLAIN", "ZERO"):
            for e in b[kind]["ends"]:
                if usable(e):
                    tr[kind].append(e["train_res"])
                    if kind == "ZERO":
                        zr.append(e.get("zero_res", float("nan")))
v3_tr = {k: (max(v) if v and all(fin(x) for x in v) else float("nan")) for k, v in tr.items()}
v3_zr = max(zr) if zr and all(fin(x) for x in zr) else float("nan")
val["V3"] = dict(passed=bool(all(fin(v3_tr[k]) and v3_tr[k] <= RES_MAX for k in tr) and fin(v3_zr) and v3_zr <= RES_MAX),
                 max_train_res=v3_tr, max_zero_res=v3_zr, n_zero=len(zr),
                 zero_check_passed=bool(fin(v3_zr) and v3_zr <= RES_MAX))
# V4
v4 = []
for cn in names:
    for kind in ("PLAIN", "ZERO"):
        for key, e in combos[cn]["small"][kind]["signs"].items():
            if e.get("split"):
                v4.append((cn, "small", kind, key, v4_ok(e["split"])))
    for b in combos[cn]["blocks"]:
        for kind in ("PLAIN", "ZERO"):
            for ln, sp in b[kind]["splits"].items():
                v4.append((cn, "tau%g" % b["tau"], kind, ln, v4_ok(sp)))
val["V4"] = dict(passed=bool(v4 and all(x[4] for x in v4)), n=len(v4), failed=[x[:4] for x in v4 if not x[4]])
# V5
v5 = lin["max_ratio"] if lin else float("nan")
val["V5"] = dict(passed=bool(fin(v5) and v5 <= V5_TOL), max_ratio=v5, ratios=lin["ratios"] if lin else None,
                 present=lin is not None)
# V6
v6 = []
for cn in names:
    for kind in ("PLAIN", "ZERO"):
        for key, e in combos[cn]["small"][kind]["signs"].items():
            for pre in ("L_", "NP_", "TH_"):
                if pre + "loss" in e:
                    v6.append(rel(e[pre + "loss"], e[pre + "loss_explicit"]))
    for b in combos[cn]["blocks"]:
        for kind in ("PLAIN", "ZERO"):
            for e in b[kind]["ends"]:
                if "loss" in e:
                    v6.append(rel(e["loss"], e["loss_explicit"]))
for e in (lin["ends"] if lin else []):
    if "loss" in e:
        v6.append(rel(e["loss"], e["loss_explicit"]))
v6_max = max(v6) if v6 and all(fin(x) for x in v6) else float("nan")
val["V6"] = dict(passed=bool(fin(v6_max) and v6_max <= V6_TOL), max=v6_max, n=len(v6))
# V7
v7 = [(cn, rel(combos[cn]["V7"]["loss"], 1.0 / (2 * C)), rel(combos[cn]["V7"]["loss_explicit"], 1.0 / (2 * C))) for cn in names]
v7_max = max(max(a, b) for _, a, b in v7) if v7 and all(fin(a, b) for _, a, b in v7) else float("nan")
val["V7"] = dict(passed=bool(fin(v7_max) and v7_max <= V7_TOL and len(v7) == len(assigned)), max=v7_max, n=len(v7))

for k in sorted(val):
    if not val[k]["passed"]:
        invalid.append("%s failed" % k)
if invalid:
    p("#" * 110)
    p("###  INVALID: this run may not be interpreted.  Reasons: %s" % "; ".join(invalid))
    p("#" * 110)
else:
    p("All validations pass; all %d worker file(s) present and complete; no test hook set." % M)
p("")
p("1. VALIDATIONS (SPEC10 sec. 7)")
P = lambda ok: "PASS" if ok else "FAIL"
p("  V1 PLAIN reproduction of the stored G4 records (DJ(1), DJ(1.5), every block): %s  max rel. difference (loss, dist) = %s "
  "over %d of %d end points (tol %g)" % (P(val["V1"]["passed"]), f(v1_max, 9, 2, False), len(v1), v1_exp, V1_TOL))
for x in v1:
    if not (fin(x[3], x[4]) and max(x[3], x[4]) <= V1_TOL):
        p("      failing: %s tau=%g %s rel loss %s rel dist %s" % (x[0], x[1], x[2], f(x[3], 9, 2, False), f(x[4], 9, 2, False)))
p("  V2 PLAIN reproduction of the stored Part A records (loss_NP, loss_TH, rho; both signs; n = N_TRAIN): %s  max rel. "
  "difference = %s over %d comparisons%s (tol %g)" % (P(val["V2"]["passed"]), f(v2_max, 9, 2, False), len(v2),
                                                      " (no stored Part A record: not applicable)" if not v2 else "", V2_TOL))
p("  V3 same training outputs: %s  max train_res of usable end points PLAIN %s, ZERO %s; ZERO explicit "
  "|(F - F0) - tau Y|/|tau Y| max %s over %d usable ZERO end points (%s) (limit G4_RES_MAX = %g)" % (
      P(val["V3"]["passed"]), f(v3_tr["PLAIN"], 9, 2, False), f(v3_tr["ZERO"], 9, 2, False), f(v3_zr, 9, 2, False), len(zr),
      P(val["V3"]["zero_check_passed"]), RES_MAX))
p("  V4 sum check of every split: %s  %d splits, %d failed %s" % (P(val["V4"]["passed"]), len(v4), len(val["V4"]["failed"]),
                                                                  val["V4"]["failed"] if val["V4"]["failed"] else ""))
p("  V5 linear control (ZERO, tau = %s): %s  max |F_test(L) - F_test(DJ(1))| / |F_test(DJ(1)) - F0t| = %s %s (tol %g)" % (
    lin["tau"] if lin else "-", P(val["V5"]["passed"]), f(v5, 9, 2, False),
    {k: "%.2e" % v for k, v in lin["ratios"].items()} if lin else "(MISSING: worker 0 records)", V5_TOL))
p("  V6 loss two ways, every end point: %s  max rel. difference %s over %d end points (tol %g)" % (
    P(val["V6"]["passed"]), f(v6_max, 9, 2, False), len(v6), V6_TOL))
p("  V7 ZERO model at theta0, loss = 1/(2C) = %g: %s  max rel. difference %s over %d combinations (tol %g)" % (
    1.0 / (2 * C), P(val["V7"]["passed"]), f(v7_max, 9, 2, False), len(v7), V7_TOL))
p("")

# ------------------------------------------------------------------ 2. rules HZ1-HZ4
lossZ, lossP, accZ, lstarZ, endZ = {}, {}, {}, {}, {}
for cn, b in blocks():
    key = (cn, b["tau"])
    endZ[key] = {e["learner"]: e for e in b["ZERO"]["ends"] if usable(e)}
    lossZ[key] = {ln: e["loss"] for ln, e in endZ[key].items()}
    accZ[key] = {ln: e["acc"] for ln, e in endZ[key].items()}
    lstarZ[key] = {ln: e["loss_star"] for ln, e in endZ[key].items()}
    lossP[key] = {ln: r["loss_L"] for ln, r in b["stored"].items()
                  if not r["missing"] and usable(dict(learner_status=r["learner_status"], train_res=r["train_res"]))}


def rule(name, needZ, needP, test):
    rows = []
    for cn, b in blocks():
        key = (cn, b["tau"])
        if all(x in lossZ[key] for x in needZ) and all(x in lossP[key] for x in needP):
            rows.append((cn, b["tau"], bool(test(lossZ[key], lossP[key]))))
    n = len(rows)
    t = thr(HZ_FRAC[name], n)
    cnt = sum(r[2] for r in rows)
    return dict(count=cnt, n_blocks=n, threshold=t, passed=bool(n > 0 and cnt >= t), rows=rows)


rules = {
    "HZ1": rule("HZ1", [Q25, REF], [], lambda z, _: z[Q25] < z[REF]),
    "HZ2": rule("HZ2", [Q15, REF], [], lambda z, _: z[Q15] > z[REF]),
    "HZ3": rule("HZ3", [Q25, REF], [Q25, REF], lambda z, pl: (z[Q25] - z[REF]) < (pl[Q25] - pl[REF])),
    "HZ4": rule("HZ4", [SL4, REF], [], lambda z, _: z[SL4] < z[REF]),
}
TEXT = {"HZ1": "lossZ[DJ(0.25)] < lossZ[DJ(1)]", "HZ2": "lossZ[DJ(1.5)] > lossZ[DJ(1)]",
        "HZ3": "lossZ[DJ(0.25)] - lossZ[DJ(1)] < lossP[DJ(0.25)] - lossP[DJ(1)]", "HZ4": "lossZ[SL(4)] < lossZ[DJ(1)]"}
p("2. PRE-REGISTERED RULES (ZERO model; a block counts only if the learners are usable)%s" % ("   [RUN INVALID]" if invalid else ""))
p("   %-5s %-66s %7s %8s %10s  %s" % ("rule", "inequality", "count", "blocks", "threshold", "verdict"))
for name, rr in rules.items():
    p("   %-5s %-66s %7d %8d %10d  %s%s" % (name, TEXT[name], rr["count"], rr["n_blocks"], rr["threshold"],
                                          "PASS" if rr["passed"] else "FAIL",
                                          "" if rr["n_blocks"] == n_expected_blocks else
                                          "   (only %d of %d blocks count: threshold = %.1f%% rounded up)" % (
                                              rr["n_blocks"], n_expected_blocks, 100 * HZ_FRAC[name])))
p("   counts per combination (holds/counted) and per tau; 'maj' = combinations in which the majority of tau values agrees:")
for name, rr in rules.items():
    pc = {cn: [r[2] for r in rr["rows"] if r[0] == cn] for cn in names}
    pt = {tau: [r[2] for r in rr["rows"] if r[1] == tau] for tau in TAUS}
    maj = sum(1 for v in pc.values() if v and sum(v) > len(v) / 2.0)
    rr.update(per_combo={cn: [sum(v), len(v)] for cn, v in pc.items()}, per_tau={str(t): [sum(v), len(v)] for t, v in pt.items()},
              majority_combos=maj)
    p("   %s: %s | %s | maj %d of %d" % (name, "  ".join("%s %d/%d" % (cn, sum(v), len(v)) for cn, v in pc.items()),
                                       "  ".join("tau=%g %d/%d" % (t, sum(v), len(v)) for t, v in pt.items()), maj, len(names)))
p("")

# ------------------------------------------------------------------ 3. expectations E1-E8
exp = {}
p("3. STATED EXPECTATIONS (report only)")
# E1
e1 = []
for cn in names:
    sm = combos[cn]["small"]
    e1.append((cn, combos[cn]["n"], sm["ZERO"]["b"], sm["PLAIN"]["b"]))
big = [x for x in e1 if x[1] != CFG["N_TRAIN"]]
nt = [x for x in e1 if x[1] == CFG["N_TRAIN"]]
c_big = sum(1 for x in big if fin(x[2]) and x[2] < 0)
c_nt_neg = sum(1 for x in nt if fin(x[2]) and x[2] < 0)
c_nt_pos = sum(1 for x in nt if fin(x[2]) and x[2] > 0)
c_pl = sum(1 for x in e1 if fin(x[3]) and x[3] > 0)
exp["E1"] = dict(rows=[dict(combo=x[0], n=x[1], b_ZERO=x[2], b_PLAIN=x[3],
                            ratio=x[2] / x[3] if fin(x[2], x[3]) and x[3] != 0 else float("nan")) for x in e1],
                 b_ZERO_neg_large_n=[c_big, len(big)], b_ZERO_sign_ntrain=[c_nt_neg, c_nt_pos, len(nt)], b_PLAIN_pos=[c_pl, len(e1)])
p("  E1 small moves: b_ZERO < 0 in %d of %d combinations with n != N_TRAIN (expected all); n = N_TRAIN: %d negative, %d positive "
  "of %d (expected mixed); b_PLAIN > 0 in %d of %d (expected all)" % (c_big, len(big), c_nt_neg, c_nt_pos, len(nt), c_pl, len(e1)))
for r in exp["E1"]["rows"]:
    p("      %-22s n=%-4d b_ZERO %s  b_PLAIN %s  b_ZERO/b_PLAIN %s" % (r["combo"], r["n"], f(r["b_ZERO"]), f(r["b_PLAIN"]),
                                                                       f(r["ratio"], 9, 3)))
# E2
e2 = [(cn, b["tau"], b["ZERO"]["splits"][Q25]) for cn, b in blocks() if Q25 in b["ZERO"]["splits"]]
n2 = len(e2)
c_sr = sum(1 for x in e2 if x[2]["dS"] + x[2]["drest"] < 0)
c_gp = sum(1 for x in e2 if x[2]["gN"] > 0)
c_gm = sum(1 for x in e2 if x[2]["gN"] < 0)
c_en = sum(1 for x in e2 if x[2]["eN"] > 0)
exp["E2"] = dict(n=n2, dS_plus_drest_neg=c_sr, gN_pos=c_gp, gN_neg=c_gm, eN_pos=c_en, threshold=thr(E2_FRAC, n2))
p("  E2 split ZERO DJ(0.25) vs DJ(1): dS + drest < 0 in %d of %d (expected >= %d); gN > 0 in %d, < 0 in %d (no fixed sign "
  "expected); eN > 0 in %d of %d (expected >= %d)" % (c_sr, n2, thr(E2_FRAC, n2), c_gp, c_gm, c_en, n2, thr(E2_FRAC, n2)))
# E3
e3 = []
for cn, b in blocks():
    key = (cn, b["tau"])
    us = endZ[key]
    best = min(us, key=lambda ln: us[ln]["loss"]) if us else None
    rho = L.spearman([us[ln]["f_nominal"] for ln in us], [us[ln]["loss"] for ln in us]) if len(us) >= E3_MIN_LEARNERS else float("nan")
    e3.append((cn, b["tau"], len(us), rho, best))
e3n = [x for x in e3 if x[2] >= E3_MIN_LEARNERS]
c3 = sum(1 for x in e3n if fin(x[3]) and x[3] < 0)
exp["E3"] = dict(n_blocks_6plus=len(e3n), count_neg=c3, threshold=thr(E3_FRAC, len(e3n)),
                 rows=[dict(combo=x[0], tau=x[1], n_usable=x[2], spearman=x[3], best=x[4]) for x in e3])
p("  E3 Spearman(f_nominal, lossZ) < 0 in %d of %d blocks with >= %d usable ZERO learners (expected >= %d); lowest lossZ per block:"
  % (c3, len(e3n), E3_MIN_LEARNERS, thr(E3_FRAC, len(e3n))))
for x in e3:
    p("      %-22s tau=%-5g usable %d  Spearman %s  lowest lossZ: %s" % (x[0], x[1], x[2], f(x[3], 7, 3), x[4]))
# E4, E5, E7
c4 = [(k, accZ[k][Q25] > accZ[k][REF]) for k in accZ if Q25 in accZ[k] and REF in accZ[k]]
c5 = [(k, lstarZ[k][Q25] < lstarZ[k][REF]) for k in lstarZ if Q25 in lstarZ[k] and REF in lstarZ[k]
      and fin(lstarZ[k][Q25], lstarZ[k][REF])]
c7 = [(k, lossZ[k][REF] < lossP[k][REF]) for k in lossZ if REF in lossZ[k] and REF in lossP[k]]
exp["E4"] = dict(count=sum(x[1] for x in c4), n=len(c4), threshold=thr(E4_FRAC, len(c4)))
exp["E5"] = dict(count=sum(x[1] for x in c5), n=len(c5), threshold=thr(E5_FRAC, len(c5)))
exp["E7"] = dict(count=sum(x[1] for x in c7), n=len(c7))
p("  E4 accZ[DJ(0.25)] > accZ[DJ(1)] in %d of %d blocks (expected >= %d)" % (exp["E4"]["count"], len(c4), exp["E4"]["threshold"]))
p("  E5 loss_star ZERO DJ(0.25) < DJ(1) in %d of %d blocks (expected >= %d)" % (exp["E5"]["count"], len(c5), exp["E5"]["threshold"]))
# E6
tau6 = TAUS[0]
c6z, c6p = [], []
for cn in names:
    b = [b for b in combos[cn]["blocks"] if b["tau"] == tau6][0]
    z = {e["learner"]: e.get("trK") for e in b["ZERO"]["ends"] if usable(e)}
    if all(x in z and fin(z[x]) for x in (Q15, REF, Q25)):
        c6z.append((cn, z[Q15] < z[REF] < z[Q25]))
    st = {ln: r.get("trK_L") for ln, r in b["stored"].items() if not r["missing"]
          and usable(dict(learner_status=r["learner_status"], train_res=r["train_res"]))}
    if all(x in st and fin(st[x]) for x in (Q15, REF, Q25)):
        c6p.append((cn, st[Q15] < st[REF] < st[Q25]))
exp["E6"] = dict(tau=tau6, zero=[sum(x[1] for x in c6z), len(c6z)], plain=[sum(x[1] for x in c6p), len(c6p)])
p("  E6 trK rises with f among DJ(1.5), DJ(1), DJ(0.25) at the smallest tau of the grid (tau = %g): ZERO %d of %d combinations "
  "(expected all), PLAIN (stored) %d of %d" % (tau6, exp["E6"]["zero"][0], len(c6z), exp["E6"]["plain"][0], len(c6p)))
p("  E7 lossZ[DJ(1)] < lossP[DJ(1)] (stored) in %d of %d blocks (expected all)" % (exp["E7"]["count"], len(c7)))
# E8
exp["E8"] = {}
p("  E8 best move size per learner (tau of the grid with the lowest lossZ, usable end points only), learners in order of f:")
for cn in names:
    per = {}
    for b in combos[cn]["blocks"]:
        for e in b["ZERO"]["ends"]:
            if usable(e):
                per.setdefault(e["learner"], dict(f=e["f_nominal"], taus={}))["taus"][b["tau"]] = e["loss"]
    rows = sorted(((v["f"], ln, min(v["taus"], key=v["taus"].get)) for ln, v in per.items()), key=lambda x: x[0])
    mono = all(rows[i][2] <= rows[i + 1][2] for i in range(len(rows) - 1)) if len(rows) > 1 else None
    strict_any = any(rows[i][2] < rows[i + 1][2] for i in range(len(rows) - 1))
    exp["E8"][cn] = dict(best=[dict(f=x[0], learner=x[1], best_tau=x[2]) for x in rows], nondecreasing_in_f=mono,
                         any_increase=strict_any)
    p("      %-22s %s  -> best tau non-decreasing in f: %s%s" % (
        cn, "  ".join("%s(f=%+.3f):%g" % (x[1], x[0], x[2]) for x in rows), mono,
        "" if not mono else (" (strictly larger somewhere)" if strict_any else " (all equal)")))
p("")

# ------------------------------------------------------------------ 4. per-block tables
p("4. PER-BLOCK TABLES")
for cn in names:
    cb = combos[cn]
    p("")
    p("=== %s  (n = %d, n_test = %d; PLAIN test loss at theta0 %.6f, accuracy %.4f; V7 ZERO loss at theta0 %.15f)" % (
        cn, cb["n"], cb["n_test"], cb["loss0_plain"], cb["acc0_plain"], cb["V7"]["loss"]))
    p("  small-move block (AJ(1), |tau| = S2_FRAC = %g; lo = TH, hi = NP; b = mean over signs of loss_NP - loss_TH)" % CFG["S2_FRAC"])
    p("   %-5s %4s %-15s %-10s %-10s %8s %10s %10s %7s %7s | %10s = %10s %10s %10s %10s %10s | %10s %10s %10s %6s" % (
        "kind", "sign", "AJ status", "NP", "TH", "rho", "loss_TH", "loss_NP", "acc_TH", "acc_NP", "NP-TH", "const", "dN", "dS",
        "drest", "quad", "eN", "eS", "gN", "R2"))
    for kind in ("PLAIN", "ZERO"):
        sm = cb["small"][kind]
        for key, e in sm["signs"].items():
            sp = e.get("split")
            tail = "%s = %s %s %s %s %s | %s %s %s %6.3f" % (
                f(sp["dl"]), f(sp["const"], 10, 2), f(sp["dN"], 10, 2), f(sp["dS"], 10, 2), f(sp["drest"], 10, 2),
                f(sp["quad"], 10, 2), f(sp["eN"], 10, 2), f(sp["eS"], 10, 2), f(sp["gN"], 10, 2), sp["R2"]) if sp else "-"
            p("   %-5s %4s %-15s %-10s %-10s %8.5f %10.7f %10.7f %7.4f %7.4f | %s" % (
                kind, key, e.get("learner_status"), e.get("np_status", "-"), e.get("th_status", "-"), e.get("rho", float("nan")),
                e.get("loss_TH", float("nan")), e.get("loss_NP", float("nan")), e.get("acc_TH", float("nan")),
                e.get("acc_NP", float("nan")), tail))
        a = sm["split_avg"]
        p("   %-5s b = %s%s" % (kind, f(sm["b"]), "   sign-averaged split: const %s dN %s dS %s drest %s quad %s | eN %s eS %s gN %s gS %s"
                                 % tuple(f(a[k], 10, 2) for k in ("const", "dN", "dS", "drest", "quad", "eN", "eS", "gN", "gS"))
                                 if a else "   (b not defined: a sign is not usable)"))
    for b in cb["blocks"]:
        p("  tau = %g" % b["tau"])
        for s in b["skips"]:
            p("     ZERO %s skipped: %s" % (s["learner"], s["reason"]))
        p("   %-5s %-11s %7s %-15s %9s %9s %9s %10s %7s %7s %10s %9s | %10s = %10s %10s %10s %10s %10s | %10s %10s %6s" % (
            "kind", "learner", "f_nom", "status", "train res", "zero res", "V1 rel", "loss", "acc", "a*", "loss*", "trK",
            "loss-lo", "const", "dN", "dS", "drest", "quad", "eN", "eS", "R2"))
        for kind in ("PLAIN", "ZERO"):
            for e in b[kind]["ends"]:
                sp = b[kind]["splits"].get(e["learner"])
                v1r = max(e["v1_rel_loss"], e["v1_rel_dist"]) if fin(e.get("v1_rel_loss"), e.get("v1_rel_dist")) else float("nan")
                tail = ("(reference)" if e["learner"] == REF else "%s = %s %s %s %s %s | %s %s %6.3f" % (
                    f(sp["dl"]), f(sp["const"], 10, 2), f(sp["dN"], 10, 2), f(sp["dS"], 10, 2), f(sp["drest"], 10, 2),
                    f(sp["quad"], 10, 2), f(sp["eN"], 10, 2), f(sp["eS"], 10, 2), sp["R2"]) if sp else "-")
                p("   %-5s %-11s %+7.4f %-15s %s %s %s %10.7f %7.4f %7.3f %10.7f %s | %s%s" % (
                    kind, e["learner"], e["f_nominal"], e.get("learner_status"), f(e.get("train_res"), 9, 2, False),
                    f(e.get("zero_res"), 9, 2, False), f(v1r, 9, 1, False) if kind == "PLAIN" else "%9s" % "-",
                    e.get("loss", float("nan")), e.get("acc", float("nan")), e.get("a_star", float("nan")),
                    e.get("loss_star", float("nan")), f(e.get("trK"), 9, 2, False), tail,
                    ("   ERROR: " + e["error"]) if e.get("error") else ""))
        p("   stored PLAIN G4 records: " + "  ".join(
            "%s %s" % (ln, "missing" if r["missing"] else "%.7f%s" % (r["loss_L"], "" if usable(dict(
                learner_status=r["learner_status"], train_res=r["train_res"])) else "(not usable)"))
            for ln, r in b["stored"].items()))
if lin:
    p("")
    p("=== linear control %s (ZERO, tau = %g): %s" % (lin["cname"], lin["tau"], "  ".join(
        "%s %s train res %s zero res %s" % (e["learner"], e.get("learner_status"), f(e.get("train_res"), 9, 2, False),
                                            f(e.get("zero_res"), 9, 2, False)) for e in lin["ends"])))
p("")
p("RUN %s" % ("INVALID: " + "; ".join(invalid) if invalid else "VALID"))

res = dict(dir=D, invalid=bool(invalid), invalid_reasons=invalid, BREAK=brk, n_workers=M, worker_files=ks,
           validations=val, rules=rules, expectations=exp, mode=recs[0]["mode"],
           data_source=sorted({r["data_source"] for r in recs}))
L.write_json_atomic(os.path.join(D, "spec10_results.json"), res)
with open(os.path.join(D, "spec10_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES) + "\n")
print("spec10_report: wrote %s and %s" % (os.path.join(D, "spec10_report.txt"), os.path.join(D, "spec10_results.json")))
