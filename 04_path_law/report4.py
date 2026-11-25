"""Reads results4.json, writes REPORT4.md (SPEC4 section 5 verdicts S1-S6, then R1-R6)."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results4.json")
OUT = os.path.join(HERE, "REPORT4.md")

LEARNERS = ["GN", "GD0.25", "GD1", "GD1.8", "MOM"]
LNAME = {"GN": "GN", "GD0.25": "GD(0.25)", "GD1": "GD(1)", "GD1.8": "GD(1.8)", "MOM": "MOM",
         "ADAM": "ADAM"}
CLOSED = ["oab", "tri+", "tri-", "walk8"]
EPS_LIST = [0.2, 0.1, 0.05, 0.025, 0.0125]
S5_EPS = [0.1, 0.05, 0.025, 0.0125]

db = json.load(open(IN))
R = {}
for r in db["records"]:
    R[(r["grid"], r["h"], r["seed"], r["eps"], r["learner"])] = r

lines = []


def P(s=""):
    lines.append(s)


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.{d}f}"


def med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def mm(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def mmr(xs, d=4):
    a, b = mm(xs)
    return f"{fmt(med(xs), d)} [{fmt(a, d)}, {fmt(b, d)}]"


def verdict(med_ok, seed_oks):
    n = len(seed_oks)
    need = 4 if n >= 5 else (2 if n == 3 else n)
    k = sum(bool(x) for x in seed_oks)
    if med_ok and k >= need:
        return "PASS", k, n
    if med_ok:
        return "INCONCLUSIVE", k, n
    return "FAIL", k, n


def get_path(grid, h, seed, eps, learner, path):
    r = R.get((grid, h, seed, eps, learner))
    if r is None or r.get("status") != "OK":
        return None, (r.get("status") if r else "MISSING")
    p = r["paths"].get(path)
    if p is None:
        return None, "MISSING"
    return p, p["status"]


def valid(p):
    return p is not None and p["status"] == "OK" and p.get("gn_status", "OK") == "OK"


# configurations: (label, grid, h, seeds, eps, learners, closed paths)
CONF_MAIN = [(f"{h}", "MAIN", h, [0, 1, 2, 3, 4], 0.025, LEARNERS, CLOSED) for h in (32, 64, 128)]
CONF_WIDE = [("512 (Wide)", "WIDE", 512, [0, 1, 2], 0.025, ["GN", "GD1"], ["oab", "tri+", "tri-"])]
CONFS = CONF_MAIN + CONF_WIDE

nonpass = []   # per-seed details for non-PASS verdicts


# ---------------------------------------------------------------------------
def rule_cos_ratio(cmpkey, cmin, rlo, rhi, sid):
    P(f"| width | learner | " + " | ".join(f"{p}: cos / ratio (median [min,max])" for p in CLOSED)
      + " | seeds holding | verdict |")
    P("|" + "---|" * (len(CLOSED) + 4))
    details = []
    for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
        for l in lrn:
            cells, med_ok = [], True
            seed_ok = {s: True for s in seeds}
            per_seed = {s: {} for s in seeds}
            for p in CLOSED:
                if p not in paths:
                    cells.append("not in grid")
                    continue
                cs, rs = [], []
                for s in seeds:
                    pp, st = get_path(grid, h, s, eps, l, p)
                    if not valid(pp):
                        seed_ok[s] = False
                        per_seed[s][p] = st
                        continue
                    c = pp["meas"] if cmpkey == "meas" else pp["apri"]["cmp"]
                    if cmpkey == "apri" and pp["apri"]["status"] != "OK":
                        seed_ok[s] = False
                        per_seed[s][p] = "APRI " + pp["apri"]["status"]
                        continue
                    cs.append(c["cos"])
                    rs.append(c["ratio"])
                    ok = c["cos"] >= cmin and rlo <= c["ratio"] <= rhi
                    seed_ok[s] &= ok
                    per_seed[s][p] = f"{c['cos']:.5f} / {c['ratio']:.4f}"
                mc, mr = med(cs), med(rs)
                med_ok &= (math.isfinite(mc) and mc >= cmin and rlo <= mr <= rhi)
                cells.append(f"{fmt(mc, 5)} [{fmt(mm(cs)[0], 4)}, {fmt(mm(cs)[1], 4)}] / "
                             f"{fmt(mr, 4)} [{fmt(mm(rs)[0], 3)}, {fmt(mm(rs)[1], 3)}]")
            v, k, n = verdict(med_ok, [seed_ok[s] for s in seeds])
            P(f"| {lab} | {LNAME[l]} | " + " | ".join(cells) + f" | {k}/{n} | **{v}** |")
            if v != "PASS":
                details.append((lab, l, per_seed, seed_ok))
    if details:
        P()
        P(f"Per-seed values (cos / ratio) where {sid} is not PASS:")
        P()
        for lab, l, per_seed, seed_ok in details:
            for s, d in per_seed.items():
                P(f"- h={lab} {LNAME[l]} seed {s} (rule {'holds' if seed_ok[s] else 'fails'}): "
                  + "; ".join(f"{p} {v}" for p, v in d.items()))
    P()


# ---------------------------------------------------------------------------
P("# REPORT4 — the path law (SPEC4.md)")
P()
meta = db.get("meta", {})
P(f"Threads: torch.set_num_threads(2) and OMP/MKL/OPENBLAS_NUM_THREADS=2 in every process "
  f"(several worker processes ran in parallel, each with 2 threads). float64, CPU.")
P()
P("Verdict rule (SPEC4 §5 / SPEC3 §8): PASS if the rule holds for the median over seeds AND for "
  "≥ 4 of 5 seeds (2 of 3 where 3 seeds are run); INCONCLUSIVE if median only; else FAIL. "
  "A seed whose path is DIVERGED / NOT CONVERGED / INVALID / not run counts as a seed on which "
  "the rule does not hold and is left out of the medians.")
P()

# S1
P("## S1 — LAW with measured C on every closed path (cos ≥ 0.98, ratio ∈ [0.9, 1.1] on each path), ε = 0.025")
P()
rule_cos_ratio("meas", 0.98, 0.9, 1.1, "S1")

# S2
P("## S2 — LAW with a-priori C on every closed path (cos ≥ 0.97, ratio ∈ [0.85, 1.15] on each path), ε = 0.025")
P()
rule_cos_ratio("apri", 0.97, 0.85, 1.15, "S2")

# S3
P("## S3 — learner vs learner (oneway), measured C: D_meas vs LAW(C_learner − C_GN) (cos ≥ 0.98, ratio ∈ [0.9, 1.1]), ε = 0.025")
P()
P("| width | learner | cos (median [min,max]) | ratio (median [min,max]) | rel_err median | ‖D_meas‖ median | a-priori C: cos / ratio (median; report only) | seeds holding | verdict |")
P("|---|---|---|---|---|---|---|---|---|")
s3det = []
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    for l in lrn:
        if l == "GN":
            continue
        cs, rs, es, ns, ac, ar, oks = [], [], [], [], [], [], []
        ps = {}
        for s in seeds:
            pp, st = get_path(grid, h, s, eps, l, "oneway")
            if not valid(pp):
                oks.append(False)
                ps[s] = st if pp is None else f"{pp['status']} (GN {pp.get('gn_status')})"
                continue
            c = pp["meas"]
            cs.append(c["cos"]); rs.append(c["ratio"]); es.append(c["rel_err"]); ns.append(c["norm_meas"])
            if "apri" in pp and "cmp" in pp["apri"]:
                ac.append(pp["apri"]["cmp"]["cos"]); ar.append(pp["apri"]["cmp"]["ratio"])
            ok = c["cos"] >= 0.98 and 0.9 <= c["ratio"] <= 1.1
            oks.append(ok)
            ps[s] = f"{c['cos']:.5f} / {c['ratio']:.4f}"
        mok = math.isfinite(med(cs)) and med(cs) >= 0.98 and 0.9 <= med(rs) <= 1.1
        v, k, n = verdict(mok, oks)
        P(f"| {lab} | {LNAME[l]} | {mmr(cs, 5)} | {mmr(rs)} | {fmt(med(es))} | {fmt(med(ns))} | "
          f"{fmt(med(ac), 5)} / {fmt(med(ar))} | {k}/{n} | **{v}** |")
        if v != "PASS":
            s3det.append((lab, l, ps))
if s3det:
    P()
    P("Per-seed values (cos / ratio) where S3 is not PASS:")
    P()
    for lab, l, ps in s3det:
        P(f"- h={lab} {LNAME[l]}: " + "; ".join(f"seed {s}: {v}" for s, v in ps.items()))
P()

# S4
P("## S4 — stiffness: Δφ_meas / Δφ_pred (measured C) ∈ [0.8, 1.25] (implies same sign) on ≥ 3 of the 4 closed paths, ε = 0.025")
P()
P("For h = 512 (Wide) only 3 closed paths exist (walk8 not in the Wide grid); the rule is applied as "
  "\"≥ 3 of the paths run\", i.e. all 3.")
P()
P("| width | learner | " + " | ".join(f"{p}: Δφ_meas/Δφ_pred median [min,max]" for p in CLOSED)
  + " | seeds holding | verdict |")
P("|" + "---|" * (len(CLOSED) + 4))
s4det = []
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    need_paths = 3
    for l in lrn:
        cells = []
        per = {s: {} for s in seeds}
        cnt = {s: 0 for s in seeds}
        med_cnt = 0
        for p in CLOSED:
            if p not in paths:
                cells.append("not in grid")
                continue
            q = []
            for s in seeds:
                pp, st = get_path(grid, h, s, eps, l, p)
                if not valid(pp):
                    per[s][p] = st
                    continue
                x = pp["dphi_meas"] / pp["dphi_pred_meas"]
                q.append(x)
                cnt[s] += int(0.8 <= x <= 1.25)
                per[s][p] = f"{x:.4f} (Δφ_meas {pp['dphi_meas']:+.3e}, pred {pp['dphi_pred_meas']:+.3e})"
            mq = med(q)
            med_cnt += int(math.isfinite(mq) and 0.8 <= mq <= 1.25)
            cells.append(mmr(q))
        oks = [cnt[s] >= need_paths for s in seeds]
        v, k, n = verdict(med_cnt >= need_paths, oks)
        P(f"| {lab} | {LNAME[l]} | " + " | ".join(cells) + f" | {k}/{n} | **{v}** |")
        if v != "PASS":
            s4det.append((lab, l, per, cnt))
if s4det:
    P()
    P("Per-seed values where S4 is not PASS:")
    P()
    for lab, l, per, cnt in s4det:
        for s, d in per.items():
            P(f"- h={lab} {LNAME[l]} seed {s} ({cnt[s]} paths in range): "
              + "; ".join(f"{p} {v}" for p, v in d.items()))
P()
# sign table
P("Sign of Δφ_meas (count of + / − over seeds) per path, measured, ε = 0.025:")
P()
P("| width | learner | " + " | ".join(CLOSED) + " |")
P("|" + "---|" * (len(CLOSED) + 2))
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    for l in lrn:
        cells = []
        for p in CLOSED:
            if p not in paths:
                cells.append("—")
                continue
            plus = minus = 0
            for s in seeds:
                pp, st = get_path(grid, h, s, eps, l, p)
                if valid(pp):
                    plus += pp["dphi_meas"] > 0
                    minus += pp["dphi_meas"] < 0
            cells.append(f"{plus}+ / {minus}−")
        P(f"| {lab} | {LNAME[l]} | " + " | ".join(cells) + " |")
P()


# S5
def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


P("## S5 — remainder is third order: slope of log rel_err on log ε over ε ∈ {0.1, 0.05, 0.025, 0.0125} in [0.7, 1.5] (measured C)")
P()
P("Per (width, learner) the rule must hold on both paths (oab and tri+). Points whose path is not "
  "OK are left out of that seed's fit; a seed with fewer than 4 valid points is counted as not holding.")
P()
P("| width | learner | oab slope median [min,max] | tri+ slope median [min,max] | seeds holding | verdict |")
P("|---|---|---|---|---|---|")
s5det = []
for h in (32, 128):
    for l in ("GN", "GD1"):
        cells = []
        okseed = {s: True for s in range(5)}
        medok = True
        per = {s: {} for s in range(5)}
        for p in ("oab", "tri+"):
            sl = []
            for s in range(5):
                xs, ys = [], []
                for e in S5_EPS:
                    pp, st = get_path("EPS", h, s, e, l, p)
                    if valid(pp):
                        xs.append(e); ys.append(pp["meas"]["rel_err"])
                if len(xs) < len(S5_EPS):
                    okseed[s] = False
                    per[s][p] = f"only {len(xs)} valid points"
                    if len(xs) < 2:
                        continue
                b = slope(xs, ys)
                sl.append(b)
                okseed[s] &= 0.7 <= b <= 1.5
                per[s][p] = f"slope {b:.3f} (rel_err " + ", ".join(f"{y:.3e}" for y in ys) + ")"
            ms = med(sl)
            medok &= math.isfinite(ms) and 0.7 <= ms <= 1.5
            cells.append(mmr(sl, 3))
        v, k, n = verdict(medok, [okseed[s] for s in range(5)])
        P(f"| {h} | {LNAME[l]} | " + " | ".join(cells) + f" | {k}/{n} | **{v}** |")
        if v != "PASS":
            s5det.append((h, l, per))
if s5det:
    P()
    P("Per-seed values where S5 is not PASS:")
    P()
    for h, l, per in s5det:
        for s, d in per.items():
            P(f"- h={h} {LNAME[l]} seed {s}: " + "; ".join(f"{p}: {v}" for p, v in d.items()))
P()
# full rel_err table
P("rel_err (median over seeds) by ε, measured C:")
P()
P("| width | learner | path | " + " | ".join(f"ε={e}" for e in EPS_LIST) + " |")
P("|" + "---|" * (len(EPS_LIST) + 3))
for h in (32, 128):
    for l in LEARNERS:
        for p in ("oab", "tri+"):
            cells = []
            for e in EPS_LIST:
                q = []
                for s in range(5):
                    pp, st = get_path("EPS", h, s, e, l, p)
                    if valid(pp):
                        q.append(pp["meas"]["rel_err"])
                cells.append(fmt(med(q)) + (f" (n={len(q)})" if len(q) < 5 else ""))
            P(f"| {h} | {LNAME[l]} | {p} | " + " | ".join(cells) + " |")
P()

# S6
P("## S6 — linear control: ‖P_v(θ_end − θ_A)‖ < 1e-10 on oab and tri+, ε = 0.1, all 5 seeds")
P()
P("tcr.LinearModel(seed), y_A = model.labels(X). Primary reading: GN = tcr.gn_learn (converged), "
  "GD(1) exactly 2,000 steps per learn; u, v scaled to the linear model's ‖y_A‖. Supplementary "
  "readings: u, v scaled to the teacher's ‖y_A‖ (SPEC3 primary choice); GN also run for exactly "
  "2,000 iterations per learn.")
P()
P("| seed | reading | GN oab | GN tri+ | GD(1) oab | GD(1) tri+ | holds |")
P("|---|---|---|---|---|---|---|")
s6ok = []
for s in range(5):
    r = R.get(("S6", None, s, 0.1, "GN+GD1"))
    if r is None or r.get("status") != "OK":
        P(f"| {s} | — | {r.get('status') if r else 'MISSING'} | | | | no |")
        s6ok.append(False)
        continue
    for tag in ("primary", "supp_uv_teacher_scale", "supp_gn_2000_iters"):
        vals = r["S6"][tag]
        vs = [vals[f"{a}|{p}"]["Pv_norm"] for a in ("GN", "GD1") for p in ("oab", "tri+")]
        hold = all(v < 1e-10 for v in vs)
        if tag == "primary":
            s6ok.append(hold)
        P(f"| {s} | {tag} | " + " | ".join(f"{v:.3e}" for v in vs) + f" | {'yes' if hold else 'NO'} |")
P()
P(f"**S6 verdict (primary reading, must hold for all 5 seeds): {'PASS' if all(s6ok) and len(s6ok) == 5 else 'FAIL'}** ({sum(s6ok)}/5 seeds)")
P()

# ---------------------------------------------------------------------------
P("## R1 — norms per (learner, path), measured C (median over seeds), ε = 0.025")
P()
P("‖V_meas‖ (for oneway: ‖D_meas‖), ‖C_sym‖_F, ‖C_area‖_F (oneway: of C_learner − C_GN), ‖V_sym‖, "
  "‖V_area‖ (LAW of the two parts of C), and the a-priori C parts.")
P()
P("| width | learner | path | ‖V_meas‖ | ‖C_sym‖_F | ‖C_area‖_F | ‖V_sym‖ | ‖V_area‖ | a-priori ‖C_sym‖_F | a-priori ‖C_area‖_F | a-priori ‖V_sym‖ | a-priori ‖V_area‖ | n valid |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    for l in lrn:
        for p in paths + ["oneway"]:
            if p == "oneway" and l == "GN":
                continue
            vals = {k: [] for k in ("vm", "cs", "ca", "vs", "va", "acs", "aca", "avs", "ava")}
            nv = 0
            for s in seeds:
                pp, st = get_path(grid, h, s, eps, l, p)
                if not valid(pp):
                    continue
                nv += 1
                if p == "oneway":
                    vals["vm"].append(pp["meas"]["norm_meas"])
                    vals["cs"].append(pp["Cdiff_sym_F"]); vals["ca"].append(pp["Cdiff_area_F"])
                else:
                    vals["vm"].append(pp["Vmeas_norm"])
                    vals["cs"].append(pp["C_sym_F"]); vals["ca"].append(pp["C_area_F"])
                    if "apri" in pp:
                        vals["avs"].append(pp["apri"].get("Vsym_norm"))
                        vals["ava"].append(pp["apri"].get("Varea_norm"))
                vals["vs"].append(pp["Vsym_norm"]); vals["va"].append(pp["Varea_norm"])
                if "apri" in pp and p != "oneway":
                    vals["acs"].append(pp["apri"]["C_sym_F"]); vals["aca"].append(pp["apri"]["C_area_F"])
            P(f"| {lab} | {LNAME[l]} | {p} | " + " | ".join(fmt(med(vals[k])) for k in
                                                            ("vm", "cs", "ca", "vs", "va", "acs", "aca", "avs", "ava"))
              + f" | {nv} |")
P()

# R2
P("## R2 — learning-rate dependence (MAIN, ε = 0.025)")
P()
P("rel diff = ‖V_meas[GD(c)] − V_meas[GD(1)]‖ / ‖V_meas[GD(1)]‖, next to ‖C_sym[GD(c)] − C_sym[GD(1)]‖_F / ‖C_sym[GD(1)]‖_F. Median [min, max] over seeds.")
P()
P("| width | path | c | V rel diff | C_sym rel diff | n |")
P("|---|---|---|---|---|---|")
for h in (32, 64, 128):
    for p in ("oab", "tri+"):
        for c in ("0.25", "1.8"):
            vr, cr = [], []
            for s in range(5):
                r = R.get(("R2", h, s, 0.025, "GD"))
                if r is None:
                    continue
                d = r["R2"].get(f"{p}|{c}")
                if d is None or d["status"] != "OK" or d["ref_status"] != "OK":
                    continue
                vr.append(d["V_rel"]); cr.append(d["Csym_rel"])
            P(f"| {h} | {p} | {c} | {mmr(vr)} | {mmr(cr)} | {len(vr)} |")
P()

# R3
P("## R3 — ε-dependence of cos and ratio (measured C), oab and tri+, median over seeds")
P()
P("| width | learner | path | " + " | ".join(f"ε={e}: cos / ratio" for e in EPS_LIST) + " |")
P("|" + "---|" * (len(EPS_LIST) + 3))
for h in (32, 128):
    for l in LEARNERS:
        for p in ("oab", "tri+"):
            cells = []
            for e in EPS_LIST:
                cs, rs = [], []
                for s in range(5):
                    pp, st = get_path("EPS", h, s, e, l, p)
                    if valid(pp):
                        cs.append(pp["meas"]["cos"]); rs.append(pp["meas"]["ratio"])
                cells.append(f"{fmt(med(cs), 5)} / {fmt(med(rs))}" + (f" (n={len(cs)})" if len(cs) < 5 else ""))
            P(f"| {h} | {LNAME[l]} | {p} | " + " | ".join(cells) + " |")
P()

# R4
P("## R4 — order-odd part a = (V_meas[tri+] − V_meas[tri−])/2 vs (ε²/2)·b_uv(θ_A), ε = 0.025")
P()
P("| width | learner | cos(a, b_uv) median [min,max] | ‖a‖/((ε²/2)‖b_uv‖) median [min,max] | n |")
P("|---|---|---|---|---|")
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    for l in lrn:
        cs, rs = [], []
        for s in seeds:
            r = R.get((grid, h, s, eps, l))
            if r is None or r.get("status") != "OK" or "R4" not in r.get("cross", {}):
                continue
            if not (valid(r["paths"].get("tri+")) and valid(r["paths"].get("tri-"))):
                continue
            cs.append(r["cross"]["R4"]["cos_a_buv"]); rs.append(r["cross"]["R4"]["ratio"])
        P(f"| {lab} | {LNAME[l]} | {mmr(cs, 5)} | {mmr(rs)} | {len(cs)} |")
P()

# R5
P("## R5 — ADAM (exploratory): LAW with measured C (plain rule), ε = 0.025")
P()
P("| h | seed | path | cos | ratio | rel_err | ‖V_meas‖ | end ‖f−y_A‖_∞ | learns: steps (status, final ‖r‖_∞) |")
P("|---|---|---|---|---|---|---|---|---|")
for h in (32, 128):
    for s in (0, 1, 2):
        r = R.get(("ADAM", h, s, 0.025, "ADAM"))
        if r is None or r.get("status") != "OK":
            P(f"| {h} | {s} | — | {r.get('status') if r else 'NOT RUN / MISSING'} | | | | | |")
            continue
        for p in ("oab", "tri+"):
            pp = r["paths"][p]
            if "meas" not in pp:
                P(f"| {h} | {s} | {p} | {pp['status']} | | | | | |")
                continue
            ls = "; ".join(f"{i['target']}: {i.get('steps')} ({i['status']}, {i.get('res_inf', float('nan')):.2e})"
                           for i in pp["learns"])
            P(f"| {h} | {s} | {p} | {pp['meas']['cos']:.5f} | {pp['meas']['ratio']:.4f} | "
              f"{pp['meas']['rel_err']:.4f} | {fmt(pp['Vmeas_norm'])} | {pp['end_res_A']:.2e} | {ls} |")
P()

# R6 and conditions
P("## Conditions that are not OK (DIVERGED / NOT CONVERGED / INVALID / ERROR / TIMEOUT / NOT RUN)")
P()
bad = []
for k, r in sorted(R.items(), key=lambda kv: str(kv[0])):
    if r.get("status") != "OK":
        bad.append(f"- {k}: job {r.get('status')} — {r.get('error', '')[:200]}")
        continue
    if r["grid"] in ("S6", "R2"):
        continue
    for p, pp in r["paths"].items():
        if pp["status"] != "OK" or pp.get("gn_status", "OK") != "OK":
            ls = "; ".join(f"{i.get('target')}: {i.get('status')} steps {i.get('steps')} res {i.get('res_inf', float('nan')):.2e}"
                           for i in pp["learns"])
            bad.append(f"- {r['grid']} h={r['h']} seed={r['seed']} ε={r['eps']} {LNAME.get(r['learner'], r['learner'])} {p}: "
                       f"{pp['status']} (GN ref {pp.get('gn_status', '—')}) [{ls}]")
        if "apri" in pp and pp["apri"]["status"] != "OK":
            bad.append(f"- {r['grid']} h={r['h']} seed={r['seed']} ε={r['eps']} {r['learner']} {p}: a-priori C {pp['apri']['status']}")
P("\n".join(bad) if bad else "None.")
P()
# expected but missing jobs
exp = []
for h in (32, 64, 128):
    for s in range(5):
        for l in LEARNERS:
            exp.append(("MAIN", h, s, 0.025, l))
for h in (32, 128):
    for s in range(5):
        for e in EPS_LIST:
            for l in LEARNERS:
                exp.append(("EPS", h, s, e, l))
for s in range(3):
    for l in ("GN", "GD1"):
        exp.append(("WIDE", 512, s, 0.025, l))
for h in (32, 128):
    for s in range(3):
        exp.append(("ADAM", h, s, 0.025, "ADAM"))
for s in range(5):
    exp.append(("S6", None, s, 0.1, "GN+GD1"))
miss = [k for k in exp if k not in R]
P("Jobs of the SPEC4 §6 grid with no record (not run): " + (", ".join(map(str, miss)) if miss else "none") + ".")
P()

P("## R6 — anomalies (measurements only)")
P()
an = []
# GN line-search failures, GD/MOM steps hitting large values, sign patterns
for k, r in sorted(R.items(), key=lambda kv: str(kv[0])):
    if r.get("status") != "OK" or r["grid"] in ("S6", "R2"):
        continue
    for p, pp in r["paths"].items():
        for i in pp["learns"]:
            if i.get("ls_fail", 0):
                an.append(f"- {r['grid']} h={r['h']} seed={r['seed']} ε={r['eps']} {r['learner']} {p}: GN line search exhausted {i['ls_fail']} times in learn {i.get('target')}")
        if "meas" in pp and r["grid"] in ("MAIN", "WIDE") and valid(pp):
            c = pp["meas"]
            if c["cos"] < 0.98 or not (0.9 <= c["ratio"] <= 1.1):
                an.append(f"- {r['grid']} h={r['h']} seed={r['seed']} {LNAME.get(r['learner'])} {p}: measured-C cos {c['cos']:.5f}, ratio {c['ratio']:.4f}, rel_err {c['rel_err']:.4f}")
        if "dphi_meas" in pp and r["grid"] in ("MAIN", "WIDE") and valid(pp):
            x = pp["dphi_meas"] / pp["dphi_pred_meas"]
            if not (0.8 <= x <= 1.25):
                an.append(f"- {r['grid']} h={r['h']} seed={r['seed']} {LNAME.get(r['learner'])} {p}: Δφ_meas/Δφ_pred = {x:.4f} (Δφ_meas {pp['dphi_meas']:+.3e}, Δφ_pred {pp['dphi_pred_meas']:+.3e})")
# determinism cross-check: EPS eps=0.025 vs MAIN
diffs = []
for h in (32, 128):
    for s in range(5):
        for l in LEARNERS:
            for p in ("oab", "tri+"):
                a, _ = get_path("MAIN", h, s, 0.025, l, p)
                b, _ = get_path("EPS", h, s, 0.025, l, p)
                if a is not None and b is not None and "Vmeas_norm" in a and "Vmeas_norm" in b:
                    diffs.append(abs(a["Vmeas_norm"] - b["Vmeas_norm"]) / a["Vmeas_norm"])
if diffs:
    an.append(f"- Reproducibility: ε = 0.025 conditions run twice (MAIN and EPS grids, separate processes): "
              f"max relative difference of ‖V_meas‖ = {max(diffs):.3e} over {len(diffs)} pairs.")
# horizontal fraction of closed-path displacement
hf = [pp.get("horiz_frac") for r in R.values() if r.get("status") == "OK" and r["grid"] in ("MAIN",)
      for p, pp in r["paths"].items() if p in CLOSED and valid(pp)]
if hf:
    an.append(f"- Closed paths (MAIN): ‖(I − P_v(θ_A))(θ_end − θ_A)‖/‖θ_end − θ_A‖ median {med(hf):.3e}, max {max(hf):.3e}.")
P("\n".join(an) if an else "None.")
P()

# step counts
P("## Learner step counts (MAIN, median over seeds of steps per learn, first learn of each path)")
P()
P("| width | learner | oab: u, A | tri+: u, v, A | walk8 total | median wall per job (s) |")
P("|---|---|---|---|---|---|")
for (lab, grid, h, seeds, eps, lrn, paths) in CONFS:
    for l in lrn:
        oa, tr, wk, wl = [], [], [], []
        for s in seeds:
            r = R.get((grid, h, s, eps, l))
            if r is None or r.get("status") != "OK":
                continue
            wl.append(r["wall"])
            o = r["paths"].get("oab")
            if o:
                oa.append([i.get("steps") for i in o["learns"]])
            t = r["paths"].get("tri+")
            if t:
                tr.append([i.get("steps") for i in t["learns"]])
            w = r["paths"].get("walk8")
            if w:
                wk.append(sum(i.get("steps") or 0 for i in w["learns"]))
        def mlist(L):
            if not L:
                return "—"
            m = len(L[0])
            return ", ".join(str(int(np.median([x[j] for x in L if len(x) > j and x[j] is not None]))) for j in range(m))
        P(f"| {lab} | {LNAME[l]} | {mlist(oa)} | {mlist(tr)} | {fmt(med(wk), 0) if wk else '—'} | {fmt(med(wl), 1)} |")
P()

# wall time
P("## Wall time")
P()
walls = {}
for r in R.values():
    walls.setdefault(r["grid"], 0.0)
    walls[r["grid"]] += r.get("wall", 0.0) or 0.0
P("Sum of per-job wall time by grid (s; jobs ran in 4 parallel worker processes): "
  + ", ".join(f"{g} {v:.0f}" for g, v in walls.items()))
P()
for m in meta.get("runs", []):
    P(f"- run: grid {m.get('grid')} pid {m.get('pid')} jobs {m.get('n_run')} wall {m.get('wall', 0):.0f}s "
      f"stopped={m.get('stopped')} threads={m.get('threads')} finished {m.get('finished')}")
P()

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa
    pass
print("\n".join(lines))
