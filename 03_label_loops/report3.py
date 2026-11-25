"""Reads results3.json, writes REPORT3.md (SPEC3.md section 8/9) and adds the L6 median
16x16 matrices to results3.json (key "L6_median"). Run: python report3.py"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import run3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "REPORT3.md")

db = run3.load()
R = [r for r in db["records"]]
SEEDS, WIDTHS, EPS = run3.SEEDS, run3.WIDTHS, run3.EPS
lines = []
VERDICTS = []
INVALID = []


def P(s=""):
    lines.append(s)


def g(grid, **kw):
    out = []
    for r in R:
        if r.get("grid") != grid:
            continue
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def one(grid, **kw):
    rs = g(grid, **kw)
    return rs[0] if rs else None


def fmt(x, p=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.{p - 1}e}"
    return f"{x:.{p}g}" if ax >= 1 else f"{x:.{p}f}"


def stat(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.median(v)), float(min(v)), float(max(v)), len(v)


def mmm(vals, p=4):
    m, lo, hi, n = stat(vals)
    s = f"{fmt(m, p)} [{fmt(lo, p)}, {fmt(hi, p)}]"
    if n != len(vals):
        s += f" (n={n})"
    return s


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(ys) < 2 or not np.all(np.isfinite(ys)) or np.any(ys <= 0):
        return float("nan")
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def verdict(seed_ok, median_ok):
    """section 8: PASS if rule holds for median AND >= 4 of 5 seeds; INCONCLUSIVE if median
    only; FAIL otherwise. seed_ok: list of 5 booleans (invalid seeds count as False)."""
    n = sum(bool(x) for x in seed_ok)
    if median_ok and n >= 4:
        return "PASS", n
    if median_ok:
        return "INCONCLUSIVE", n
    return "FAIL", n


def table(header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P()


def ok_status(r):
    return r is not None and r.get("status") == "OK"


def per_seed_str(vals, p=4):
    return ", ".join(f"s{s}={fmt(v, p) if not isinstance(v, str) else v}" for s, v in zip(SEEDS, vals))


def add_verdict(pid, h, v, n, detail=""):
    VERDICTS.append((pid, h, v, n, detail))


# ---------------------------------------------------------------------------
# collect INVALID / NOT CONVERGED
# ---------------------------------------------------------------------------
for r in R:
    st = r.get("status")
    if st not in ("OK",):
        INVALID.append(f"{r.get('grid')} h={r.get('h')} seed={r.get('seed')} eps={r.get('eps')} "
                       f"L={r.get('L')}: {st} {r.get('error', '')} invalid={r.get('invalid', '')}")
    for sub in ("polys", "tri"):
        if isinstance(r.get(sub), dict):
            for k, v in r[sub].items():
                if isinstance(v, dict) and v.get("status") not in (None, "OK"):
                    INVALID.append(f"{r.get('grid')} h={r.get('h')} seed={r.get('seed')} "
                                   f"eps={r.get('eps')} {k}: {v.get('status')} {v.get('error')}")
    if r.get("grid") == "K5" and r.get("status") == "OK":
        for sn, st_ in r["states"].items():
            for run in st_["runs"]:
                if not run["converged"]:
                    INVALID.append(f"K5 h={r['h']} seed={r['seed']} {sn} j={run['j']}: NOT CONVERGED "
                                   f"(cap {run['steps']} steps, res_inf={run['res_inf']:.3e})")

P("# REPORT3 — label-space loops (SPEC3.md)")
P()
P("All numbers: median over seeds [min, max] unless stated. GN learner unless stated.")
P("Verdict rule (SPEC3 section 8): PASS if the rule holds for the median over seeds AND for at "
  "least 4 of 5 seeds; INCONCLUSIVE if it holds for the median only; FAIL otherwise.")
P()

# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------
P("## L1 — ‖a_tri‖ ∝ ε² (slope of log‖a_tri‖ on log ε, 5 ε; rule: in [1.8, 2.2])")
P()
rows = []
for h in WIDTHS:
    sl = []
    for s in SEEDS:
        recs = [one("LE", h=h, seed=s, eps=e) for e in EPS]
        if all(ok_status(r) for r in recs):
            sl.append(slope(EPS, [r["polys"]["tri"]["Na"] for r in recs]))
        else:
            sl.append(float("nan"))
    med = stat(sl)[0]
    v, n = verdict([1.8 <= x <= 2.2 for x in sl], 1.8 <= med <= 2.2)
    add_verdict("L1", h, v, n, per_seed_str(sl))
    rows.append([h, "2 ([1.8, 2.2])", mmm(sl), v, f"{n}/5"])
table(["h", "predicted", "measured slope", "verdict", "seeds holding"], rows)
P("‖a_tri‖ per ε (weight space):")
P()
rows = []
for h in WIDTHS:
    rows.append([h] + [mmm([one("LE", h=h, seed=s, eps=e)["polys"]["tri"]["Na"]
                            if ok_status(one("LE", h=h, seed=s, eps=e)) else float("nan")
                            for s in SEEDS]) for e in EPS])
table(["h"] + [f"ε={e}" for e in EPS], rows)

# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------
P("## L2 — a_tri = +(ε²/2) b_uv(θ_A) (rule at ε = 0.0125: cos ≥ 0.99 AND ratio in [0.97, 1.03])")
P()
rows = []
for h in WIDTHS:
    cs, rs = [], []
    for s in SEEDS:
        r = one("LE", h=h, seed=s, eps=0.0125)
        cs.append(r["polys"]["tri"]["cos_b"] if ok_status(r) else float("nan"))
        rs.append(r["polys"]["tri"]["ratio_b"] if ok_status(r) else float("nan"))
    mc, mr = stat(cs)[0], stat(rs)[0]
    v, n = verdict([c >= 0.99 and 0.97 <= x <= 1.03 for c, x in zip(cs, rs)],
                   mc >= 0.99 and 0.97 <= mr <= 1.03)
    add_verdict("L2", h, v, n, "cos: " + per_seed_str(cs, 6) + "; ratio: " + per_seed_str(rs, 6))
    rows.append([h, "cos 1 (≥0.99)", mmm(cs, 6), "ratio 1 ([0.97,1.03])", mmm(rs, 6), v, f"{n}/5"])
table(["h", "pred cos", "measured cos", "pred ratio", "measured ratio", "verdict", "seeds"], rows)
P("All ε (no rule; context): cos(a_tri, b_uv) / ratio ‖a_tri‖/((ε²/2)‖b_uv‖)")
P()
rows = []
for h in WIDTHS:
    row = [h]
    for e in EPS:
        rr = [one("LE", h=h, seed=s, eps=e) for s in SEEDS]
        row.append(mmm([r["polys"]["tri"]["cos_b"] if ok_status(r) else float("nan") for r in rr], 5)
                   + " / " + mmm([r["polys"]["tri"]["ratio_b"] if ok_status(r) else float("nan")
                                  for r in rr], 5))
    rows.append(row)
table(["h"] + [f"ε={e}" for e in EPS], rows)

# ---------------------------------------------------------------------------
# L3a, L3b, L3e
# ---------------------------------------------------------------------------
L3N, L3C = run3.L3_N, run3.L3_C


def l3_vals(h, eps, name, field):
    out = []
    for s in SEEDS:
        r = one("L3", h=h, seed=s, eps=eps)
        if r is not None and r.get("polys", {}).get(name, {}).get("status") == "OK":
            out.append(r["polys"][name].get(field, float("nan")))
        else:
            out.append(float("nan"))
    return out


for pid, fam_, names, preds in (
        ("L3a", "circ(N)", [f"circ{N}" for N in L3N], [N * math.sin(2 * math.pi / N) for N in L3N]),
        ("L3b", "rect(c)", [f"rect{c}" for c in L3C], [2 * c for c in L3C])):
    P(f"## {pid} — ‖a_{fam_}‖/‖a_tri‖ (rule at ε = 0.025: every member within 5% of prediction)")
    P()
    for eps in run3.L3_EPS:
        P(f"ε = {eps}" + (" (rule applies)" if eps == 0.025 else " (report only)"))
        P()
        rows = []
        for h in WIDTHS:
            row = [h]
            seed_ok = [True] * 5
            med_ok = True
            for nm, pr in zip(names, preds):
                vals = l3_vals(h, eps, nm, "ratio")
                row.append(f"{fmt(pr)} / {mmm(vals)}")
                med_ok &= abs(stat(vals)[0] / pr - 1) <= 0.05
                for i, x in enumerate(vals):
                    seed_ok[i] &= (math.isfinite(x) and abs(x / pr - 1) <= 0.05)
            if eps == 0.025:
                v, n = verdict(seed_ok, med_ok)
                det = "; ".join(f"{nm}: " + per_seed_str(l3_vals(h, eps, nm, "ratio"))
                                for nm in names)
                add_verdict(pid, h, v, n, det)
                row += [v, f"{n}/5"]
            rows.append(row)
        hdr = ["h"] + [f"{nm} pred / measured" for nm in names]
        if eps == 0.025:
            hdr += ["verdict", "seeds"]
        table(hdr, rows)

P("## L3e — cos(a_poly, a_tri) for circ and rect (rule at ε = 0.025: every one ≥ 0.98)")
P()
allnames = [f"circ{N}" for N in L3N] + [f"rect{c}" for c in L3C]
for eps in run3.L3_EPS:
    P(f"ε = {eps}" + (" (rule applies)" if eps == 0.025 else " (report only)"))
    P()
    rows = []
    for h in WIDTHS:
        row = [h]
        seed_ok = [True] * 5
        med_ok = True
        for nm in allnames:
            vals = l3_vals(h, eps, nm, "cos")
            row.append(mmm(vals, 5))
            med_ok &= stat(vals)[0] >= 0.98
            for i, x in enumerate(vals):
                seed_ok[i] &= (math.isfinite(x) and x >= 0.98)
        if eps == 0.025:
            v, n = verdict(seed_ok, med_ok)
            add_verdict("L3e", h, v, n, "; ".join(f"{nm}: " + per_seed_str(l3_vals(h, eps, nm, "cos"), 5)
                                                  for nm in allnames))
            row += [v, f"{n}/5"]
        rows.append(row)
    hdr = ["h"] + [f"{nm} (pred 1)" for nm in allnames]
    if eps == 0.025:
        hdr += ["verdict", "seeds"]
    table(hdr, rows)

# ---------------------------------------------------------------------------
# L3c, L3d
# ---------------------------------------------------------------------------
P("## L3c — fig8: ‖a_fig8‖/‖a_tri‖ → 0 (rule: slope of log ratio on log ε > 0.5)")
P()
rows = []
for h in WIDTHS:
    sl = []
    per_eps = {e: [] for e in EPS}
    for s in SEEDS:
        recs = [one("LE", h=h, seed=s, eps=e) for e in EPS]
        vals = [r["polys"]["fig8"].get("ratio_tri", float("nan")) if ok_status(r) else float("nan")
                for r in recs]
        for e, x in zip(EPS, vals):
            per_eps[e].append(x)
        sl.append(slope(EPS, vals))
    med = stat(sl)[0]
    v, n = verdict([x > 0.5 for x in sl], med > 0.5)
    add_verdict("L3c", h, v, n, per_seed_str(sl))
    rows.append([h, "> 0.5", mmm(sl), v, f"{n}/5"] + [mmm(per_eps[e]) for e in EPS])
table(["h", "predicted slope", "measured slope", "verdict", "seeds"] + [f"ratio ε={e}" for e in EPS], rows)

P("## L3d — star: ‖a_star‖ ∝ ε⁴ (rule: slope in [3.4, 4.4])")
P()
rows = []
for h in WIDTHS:
    sl = []
    per_eps = {e: [] for e in EPS}
    for s in SEEDS:
        recs = [one("LE", h=h, seed=s, eps=e) for e in EPS]
        vals = [r["polys"]["star"]["Na"] if ok_status(r) else float("nan") for r in recs]
        for e, x in zip(EPS, vals):
            per_eps[e].append(x)
        sl.append(slope(EPS, vals))
    med = stat(sl)[0]
    v, n = verdict([3.4 <= x <= 4.4 for x in sl], 3.4 <= med <= 4.4)
    add_verdict("L3d", h, v, n, per_seed_str(sl))
    rows.append([h, "4 ([3.4, 4.4])", mmm(sl), v, f"{n}/5"] + [mmm(per_eps[e]) for e in EPS])
table(["h", "predicted", "measured slope", "verdict", "seeds"] + [f"‖a_star‖ ε={e}" for e in EPS], rows)

# ---------------------------------------------------------------------------
# L4
# ---------------------------------------------------------------------------
P("## L4 — width dependence (report only): fitted exponent on h (log-log, all 5 widths)")
P()
eb, ea = [], []
for s in SEEDS:
    bb = [one("BASE", h=h, seed=s) for h in WIDTHS]
    aa = [one("LE", h=h, seed=s, eps=0.05) for h in WIDTHS]
    eb.append(slope(WIDTHS, [r["norm_b_uv"] for r in bb]) if all(ok_status(r) for r in bb) else float("nan"))
    ea.append(slope(WIDTHS, [r["polys"]["tri"]["Na"] for r in aa]) if all(ok_status(r) for r in aa) else float("nan"))
table(["quantity", "exponent median [min, max]", "per seed"],
      [["‖b_uv(θ_A)‖", mmm(eb), per_seed_str(eb)], ["‖a_tri‖ (ε=0.05)", mmm(ea), per_seed_str(ea)]])
rows = []
for h in WIDTHS:
    rows.append([h, mmm([one("BASE", h=h, seed=s)["norm_b_uv"] for s in SEEDS]),
                 mmm([one("LE", h=h, seed=s, eps=0.05)["polys"]["tri"]["Na"] for s in SEEDS]),
                 mmm([one("BASE", h=h, seed=s)["norm_c_u"] for s in SEEDS]),
                 mmm([one("BASE", h=h, seed=s)["vert_b"] for s in SEEDS], 3)])
table(["h", "‖b_uv(θ_A)‖", "‖a_tri‖ ε=0.05", "‖c_u(θ_A)‖ (context)", "verticality of b_uv (context)"], rows)

# ---------------------------------------------------------------------------
# L5 / K6
# ---------------------------------------------------------------------------
P("## L5 and K6 — linear control (tcr.LinearModel; must hold: every norm < 1e-10)")
P()
rows = []
l5ok, k6ok = True, True
for s in SEEDS:
    r = one("LIN", seed=s)
    for tag in ("primary_uv_teacher_norm", "suppl_uv_linear_norm"):
        q = r[tag]
        rows.append([s, tag, fmt(q["L5_Na_tri_eps0.2"]), fmt(max(q["K6_d_ret"].values())),
                     fmt(max(q["K6_d_sub4"].values())), fmt(q["K6_d_K2_m8"])])
        if tag.startswith("primary"):
            l5ok &= q["L5_Na_tri_eps0.2"] < 1e-10
            k6ok &= max(q["K6_d_ret"].values()) < 1e-10 and max(q["K6_d_sub4"].values()) < 1e-10 \
                and q["K6_d_K2_m8"] < 1e-10
table(["seed", "u,v scaling", "L5 ‖a_tri‖ ε=0.2", "K6 max‖d_ret‖ (5 ε)", "K6 max‖d_sub(4)‖ (L=0.1,0.05)",
       "K6 ‖d_K2(m=8)‖"], rows)
P(f"L5: {'PASS' if l5ok else 'FAIL'} (all seeds). K6: {'PASS' if k6ok else 'FAIL'} (all seeds).")
P()
VERDICTS.append(("L5", "linear", "PASS" if l5ok else "FAIL", 5, ""))
VERDICTS.append(("K6", "linear", "PASS" if k6ok else "FAIL", 5, ""))

# ---------------------------------------------------------------------------
# L6
# ---------------------------------------------------------------------------
P("## L6 — where the curvature lives (report only; median over seeds [min, max])")
P()
P("Eigen-indices 1..16 with λ_1 ≥ … ≥ λ_16. Blocks over pairs i<j: top×top (i,j ≤ 4, 6 pairs), "
  "bottom×bottom (i,j ≥ 13, 6 pairs), top×bottom (i ≤ 4, j ≥ 13, 16 pairs). Spearman over all 120 pairs.")
P()
iu = [(i, j) for i in range(16) for j in range(i + 1, 16)]
blocks = {"top×top": [(i, j) for i, j in iu if i < 4 and j < 4],
          "bot×bot": [(i, j) for i, j in iu if i >= 12 and j >= 12],
          "top×bot": [(i, j) for i, j in iu if i < 4 and j >= 12]}
L6med = {}
rows = []
for h in WIDTHS:
    acc = {f"{q}_{b}": [] for q in ("kappa", "B") for b in blocks}
    acc.update(sp_kappa=[], sp_B=[])
    mats_k, mats_B, lams = [], [], []
    for s in SEEDS:
        r = one("BASE", h=h, seed=s)
        K = np.array(r["L6"]["kappa"])
        B = np.array(r["L6"]["B"])
        mats_k.append(K)
        mats_B.append(B)
        lams.append(r["L6"]["lam"])
        for b, prs in blocks.items():
            acc[f"kappa_{b}"].append(float(np.mean([K[i, j] for i, j in prs])))
            acc[f"B_{b}"].append(float(np.mean([B[i, j] for i, j in prs])))
        ij = [i + j + 2 for i, j in iu]
        acc["sp_kappa"].append(float(spearmanr([K[i, j] for i, j in iu], ij)[0]))
        acc["sp_B"].append(float(spearmanr([B[i, j] for i, j in iu], ij)[0]))
    L6med[str(h)] = {"kappa_median": np.median(np.stack(mats_k), 0).tolist(),
                     "B_median": np.median(np.stack(mats_B), 0).tolist(),
                     "lam_median": np.median(np.array(lams), 0).tolist()}
    rows.append([h] + [mmm(acc[f"kappa_{b}"]) for b in blocks] + [mmm(acc["sp_kappa"], 3)]
                + [mmm(acc[f"B_{b}"]) for b in blocks] + [mmm(acc["sp_B"], 3)])
table(["h"] + [f"κ {b}" for b in blocks] + ["Spearman(κ, i+j)"] + [f"B {b}" for b in blocks]
      + ["Spearman(B, i+j)"], rows)
rows = []
for h in WIDTHS:
    lam = np.array([one("BASE", h=h, seed=s)["L6"]["lam"] for s in SEEDS])
    rows.append([h, mmm(list(lam[:, 0])), mmm(list(lam[:, 3])), mmm(list(lam[:, 12])), mmm(list(lam[:, 15])),
                 mmm([one("BASE", h=h, seed=s)["L6"]["kappa_alt_max_rel_diff"] for s in SEEDS], 2)])
table(["h", "λ_1", "λ_4", "λ_13", "λ_16", "max rel diff κ (two formulas)"], rows)
P("Full 16×16 median matrices (B, κ, and median λ) saved in results3.json under key \"L6_median\".")
P()

# ---------------------------------------------------------------------------
# L7
# ---------------------------------------------------------------------------
P("## L7 — kernel and feature change of the tri loop (report only)")
P()
qs = [("K_odd", "‖K₊−K₋‖_F/‖K_A‖_F"), ("K_even", "‖(K₊+K₋)/2−K_A‖_F/‖K_A‖_F"),
      ("probe_odd_per_input", "‖a_fn‖/16"), ("one_minus_cka_plus_minus", "1−CKA(θ₊,θ₋)"),
      ("one_minus_cka_A_plus", "1−CKA(θ_A,θ₊)")]
for q, lab in qs:
    P(f"### {lab}")
    P()
    rows = []
    for h in WIDTHS:
        row = [h]
        sl = []
        for s in SEEDS:
            recs = [one("LE", h=h, seed=s, eps=e) for e in EPS]
            vals = [r["polys"]["tri"]["L7"][q] if ok_status(r) else float("nan") for r in recs]
            sl.append(slope(EPS, vals))
        for e in EPS:
            row.append(mmm([one("LE", h=h, seed=s, eps=e)["polys"]["tri"]["L7"][q] for s in SEEDS], 3))
        row.append(mmm(sl, 3) if q in ("K_odd", "K_even", "probe_odd_per_input") else "—")
        rows.append(row)
    table(["h"] + [f"ε={e}" for e in EPS] + ["slope in ε"], rows)

# ---------------------------------------------------------------------------
# L8
# ---------------------------------------------------------------------------
P("## L8 — many cycles of circ(16), ε = 0.05 (report only)")
P()
L8M = run3.L8_M
for q, lab in (("ratio", "‖a_m‖/(m‖a_1‖)"), ("cos_a1", "cos(a_m, a_1)"), ("Ns", "‖s_m‖"),
               ("fwd_trKinv", "tr K⁻¹ (fwd endpoint)"), ("fwd_lam_min", "λ_min(K) (fwd endpoint)"),
               ("fwd_phi_u", "φ_u (fwd endpoint)")):
    P(f"### {lab}")
    P()
    rows = []
    base_key = {"fwd_trKinv": "trKinv", "fwd_lam_min": "lam_min", "fwd_phi_u": "phi_u"}.get(q)
    for h in WIDTHS:
        recs = [one("L8", h=h, seed=s) for s in SEEDS]
        row = [h]
        if base_key:
            row.append(mmm([r["m"]["0"][base_key] if ok_status(r) else float("nan") for r in recs]))
        for mm in L8M:
            row.append(mmm([r["m"][str(mm)][q] if ok_status(r) else float("nan") for r in recs]))
        rows.append(row)
    table(["h"] + (["m=0 (θ_A)"] if base_key else []) + [f"m={mm}" for mm in L8M], rows)

# ---------------------------------------------------------------------------
# K1
# ---------------------------------------------------------------------------
P("## K1a — d_ret = −ε² c_u(θ_A) (rule at ε = 0.0125: cos ≤ −0.99 AND ratio in [0.95, 1.05])")
P()
rows = []
for h in WIDTHS:
    rr = [one("K1", h=h, seed=s, eps=0.0125) for s in SEEDS]
    cs = [r["cos_c"] if ok_status(r) else float("nan") for r in rr]
    rs = [r["ratio_c"] if ok_status(r) else float("nan") for r in rr]
    mc, mr = stat(cs)[0], stat(rs)[0]
    v, n = verdict([c <= -0.99 and 0.95 <= x <= 1.05 for c, x in zip(cs, rs)],
                   mc <= -0.99 and 0.95 <= mr <= 1.05)
    add_verdict("K1a", h, v, n, "cos: " + per_seed_str(cs, 6) + "; ratio: " + per_seed_str(rs, 6))
    rows.append([h, "−1 (≤ −0.99)", mmm(cs, 6), "1 ([0.95,1.05])", mmm(rs, 6), v, f"{n}/5"])
table(["h", "pred cos", "measured cos", "pred ratio", "measured ratio", "verdict", "seeds"], rows)
P("All ε (context): cos(d_ret, c_u) / ‖d_ret‖/(ε²‖c_u‖)")
P()
rows = []
for h in WIDTHS:
    row = [h]
    for e in EPS:
        rr = [one("K1", h=h, seed=s, eps=e) for s in SEEDS]
        row.append(mmm([r["cos_c"] for r in rr], 5) + " / " + mmm([r["ratio_c"] for r in rr], 5))
    rows.append(row)
table(["h"] + [f"ε={e}" for e in EPS], rows)

P("## K1b — φ_u(θ_ret) − φ_u(θ_A) > 0 (statement: every seed, width, ε)")
P()
P("Section 8 applied per width: a seed holds if Δφ_u > 0 at all 5 ε; the median holds if the "
  "median Δφ_u > 0 at every ε.")
P()
rows = []
k1b_all = True
for h in WIDTHS:
    seed_ok, row = [], [h]
    meds = []
    for s in SEEDS:
        seed_ok.append(all(ok_status(one("K1", h=h, seed=s, eps=e)) and one("K1", h=h, seed=s, eps=e)["dphi"] > 0
                           for e in EPS))
    for e in EPS:
        vals = [one("K1", h=h, seed=s, eps=e)["dphi"] for s in SEEDS]
        meds.append(stat(vals)[0])
        row.append(mmm(vals))
    v, n = verdict(seed_ok, all(m > 0 for m in meds))
    k1b_all &= all(seed_ok)
    add_verdict("K1b", h, v, n, "seeds with Δφ>0 at all ε: " + per_seed_str([str(x) for x in seed_ok]))
    rows.append(row + [v, f"{n}/5"])
table(["h"] + [f"Δφ_u ε={e}" for e in EPS] + ["verdict", "seeds"], rows)
P(f"Literal statement (every seed, every width, every ε): {'HOLDS' if k1b_all else 'DOES NOT HOLD'}.")
P()

P("## K1c — φ_u(θ_ret) − φ_u(θ_A) = ε²‖c_u‖² (rule at ε = 0.0125: ratio in [0.9, 1.1])")
P()
rows = []
for h in WIDTHS:
    rs = [one("K1", h=h, seed=s, eps=0.0125)["dphi_ratio"] for s in SEEDS]
    mr = stat(rs)[0]
    v, n = verdict([0.9 <= x <= 1.1 for x in rs], 0.9 <= mr <= 1.1)
    add_verdict("K1c", h, v, n, per_seed_str(rs, 6))
    row = [h, "1 ([0.9, 1.1])", mmm(rs, 6), v, f"{n}/5"]
    for e in EPS:
        row.append(mmm([one("K1", h=h, seed=s, eps=e)["dphi_ratio"] for s in SEEDS]))
    rows.append(row)
table(["h", "predicted", "measured ε=0.0125", "verdict", "seeds"] + [f"ratio ε={e}" for e in EPS], rows)

# ---------------------------------------------------------------------------
# K2 / K3
# ---------------------------------------------------------------------------
QN = [("phi_u", "φ_u"), ("phi_v", "φ_v"), ("trKinv", "tr K⁻¹"), ("lam_min", "λ_min(K)"),
      ("lam_max", "λ_max(K)"), ("norm_theta", "‖θ‖"), ("dist_thetaA", "‖θ−θ_A‖"),
      ("probe_change", "‖f(X_probe;θ)−f(X_probe;θ_A)‖/16")]


def mtables(grid, Ms):
    for q, lab in QN:
        P(f"### {grid}: {lab}")
        P()
        rows = []
        for h in WIDTHS:
            recs = [one(grid, h=h, seed=s) for s in SEEDS]
            row = [h]
            for mm in [0] + Ms:
                row.append(mmm([r["m"][str(mm)][q] if ok_status(r) else float("nan") for r in recs]))
            if q in ("phi_u", "phi_v", "trKinv", "lam_min", "lam_max"):
                row.append(mmm([r["m"][str(Ms[-1])][q] / r["m"]["0"][q] if ok_status(r) else float("nan")
                                for r in recs]))
            rows.append(row)
        hdr = ["h"] + [f"m={mm}" for mm in [0] + Ms]
        if q in ("phi_u", "phi_v", "trKinv", "lam_min", "lam_max"):
            hdr.append(f"ratio m={Ms[-1]}/m=0")
        table(hdr, rows)


P("## K2 — repeated excursions along u, ε = 0.1 (report + rule)")
P()
P("Rule K2: φ_u strictly increasing in m in at least 90% of (seed, width) pairs "
  "(checked over the recorded m ∈ {1,2,4,…,64}; also shown including m = 0).")
P()
cnt, cnt0, tot, fails = 0, 0, 0, []
for h in WIDTHS:
    for s in SEEDS:
        r = one("K2", h=h, seed=s)
        tot += 1
        if not ok_status(r):
            fails.append((h, s, "INVALID"))
            continue
        ph = [r["m"][str(mm)]["phi_u"] for mm in run3.K2_M]
        inc = all(b > a for a, b in zip(ph, ph[1:]))
        inc0 = inc and ph[0] > r["m"]["0"]["phi_u"]
        cnt += inc
        cnt0 += inc0
        if not inc:
            fails.append((h, s, [fmt(x, 6) for x in ph]))
k2v = "PASS" if cnt >= 0.9 * tot else "FAIL"
P(f"φ_u strictly increasing over m=1..64: {cnt}/{tot} pairs ({100 * cnt / tot:.0f}%); "
  f"including m=0: {cnt0}/{tot}. Rule K2: **{k2v}**.")
P()
for f_ in fails:
    P(f"- not strictly increasing: h={f_[0]} seed={f_[1]}: {f_[2]}")
P()
VERDICTS.append(("K2", "pooled", k2v, cnt, f"{cnt}/{tot} pairs"))
mtables("K2", run3.K2_M)

P("## K3 — random excursions, ε = 0.1 (report + rule)")
P()
cnt, tot, fails = 0, 0, []
for h in WIDTHS:
    for s in SEEDS:
        r = one("K3", h=h, seed=s)
        tot += 1
        if ok_status(r) and r["m"]["128"]["trKinv"] > r["m"]["0"]["trKinv"]:
            cnt += 1
        else:
            fails.append((h, s, r["m"]["128"]["trKinv"] / r["m"]["0"]["trKinv"] if ok_status(r) else "INVALID"))
k3v = "PASS" if cnt == tot else "FAIL"
P(f"Rule K3 (tr K⁻¹ at m=128 > start, every (seed, width)): {cnt}/{tot}. **{k3v}**.")
P()
for f_ in fails:
    P(f"- h={f_[0]} seed={f_[1]}: ratio {f_[2]}")
VERDICTS.append(("K3", "pooled", k3v, cnt, f"{cnt}/{tot} pairs"))
P()
mtables("K3", run3.K3_M)

# ---------------------------------------------------------------------------
# K4
# ---------------------------------------------------------------------------
P("## K4a — d_sub(N) = +(L²/2)(1−1/N) c_u(θ_A) (rule at L = 0.05, N ∈ {2,4,16}: cos ≥ 0.99 AND ratio in [0.95, 1.05])")
P()
for L in run3.K4_L:
    P(f"L = {L}" + (" (rule applies to N ∈ {2,4,16})" if L == 0.05 else " (report only)"))
    P()
    rows = []
    for h in WIDTHS:
        recs = [one("K4", h=h, seed=s, L=L) for s in SEEDS]
        row = [h]
        seed_ok = [True] * 5
        med_ok = True
        for N in run3.K4_N[1:]:
            cs = [r["N"][str(N)]["cos_c"] if ok_status(r) else float("nan") for r in recs]
            rs = [r["N"][str(N)]["ratio_c"] if ok_status(r) else float("nan") for r in recs]
            row.append(mmm(cs, 5) + " / " + mmm(rs, 5))
            if L == 0.05 and N in (2, 4, 16):
                med_ok &= stat(cs)[0] >= 0.99 and 0.95 <= stat(rs)[0] <= 1.05
                for i in range(5):
                    seed_ok[i] &= cs[i] >= 0.99 and 0.95 <= rs[i] <= 1.05
        if L == 0.05:
            v, n = verdict(seed_ok, med_ok)
            det = "; ".join(f"N={N} cos: " + per_seed_str([r["N"][str(N)]["cos_c"] for r in recs], 5)
                            + " ratio: " + per_seed_str([r["N"][str(N)]["ratio_c"] for r in recs], 5)
                            for N in (2, 4, 16))
            add_verdict("K4a", h, v, n, det)
            row += [v, f"{n}/5"]
        rows.append(row)
    hdr = ["h"] + [f"N={N}: cos (pred 1) / ratio (pred 1)" for N in run3.K4_N[1:]]
    if L == 0.05:
        hdr += ["verdict", "seeds"]
    table(hdr, rows)

P("## K4b — φ_u(θ^(N)) < φ_u(θ^(1)) (statement: every seed, width, N at L ∈ {0.1, 0.05})")
P()
P("Section 8 per width: a seed holds if the inequality holds for all N ∈ {2,4,8,16} and both L; "
  "the median holds if the median of φ_u(θ^(N)) − φ_u(θ^(1)) is < 0 for every (N, L).")
P()
rows = []
k4b_all = True
for h in WIDTHS:
    seed_ok = [True] * 5
    med_ok = True
    row = [h]
    for L in run3.K4_L:
        recs = [one("K4", h=h, seed=s, L=L) for s in SEEDS]
        for N in run3.K4_N[1:]:
            vals = [r["N"][str(N)]["dphi_vs_jump"] for r in recs]
            row.append(mmm(vals))
            med_ok &= stat(vals)[0] < 0
            for i in range(5):
                seed_ok[i] &= vals[i] < 0
    v, n = verdict(seed_ok, med_ok)
    k4b_all &= all(seed_ok)
    add_verdict("K4b", h, v, n, per_seed_str([str(x) for x in seed_ok]))
    rows.append(row + [v, f"{n}/5"])
table(["h"] + [f"Δφ L={L} N={N}" for L in run3.K4_L for N in run3.K4_N[1:]] + ["verdict", "seeds"], rows)
P(f"Literal statement (every seed, width, N, L): {'HOLDS' if k4b_all else 'DOES NOT HOLD'}.")
P()

# ---------------------------------------------------------------------------
# K5
# ---------------------------------------------------------------------------
P("## K5 — GD steps to fit fresh targets from S0 = θ_A and S1 = K3 state after 128 excursions")
P()
P("GD learner SPEC.md §3.2 (tol 1e-8, cap 400,000, lr = n/λ_max at the start state). Rule K5: "
  "median over (seed, j) of steps(S1)/steps(S0) > 1 at both widths.")
P()
rows = []
k5ok = True
for h in run3.K5_WIDTHS:
    ratios = []
    for s in run3.K5_SEEDS:
        r = one("K5", h=h, seed=s)
        if not ok_status(r):
            rows.append([h, s, r.get("status") if r else "MISSING"] + [""] * 7)
            continue
        st = r["states"]
        for j in range(3):
            ratios.append(r["ratios"][j])
        rows.append([h, s, fmt(st["S0"]["kernel"]["lam_max"]), fmt(st["S0"]["kernel"]["lam_min"]),
                     fmt(st["S1"]["kernel"]["lam_max"]), fmt(st["S1"]["kernel"]["lam_min"]),
                     ", ".join(("> cap " if not x["converged"] else "") + str(x["steps"]) for x in st["S0"]["runs"]),
                     ", ".join(("> cap " if not x["converged"] else "") + str(x["steps"]) for x in st["S1"]["runs"]),
                     ", ".join(fmt(x) for x in r["ratios"]), fmt(r["dist_S1_S0"])])
    med = float(np.median(ratios)) if ratios else float("nan")
    k5ok &= med > 1
    VERDICTS.append(("K5", h, "holds" if med > 1 else "does not hold", len(ratios),
                     f"median ratio {fmt(med)} over {len(ratios)} (seed, j)"))
    P(f"h={h}: median steps(S1)/steps(S0) over (seed, j) = {fmt(med)} (n={len(ratios)})")
    P()
table(["h", "seed", "λ_max S0", "λ_min S0", "λ_max S1", "λ_min S1", "steps S0 (j=1,2,3)",
       "steps S1 (j=1,2,3)", "ratios S1/S0", "‖θ_S1−θ_S0‖"], rows)
P(f"Rule K5 (both widths): **{'PASS' if k5ok else 'FAIL'}**.")
P()
VERDICTS.append(("K5", "both", "PASS" if k5ok else "FAIL", 0, ""))

# ---------------------------------------------------------------------------
# G
# ---------------------------------------------------------------------------
P("## G — the two laws under plain GD (h = 128, report only)")
P()
rows = []
for s in run3.G_SEEDS:
    for e in run3.G_EPS:
        r = one("G", h=128, seed=s, eps=e)
        if r is None:
            rows.append([s, e, "NOT RUN"] + [""] * 8)
            continue
        if r["status"] in ("TIMEOUT", "ERROR"):
            rows.append([s, e, r["status"]] + [""] * 8)
            continue
        rows.append([s, e, r["status"], fmt(r["cos_b"], 5), fmt(r["ratio_b"], 5), fmt(r["cos_c"], 5),
                     fmt(r["ratio_c"], 5), fmt(r["base_dist_gd_gn"]),
                     r["learn_tri"]["total_steps"], r["learn_ret"]["total_steps"],
                     fmt(max(r["learn_tri"]["max_res_inf"], r["learn_ret"]["max_res_inf"]))])
table(["seed", "ε", "status", "cos(a_tri^GD, b_uv)", "‖a‖/((ε²/2)‖b‖)", "cos(d_ret^GD, c_u)",
       "‖d‖/(ε²‖c_u‖)", "‖θ_A^GD − θ_A^GN‖", "GD steps tri±", "GD steps ret", "max end res_inf"], rows)
for e in run3.G_EPS:
    rr = [one("G", h=128, seed=s, eps=e) for s in run3.G_SEEDS]
    rr = [r for r in rr if r is not None and r["status"] in ("OK", "NOT CONVERGED")]
    if rr:
        P(f"ε={e}: median [min,max] over seeds — cos_b {mmm([r['cos_b'] for r in rr], 5)}, "
          f"ratio_b {mmm([r['ratio_b'] for r in rr], 5)}, cos_c {mmm([r['cos_c'] for r in rr], 5)}, "
          f"ratio_c {mmm([r['ratio_c'] for r in rr], 5)}")
        P()

# ---------------------------------------------------------------------------
# M
# ---------------------------------------------------------------------------
P("## M — mixed loop: B moves inputs, C moves labels (report only)")
P()
rows = []
for h in run3.M_WIDTHS:
    sl = []
    for s in SEEDS:
        recs = [one("M", h=h, seed=s, eps=e) for e in EPS]
        sl.append(slope(EPS, [r["tri"]["mixed"]["Na"] if ok_status(r) else float("nan") for r in recs]))
    row = [h, mmm(sl, 3)]
    for e in EPS:
        rr = [one("M", h=h, seed=s, eps=e) for s in SEEDS]
        row.append(mmm([r["tri"]["mixed"]["Na"] if ok_status(r) else float("nan") for r in rr], 3))
    rows.append(row)
table(["h", "slope of ‖a_mixed‖ in ε"] + [f"‖a_mixed‖ ε={e}" for e in EPS], rows)
for q, lab in (("ratio_input", "‖a_mixed‖/‖a_input-tri‖"), ("ratio_label", "‖a_mixed‖/‖a_label-tri‖")):
    rows = []
    for h in run3.M_WIDTHS:
        row = [h]
        for e in EPS:
            rr = [one("M", h=h, seed=s, eps=e) for s in SEEDS]
            row.append(mmm([r["tri"]["mixed"].get(q, float("nan")) if r else float("nan") for r in rr], 3))
        rows.append(row)
    P(f"### {lab}")
    P()
    table(["h"] + [f"ε={e}" for e in EPS], rows)
rows = []
for h in run3.M_WIDTHS:
    row = [h]
    for e in EPS:
        rr = [one("M", h=h, seed=s, eps=e) for s in SEEDS]
        row.append(mmm([r["tri"]["input"]["Na"] for r in rr], 3) + " / " + mmm([r["tri"]["label"]["Na"] for r in rr], 3))
    rows.append(row)
P("### ‖a_input-tri‖ / ‖a_label-tri‖ (context)")
P()
table(["h"] + [f"ε={e}" for e in EPS], rows)

# ---------------------------------------------------------------------------
# verdict summary, invalid, run meta
# ---------------------------------------------------------------------------
sup = g("SUPP_THREADS")
if sup:
    P("## SUPPLEMENTARY (not part of the specified grid) — sensitivity of long GN sequences to BLAS thread count")
    P()
    P("All main grids ran with torch.set_num_threads(2). Here the K2 (m=64) and K3 (m=64, 128) endpoints "
      "for h ∈ {32, 64} were recomputed with 1 and 8 threads (different floating-point summation order, "
      "same code and data). rel = ‖θ_T − θ_T2‖/‖θ_T2‖.")
    P()
    rows = []
    for r in sorted(sup, key=lambda r: (r["h"], r["seed"])):
        row = [r["h"], r["seed"]]
        for q in ("K2_64", "K3_64", "K3_128"):
            cells = []
            for t in ("T1", "T2", "T8"):
                x = r[q][t]
                if "error" in x:
                    cells.append(f"{t}: INVALID")
                else:
                    cells.append(f"{t}: φ_u {fmt(x['phi_u'])}, trK⁻¹ {fmt(x['trKinv'])}"
                                 + (f", rel {x['rel_diff_vs_T2']:.1e}" if "rel_diff_vs_T2" in x else ""))
            row.append("; ".join(cells))
        rows.append(row)
    table(["h", "seed", "K2 m=64", "K3 m=64", "K3 m=128"], rows)
for T in (1, 8):
    r = one(f"SUPP_K5_T{T}")
    if r:
        if T == 1:
            P("## SUPPLEMENTARY — K5 for (h=32, seed=2) with S1 recomputed under other thread counts")
            P()
            P("The main-run S1 (2 threads) is INVALID (GN not converged during the K3 excursions). "
              "Not included in the K5 rule above.")
            P()
        P(f"threads={T}: λ_max S0 {fmt(r['S0']['lam_max'])}, λ_min S0 {fmt(r['S0']['lam_min'])}, "
          f"λ_max S1 {fmt(r['S1']['lam_max'])}, λ_min S1 {fmt(r['S1']['lam_min'])}; steps S0 "
          f"{[x[0] for x in r['S0']['steps']]} (converged {[x[1] for x in r['S0']['steps']]}); steps S1 "
          f"{[x[0] for x in r['S1']['steps']]} (converged {[x[1] for x in r['S1']['steps']]}); ratios "
          f"{[fmt(x) for x in r['ratios']]}")
        P()

P("## Verdict summary (section 8 rule)")
P()
table(["prediction", "width", "verdict", "seeds holding", "per-seed values (shown when not PASS)"],
      [[p, h, v, n, (d if v not in ("PASS",) else "")] for p, h, v, n, d in VERDICTS])

P("## INVALID / NOT CONVERGED / TIMEOUT conditions")
P()
if INVALID:
    for x in INVALID:
        P(f"- {x}")
else:
    P("None.")
P()
P("## Run log (per grid process)")
P()
table(["grid", "widths", "conditions run", "wall s", "stopped at cap", "finished"],
      [[m["grid"], m["widths"], m["n_run"], f"{m['wall']:.1f}", m["stopped_at_cap"], m["finished"]]
       for m in db["meta"]["runs"]])
gw = {}
for r in R:
    gw[r["grid"]] = gw.get(r["grid"], 0) + r.get("wall", 0)
P("Sum of per-condition wall time by grid (s): " + ", ".join(f"{k} {v:.0f}" for k, v in gw.items()))
P()

with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

# save L6 medians into results3.json
run3._lock()
try:
    dbw = run3.load()
    dbw["L6_median"] = L6med
    tmp = run3.OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dbw, fh, indent=1)
    os.replace(tmp, run3.OUT)
finally:
    run3._unlock()
print(f"wrote {OUT_MD} ({len(lines)} lines)")
