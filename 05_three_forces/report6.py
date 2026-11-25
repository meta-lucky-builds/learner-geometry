"""Reads results6.json and writes REPORT6.md (SPEC6 section 8). Run: python report6.py [test_output_file]"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "REPORT6.md")
WIDTHS = [32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
C5 = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 1.95]
EPS5 = [0.1, 0.05, 0.025]
STATES = ["S0", "u4", "u8", "u16", "v4", "v8", "v16", "GD4", "GD8", "GD16"]
TT = [50, 200, 800, 3200]
GAM = ["0.0", "0.5", "0.9", "0.99"]
PIS = ["16", "64", "256"]
MS = ["1", "2", "4", "8", "16"]

db = json.load(open(os.path.join(HERE, "results6.json")))
REC = {(r["test"], r["h"], r["seed"]): r for r in db["records"]}
lines = []
anomalies = []
notrun = []


def P(s=""):
    lines.append(s)


def g(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(x)
    if not np.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e4 or a < 1e-3:
        return f"{x:.{max(nd - 1, 1)}e}"
    return f"{x:.{nd}f}" if a < 10 else f"{x:.{max(nd - 2, 1)}f}"


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def mmm(v, nd=4):
    w = [x for x in v if x is not None and np.isfinite(x)]
    if not w:
        return "—"
    return f"{g(float(np.median(w)), nd)} [{g(min(w), nd)}, {g(max(w), nd)}]"


def verdict(per_seed, median_ok):
    """SPEC6 sec. 0: PASS if rule holds for the median AND >= 4 of 5 seeds; INCONCLUSIVE if median only."""
    k = sum(1 for s in SEEDS if per_seed.get(s) is True)
    if median_ok and k >= 4:
        return "**PASS**", k
    if median_ok:
        return "**INCONCLUSIVE**", k
    return "**FAIL**", k


def recs(test, h):
    out = {}
    for s in SEEDS:
        r = REC.get((test, h, s))
        if r is None:
            notrun.append(f"{test} h={h} seed={s}: no record (not run)")
            continue
        if r["status"] != "OK":
            notrun.append(f"{test} h={h} seed={s}: {r['status']} {r.get('error', '')}")
            continue
        out[s] = r
    return out


def inr(x, lo, hi):
    return x is not None and np.isfinite(x) and lo <= x <= hi


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


# ---------------------------------------------------------------------------
P("# REPORT6 — five CPU tests of the three-force picture (SPEC6.md, Tests 5–9)")
P()
th = sorted({r.get("threads") for r in db["records"] if "threads" in r})
P(f"float64, CPU. Threads: torch.set_num_threads(2) and OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=2 in "
  f"every process (recorded per job: {th}); up to 4 worker processes ran in parallel. Widths h ∈ {{32, 64, 128}}, "
  f"seeds 0–4 for every test.")
P()
P("Verdict rule (SPEC6 §0): PASS if the rule holds for the median over seeds AND for at least 4 of 5 seeds; "
  "INCONCLUSIVE if median only; FAIL otherwise. A seed with a non-converged / missing run counts as not holding. "
  "\"median\" = the rule applied to the per-quantity medians over seeds. Tables give median [min, max] over seeds.")
P()
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    P("## test_part6.py output (verbatim)")
    P()
    P("```")
    P(open(sys.argv[1]).read().rstrip())
    P("```")
    P()

# ===========================================================================
# Test 5
# ===========================================================================
P("## Test 5 — large steps along the strongest direction lower the top eigenvalue")
P()
P("Claim: GD with step factor c > 1 zigzags along q_1 and lowers λ_1 relative to the GN solution with the same "
  "outputs; c < 1 raises it. Δλ_1(c) = λ_1(K(θ_c)) − λ_1(K(θ_GN)), target y_A + ε q_1(θ_A).")
P()
P("‖P_v∇λ_1‖² (for pred2) was computed as SPEC6 says: orthonormal basis b_k of span{T_ij} (SVD of the P×256 matrix of "
  "T_ij = P_v[Dξ_{e_i}·ξ_{e_j}](θ_A), singular values > 1e-10·s_max kept), c_k = Dλ_1[b_k] by central FD "
  "(step 1e-4), g = Σ c_k b_k, ‖P_v∇λ_1‖² := Dλ_1[g] (FD, step 1e-4/‖g‖, times ‖g‖). Cross-checks per job: Σ c_k², "
  "and g vs the identity P_v∇λ_1 = 2λ_1² LAW(q_1 q_1ᵀ).")
P()
P("| width | rank of span{T_ij} | Dλ_1[g] vs Σc_k² (max rel diff) | cos(g, 2λ_1²LAW(q_1q_1ᵀ)) min | ‖g‖/‖2λ_1²LAW(q_1q_1ᵀ)‖ median | λ_1(θ_A) median | λ_16(θ_A) median |")
P("|---|---|---|---|---|---|---|")
for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    b = [R[s]["base"] for s in R]
    P(f"| {h} | {sorted({x['rank'] for x in b})} | {g(max(abs(x['G'] - x['sumsq']) / x['G'] for x in b))} | "
      f"{g(min(x['cos_g_identity'] for x in b), 8)} | {g(med([x['ratio_g_identity'] for x in b]), 8)} | "
      f"{g(med([x['lam1_A'] for x in b]))} | {g(med([x['lam16_A'] for x in b]))} |")
P()
cnr = []
for h in WIDTHS:
    for s, r in recs("t5", h).items():
        for e in EPS5:
            for c in C5:
                x = r["A"][str(e)]["rows"][str(c)]
                if x["status"] != "CONVERGED":
                    cnr.append(f"h={h} seed={s} ε={e} c={c}: {x['status']} after {x['steps']} steps (‖r‖_∞ {x['res_inf']:.2e})")
P("Non-converged GD runs (Protocol A): " + ("; ".join(cnr) if cnr else "none (every GD(c) run reached ‖r‖_∞ ≤ 1e-8)."))
P()


def t5_rows(r, e):
    return r["A"][str(e)]["rows"]


for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    P(f"### Test 5, h = {h}, ε = 0.025: per c (median [min, max] over seeds)")
    P()
    P("| c | Δλ_1 | pred1 = Dλ_1[LAW(C_c − C_GN)] | pred2 [derived] | Δλ_1/pred1 | Δλ_1/pred2 | GD steps (median) |")
    P("|---|---|---|---|---|---|---|")
    for c in C5:
        xs = [t5_rows(R[s], 0.025)[str(c)] for s in R]
        r1 = [x["dlam1"] / x["pred1"] if x["pred1"] else float("nan") for x in xs]
        r2 = [x["dlam1"] / x["pred2"] if x["pred2"] else float("nan") for x in xs]
        P(f"| {c} | {mmm([x['dlam1'] for x in xs])} | {mmm([x['pred1'] for x in xs])} | {mmm([x['pred2'] for x in xs])} | "
          f"{mmm(r1)} | {mmm(r2)} | {g(med([x['steps'] for x in xs]))} |")
    P()

P("### Test 5 H-table (ε = 0.025)")
P()
P("| id | rule | width | measured (per-seed detail if not PASS) | seeds holding | verdict |")
P("|---|---|---|---|---|---|")
H5 = {}
for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    rows = {s: t5_rows(R[s], 0.025) for s in R}
    ok_conv = {s: all(rows[s][str(c)]["status"] == "CONVERGED" for c in C5) for s in rows}
    cs_ne1 = [c for c in C5 if c != 1.0]
    cs_far = [c for c in C5 if abs(c - 1) >= 0.25]
    # H5a
    per = {s: ok_conv[s] and all(np.sign(rows[s][str(c)]["dlam1"]) == np.sign(1 - c) for c in cs_ne1) for s in rows}
    mdl = {c: med([rows[s][str(c)]["dlam1"] for s in rows]) for c in cs_ne1}
    mok = all(np.sign(mdl[c]) == np.sign(1 - c) for c in cs_ne1)
    v, k = verdict(per, mok)
    det = "median Δλ_1 signs: " + ", ".join(f"c={c}:{'+' if mdl[c] > 0 else '−'}" for c in cs_ne1)
    if v != "**PASS**":
        det += " | per seed Δλ_1: " + "; ".join(
            f"s{s}: " + ", ".join(f"{c}:{g(rows[s][str(c)]['dlam1'], 3)}" for c in cs_ne1) for s in rows)
    P(f"| H5a | sign(Δλ_1) = sign(1 − c), every c ≠ 1 | {h} | {det} | {k}/5 | {v} |")
    H5[("H5a", h)] = v
    # H5b, H5c
    for hid, key_, lo, hi, tag in (("H5b", "pred1", 0.8, 1.25, ""), ("H5c", "pred2", 0.67, 1.5, " [derived]")):
        rat = {s: {c: rows[s][str(c)]["dlam1"] / rows[s][str(c)][key_] for c in cs_far} for s in rows}
        per = {s: ok_conv[s] and all(inr(rat[s][c], lo, hi) for c in cs_far) for s in rows}
        mr = {c: med([rat[s][c] for s in rows]) for c in cs_far}
        mok = all(inr(mr[c], lo, hi) for c in cs_far)
        v, k = verdict(per, mok)
        det = "median ratio: " + ", ".join(f"{c}:{g(mr[c], 3)}" for c in cs_far)
        det += f"; range over seeds and c [{g(min(min(d.values()) for d in rat.values()), 3)}, {g(max(max(d.values()) for d in rat.values()), 3)}]"
        if v != "**PASS**":
            det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(f"{c}:{g(rat[s][c], 3)}" for c in cs_far) for s in rows)
        P(f"| {hid}{tag} | Δλ_1/{key_} ∈ [{lo}, {hi}] for every c with \\|c−1\\| ≥ 0.25 | {h} | {det} | {k}/5 | {v} |")
        H5[(hid, h)] = v
    # H5d
    rr = {s: abs(rows[s]["1.95"]["dlam1"]) / abs(rows[s]["1.5"]["dlam1"]) for s in rows}
    per = {s: ok_conv[s] and rr[s] >= 5 for s in rows}
    v, k = verdict(per, med(list(rr.values())) >= 5)
    det = f"\\|Δλ_1(1.95)\\|/\\|Δλ_1(1.5)\\| = {mmm(list(rr.values()), 3)} (predicted 19 from (1−c)/(2−c))"
    if v != "**PASS**":
        det += " | per seed: " + ", ".join(f"s{s}: {g(rr[s], 3)}" for s in rows)
    P(f"| H5d | \\|Δλ_1(1.95)\\| ≥ 5 \\|Δλ_1(1.5)\\| | {h} | {det} | {k}/5 | {v} |")
    H5[("H5d", h)] = v
P()
P("Report only — Δλ_1/pred1 and Δλ_1/pred2 at the larger ε (median over seeds, c with |c−1| ≥ 0.25 pooled: median [min, max]):")
P()
P("| width | ε | Δλ_1/pred1 | Δλ_1/pred2 | sign(Δλ_1) = sign(1−c) in all (seed, c≠1) |")
P("|---|---|---|---|---|")
for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    for e in EPS5:
        r1, r2, sg = [], [], True
        for s in R:
            rows = t5_rows(R[s], e)
            for c in C5:
                x = rows[str(c)]
                if abs(c - 1) >= 0.25:
                    r1.append(x["dlam1"] / x["pred1"])
                    r2.append(x["dlam1"] / x["pred2"])
                if c != 1.0:
                    sg &= np.sign(x["dlam1"]) == np.sign(1 - c)
        P(f"| {h} | {e} | {mmm(r1)} | {mmm(r2)} | {sg} |")
P()
P("Report only — c = 1 (no prediction rule): Δλ_1(c=1) and the measured q_1q_1 entry of C_c − C_GN vs ε²(1−c)/(2−c) (ε = 0.025):")
P()
P("| width | Δλ_1(c=1) | \\|Δλ_1(1)\\|/\\|Δλ_1(1.5)\\| | q_1ᵀ(C_c−C_GN)q_1 / [ε²(1−c)/(2−c)] at c = 0.25, 1.5, 1.95 |")
P("|---|---|---|---|")
for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    rows = {s: t5_rows(R[s], 0.025) for s in R}
    P(f"| {h} | {mmm([rows[s]['1.0']['dlam1'] for s in rows], 3)} | "
      f"{mmm([abs(rows[s]['1.0']['dlam1']) / abs(rows[s]['1.5']['dlam1']) for s in rows], 3)} | "
      + ", ".join(mmm([rows[s][str(c)]['Cdiff_q1q1'] / rows[s][str(c)]['Cdiff_q1q1_pred'] for s in rows], 4) for c in (0.25, 1.5, 1.95))
      + " |")
P()
P("### Test 5 Protocol B (report only): fixed target y_A + 0.05·u, GD for 3000 steps, η = c·n/λ_1(θ_A)")
P()
P("Regression: OLS with intercept of (λ_1,k+1 − λ_1,k) on x_k² over the steps with c_eff,k > 1.5. "
  "Predicted slope [derived] = −(c_eff²/4)·‖P_v∇λ_1‖²/λ_1² at θ_A, with c_eff = c (its value at θ_A); the same with "
  "c_eff = median c_eff over the regression steps is also given.")
P()
P("| width | c | seed | stopped (‖r‖>1e6) | λ_1: start → min → end | c_eff: start → end | first k with c_eff<2 | frac Δλ_1<0 before it | loss max / end | reg n | slope | R² | pred slope (c_eff=c) | pred (median c_eff) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for h in WIDTHS:
    R = recs("t5", h)
    if not R:
        continue
    for c in ("2.05", "1.9"):
        for s in R:
            b = R[s]["B"][c]
            P(f"| {h} | {c} | {s} | {b['stopped_at']} | {g(b['lam1_0'])} → {g(b['lam1_min'])} → {g(b['lam1_end'])} | "
              f"{g(b['c_eff_0'])} → {g(b['c_eff_end'])} | {b['first_k_ceff_below_2']} | {g(b['frac_dlam_neg_before_k2'], 3)} | "
              f"{g(b['loss_max'], 3)} / {g(b['loss_end'], 3)} | {b['reg_n']} | {g(b['reg_slope'], 4)} | {g(b['reg_R2'], 3)} | "
              f"{g(b['pred_slope_ceff_c'], 4)} | {g(b['pred_slope_ceff_median'], 4)} |")
P()

# ===========================================================================
# Test 6
# ===========================================================================
P("## Test 6 — stiffening protects what was practised (consolidation)")
P()
P("Claim: repeated exact out-and-back practice along u stiffens the kernel along u, so brief later training on "
  "other targets moves the outputs along u less (hard to learn ⇔ hard to overwrite).")
P()
P("States: S0 = θ_A; u_m / v_m = m GN out-and-back excursions along u / v (ε = 0.1); GD_m = the same along u with "
  "every learn by GD(1) (tcr.gd_learn: η = n/λ_max(K(θ_in)), stop at ‖r‖_∞ ≤ 1e-8). Probe: 16 unit b_j "
  "(Generator(6000+seed)), 50 GD steps with η_0 = 0.5·n/λ_max(K(θ_A)) toward y_A + δb_j, δ = 0.05‖y_A‖.")
P()
gdinfo = []
for h in WIDTHS:
    for s, r in recs("t6", h).items():
        pi = r["practice"]["GD"]
        if not pi["all_converged"]:
            gdinfo.append(f"h={h} s={s}")
P("GD(1) practice learns not converged: " + (", ".join(gdinfo) if gdinfo else "none") + ".")
P()
for h in WIDTHS:
    R = recs("t6", h)
    if not R:
        continue
    P(f"### Test 6, h = {h}: per state (median [min, max] over seeds)")
    P()
    P("| state | ν_u/ν_u(S0) | ν_v/ν_v(S0) | ν_u | ν_u^lin | ν_u/ν_u^lin | ν_u/ν_u^lin,sample (supp.) | π_u | π_u^lin | π_u/π_u^lin | generic plasticity | φ_u | φ_v | λ_min | tr K⁻¹ |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in STATES:
        X = {s: R[s]["states"][st] for s in R}
        Z = {s: R[s]["states"]["S0"] for s in R}
        P(f"| {st} | {mmm([X[s]['nu_u'] / Z[s]['nu_u'] for s in R])} | {mmm([X[s]['nu_v'] / Z[s]['nu_v'] for s in R])} | "
          f"{mmm([X[s]['nu_u'] for s in R])} | {mmm([X[s]['nu_u_lin'] for s in R])} | "
          f"{mmm([X[s]['nu_u'] / X[s]['nu_u_lin'] for s in R])} | {mmm([X[s]['nu_u'] / X[s]['nu_u_lin_sample'] for s in R])} | "
          f"{mmm([X[s]['pi_u'] for s in R])} | {mmm([X[s]['pi_u_lin'] for s in R])} | {mmm([X[s]['pi_u'] / X[s]['pi_u_lin'] for s in R])} | "
          f"{mmm([X[s]['pi_generic'] for s in R])} | {mmm([X[s]['phi_u'] for s in R])} | {mmm([X[s]['phi_v'] for s in R])} | "
          f"{mmm([X[s]['lam_min'] for s in R])} | {mmm([X[s]['trKinv'] for s in R])} |")
    P()
    P(f"Price of protection (generic plasticity u16 / S0), h = {h}: "
      f"{mmm([R[s]['states']['u16']['pi_generic'] / R[s]['states']['S0']['pi_generic'] for s in R])}; "
      f"(v16 / S0): {mmm([R[s]['states']['v16']['pi_generic'] / R[s]['states']['S0']['pi_generic'] for s in R])}; "
      f"(GD16 / S0): {mmm([R[s]['states']['GD16']['pi_generic'] / R[s]['states']['S0']['pi_generic'] for s in R])}")
    P()

P("### Test 6 H-table")
P()
P("| id | rule | width | measured (per-seed detail if not PASS) | seeds holding | verdict |")
P("|---|---|---|---|---|---|")
H6 = {}
pool_nu, pool_pi, seed_pool = [], [], {s: ([], []) for s in SEEDS}
width_pool = {}
for h in WIDTHS:
    R = recs("t6", h)
    if not R:
        continue
    st = {s: R[s]["states"] for s in R}
    ru = {s: {m: st[s][f"u{m}"]["nu_u"] / st[s]["S0"]["nu_u"] for m in (4, 8, 16)} for s in st}
    per = {s: (1 > ru[s][4] > ru[s][8] > ru[s][16]) and ru[s][16] < 0.8 for s in st}
    mr = {m: med([ru[s][m] for s in st]) for m in (4, 8, 16)}
    v, k = verdict(per, (1 > mr[4] > mr[8] > mr[16]) and mr[16] < 0.8)
    det = "median ν_u(u_m)/ν_u(S0), m=4,8,16: " + ", ".join(g(mr[m], 4) for m in (4, 8, 16))
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(g(ru[s][m], 4) for m in (4, 8, 16)) for s in st)
    P(f"| H6a | ν_u(u_m)/ν_u(S0) decreasing in m (1 > m4 > m8 > m16) and < 0.8 at m = 16 | {h} | {det} | {k}/5 | {v} |")
    H6[("H6a", h)] = v
    # H6b
    q = {s: (st[s]["u16"]["nu_u"] / st[s]["S0"]["nu_u"], st[s]["u16"]["nu_v"] / st[s]["S0"]["nu_v"],
             st[s]["v16"]["nu_v"] / st[s]["S0"]["nu_v"], st[s]["v16"]["nu_u"] / st[s]["S0"]["nu_u"]) for s in st}
    per = {s: q[s][0] < q[s][1] and q[s][2] < q[s][3] for s in st}
    mq = [med([q[s][i] for s in st]) for i in range(4)]
    v, k = verdict(per, mq[0] < mq[1] and mq[2] < mq[3])
    det = (f"median: u16 ν_u-ratio {g(mq[0], 4)} vs ν_v-ratio {g(mq[1], 4)}; v16 ν_v-ratio {g(mq[2], 4)} vs ν_u-ratio {g(mq[3], 4)}")
    if v != "**PASS**":
        det += " | per seed (u16: ν_u,ν_v; v16: ν_v,ν_u): " + "; ".join(f"s{s}: " + ", ".join(g(x, 3) for x in q[s]) for s in st)
    P(f"| H6b | specificity (both inequalities) | {h} | {det} | {k}/5 | {v} |")
    H6[("H6b", h)] = v
    # pool for H6c
    for s in st:
        for nm in STATES:
            pool_nu.append(st[s][nm]["nu_u"])
            pool_pi.append(st[s][nm]["pi_u"])
            seed_pool[s][0].append(st[s][nm]["nu_u"])
            seed_pool[s][1].append(st[s][nm]["pi_u"])
            width_pool.setdefault(h, ([], []))
            width_pool[h][0].append(st[s][nm]["nu_u"])
            width_pool[h][1].append(st[s][nm]["pi_u"])
    # H6d
    for tag, nk in (("", "nu_u_lin"), (" (supplementary: ν_u^lin evaluated on the same 16 b_j)", "nu_u_lin_sample")):
        rat = {s: {nm: (st[s][nm]["nu_u"] / st[s][nm][nk], st[s][nm]["pi_u"] / st[s][nm]["pi_u_lin"]) for nm in STATES} for s in st}
        per = {s: all(inr(a, 0.8, 1.25) and inr(b, 0.8, 1.25) for a, b in rat[s].values()) for s in st}
        mrat = {nm: (med([rat[s][nm][0] for s in st]), med([rat[s][nm][1] for s in st])) for nm in STATES}
        mok = all(inr(a, 0.8, 1.25) and inr(b, 0.8, 1.25) for a, b in mrat.values())
        v, k = verdict(per, mok)
        allnu = [rat[s][nm][0] for s in st for nm in STATES]
        allpi = [rat[s][nm][1] for s in st for nm in STATES]
        det = (f"median over seeds per state: ν-ratio range [{g(min(a for a, _ in mrat.values()), 3)}, {g(max(a for a, _ in mrat.values()), 3)}], "
               f"π-ratio range [{g(min(b for _, b in mrat.values()), 3)}, {g(max(b for _, b in mrat.values()), 3)}]; "
               f"all (seed,state): ν {g(min(allnu), 3)}–{g(max(allnu), 3)}, π {g(min(allpi), 3)}–{g(max(allpi), 3)}")
        if v != "**PASS**":
            det += " | per seed: " + "; ".join(
                f"s{s}: ν∈[{g(min(a for a, _ in rat[s].values()), 3)},{g(max(a for a, _ in rat[s].values()), 3)}] "
                f"π∈[{g(min(b for _, b in rat[s].values()), 3)},{g(max(b for _, b in rat[s].values()), 3)}]" for s in st)
        P(f"| H6d{tag} | ν_u/ν_u^lin and π_u/π_u^lin ∈ [0.8, 1.25] for every state | {h} | {det} | {k}/5 | {v} |")
        H6[("H6d" + ("supp" if tag else ""), h)] = v
    # H6e
    lr = {s: (abs(math.log(st[s]["GD16"]["nu_u"] / st[s]["S0"]["nu_u"])),
              abs(math.log(st[s]["u16"]["nu_u"] / st[s]["S0"]["nu_u"]))) for s in st}
    per = {s: lr[s][0] < 0.25 * lr[s][1] for s in st}
    m0, m1 = med([lr[s][0] for s in st]), med([lr[s][1] for s in st])
    v, k = verdict(per, m0 < 0.25 * m1)
    det = f"median \\|log ratio\\| GD16 {g(m0, 4)} vs 0.25×(u16 {g(m1, 4)}) = {g(0.25 * m1, 4)}"
    if v != "**PASS**":
        det += " | per seed (GD16, u16): " + "; ".join(f"s{s}: {g(lr[s][0], 3)}, {g(lr[s][1], 3)}" for s in st)
    P(f"| H6e | \\|log(ν_u(GD16)/ν_u(S0))\\| < 0.25·\\|log(ν_u(u16)/ν_u(S0))\\| | {h} | {det} | {k}/5 | {v} |")
    H6[("H6e", h)] = v
rho_all = spearman(pool_nu, pool_pi) if pool_nu else float("nan")
rho_seed = {s: spearman(*seed_pool[s]) for s in SEEDS if seed_pool[s][0]}
rho_w = {h: spearman(*width_pool[h]) for h in width_pool}
v6c = "**PASS**" if rho_all >= 0.9 else "**FAIL**"
vs, ks = verdict({s: rho_seed[s] >= 0.9 for s in rho_seed}, med(list(rho_seed.values())) >= 0.9)
P(f"| H6c | pooled over all states, widths, seeds: Spearman(ν_u, π_u) ≥ 0.9 | all | pooled ρ = {g(rho_all, 4)} (n = {len(pool_nu)}); "
  f"per seed (pooled over states and widths): " + ", ".join(f"s{s}: {g(v, 3)}" for s, v in rho_seed.items()) +
  f"; per width: " + ", ".join(f"h{h}: {g(v, 3)}" for h, v in rho_w.items()) +
  f" | (§0 rule on per-seed ρ: {ks}/5 → {vs}) | {v6c} |")
H6[("H6c", "all")] = v6c
P()
P("H6c note: the rule pools seeds, so the §0 per-seed clause cannot be applied literally; the verdict shown is the pooled value; "
  "the §0 rule applied to per-seed pooled ρ is given in brackets.")
P()

# ===========================================================================
# Test 7
# ===========================================================================
P("## Test 7 — order matters for the weak-kernel parts, not the strong ones")
P()
P("Claim: at convergence the order of two label moves leaves a trace only through the area (bracket) term, much larger "
  "for weak kernel directions; at a fixed step budget order changes the training outputs only in weak directions.")
P()
P("s, w, s′, w′: coefficients from Generator(7000+seed) drawn in the order s, w, s′, w′ (torch.randn(4) each) on "
  "q_1..q_4 resp. q_13..q_16 of K(θ_A); s′ ⊥ s, w′ ⊥ w; all scaled to ‖y_A‖.")
P()
kap_signs = []
for h in WIDTHS:
    for s, r in recs("t7", h).items():
        for pn in ("ss'", "ww'", "sw"):
            kap_signs.append(np.sign(r["part_a"][pn]["0.025"]["cos_O_b"]))
kpos, kneg = sum(1 for x in kap_signs if x > 0), sum(1 for x in kap_signs if x < 0)
P(f"**κ found:** sign of cos(O_pq, b_pq) at ε = 0.025 is + in {kpos} and − in {kneg} of {len(kap_signs)} (width, seed, pair) "
  f"cases → κ = {'+1' if kpos > kneg else '−1'}. (Independent check: LAW(qpᵀ − pqᵀ) = +b_pq, cos 1.000000 for every job — "
  f"the path matrix of p-first minus q-first is ε²(q pᵀ − p qᵀ).)")
P()
for h in WIDTHS:
    R = recs("t7", h)
    if not R:
        continue
    P(f"### Test 7(a), h = {h} (median [min, max] over seeds)")
    P()
    P("| pair | ε | ‖O_pq‖ | ‖b_pq‖ | cos(O, b) | ‖O‖/(ε²‖b‖) | probe RMS o_pq |")
    P("|---|---|---|---|---|---|---|")
    for pn in ("ss'", "ww'", "sw"):
        for e in ("0.1", "0.05", "0.025"):
            xs = [R[s]["part_a"][pn][e] for s in R]
            P(f"| {pn} | {e} | {mmm([x['O_norm'] for x in xs])} | {mmm([R[s]['part_a'][pn]['b_norm'] for s in R])} | "
              f"{mmm([x['cos_O_b'] for x in xs], 5)} | {mmm([x['ratio'] for x in xs])} | {mmm([x['probe_rms'] for x in xs])} |")
    P()
P("### Test 7 H-table")
P()
P("| id | rule | width | measured (per-seed detail if not PASS) | seeds holding | verdict |")
P("|---|---|---|---|---|---|")
H7 = {}
for h in WIDTHS:
    R = recs("t7", h)
    if not R:
        continue
    A = {s: R[s]["part_a"] for s in R}
    pairs = ("ss'", "ww'", "sw")
    per = {s: all(abs(A[s][p]["0.025"]["cos_O_b"]) >= 0.99 and inr(A[s][p]["0.025"]["ratio"], 0.95, 1.05) for p in pairs) for s in A}
    mc = {p: med([abs(A[s][p]["0.025"]["cos_O_b"]) for s in A]) for p in pairs}
    mr = {p: med([A[s][p]["0.025"]["ratio"] for s in A]) for p in pairs}
    v, k = verdict(per, all(mc[p] >= 0.99 and inr(mr[p], 0.95, 1.05) for p in pairs))
    det = "median \\|cos\\| / ratio: " + "; ".join(f"{p}: {g(mc[p], 5)} / {g(mr[p], 4)}" for p in pairs)
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(f"{p} {g(A[s][p]['0.025']['cos_O_b'], 4)}/{g(A[s][p]['0.025']['ratio'], 4)}" for p in pairs) for s in A)
    P(f"| H7a | \\|cos(O,b)\\| ≥ 0.99 and ‖O‖/(ε²‖b‖) ∈ [0.95, 1.05], all three pairs, ε = 0.025 | {h} | {det} | {k}/5 | {v} |")
    H7[("H7a", h)] = v
    q = {s: (A[s]["ww'"]["0.025"]["O_norm"] / A[s]["ss'"]["0.025"]["O_norm"],
             A[s]["ww'"]["0.025"]["probe_rms"] / A[s]["ss'"]["0.025"]["probe_rms"]) for s in A}
    per = {s: q[s][0] >= 10 and q[s][1] >= 5 for s in A}
    m0, m1 = med([q[s][0] for s in A]), med([q[s][1] for s in A])
    v, k = verdict(per, m0 >= 10 and m1 >= 5)
    det = f"‖O_ww′‖/‖O_ss′‖ {mmm([q[s][0] for s in A], 3)}; RMS ratio {mmm([q[s][1] for s in A], 3)}"
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: {g(q[s][0], 3)}, {g(q[s][1], 3)}" for s in A)
    P(f"| H7b | ‖O_ww′‖ ≥ 10‖O_ss′‖ and RMS o_ww′ ≥ 5 RMS o_ss′ (ε = 0.025) | {h} | {det} | {k}/5 | {v} |")
    H7[("H7b", h)] = v
    B = {s: R[s]["part_b"] for s in R}
    ws = {s: {T: B[s][str(T)]["pairs"]["EASY-HARD"]["weak_share"] for T in TT} for s in B}
    per = {s: all(ws[s][T] >= 0.9 for T in TT) for s in B}
    mw = {T: med([ws[s][T] for s in B]) for T in TT}
    v, k = verdict(per, all(mw[T] >= 0.9 for T in TT))
    det = "median weak share per T_tot: " + ", ".join(f"{T}: {g(mw[T], 4)}" for T in TT)
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(g(ws[s][T], 3) for T in TT) for s in B)
    P(f"| H7c | weak share ≥ 0.9, EASY-FIRST vs HARD-FIRST, every T_tot | {h} | {det} | {k}/5 | {v} |")
    H7[("H7c", h)] = v
    for tag, prs in (("", ("EASY-HARD", "JOINT-EASY", "JOINT-HARD")), (" (alt. reading: EASY vs HARD only)", ("EASY-HARD",))):
        cs = {s: {(p, T): B[s][str(T)]["pairs"][p]["cos_d_dlin"] for p in prs for T in TT} for s in B}
        per = {s: all(x >= 0.95 for x in cs[s].values()) for s in B}
        mcs = {kk: med([cs[s][kk] for s in B]) for kk in cs[next(iter(B))]}
        v, k = verdict(per, all(x >= 0.95 for x in mcs.values()))
        det = "median cos per (pair, T): " + "; ".join(f"{p}@{T}: {g(mcs[(p, T)], 4)}" for p in prs for T in TT)
        if v != "**PASS**":
            det += " | per seed min over (pair,T): " + ", ".join(f"s{s}: {g(min(cs[s].values()), 4)}" for s in B)
        P(f"| H7d{tag} | cos(d, d_lin) ≥ 0.95 at every T_tot{'' if tag else ' (every pair of curricula)'} | {h} | {det} | {k}/5 | {v} |")
        H7[("H7d" + ("alt" if tag else ""), h)] = v
P()
P("### Test 7(b) report only (median [min, max] over seeds)")
P()
P("| width | T_tot | probe RMS EASY−HARD | converged-order probe RMS o_sw (part a, ε = 0.1) | ‖d‖ EASY−HARD | ‖d_lin‖ | weak share (lin) | loss_s: JOINT / EASY / HARD | loss_w: JOINT / EASY / HARD |")
P("|---|---|---|---|---|---|---|---|---|")
for h in WIDTHS:
    R = recs("t7", h)
    if not R:
        continue
    for T in TT:
        B = {s: R[s]["part_b"][str(T)] for s in R}
        P(f"| {h} | {T} | {mmm([B[s]['probe_rms_EH'] for s in B])} | {mmm([R[s]['part_a']['sw']['0.1']['probe_rms'] for s in R])} | "
          f"{mmm([B[s]['pairs']['EASY-HARD']['d_norm'] for s in B])} | {mmm([B[s]['pairs']['EASY-HARD']['dlin_norm'] for s in B])} | "
          f"{mmm([B[s]['pairs']['EASY-HARD']['weak_share_lin'] for s in B])} | "
          + " / ".join(g(med([B[s]['losses'][c]['loss_s'] for s in B]), 3) for c in ("JOINT", "EASY", "HARD")) + " | "
          + " / ".join(g(med([B[s]['losses'][c]['loss_w'] for s in B]), 3) for c in ("JOINT", "EASY", "HARD")) + " |")
P()
P("loss_s = (ŝ·(F − y*))²/(2n), loss_w likewise with ŵ (final training outputs F, target y* = y_A + 0.1(s + w)). "
  "Curriculum runs use exactly the stated budgets (no stop rule); η of GD(1) recomputed at the start of each phase.")
P()

# ===========================================================================
# Test 8
# ===========================================================================
P("## Test 8 — bootstrapped targets: the extra drift, three ways")
P()
P("Claim: under noisy semi-gradient (TD-like) updates the mean extra drift is LAW(E) with E = −AΣ_y the per-step "
  "expected path matrix; equivalently a direct Hessian form and a potential form with coefficient κ = 2 on the "
  "stop-gradient term.")
P()
P("P: one P per seed, drawn sequentially from Generator(8000+seed) until the eigenvalue condition holds for all four γ "
  "simultaneously. Noise: Generator(8100 + seed + stream), ζ drawn as one (4000, 16) tensor per stream, identical for every γ.")
P()
for h in WIDTHS:
    R = recs("t8", h)
    if not R:
        continue
    P(f"### Test 8, h = {h} (median [min, max] over seeds)")
    P()
    P("| γ | ρ(I−A) | ‖d_law‖ | rel ‖d_law−d_dir‖ | rel ‖d_pot(1)−d_law‖ | rel ‖d_pot(2)−d_law‖ | κ* | ‖LAW(E_area)‖/‖LAW(E_sym)‖ | cos(V, N d_law) | ‖V‖/‖N d_law‖ | ‖V‖ | SE of mean ‖·‖ | cos(V, transient pred) | ‖V‖/‖transient‖ | cos(V, LAW(mean meas. C)) | ratio |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for gm in GAM:
        X = [R[s]["gammas"][gm] for s in R]
        P(f"| {gm} | {mmm([x['spec_rad_IA'] for x in X], 6)} | {mmm([x['d_law_norm'] for x in X])} | {mmm([x['H8a_rel'] for x in X])} | "
          f"{mmm([x['rel_pot1'] for x in X])} | {mmm([x['rel_pot2'] for x in X])} | {mmm([x['kappa_star'] for x in X], 5) if gm != '0.0' else '— (γ = 0)'} | "
          f"{mmm([x['law_area_over_sym'] for x in X])} | {mmm([x['cos_V_Ndlaw'] for x in X])} | {mmm([x['ratio_V_Ndlaw'] for x in X])} | "
          f"{mmm([x['V_norm'] for x in X])} | {mmm([x['V_se_norm'] for x in X])} | {mmm([x['cos_V_transient'] for x in X])} | "
          f"{mmm([x['ratio_V_transient'] for x in X])} | {mmm([x['cos_V_meanC'] for x in X])} | {mmm([x['ratio_V_meanC'] for x in X])} |")
    P()
P("‘transient pred’ (supplementary, not in SPEC6): LAW of the exact expected path matrix of the linearised dynamics started "
  "at x_0 = 0 for N = 4000 steps plus the GN return step, E[C] = −A Σ_{k<N} Σ_k − Σ_N; ‘LAW(mean meas. C)’: LAW of the "
  "measured path matrices averaged over the 32 streams (including the return).")
P()
P("### Test 8 H-table")
P()
P("| id | rule | width | measured (per-seed detail if not PASS) | seeds holding | verdict |")
P("|---|---|---|---|---|---|")
H8 = {}
for h in WIDTHS:
    R = recs("t8", h)
    if not R:
        continue
    Gm = {s: R[s]["gammas"] for s in R}

    def rule(hid, text, f, gl, extra=""):
        per = {s: all(f(Gm[s][gg]) for gg in gl) for s in Gm}
        medd = {gg: {kk: med([Gm[s][gg][kk] for s in Gm]) for kk in Gm[next(iter(Gm))][gg] if isinstance(Gm[next(iter(Gm))][gg][kk], (int, float)) and Gm[next(iter(Gm))][gg][kk] is not None} for gg in gl}
        v, k = verdict(per, all(f(medd[gg]) for gg in gl))
        return per, medd, v, k

    # H8a
    per, md, v, k = rule("H8a", "", lambda x: x["H8a_rel"] < 1e-5, GAM)
    det = "median rel: " + ", ".join(f"γ={gg}: {g(md[gg]['H8a_rel'], 3)}" for gg in GAM)
    if v != "**PASS**":
        det += " | per seed max over γ: " + ", ".join(f"s{s}: {g(max(Gm[s][gg]['H8a_rel'] for gg in GAM), 3)}" for s in Gm)
    P(f"| H8a [derived] | d_law = d_dir, relative difference < 1e-5, every γ | {h} | {det} | {k}/5 | {v} |")
    H8[("H8a", h)] = v
    # H8b
    per, md, v, k = rule("H8b", "", lambda x: x["H8b_rel_symAS_halfBB"] < 1e-8 and x["H8b_law_Esym_rel_vs_gamma0"] < 1e-6, GAM)
    det = ("median ‖Sym(AΣ)−½BBᵀ‖/‖½BBᵀ‖: " + ", ".join(f"γ={gg}: {g(md[gg]['H8b_rel_symAS_halfBB'], 3)}" for gg in GAM) +
           "; median ‖LAW(E_sym,γ)−LAW(E_sym,0)‖/‖LAW(E_sym,0)‖: " + ", ".join(f"γ={gg}: {g(md[gg]['H8b_law_Esym_rel_vs_gamma0'], 3)}" for gg in GAM) +
           "; supplementary ‖Sym(AΣ)−½(BBᵀ+AΣAᵀ)‖/‖Sym(AΣ)‖ max over seeds, γ: " +
           g(max(Gm[s][gg]["supp_rel_symAS_half_BB_plus_ASAt"] for s in Gm for gg in GAM), 3))
    if v != "**PASS**":
        det += " | per seed (max over γ of first quantity): " + ", ".join(f"s{s}: {g(max(Gm[s][gg]['H8b_rel_symAS_halfBB'] for gg in GAM), 3)}" for s in Gm)
    P(f"| H8b | Sym(AΣ_y) = ½BBᵀ to 1e-8, and LAW(E_sym) identical across γ to 1e-6 | {h} | {det} | {k}/5 | {v} |")
    H8[("H8b", h)] = v
    # H8c
    gl = ["0.5", "0.9", "0.99"]
    per, md, v, k = rule("H8c", "", lambda x: inr(x["kappa_star"], 1.9, 2.1), gl)
    det = "median κ*: " + ", ".join(f"γ={gg}: {g(md[gg]['kappa_star'], 6)}" for gg in gl)
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(g(Gm[s][gg]['kappa_star'], 5) for gg in gl) for s in Gm)
    P(f"| H8c [derived] | κ* ∈ [1.9, 2.1] for γ ∈ {{0.5, 0.9, 0.99}} | {h} | {det} | {k}/5 | {v} |")
    H8[("H8c", h)] = v
    # H8d
    for tag, ck, rk in (("", "cos_V_Ndlaw", "ratio_V_Ndlaw"), (" (supp.: transient-exact prediction)", "cos_V_transient", "ratio_V_transient"),
                        (" (supp.: LAW of mean measured C)", "cos_V_meanC", "ratio_V_meanC")):
        per, md, v, k = rule("H8d", "", lambda x, ck=ck, rk=rk: x[ck] >= 0.95 and inr(x[rk], 0.8, 1.25), GAM)
        det = "median cos / ratio: " + ", ".join(f"γ={gg}: {g(md[gg][ck], 4)} / {g(md[gg][rk], 4)}" for gg in GAM)
        if v != "**PASS**":
            det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(f"{g(Gm[s][gg][ck], 3)}/{g(Gm[s][gg][rk], 3)}" for gg in GAM) for s in Gm)
        P(f"| H8d{tag} | cos(V_γ, pred) ≥ 0.95 and ratio ∈ [0.8, 1.25], every γ | {h} | {det} | {k}/5 | {v} |")
        H8[("H8d" + tag, h)] = v
    # H8e
    gl = ["0.9", "0.99"]
    for tag, ck in (("", "cos_dV_dpred"), (" (supp.: transient-exact prediction)", "cos_dV_dtransient")):
        per, md, v, k = rule("H8e", "", lambda x, ck=ck: x[ck] >= 0.8, gl)
        det = "median cos: " + ", ".join(f"γ={gg}: {g(md[gg][ck], 4)}" for gg in gl)
        det += "; ‖V_γ−V_0‖ / SE (paired): " + ", ".join(f"γ={gg}: {g(md[gg]['dV_norm'], 3)} / {g(md[gg]['dV_se_norm'], 3)}" for gg in gl)
        if v != "**PASS**":
            det += " | per seed: " + "; ".join(f"s{s}: " + ", ".join(g(Gm[s][gg][ck], 3) for gg in gl) for s in Gm)
        P(f"| H8e{tag} | cos(V_γ − V_0, Δpred) ≥ 0.8 for γ ∈ {{0.9, 0.99}} | {h} | {det} | {k}/5 | {v} |")
        H8[("H8e" + tag, h)] = v
P()
ks_all = [R2["gammas"][gg]["kappa_star"] for h in WIDTHS for R2 in recs("t8", h).values() for gg in ("0.5", "0.9", "0.99")]
if ks_all:
    P(f"**κ* found:** median {g(med(ks_all), 6)}, range [{g(min(ks_all), 6)}, {g(max(ks_all), 6)}] over all widths, seeds and γ ∈ {{0.5, 0.9, 0.99}}.")
P()
P("### Test 8 report only — kernel alignment between successive states and tr K, per step (median [min, max] over seeds)")
P()
P("| width | γ | align: measured / step | align: predicted Dalign[d_law] | tr K: measured / step | tr K: predicted |")
P("|---|---|---|---|---|---|")
for h in WIDTHS:
    R = recs("t8", h)
    if not R:
        continue
    for gm in GAM:
        X = [R[s]["gammas"][gm] for s in R]
        P(f"| {h} | {gm} | {mmm([x['align_meas_per_step'] for x in X], 3)} | {mmm([x['align_pred_per_step'] for x in X], 3)} | "
          f"{mmm([x['trK_meas_per_step'] for x in X], 3)} | {mmm([x['trK_pred_per_step'] for x in X], 3)} |")
P()
P("Measured = mean over the 32 streams of [q(θ_end) − q(θ_A)]/N with θ_end after the GN return; predicted = directional "
  "derivative of q along d_law. align = (1/16) Σ_ab P_ab K_ab/√(K_aa K_bb).")
P()

# ===========================================================================
# Test 9
# ===========================================================================
P("## Test 9 — rotating targets give steady drift whose direction follows the rotation")
P()
P("Claim: targets rotating in a plane give a weight drift whose odd part (ROT+ vs ROT−) is LAW of the enclosed-area "
  "path matrix, grows linearly with the number of periods, vanishes for a back-and-forth drive, and can be aimed at a "
  "chosen unseen input.")
P()
P("Primary (literal) predictions: odd part vs LAW(C_area) and even part vs LAW(C_sym), where C is the a-priori path matrix of "
  "the ROT+ drive (linearised recursion with K(θ_A), including the a-priori GN return). Supplementary ('combo'): odd vs "
  "LAW((C₊ − C₋)/2), even vs LAW((C₊ + C₋)/2) with the a-priori matrices of ROT+ and ROT−.")
P()
H9 = {}
for pl in ("uv", "steered"):
    for h in WIDTHS:
        R = recs("t9", h)
        if not R:
            continue
        P(f"### Test 9, plane = {pl}, h = {h} (m = 8; median [min, max] over seeds)")
        P()
        P("| Π | ‖odd‖ | cos(odd, LAW(C_area)) | ‖odd‖/‖LAW(C_area)‖ | cos(even, LAW(C_sym)) | ‖even‖/‖LAW(C_sym)‖ | combo: cos odd / ratio | combo: cos even / ratio | ‖odd₁₆‖/‖odd₈‖ | ‖(V_L,ω−V_L,−ω)/2‖ | ‖V_LINE‖/‖odd‖ | ‖LAW(C_area,LINE)‖/‖odd‖ |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for Pi in PIS:
            X = {s: R[s][pl][Pi]["8"] for s in R}
            X16 = {s: R[s][pl][Pi]["16"] for s in R}
            P(f"| {Pi} | {mmm([X[s]['odd_norm'] for s in R])} | {mmm([X[s]['cos_odd_lit'] for s in R])} | {mmm([X[s]['ratio_odd_lit'] for s in R])} | "
              f"{mmm([X[s]['cos_even_lit'] for s in R])} | {mmm([X[s]['ratio_even_lit'] for s in R])} | "
              f"{g(med([X[s]['cos_odd_combo'] for s in R]), 5)} / {g(med([X[s]['ratio_odd_combo'] for s in R]), 4)} | "
              f"{g(med([X[s]['cos_even_combo'] for s in R]), 5)} / {g(med([X[s]['ratio_even_combo'] for s in R]), 4)} | "
              f"{mmm([X16[s]['odd_norm'] / X[s]['odd_norm'] for s in R])} | {mmm([X[s]['line_odd_lit_norm'] for s in R])} | "
              f"{mmm([X[s]['line_V_norm'] / X[s]['odd_norm'] for s in R])} | {mmm([X[s]['line_law_area_apri_norm'] / X[s]['odd_norm'] for s in R])} |")
        P()
P("### Test 9 H-table (m = 8 unless stated; one row per plane, Π, width)")
P()
P("| id | rule | plane | Π | width | measured (per-seed detail if not PASS) | seeds holding | verdict |")
P("|---|---|---|---|---|---|---|---|")


def h9row(hid, text, pl, Pi, h, R, f, fmt):
    X = {s: R[s][pl][Pi] for s in R}
    per = {s: f(X[s]) for s in X}
    keys = [k_ for k_, v_ in X[next(iter(X))]["8"].items() if isinstance(v_, (int, float))]
    mx = {m: {k_: med([X[s][m][k_] for s in X]) for k_ in keys} for m in ("8", "16")}
    v, k = verdict(per, f(mx))
    det = fmt(mx)
    if v != "**PASS**":
        det += " | per seed: " + "; ".join(f"s{s}: {fmt(X[s])}" for s in X)
    P(f"| {hid} | {text} | {pl} | {Pi} | {h} | {det} | {k}/5 | {v} |")
    H9[(hid, pl, Pi, h)] = v


for pl in ("uv", "steered"):
    for Pi in PIS:
        for h in WIDTHS:
            R = recs("t9", h)
            if not R:
                continue
            h9row("H9a", "odd vs LAW(C_area): cos ≥ 0.98, ratio ∈ [0.9, 1.1]", pl, Pi, h, R,
                  lambda x: x["8"]["cos_odd_lit"] >= 0.98 and inr(x["8"]["ratio_odd_lit"], 0.9, 1.1),
                  lambda x: f"cos {g(x['8']['cos_odd_lit'], 4)}, ratio {g(x['8']['ratio_odd_lit'], 4)}")
            h9row("H9a-supp", "combo: odd vs LAW((C₊−C₋)/2), same thresholds", pl, Pi, h, R,
                  lambda x: x["8"]["cos_odd_combo"] >= 0.98 and inr(x["8"]["ratio_odd_combo"], 0.9, 1.1),
                  lambda x: f"cos {g(x['8']['cos_odd_combo'], 5)}, ratio {g(x['8']['ratio_odd_combo'], 4)}")
            h9row("H9b", "even vs LAW(C_sym): cos ≥ 0.98, ratio ∈ [0.9, 1.1]", pl, Pi, h, R,
                  lambda x: x["8"]["cos_even_lit"] >= 0.98 and inr(x["8"]["ratio_even_lit"], 0.9, 1.1),
                  lambda x: f"cos {g(x['8']['cos_even_lit'], 4)}, ratio {g(x['8']['ratio_even_lit'], 4)}")
            h9row("H9b-supp", "combo: even vs LAW((C₊+C₋)/2), same thresholds", pl, Pi, h, R,
                  lambda x: x["8"]["cos_even_combo"] >= 0.98 and inr(x["8"]["ratio_even_combo"], 0.9, 1.1),
                  lambda x: f"cos {g(x['8']['cos_even_combo'], 5)}, ratio {g(x['8']['ratio_even_combo'], 4)}")
            h9row("H9c", "‖odd(16)‖/‖odd(8)‖ ∈ [1.6, 2.2]", pl, Pi, h, R,
                  lambda x: inr(x["16"]["odd_norm"] / x["8"]["odd_norm"], 1.6, 2.2),
                  lambda x: f"{g(x['16']['odd_norm'] / x['8']['odd_norm'], 4)}")
            h9row("H9d", "‖(V_LINE,ω − V_LINE,−ω)/2‖ < 0.02‖odd‖ (literal; identical drives)", pl, Pi, h, R,
                  lambda x: x["8"]["line_odd_lit_norm"] < 0.02 * x["8"]["odd_norm"],
                  lambda x: f"{g(x['8']['line_odd_lit_norm'], 3)} vs 0.02‖odd‖ = {g(0.02 * x['8']['odd_norm'], 3)}")
            h9row("H9d-supp", "‖LAW(C_area, LINE a-priori)‖ < 0.02‖odd‖", pl, Pi, h, R,
                  lambda x: x["8"]["line_law_area_apri_norm"] < 0.02 * x["8"]["odd_norm"],
                  lambda x: f"ratio {g(x['8']['line_law_area_apri_norm'] / x['8']['odd_norm'], 4)}")
            if pl == "steered":
                h9row("H9e", "f(x_1) odd change: sign as predicted, ratio to J_x1·LAW(C_area) ∈ [0.8, 1.25], ≥ 3× median \\|odd\\| of other 255 probes", pl, Pi, h, R,
                      lambda x: inr(x["8"]["f1_ratio_lit"], 0.8, 1.25) and x["8"]["f1_over_others_median"] >= 3,
                      lambda x: f"ratio {g(x['8']['f1_ratio_lit'], 4)}, ×others {g(x['8']['f1_over_others_median'], 3)}")
                h9row("H9e-supp", "same with combo prediction J_x1·LAW((C₊−C₋)/2)", pl, Pi, h, R,
                      lambda x: inr(x["8"]["f1_ratio_combo"], 0.8, 1.25) and x["8"]["f1_over_others_median"] >= 3,
                      lambda x: f"ratio {g(x['8']['f1_ratio_combo'], 4)}, ×others {g(x['8']['f1_over_others_median'], 3)}")
P()
P("### Test 9 report only — frequency response: ‖odd‖ at m = 8 vs Π, measured and predicted (median over seeds)")
P()
P("| plane | width | Π=16: meas / LAW(C_area) / combo pred | Π=64 | Π=256 |")
P("|---|---|---|---|---|")
for pl in ("uv", "steered"):
    for h in WIDTHS:
        R = recs("t9", h)
        if not R:
            continue
        cells = []
        for Pi in PIS:
            X = [R[s][pl][Pi]["8"] for s in R]
            cells.append(f"{g(med([x['odd_norm'] for x in X]), 3)} / {g(med([x['pred_odd_lit_norm'] for x in X]), 3)} / "
                         f"{g(med([x['odd_norm'] / x['ratio_odd_combo'] for x in X]), 3)}")
        P(f"| {pl} | {h} | " + " | ".join(cells) + " |")
P()
P("### Test 9 report only — STEERED plane: odd change of f(x_1) vs m (median over seeds; measured / J_x1·LAW(C_area) / ×median of others)")
P()
P("| width | Π | " + " | ".join(f"m={m}" for m in MS) + " |")
P("|---|---|" + "---|" * len(MS))
for h in WIDTHS:
    R = recs("t9", h)
    if not R:
        continue
    for Pi in PIS:
        cells = []
        for m in MS:
            X = [R[s]["steered"][Pi][m] for s in R]
            cells.append(f"{g(med([x['f1_odd'] for x in X]), 3)} / {g(med([x['f1_pred_lit'] for x in X]), 3)} / {g(med([x['f1_over_others_median'] for x in X]), 3)}")
        P(f"| {h} | {Pi} | " + " | ".join(cells) + " |")
P()
P("Steered-plane construction (per job): eigenvalue μ_max of Ā, cos(p, q):")
P()
for h in WIDTHS:
    R = recs("t9", h)
    if not R:
        continue
    P(f"- h={h}: " + "; ".join(f"s{s}: μ_max {g(R[s]['steer_info']['mu_max'], 4)}, top imag eigs {[round(x, 4) for x in R[s]['steer_info']['eig_imag_sorted']]}, cos(p,q) {g(R[s]['steer_info']['pq_cos'], 2)}" for s in R))
P()

# ===========================================================================
# automatic anomaly scan (measurements only)
for h in WIDTHS:
    for s, r in recs("t5", h).items():
        for e in EPS5:
            for c in C5:
                x = r["A"][str(e)]["rows"][str(c)]
                if c != 1.0 and np.sign(x["dlam1"]) != np.sign(1 - c):
                    anomalies.append(f"T5 h={h} s={s} ε={e} c={c}: sign(Δλ_1) ≠ sign(1−c) (Δλ_1 = {g(x['dlam1'], 3)}, pred1 = {g(x['pred1'], 3)})")
                elif abs(c - 1) >= 0.25 and not inr(x["dlam1"] / x["pred1"], 0.8, 1.25):
                    anomalies.append(f"T5 h={h} s={s} ε={e} c={c}: Δλ_1/pred1 = {g(x['dlam1'] / x['pred1'], 3)}")
        for c in ("2.05", "1.9"):
            b = r["B"][c]
            if b["stopped_at"] is not None:
                anomalies.append(f"T5-B h={h} s={s} c={c}: run stopped at step {b['stopped_at']} (‖r‖ > 1e6)")
    for s, r in recs("t6", h).items():
        st = r["states"]
        for m in (4, 8, 16):
            if st[f"u{m}"]["nu_u"] >= st["S0"]["nu_u"]:
                anomalies.append(f"T6 h={h} s={s}: ν_u(u{m}) ≥ ν_u(S0) (ratio {g(st[f'u{m}']['nu_u'] / st['S0']['nu_u'], 4)})")
        if st["GD16"]["nu_u"] > st["S0"]["nu_u"]:
            anomalies.append(f"T6 h={h} s={s}: GD practice raised ν_u (GD16/S0 = {g(st['GD16']['nu_u'] / st['S0']['nu_u'], 4)})")
    for s, r in recs("t7", h).items():
        for pn in ("ss'", "ww'", "sw"):
            x = r["part_a"][pn]["0.025"]
            if x["cos_O_b"] < 0.99:
                anomalies.append(f"T7 h={h} s={s} pair {pn} ε=0.025: cos(O,b) = {g(x['cos_O_b'], 4)}, ratio {g(x['ratio'], 4)}")
    for s, r in recs("t8", h).items():
        for gm in GAM:
            x = r["gammas"][gm]
            if x["blew_up"]:
                anomalies.append(f"T8 h={h} s={s} γ={gm}: {x['blew_up']} streams non-finite")
            if gm != "0.0" and not inr(x["kappa_star"], 1.9, 2.1):
                anomalies.append(f"T8 h={h} s={s} γ={gm}: κ* = {g(x['kappa_star'], 5)}")
    for s, r in recs("t9", h).items():
        for pl in ("uv", "steered"):
            for Pi in PIS:
                x = r[pl][Pi]["8"]
                if x["cos_odd_combo"] < 0.98:
                    anomalies.append(f"T9 h={h} s={s} {pl} Π={Pi}: cos(odd, combo pred) = {g(x['cos_odd_combo'], 4)}")
                if x["cos_even_lit"] < 0:
                    anomalies.append(f"T9 h={h} s={s} {pl} Π={Pi}: cos(even, LAW(C_sym of ROT+)) negative ({g(x['cos_even_lit'], 3)})")
P("## [derived] predictions: confirmed or contradicted")
P()
P("- H5c (pred2 = ε²(1−c)/(2(2−c))·‖P_v∇λ_1‖²/λ_1²): **confirmed** — Δλ_1/pred2 within [0.946, 1.033] at ε = 0.025 over every (width, seed, c), PASS at all widths; also the measured q_1q_1 entry of C_c − C_GN equals ε²(1−c)/(2−c) to < 0.3 %.")
P("- Protocol B slope −(c_eff²/4)‖P_v∇λ_1‖²/λ_1² (report only): at c = 2.05 the fitted slope / predicted slope (c_eff = c) is 0.913–1.045 in all 15 runs, but with low R² (0.018–0.903); at c = 1.9 the fitted slopes (−17.5 … +28.4) do **not** match the prediction (≈ −0.01 … −0.03); there the q_1 residual decays within a few steps and the regression over all 3000 steps is not dominated by the zigzag. Stated as measured.")
P("- H8a (d_law = d_dir): **confirmed**, relative difference 9.3e-9 – 5.4e-8. (Algebraically P_v[Dξ_{e_i}·ξ_{e_j}] = P_v Σ_a (K⁻¹)_{ai} H_a ξ_{e_j}, so LAW(−AΣ_y) = −η′P_v Σ_a H_a Ξ Σ_y Mᵀe_a = d_dir exactly; the numerical check confirms the identity and the finite-difference code.)")
P("- H8c (κ* = 2): **confirmed**, κ* = 2.000000 for every width, seed and γ > 0 (d_pot(κ) = −η′P_v t_1 + (κγη′/2)P_v t_2 and d_dir = −η′P_v(t_1 − γ t_2), so κ = 2 is exact; d_pot(1) differs from d_law by 30–34 % (γ = 0.5), 252–295 % (γ = 0.9), 2000–2930 % (γ = 0.99), relative).")
P()
P("## Spec items that fail for an identifiable algebraic reason (not marked [derived] in SPEC6)")
P()
P("- Unit test 5b (LINE has zero a-priori area matrix): fails, ‖C_area(LINE)‖/‖C_area(ROT+)‖ = 0.446 (h=32, seed 0). Debug check (scratchpad dbg_line_area.py): with K_A replaced by λ_1·I, or by a kernel whose eigenvectors include (p+q)/‖p+q‖, the LINE area is 1.4e-16 / 1.1e-16; the target path itself has area 5.7e-17. With the real anisotropic K_A the linearised outputs respond to a line drive with different phase lags in different kernel modes, so the output path encloses area. Not treated as a code bug; the grid was run anyway.")
P("- H8b: the discrete Lyapunov equation gives Sym(AΣ_y) = ½(BBᵀ + AΣ_yAᵀ) exactly (verified to ≤ 2e-14); ½BBᵀ alone is the continuous-time identity. The AΣAᵀ term is 11 % of ½BBᵀ at γ = 0, and LAW(E_sym) changes with γ by 3–7 %.")
P("- H9a/H9b (literal reading: odd/even part vs LAW of the antisymmetric/symmetric part of the ROT+ a-priori C): fail on the (u,v) plane and H9b fails everywhere, while LAW((C₊ ∓ C₋)/2) matches the odd/even parts with cos ≥ 0.982 and ratio 0.979–1.044 over every (plane, Π, width, seed) at m = 8 (medians cos ≥ 0.995). The mirror map ROT+ → ROT− is not a symmetry of the label dynamics (K_A is not invariant under q → −q), so C₋ ≠ mirror(C₊) and C_area(ROT+) ≠ (C₊ − C₋)/2.")
P("- H9d (literal): the LINE drive cos(ωk) is even in ω, so V_LINE,ω = V_LINE,−ω exactly and the rule holds trivially (value 0). Supplementary: the LINE drive's own a-priori area term LAW(C_area,LINE) is 3.4 %–128 % of ‖odd(ROT)‖ (never < 2 %), and ‖V_LINE‖ is 4.4 %–120 % of ‖odd(ROT)‖.")
P("- H6d (literal): ν_u^lin = ‖(I−A)û‖²/16 is the expectation over uniformly random unit b, while ν_u is a mean over 16 specific b_j; per (width, seed) the ratio stays in a narrow band across all 10 states (bands 0.30–0.47, 0.95–1.02 or 1.14–1.61 depending on the seed — a property of that seed's 16 draws). Evaluated on the same 16 b_j the ratio is 0.997–1.003 (supplementary row, PASS).")
P()
P("## Decisions and readings")
P()
for d in [
    "Threads: tcr4 is imported first in every process (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=2, torch.set_num_threads(2)); up to 4 processes ran concurrently on an 8-core machine.",
    "Reuse: tcr4.Law (T tensor; LAW(C) = Σ_ij C_ij T_ij with T_ij = P_v[Dξ_{e_i}·ξ_{e_j}], unit-tested), tcr4.fo_learn_C (GD(c) with SPEC4 stop rule 1e-8, cap 4e5, DIVERGED rule) and gn_learn_C (C bookkeeping), tcr3.bracket / kernel_stats / out_and_back, tcr.gd_learn (= GD(1)) for the Test-6 GD practice, tcr.gn_learn.",
    "Test 0(ii): random symmetric C = (R+Rᵀ)/2, R ~ N(0,1) from Generator(60); 5 directions w = P_v(N(0,I)) from the same generator. Unit test 1: (i,j) from Generator(61). Unit test 3: 'converges in one step in the q_1 direction' judged by |q_1·r_1|/|q_1·r_0| < 1e-2 (threshold chosen; measured 6.9e-6).",
    "Test 5: η = c·n/λ_max(K(θ_A)) fixed for the learn (θ_in = θ_A); q_1 = top eigenvector of K(θ_A) with torch.linalg.eigh's sign. Protocol B: OLS with intercept; predicted slope uses c_eff = c (its value at θ_A) and, separately, the median c_eff over the regression steps; the column 'frac Δλ_1<0 before it' is over all steps when c_eff < 2 from step 0 (all c = 1.9 runs).",
    "Test 6: S^GD learns by tcr.gd_learn (η = n/λ_max(K(θ_in)) recomputed per learn, tol 1e-8); the 1e-12 end-of-sequence assert is applied to GN states only. 'Decreasing in m' read as 1 > r_4 > r_8 > r_16. H6d: per width, median over seeds per state must lie in range for every state; per seed: every state in range. H6c: verdict on the single pooled Spearman ρ; the §0 per-seed rule applied to per-seed pooled ρ is shown alongside. π_v and generic plasticity computed from the same probe runs.",
    "Test 7: coefficient draws from Generator(7000+seed) in the order s, w, s′, w′ (torch.randn(4) each). Part (b): η of GD(1) recomputed at the start of each phase (each phase is a 'learn'); JOINT is one learn of T_tot steps; frozen-kernel prediction uses η′ = 1/λ_max(K_A) for all phases. H7d primary reading applies to every pair of curricula (SPEC6: 'for each pair of curricula let d = …'); the EASY-vs-HARD-only reading is shown as an alternative. loss_s = (ŝ·r)²/(2n), loss_w likewise.",
    "Test 8: one P per seed, drawn sequentially from Generator(8000+seed) until the eigenvalue condition holds for all four γ at once (all seeds accepted the first draw), so the same P serves every γ. Noise: Generator(8100 + seed + stream) (literal reading of 'Generator(8100+seed)+stream'; streams of seed s+1 coincide with streams of seed s shifted by one), drawn as one (4000, 16) tensor. Σ_θ = ΞΣ_yΞᵀ is never formed (Ξ = JᵀK⁻¹); H_a w by central FD of row a of J along w (step 1e-4/‖w‖); d_pot uses ∇[g_aᵀWg_a] = 2H_aWg_a and ∇[g_aᵀ W sg(ḡ_a)] = H_aWḡ_a; κ* by least squares over all P components. V_γ = mean of θ_end − θ_A after the GN return (not projected).",
    "Test 9: step k = 0 … mΠ−1 uses target y_k; after m full periods a GN return to y_A is run from a copy (m ∈ {1,2,4,8,16} from one continuous run). The a-priori C includes the a-priori GN return step. Primary H9a/H9b use LAW of the antisymmetric / symmetric part of the ROT+ a-priori C; the combo predictions are supplementary. H9a–H9d evaluated for both planes, H9e for the STEERED plane; one verdict per (plane, Π, width). Steered plane: eigenvector of Ā for the eigenvalue with the largest imaginary part (numpy.linalg.eig), p = √2 Re z, q = √2 Im z, each normalised to ‖y_A‖ (|cos(p,q)| ≤ 2e-15).",
]:
    P(f"- {d}")
P()
P("## Anomalies (automatic scan; measurements only)")
P()
P("\n".join(f"- {a}" for a in anomalies) if anomalies else "None flagged by the scan.")
P()
P("## Summary of verdicts")
P()
for name, D in (("Test 5", H5), ("Test 6", H6), ("Test 7", H7), ("Test 8", H8), ("Test 9", H9)):
    P(f"- {name}: " + "; ".join(f"{'/'.join(str(x) for x in k)} {v.strip('*')}" for k, v in D.items()))
P()
P("## Conditions not run / not OK")
P()
P("\n".join(f"- {x}" for x in sorted(set(notrun))) if notrun else "None: every (test, width, seed) job finished with status OK.")
P()
P("## Wall time")
P()
for t in ("t5", "t6", "t7", "t8", "t9"):
    ws = [r["wall"] for r in db["records"] if r["test"] == t and "wall" in r]
    if ws:
        P(f"- {t}: {len(ws)} jobs, sum of job wall {sum(ws):.0f} s, max job {max(ws):.0f} s")
for m in db["meta"]["runs"]:
    P(f"- run: test {m['test']} pid {m['pid']} jobs {m['n_run']} wall {m['wall']:.0f}s stopped={m['stopped']} threads={m['threads']} finished {m['finished']}")
P()
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", OUT, len(lines), "lines")
