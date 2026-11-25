"""Reads results.json, prints H1-H7 tables + G4 comparison, writes REPORT.md.

PASS / FAIL / INCONCLUSIVE rule (decision, see REPORT.md "Decisions"):
  each prediction is a predicate on a per-seed quantity.
  PASS          predicate holds for the median over seeds AND for every seed
  INCONCLUSIVE  holds for the median but not for every seed, or < 3 valid seeds
  FAIL          does not hold for the median
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTHS = [32, 128, 512]
SEEDS = [0, 1, 2, 3, 4]
EPS = [0.2, 0.1, 0.05, 0.025, 0.0125]
GAMMAS = {"pi/6": math.pi / 6, "pi/4": math.pi / 4, "pi/3": math.pi / 3,
          "pi/2": math.pi / 2, "2pi/3": 2 * math.pi / 3}
W_H1 = (1.6, 2.2)      # same windows as SPEC section 7 test 5
W_H2 = (3.4, 4.3)
W_H3 = (2.5, 3.3)
W_2 = (1.5, 2.5)       # "slope ~ 2" (H5, H7): 2 +- 0.5
H4_TOL = 0.02
H6_TOL = 0.1

db = json.load(open(os.path.join(HERE, "results.json")))
R = db["records"]
L = []


def P(s=""):
    L.append(s)
    print(s)


def ok(r):
    return r.get("status") == "OK"


def get(grid, h, seed, eps=None, gamma=None, alpha=None):
    for r in R:
        if (r["grid"] == grid and r["h"] == h and r["seed"] == seed
                and (eps is None or r["eps"] == eps) and (gamma is None or r["gamma"] == gamma)
                and (alpha is None or r["alpha"] == alpha)):
            return r
    return None


def fit_slope(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y) and y > 0]
    if len(pts) < 3:
        return None
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def series(grid, h, seed, qty, alpha=None, gamma="pi/2"):
    ys = []
    for e in EPS:
        r = get(grid, h, seed, e, gamma=gamma if alpha is None else None, alpha=alpha)
        ys.append(r["scalars"].get(qty) if r is not None and ok(r) else None)
    return ys


def ratio_series(a, b):
    return [x / y if (x is not None and y is not None and y > 0) else None for x, y in zip(a, b)]


def med_rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "n/a", None
    return f"{np.median(v):.4g} [{min(v):.4g}, {max(v):.4g}] (n={len(v)})", float(np.median(v))


def verdict(per_seed_ok, median_ok, n_valid):
    if n_valid < 3:
        return "INCONCLUSIVE"
    if not median_ok:
        return "FAIL"
    return "PASS" if all(per_seed_ok) else "INCONCLUSIVE"


def in_w(x, w):
    return x is not None and w[0] <= x <= w[1]


def per_eps_table(title, grid, qtys, alpha=None, gamma="pi/2", ratio_of=None):
    P(f"\n{title}: median [min, max] over seeds, per eps")
    P("| h | quantity | " + " | ".join(f"eps={e}" for e in EPS) + " |")
    P("|---|---|" + "---|" * len(EPS))
    for h in WIDTHS:
        for q in qtys:
            cells = []
            for i, e in enumerate(EPS):
                vals = []
                for s in SEEDS:
                    if isinstance(q, tuple):
                        a = series(grid, h, s, q[0], alpha, gamma)[i]
                        b = series(*q[1](h, s))[i]
                        vals.append(a / b if a is not None and b else None)
                    else:
                        vals.append(series(grid, h, s, q, alpha, gamma)[i])
                v = [x for x in vals if x is not None and np.isfinite(x)]
                cells.append(f"{np.median(v):.3g} [{min(v):.3g}, {max(v):.3g}]" if v else "n/a")
            name = q if not isinstance(q, tuple) else q[2]
            P(f"| {h} | {name} | " + " | ".join(cells) + " |")


verdicts = {}

# ------------------------------- H1 ---------------------------------------
P("# Task-cycle rotation: results\n")
P("## H1: N_a ~ eps^p, p ~ 2 (generic, gamma=pi/2; G1). PASS window p in [1.6, 2.2]")
P("| h | slope N_a (weight) | slope N_a (function) | verdict (weight) |")
P("|---|---|---|---|")
for h in WIDTHS:
    sw = [fit_slope(EPS, series("G1", h, s, "N_a")) for s in SEEDS]
    sf = [fit_slope(EPS, series("G1", h, s, "N_a_fn")) for s in SEEDS]
    tw, mw = med_rng(sw)
    tf, _ = med_rng(sf)
    v = [x for x in sw if x is not None]
    vd = verdict([in_w(x, W_H1) for x in v], in_w(mw, W_H1), len(v))
    verdicts[("H1", h)] = vd
    P(f"| {h} | {tw} | {tf} | {vd} |")
per_eps_table("N_a", "G1", ["N_a", "N_a_fn"])

# ------------------------------- H2 ---------------------------------------
P("\n## H2: N_star ~ eps^q, q ~ 4, and N_star/N_a -> 0 (G1). PASS: q in [3.4, 4.3] and slope of log(N_star/N_a) > 0")
P("| h | slope N_star (weight) | slope N_star (function) | slope N_star/N_a (weight) | verdict |")
P("|---|---|---|---|---|")
for h in WIDTHS:
    q = [fit_slope(EPS, series("G1", h, s, "N_star")) for s in SEEDS]
    qf = [fit_slope(EPS, series("G1", h, s, "N_star_fn")) for s in SEEDS]
    rs = [fit_slope(EPS, ratio_series(series("G1", h, s, "N_star"), series("G1", h, s, "N_a")))
          for s in SEEDS]
    tq, mq = med_rng(q)
    tqf, _ = med_rng(qf)
    tr, mr = med_rng(rs)
    pairs = [(a, b) for a, b in zip(q, rs) if a is not None and b is not None]
    vd = verdict([in_w(a, W_H2) and b > 0 for a, b in pairs],
                 in_w(mq, W_H2) and mr is not None and mr > 0, len(pairs))
    verdicts[("H2", h)] = vd
    P(f"| {h} | {tq} | {tqf} | {tr} | {vd} |")
per_eps_table("N_star and N_star/N_a", "G1",
              ["N_star", "N_star_fn",
               ("N_star", lambda h, s: ("G1", h, s, "N_a"), "N_star/N_a")])

# ------------------------------- H3 ---------------------------------------
P("\n## H3: collinear loops N_a^col ~ eps^p, p ~ 3, and N_a^col/N_a^generic -> 0 (G2 vs G1). "
  "PASS: p in [2.5, 3.3] and slope of log(ratio) > 0")
P("| h | alpha | slope N_a^col (weight) | slope N_a^col (function) | slope N_a^col/N_a^gen | verdict |")
P("|---|---|---|---|---|---|")
for h in WIDTHS:
    for al in (2.0, -1.0):
        p = [fit_slope(EPS, series("G2", h, s, "N_a", alpha=al)) for s in SEEDS]
        pf = [fit_slope(EPS, series("G2", h, s, "N_a_fn", alpha=al)) for s in SEEDS]
        rs = [fit_slope(EPS, ratio_series(series("G2", h, s, "N_a", alpha=al),
                                          series("G1", h, s, "N_a"))) for s in SEEDS]
        tp, mp = med_rng(p)
        tpf, _ = med_rng(pf)
        tr, mr = med_rng(rs)
        pairs = [(a, b) for a, b in zip(p, rs) if a is not None and b is not None]
        vd = verdict([in_w(a, W_H3) and b > 0 for a, b in pairs],
                     in_w(mp, W_H3) and mr is not None and mr > 0, len(pairs))
        verdicts[(f"H3 alpha={al:g}", h)] = vd
        P(f"| {h} | {al:g} | {tp} | {tpf} | {tr} | {vd} |")
for al in (2.0, -1.0):
    per_eps_table(f"collinear alpha={al:g}", "G2",
                  ["N_a", ("N_a", lambda h, s: ("G1", h, s, "N_a"), "N_a^col/N_a^gen")],
                  alpha=al, gamma=None)

# ------------------------------- H4 ---------------------------------------
P(f"\n## H4: N_a(gamma)/N_a(pi/2) ~ sin(gamma) at eps=0.05 (G3). PASS: |ratio - sin gamma| <= {H4_TOL} for every gamma")
P("| h | gamma | sin gamma | ratio (weight) | ratio (function) |")
P("|---|---|---|---|---|")
for h in WIDTHS:
    per_seed_dev = {s: [] for s in SEEDS}
    med_dev = []
    for g in ("pi/6", "pi/4", "pi/3", "2pi/3"):
        rw, rf = [], []
        for s in SEEDS:
            r1, r0 = get("G3", h, s, 0.05, gamma=g), get("G3", h, s, 0.05, gamma="pi/2")
            if r1 and r0 and ok(r1) and ok(r0):
                x = r1["scalars"]["N_a"] / r0["scalars"]["N_a"]
                rw.append(x)
                rf.append(r1["scalars"]["N_a_fn"] / r0["scalars"]["N_a_fn"])
                per_seed_dev[s].append(abs(x - math.sin(GAMMAS[g])))
            else:
                rw.append(None)
                rf.append(None)
                per_seed_dev[s].append(None)
        tw, mw = med_rng(rw)
        tf, _ = med_rng(rf)
        med_dev.append(abs(mw - math.sin(GAMMAS[g])) if mw is not None else None)
        P(f"| {h} | {g} | {math.sin(GAMMAS[g]):.4f} | {tw} | {tf} |")
    seed_ok = [all(d is not None and d <= H4_TOL for d in v) for v in per_seed_dev.values()
               if all(d is not None for d in v)]
    med_ok = all(d is not None and d <= H4_TOL for d in med_dev)
    vd = verdict(seed_ok, med_ok, len(seed_ok))
    verdicts[("H4", h)] = vd
    md = max(d for d in med_dev if d is not None) if any(d is not None for d in med_dev) else None
    sd = [max(v) for v in per_seed_dev.values() if all(d is not None for d in v)]
    P(f"| {h} | max dev | | median-ratio max |dev|={md:.4f}; per-seed max |dev| = "
      + ", ".join(f"{x:.4f}" for x in sd) + f" | verdict: {vd} |")

# ------------------------------- H5 ---------------------------------------
P("\n## H5: E_add -> 0 and E_shrink -> 0 as eps -> 0, slope ~ 2 (G1). PASS: both slopes in [1.5, 2.5]")
P("| h | slope E_add (w) | slope E_shrink (w) | slope E_add (fn) | slope E_shrink (fn) | verdict (weight) |")
P("|---|---|---|---|---|---|")
for h in WIDTHS:
    ea = [fit_slope(EPS, series("G1", h, s, "E_add")) for s in SEEDS]
    es = [fit_slope(EPS, series("G1", h, s, "E_shrink")) for s in SEEDS]
    eaf = [fit_slope(EPS, series("G1", h, s, "E_add_fn")) for s in SEEDS]
    esf = [fit_slope(EPS, series("G1", h, s, "E_shrink_fn")) for s in SEEDS]
    t1, m1 = med_rng(ea)
    t2, m2 = med_rng(es)
    t3, _ = med_rng(eaf)
    t4, _ = med_rng(esf)
    pairs = [(a, b) for a, b in zip(ea, es) if a is not None and b is not None]
    vd = verdict([in_w(a, W_2) and in_w(b, W_2) for a, b in pairs],
                 in_w(m1, W_2) and in_w(m2, W_2), len(pairs))
    verdicts[("H5", h)] = vd
    P(f"| {h} | {t1} | {t2} | {t3} | {t4} | {vd} |")
per_eps_table("E_add / E_shrink", "G1", ["E_add", "E_shrink", "E_add_fn", "E_shrink_fn"])

# ------------------------------- H6 ---------------------------------------
P(f"\n## H6: tang << 1 (G1). PASS: per-seed max over eps of tang <= {H6_TOL}")
P("| h | max_eps tang per seed: median [min, max] | verdict |")
P("|---|---|---|")
for h in WIDTHS:
    mx = []
    for s in SEEDS:
        v = [x for x in series("G1", h, s, "tang") if x is not None]
        mx.append(max(v) if v else None)
    t, m = med_rng(mx)
    v = [x for x in mx if x is not None]
    vd = verdict([x <= H6_TOL for x in v], m is not None and m <= H6_TOL, len(v))
    verdicts[("H6", h)] = vd
    P(f"| {h} | {t} | {vd} |")
per_eps_table("tang", "G1", ["tang"])

# ------------------------------- H7 ---------------------------------------
P("\n## H7 (reference only): N_ret ~ eps^2 and N_s ~ eps^2 (G1). PASS: both slopes in [1.5, 2.5]")
P("| h | slope N_ret (w) | slope N_s (w) | slope N_ret (fn) | slope N_s (fn) | verdict (weight) |")
P("|---|---|---|---|---|---|")
for h in WIDTHS:
    a = [fit_slope(EPS, series("G1", h, s, "N_ret")) for s in SEEDS]
    b = [fit_slope(EPS, series("G1", h, s, "N_s")) for s in SEEDS]
    af = [fit_slope(EPS, series("G1", h, s, "N_ret_fn")) for s in SEEDS]
    bf = [fit_slope(EPS, series("G1", h, s, "N_s_fn")) for s in SEEDS]
    t1, m1 = med_rng(a)
    t2, m2 = med_rng(b)
    t3, _ = med_rng(af)
    t4, _ = med_rng(bf)
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    vd = verdict([in_w(x, W_2) and in_w(y, W_2) for x, y in pairs],
                 in_w(m1, W_2) and in_w(m2, W_2), len(pairs))
    verdicts[("H7", h)] = vd
    P(f"| {h} | {t1} | {t2} | {t3} | {t4} | {vd} |")
per_eps_table("N_ret / N_s", "G1", ["N_ret", "N_s"])

# --------------------------- section 4.1 + lin -----------------------------
P("\n## Linearised comparator (section 4.1) and distance from linear regime (G1)")
per_eps_table("cos_lin / ratio_lin / lin", "G1", ["cos_lin", "ratio_lin", "lin"])

# ------------------------------- G4 ---------------------------------------
P("\n## G4: GD vs GN (h=128, gamma=pi/2)")
P("| seed | eps | status | N_a GD | N_a GN | N_star GD | N_star GN | N_ret GD | N_ret GN | "
  "cos(a_GD,a_GN) | ||thA_GD-thA_GN|| | max ||r||_inf GD | GD steps | wall s |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted([r for r in R if r["grid"] == "G4"], key=lambda r: (r["seed"], -r["eps"])):
    sc, sg = r.get("scalars") or {}, r.get("scalars_gn") or {}
    f = lambda d, k: f"{d[k]:.4g}" if k in d else "n/a"
    P(f"| {r['seed']} | {r['eps']} | {r['status']} | {f(sc,'N_a')} | {f(sg,'N_a')} | "
      f"{f(sc,'N_star')} | {f(sg,'N_star')} | {f(sc,'N_ret')} | {f(sg,'N_ret')} | "
      f"{r.get('cos_gd_gn', float('nan')):.4f} | {r.get('base_dist_gd_gn', float('nan')):.3g} | "
      f"{r.get('max_res_inf', float('nan')):.2e} | {r.get('total_gd_steps', 'n/a')} | {r.get('wall', 0):.0f} |")
runs_g4 = [m for m in db["meta"]["runs"] if "g4_wall" in m]
for m in runs_g4:
    P(f"\nG4 wall time {m['g4_wall']:.0f} s; stopped by 40-min cap: {m['g4_stopped_by_cap']}")

# ------------------------- invalid / not converged -------------------------
P("\n## INVALID / NOT CONVERGED / TIMEOUT conditions")
bad = [r for r in R if r.get("status") != "OK"]
if not bad:
    P("none")
for r in bad:
    P(f"- {r['grid']} h={r['h']} seed={r['seed']} eps={r['eps']} gamma={r['gamma']} "
      f"alpha={r['alpha']}: {r['status']} {r.get('error', '')} "
      f"{r.get('nonconverged_learns', '')}")

# ------------------------------- summary -----------------------------------
okr = [r for r in R if ok(r) and r["learner"] == "gn"]
P("\n## Convergence summary (GN)")
P(f"valid GN conditions: {len(okr)}; max GN iterations in any learn: "
  f"{max(r['max_gn_iters'] for r in okr) if okr else 'n/a'}; "
  f"line-search exhaustions (30 halvings): {sum(r['ls_fail_total'] for r in okr)}")
base_iters = sorted({(r['h'], r['seed'], r['base_info']['iters']) for r in okr if r.get('base_info')})
P("base-point GN iterations (h, seed, iters): " + ", ".join(str(x) for x in base_iters))
tot = sum(m.get("end", m["start"]) - m["start"] for m in db["meta"]["runs"])
P(f"total wall time of run.py invocations: {tot:.0f} s ({tot/60:.1f} min)")

P("\n## Verdicts")
P("| prediction | " + " | ".join(f"h={h}" for h in WIDTHS) + " |")
P("|---|---|---|---|")
for hid in ["H1", "H2", "H3 alpha=2", "H3 alpha=-1", "H4", "H5", "H6", "H7"]:
    P(f"| {hid} | " + " | ".join(verdicts.get((hid, h), "n/a") for h in WIDTHS) + " |")

DEC = """
## Decisions / rule for verdicts
- Per-seed OLS slope of log(quantity) on log(eps) over valid points (>= 3 points needed).
- PASS = predicate true for the median over seeds and for every seed; INCONCLUSIVE = true for
  the median but not every seed (or < 3 valid seeds); FAIL = false for the median.
- Windows: H1 [1.6,2.2], H2 [3.4,4.3], H3 [2.5,3.3] (the SPEC section 7 test-5 windows);
  H5, H7 slope in [1.5,2.5]; "-> 0" = positive slope of log(ratio) vs log(eps);
  H4 |ratio - sin gamma| <= 0.02 for all four gammas; H6 tang <= 0.1 at every eps.
- Weight space is primary for verdicts; function-space numbers are shown alongside.
"""
P(DEC)
try:
    import matplotlib  # noqa: F401
    P("(matplotlib installed but no plots implemented)")
except ImportError:
    P("(matplotlib not installed: no plots)")

with open(os.path.join(HERE, "REPORT.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
