"""Writes REPORT7.md from results7.json (+ results7_mats/*.npz for Part C c_H samples).

Usage: python report7.py [--tests FILE_WITH_test_part7_OUTPUT]
All rules evaluated exactly as pre-registered in SPEC7.md (literal), plus clearly labelled
SUPPLEMENTARY versions where the spec's reference point is wrong (see section 5 of the report).
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results7.json")
MATDIR = os.path.join(HERE, "results7_mats")
OUT = os.path.join(HERE, "REPORT7.md")
N = 16
M_SL = 64
CS = [0.25, 1.0, 1.5, 1.9]
SIZES = [0.025, 0.05, 0.1]
WIDTHS = [32, 64, 128]
KAPPAS = [0.5, 1.0, 2.0, 4.0]
RHOS = [0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, "final"]
EIG = [1, 2, 4, 7, 10, 13, 16]


def phi(c):
    return (1.0 - c) / (2.0 - c)


def f(x, p=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not math.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.{p}e}"
    return f"{x:.{p + 1}g}"


def frac(mask):
    mask = list(mask)
    return (sum(bool(m) for m in mask) / len(mask)) if mask else float("nan")


def med(x):
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def q(x, p):
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.percentile(x, p)) if x else float("nan")


def verdict(ok):
    return "**PASS**" if ok else "**FAIL**"


class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        self.lines.append(s)

    def table(self, head, rows):
        self("| " + " | ".join(head) + " |")
        self("|" + "---|" * len(head))
        for r in rows:
            self("| " + " | ".join((f(c) if not isinstance(c, str) else c).replace("|", "¦") for c in r) + " |")
        self()


def load():
    with open(RES) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def a_tables(rec):
    """Returns per-record derived structure."""
    lamA = rec["kA"]["lam1"]
    trA = rec["kA"]["trK"]
    lam = rec["lamA"]
    R = {}
    for r in rec["runs"]:
        R[(r["dir"], r["s"], r["sign"], r["learner"])] = r
    return lamA, trA, lam, R


def analyse_A(db, part="A"):
    out = {}
    fails = []
    for h in WIDTHS:
        recs = [r for r in db["records"] if r["part"] == part and r["h"] == h]
        if not recs:
            continue
        W = {"seeds": [], "rows": []}
        for rec in sorted(recs, key=lambda r: r["seed"]):
            if rec["status"] != "OK":
                fails.append(f"Part {part} h={h} seed={rec['seed']}: {rec['status']} {rec.get('error', '')}")
                continue
            seed = rec["seed"]
            W["seeds"].append(seed)
            lamA, trA, lam, R = a_tables(rec)
            for r in rec["runs"]:
                if r["status"] not in ("CONVERGED", "CONVERGED-CONT"):
                    fails.append(f"Part {part} h={h} seed={seed} dir={r['dir']} s={r['s']} sign={r['sign']} "
                                 f"learner={r['learner']}: {r['status']} (steps {r.get('steps')}, res {f(r.get('res_inf'))})")
            for dname, di in rec["dirs"].items():
                G = di["G"]
                i = di["eig_index"]
                for s in SIZES:
                    keyp = (dname, s, 1, "NP")
                    if keyp not in R:
                        continue
                    row = {"h": h, "seed": seed, "dir": dname, "i": i, "s": s, "G": G,
                           "lam_i": di["lam_i"], "Dlam_xi": di["Dlam_xi"], "Dtr_xi": di["Dtr_xi"]}
                    ok_np = all(R[(dname, s, sg, "NP")]["status"] in ("CONVERGED", "CONVERGED-CONT") for sg in (1, -1))
                    row["np_ok"] = ok_np
                    for L in ["NP", "GN", "SL"] + [f"GD{c}" for c in CS]:
                        rp, rm = R[(dname, s, 1, L)], R[(dname, s, -1, L)]
                        if "lam1" not in rp or "lam1" not in rm:
                            row[L] = None
                            continue
                        d = {"Dp": rp["lam1"] - lamA, "Dm": rm["lam1"] - lamA,
                             "Tp": rp["trK"] - trA, "Tm": rm["trK"] - trA,
                             "pl_p": rp.get("pred_lit"), "pl_m": rm.get("pred_lit"),
                             "ps_p": rp.get("pred_supp"), "ps_m": rm.get("pred_supp"),
                             "status": (rp["status"], rm["status"])}
                        for sg_, rr_ in (("p", rp), ("m", rm)):
                            dv = rr_.get("dist", rr_.get("np_dist"))
                            if dv is None:
                                dv = ADIST.get((h, seed, dname, s, 1 if sg_ == "p" else -1, L))
                            d["dist_" + sg_] = dv
                        d["E"] = 0.5 * (d["Dp"] + d["Dm"])
                        d["O"] = 0.5 * (d["Dp"] - d["Dm"])
                        d["Etr"] = 0.5 * (d["Tp"] + d["Tm"])
                        row[L] = d
                    if ok_np and row["NP"] is not None:
                        npd = row["NP"]
                        for L in ["GN", "SL"] + [f"GD{c}" for c in CS]:
                            d = row[L]
                            if d is None:
                                continue
                            d["Pp"] = d["Dp"] - npd["Dp"]
                            d["Pm"] = d["Dm"] - npd["Dm"]
                            d["EP"] = 0.5 * (d["Pp"] + d["Pm"])
                            d["EPtr"] = d["Etr"] - npd["Etr"]
                    # closed forms
                    cf = {}
                    for c in CS:
                        if i is not None:
                            ci = c * lam[i - 1] / lam[0]
                            cf[f"GD{c}"] = (s * s * phi(ci) * G, s * s * (phi(ci) - 1.0) * G, phi(ci))
                    cf["SL"] = (s * s * 0.5 * (1 - 1 / M_SL) * G, -s * s * 0.5 * (1 + 1 / M_SL) * G, None)
                    row["cf"] = cf
                    W["rows"].append(row)
        out[h] = W
    return out, fails


def rules_A(A, o, sup=False):
    """H10a-e for each width; returns list of H-table rows."""
    rows = []
    Ls = ["SL"] + [f"GD{c}" for c in CS]
    pk = ("ps_p", "ps_m") if sup else ("pl_p", "pl_m")
    tagS = " SUPP" if sup else ""
    for h, W in A.items():
        R = [r for r in W["rows"] if r["np_ok"]]
        nexcl_np = len([r for r in W["rows"] if not r["np_ok"]])
        # H10a
        items = []
        for r in R:
            if r["s"] != 0.025:
                continue
            for L in Ls:
                d = r[L]
                if d is None or "Pp" not in d:
                    continue
                items.append((d["Pp"], d[pk[0]]))
                items.append((d["Pm"], d[pk[1]]))
        mp = med([abs(p) for _, p in items])
        kept = [(m, p) for m, p in items if abs(p) >= 1e-3 * mp]
        ratios = [m / p for m, p in kept]
        fr = frac([0.8 <= x <= 1.25 for x in ratios])
        rows.append((f"H10a{tagS}", h, f"in [0.8,1.25]: {fr:.3f} of {len(kept)} (excluded |pred| small: {len(items) - len(kept)}; "
                     f"NP-failed cells excl.: {nexcl_np}); median ratio {f(med(ratios))}; "
                     f"frac ratio<0 {frac([x < 0 for x in ratios]):.3f}", fr >= 0.9))
        # H10b
        it_gd, it_sl = [], []
        for r in R:
            if r["s"] != 0.025:
                continue
            for c in CS:
                L = f"GD{c}"
                if r["i"] is None or r[L] is None or "EP" not in r[L]:
                    continue
                lit, supv, ph = r["cf"][L]
                if abs(ph if not sup else ph - 1.0) < 0.05:
                    continue
                it_gd.append(r[L]["EP"] / (supv if sup else lit))
            if r["SL"] is not None and "EP" in r["SL"]:
                lit, supv, _ = r["cf"]["SL"]
                it_sl.append(r["SL"]["EP"] / (supv if sup else lit))
        f1 = frac([0.67 <= x <= 1.5 for x in it_gd])
        f2 = frac([0.8 <= x <= 1.25 for x in it_sl])
        rows.append((f"H10b{tagS}", h, f"GD: {f1:.3f} of {len(it_gd)} in [0.67,1.5] (median {f(med(it_gd))}); "
                     f"SL: {f2:.3f} of {len(it_sl)} in [0.8,1.25] (median {f(med(it_sl))})", f1 >= 0.9 and f2 >= 0.9))
        if sup:
            continue
        # H10c
        Gs = {(r["seed"], r["dir"]): r["G"] for r in W["rows"]}
        fG = frac([g > 0 for g in Gs.values()])
        parts = [f"G_e>0: {fG:.3f} of {len(Gs)}"]
        ok = fG >= 0.9
        for s in SIZES:
            sl = [r["SL"]["EP"] > 0 for r in R if r["s"] == s and r["SL"] and "EP" in r["SL"]]
            gd = [r["GD0.25"]["EP"] > 0 for r in R if r["s"] == s and r["GD0.25"] and "EP" in r["GD0.25"]]
            parts.append(f"s={s}: EP_SL>0 {frac(sl):.3f}, EP_GD.25>0 {frac(gd):.3f}")
            ok &= frac(sl) >= 0.9 and frac(gd) >= 0.9
        rows.append(("H10c", h, "; ".join(parts), ok))
        # H10d
        slopes = []
        for seed in W["seeds"]:
            pts = sorted({(r["i"], r["lam_i"], r["G"]) for r in W["rows"] if r["seed"] == seed and r["i"] is not None})
            if any(g <= 0 for _, _, g in pts):
                slopes.append(float("nan"))
            else:
                slopes.append(float(np.polyfit(np.log([p[1] for p in pts]), np.log([p[2] for p in pts]), 1)[0]))
        n_ok = sum(1 for x in slopes if math.isfinite(x) and x <= -1)
        rows.append(("H10d", h, "slopes per seed: " + ", ".join(f(x) for x in slopes) + f"; seeds with slope <= -1: {n_ok}/{len(slopes)}",
                     n_ok >= 4))
        # H10e
        rat = []
        for r in R:
            if r["i"] in (7, 10, 13, 16) and r["GD1.9"] and r["GD0.25"] and "EP" in r["GD1.9"]:
                rat.append(r["GD1.9"]["EP"] / r["GD0.25"]["EP"])
        f1 = frac([x >= 0.8 for x in rat])
        neg = {}
        for s in SIZES:
            neg[s] = sum(1 for r in R if r["i"] == 1 and r["s"] == s and r["GD1.9"] and "EP" in r["GD1.9"] and r["GD1.9"]["EP"] < 0)
        ok = f1 >= 0.9 and all(v == 5 for v in neg.values())
        rows.append(("H10e", h, f"weak i: ratio>=0.8 in {f1:.3f} of {len(rat)} (median {f(med(rat))}); i=1 EP_GD1.9<0 seeds: "
                     + ", ".join(f"s={s}: {v}/5" for s, v in neg.items()), ok))
    return rows


def report_A(o, A):
    o("### R10 — sign and size of the position part (report only)")
    o()
    hd = ["h", "s", "frac E_NP>0", "median E_NP/s²", "median EP_GD.25/s²", "median EP/(E_NP+EP) [q25,q75]",
          "frac trK E_NP>0", "median trK E_NP/s²", "median trK EP_GD.25/s²", "trK ratio median [q25,q75]",
          "Pearson(E_NP/s², G_e)"]
    rows = []
    for h, W in A.items():
        for s in SIZES + ["all"]:
            R = [r for r in W["rows"] if r["np_ok"] and (s == "all" or r["s"] == s)]
            if not R:
                continue
            E = [r["NP"]["E"] / r["s"] ** 2 for r in R]
            EP = [r["GD0.25"]["EP"] / r["s"] ** 2 for r in R if r["GD0.25"] and "EP" in r["GD0.25"]]
            ratio = [r["GD0.25"]["EP"] / (r["NP"]["E"] + r["GD0.25"]["EP"]) for r in R if r["GD0.25"] and "EP" in r["GD0.25"]]
            Et = [r["NP"]["Etr"] / r["s"] ** 2 for r in R]
            EPt = [r["GD0.25"]["EPtr"] / r["s"] ** 2 for r in R if r["GD0.25"] and "EPtr" in r["GD0.25"]]
            rt = [r["GD0.25"]["EPtr"] / (r["NP"]["Etr"] + r["GD0.25"]["EPtr"]) for r in R if r["GD0.25"] and "EPtr" in r["GD0.25"]]
            pr = stats.pearsonr(E, [r["G"] for r in R])[0] if len(R) > 2 else float("nan")
            rows.append([str(h), str(s), frac([e > 0 for e in E]), med(E), med(EP),
                         f"{f(med(ratio))} [{f(q(ratio, 25))}, {f(q(ratio, 75))}]",
                         frac([e > 0 for e in Et]), med(Et), med(EPt),
                         f"{f(med(rt))} [{f(q(rt, 25))}, {f(q(rt, 75))}]", pr])
    o.table(hd, rows)
    o("SUPPLEMENTARY to R10 — the position part split at the C = 0 reference point (GN = straight first-order "
      "step, SPEC4): E_NP = E_GN + (E_NP − E_GN), where theory (U3-SUPP) predicts E_NP − E_GN ≈ s²·G_e.")
    o()
    rows = []
    for h, W in A.items():
        for s in SIZES:
            R = [r for r in W["rows"] if r["np_ok"] and r["s"] == s and r["GN"]]
            gn = [r["GN"]["E"] / s ** 2 for r in R]
            dnp = [(r["NP"]["E"] - r["GN"]["E"]) / (s * s * r["G"]) for r in R if r["G"] != 0]
            dGD = [r["GD0.25"]["E"] / s ** 2 for r in R if r["GD0.25"]]
            rows.append([str(h), str(s), frac([x > 0 for x in gn]), med(gn), med(dnp), frac([r["GD0.25"]["E"] > 0 for r in R if r["GD0.25"]]), med(dGD)])
    o.table(["h", "s", "frac E_GN>0", "median E_GN/s²", "median (E_NP−E_GN)/(s²G_e)", "frac E_GD.25>0 (total even rise)", "median E_GD.25/s²"], rows)
    o("### R10b — odd parts vs first-order prediction s·Dλ₁[ξ_e] (report only)")
    o()
    rows = []
    for h, W in A.items():
        for s in SIZES:
            R = [r for r in W["rows"] if r["s"] == s]
            a = [r["NP"]["O"] / (s * r["Dlam_xi"]) for r in R if r["NP"] and r["Dlam_xi"] != 0]
            b = [r["GD0.25"]["O"] / (s * r["Dlam_xi"]) for r in R if r["GD0.25"] and r["Dlam_xi"] != 0]
            rows.append([str(h), str(s), med(a), q(a, 10), q(a, 90), med(b), q(b, 10), q(b, 90)])
    o.table(["h", "s", "median O_NP/pred1", "q10", "q90", "median O_GD.25/pred1", "q10", "q90"], rows)
    o("### G_i·λ_i² per eigen-direction (H10d detail; median over seeds, and per-seed G_i)")
    o()
    rows = []
    for h, W in A.items():
        for i in EIG:
            R = {r["seed"]: r for r in W["rows"] if r["i"] == i and r["s"] == 0.025}
            gl = [R[s]["G"] * R[s]["lam_i"] ** 2 for s in sorted(R)]
            rows.append([str(h), f"q{i}", med([R[s]["lam_i"] for s in R]), med([R[s]["G"] for s in R]), med(gl),
                         ", ".join(f(R[s]["G"]) for s in sorted(R))])
    o.table(["h", "dir", "median λ_i", "median G_i", "median G_i λ_i²", "G_i per seed"], rows)
    o("### Part A detail: medians of the even path part vs predictions at s = 0.025 (per learner)")
    o()
    rows = []
    for h, W in A.items():
        for L in ["GN", "SL"] + [f"GD{c}" for c in CS]:
            R = [r for r in W["rows"] if r["np_ok"] and r["s"] == 0.025 and r[L] and "EP" in r[L]]
            rl = [r[L]["Pp"] / r[L]["pl_p"] for r in R if r[L]["pl_p"]] + [r[L]["Pm"] / r[L]["pl_m"] for r in R if r[L]["pl_m"]]
            rs = [r[L]["Pp"] / r[L]["ps_p"] for r in R if r[L]["ps_p"]] + [r[L]["Pm"] / r[L]["ps_m"] for r in R if r[L]["ps_m"]]
            rows.append([str(h), L, med([r[L]["EP"] / 0.025 ** 2 for r in R]), frac([r[L]["EP"] > 0 for r in R]),
                         med(rl), med(rs), frac([0.8 <= x <= 1.25 for x in rs])])
    o.table(["h", "learner", "median EP/s²", "frac EP>0", "median P/pred_lit", "median P/pred_supp", "frac P/pred_supp in [0.8,1.25]"], rows)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def analyse_B(db, part="B"):
    cells = defaultdict(dict)
    fails = []
    for r in db["records"]:
        if r["part"] != part:
            continue
        if r["status"] != "OK":
            fails.append(f"Part {part} h={r['h']} seed={r['seed']} kappa={r['kappa']} sign={r['sign']}: {r['status']} {r.get('error', '')}")
            continue
        g = r["gd"]
        if g["status"] != "CONVERGED":
            fails.append(f"Part {part} h={r['h']} seed={r['seed']} kappa={r['kappa']} sign={r['sign']}: GD(0.1) {g['status']} "
                         f"after {g['steps']} steps (res_inf {f(g['res_inf'])}); rho not reached: {g.get('rhos_not_reached')}")
        for c in r["checkpoints"]:
            ni = c["NP_info"]
            if ni["status"] != "CONVERGED":
                fails.append(f"Part {part} h={r['h']} seed={r['seed']} kappa={r['kappa']} sign={r['sign']} rho={c['rho']}: NP {ni['status']} "
                             f"(iters {ni['iters']}, res {f(ni['res_inf'])}, stat {f(ni['stat'])})")
            if c["SL_info"]["status"] != "CONVERGED":
                fails.append(f"Part {part} h={r['h']} seed={r['seed']} kappa={r['kappa']} sign={r['sign']} rho={c['rho']}: SL {c['SL_info']['status']} {c['SL_info'].get('error', '')}")
        cells[(r["h"], r["seed"], r["kappa"])][r["sign"]] = r
    return cells, fails


def ck_of(rec, rho):
    for c in rec["checkpoints"]:
        if c["rho"] == rho:
            return c
    return None


def even(cp, cm, qty, comp):
    a = cp.get("split_" + qty, {}).get(comp) if cp else None
    b = cm.get("split_" + qty, {}).get(comp) if cm else None
    if a is None or b is None:
        return None
    return 0.5 * (a + b)


def rules_B(cells, o):
    rows = []
    for h in WIDTHS:
        C = {k: v for k, v in cells.items() if k[0] == h and "+" in v and "-" in v}
        if not C:
            continue
        te, tr_share, tr_pos = [], [], []
        miss = 0
        for k, v in sorted(C.items()):
            cp, cm = ck_of(v["+"], "final"), ck_of(v["-"], "final")
            t = even(cp, cm, "lam1", "total")
            te.append(t)
            tr = None
            pos = even(cp, cm, "lam1", "position")
            if pos is not None:
                tr = t - pos
                tr_pos.append(tr > 0)
                if t > 0:
                    tr_share.append(tr / t)
            else:
                miss += 1
        f1 = frac([x > 0 for x in te])
        rows.append(("H11a", h, f"total_even(final)>0 in {f1:.3f} of {len(te)} cells; median total_even {f(med(te))}", f1 >= 0.8))
        f2 = frac(tr_pos)
        m2 = med(tr_share)
        rows.append(("H11b", h, f"(travel+route)_even>0 in {f2:.3f} of {len(tr_pos)} cells (NP missing: {miss}); "
                     f"median (travel+route)_even/total_even = {f(m2)} over {len(tr_share)} cells with total_even>0",
                     f2 >= 0.9 and m2 >= 0.5))
        # H11c
        sl = {}
        for seed in range(5):
            T = {}
            for kp in KAPPAS:
                v = C.get((h, seed, kp))
                if v:
                    T[kp] = even(ck_of(v["+"], "final"), ck_of(v["-"], "final"), "lam1", "total")
            def slope(a, b):
                if a in T and b in T and T[a] and T[b] and T[a] > 0 and T[b] > 0:
                    return math.log(T[b] / T[a]) / math.log(b / a)
                return float("nan")
            sl[seed] = (slope(0.5, 1.0), slope(1.0, 2.0), slope(2.0, 4.0))
        n_ok = sum(1 for v in sl.values() if math.isfinite(v[0]) and 1.6 <= v[0] <= 2.4)
        rows.append(("H11c", h, "slope 0.5→1 per seed: " + ", ".join(f(v[0]) for v in sl.values())
                     + f" ({n_ok}/5 in [1.6,2.4]); 1→2: " + ", ".join(f(v[1]) for v in sl.values())
                     + "; 2→4: " + ", ".join(f(v[2]) for v in sl.values()), n_ok >= 4))
        # H11d
        rat, rats, rr, sg = [], [], [], []
        for seed in range(5):
            v = C.get((h, seed, 0.5))
            if not v:
                continue
            for s_ in ("+", "-"):
                c = ck_of(v[s_], 0.9)
                if c is None or "travel" not in c["split_lam1"]:
                    continue
                rat.append(c["split_lam1"]["travel"] / c["travel_pred"])
                rats.append(c["split_lam1"]["travel"] / c["travel_pred_supp"])
                rr.append(c["split_lam1"]["route"] / c["route_pred"])
                sg.append(np.sign(c["split_lam1"]["route"]) == np.sign(c["route_pred"]))
        f4 = frac([0.67 <= x <= 1.5 for x in rat])
        rows.append(("H11d", h, f"travel/travel_pred in [0.67,1.5]: {f4:.3f} of {len(rat)} (ratios {', '.join(f(x) for x in rat)}); "
                     f"route/route_pred median {f(med(rr))}, sign agreement {frac(sg):.3f}", f4 >= 0.8))
        f5 = frac([0.67 <= x <= 1.5 for x in rats])
        rows.append(("H11d SUPP", h, f"travel/travel_pred_supp [pred(C_SL-ΔYΔYᵀ) form] in [0.67,1.5]: {f5:.3f} of {len(rats)} "
                     f"(ratios {', '.join(f(x) for x in rats)})", f5 >= 0.8))
    return rows


def report_B(cells, o):
    o("### Part B per-cell final-point numbers (λ₁)")
    o()
    rows = []
    for k in sorted(cells):
        v = cells[k]
        if "+" not in v or "-" not in v:
            continue
        cp, cm = ck_of(v["+"], "final"), ck_of(v["-"], "final")
        lp = v["+"]["init"]["lam1"]
        sp, sm = cp["split_lam1"], cm["split_lam1"]
        rows.append([str(k[0]), str(k[1]), str(k[2]), lp, sp["total"], sm["total"],
                     cp["GD"]["lam1"] / lp - 1, cm["GD"]["lam1"] / lp - 1,
                     even(cp, cm, "lam1", "total"), even(cp, cm, "lam1", "position"),
                     even(cp, cm, "lam1", "travel"), even(cp, cm, "lam1", "route"),
                     f"{v['+']['gd']['steps']}/{v['-']['gd']['steps']}"])
    o.table(["h", "seed", "κ", "λ₁(init)", "total⁺", "total⁻", "rel rise⁺", "rel rise⁻", "total_even", "position_even",
             "travel_even", "route_even", "GD steps +/−"], rows)
    o("### R11 — medians over seeds of the even parts along the run (λ₁, tr K, S_H)")
    o()
    for h in WIDTHS:
        for kp in (1.0, 4.0):
            rows = []
            for rho in RHOS:
                row = [str(h), str(kp), str(rho)]
                for qty in ("lam1", "trK", "S_H"):
                    for comp in ("position", "travel", "route", "total"):
                        vals = []
                        for seed in range(5):
                            v = cells.get((h, seed, kp))
                            if not v or "+" not in v or "-" not in v:
                                continue
                            vals.append(even(ck_of(v["+"], rho), ck_of(v["-"], rho), qty, comp))
                        row.append(med([x for x in vals if x is not None]))
                rows.append(row)
            if rows:
                o.table(["h", "κ", "ρ"] + [f"{q_}:{c[:4]}" for q_ in ("λ₁", "trK", "S_H") for c in ("position", "travel", "route", "total")], rows)
    o("### R11 — other quantities at the final point")
    o()
    rows = []
    for h in WIDTHS:
        pos = []
        dG, dN, dS, shr = [], [], [], []
        for k, v in cells.items():
            if k[0] != h or "+" not in v or "-" not in v:
                continue
            for rho in RHOS:
                e = even(ck_of(v["+"], rho), ck_of(v["-"], rho), "lam1", "position")
                if e is not None:
                    pos.append(e > 0)
            for s_ in ("+", "-"):
                c = ck_of(v[s_], "final")
                dG.append(c["GD"]["dist"])
                if "NP" in c:
                    dN.append(c["NP"]["dist"])
                if "SL" in c:
                    dS.append(c["SL"]["dist"])
                shr.append(c["GD"]["S_H"] / (c["GD"]["lam1"] / N))
        rows.append([str(h), frac(pos), len(pos), med(dG), med(dN), med(dS), med(shr), min(shr) if shr else None, max(shr) if shr else None])
    o.table(["h", "frac position_even>0 (all ρ, cells)", "n", "median ‖θ_GD−θ_init‖", "median ‖θ_NP−θ_init‖",
             "median ‖θ_SL−θ_init‖", "median S_H/(λ₁/n) final", "min", "max"], rows)
    o("### R11b — Spearman correlations across cells (σ = +, final point)")
    o()
    rows = []
    for h in WIDTHS:
        T, D, M2 = [], [], []
        for k, v in cells.items():
            if k[0] != h or "+" not in v:
                continue
            c = ck_of(v["+"], "final")
            T.append(c["split_lam1"]["total"])
            D.append(v["+"]["difficulty_D"])
            M2.append(v["+"]["move_sq"])
        if len(T) > 2:
            rows.append([str(h), len(T), stats.spearmanr(T, D)[0], stats.spearmanr(T, M2)[0]])
    o.table(["h", "cells", "Spearman(total⁺, D)", "Spearman(total⁺, ‖y⁺−F₀‖²)"], rows)
    o("### Part B: second-order predictions along the run (median over seeds and σ of measured/predicted)")
    o()
    rows = []
    for h in WIDTHS:
        for kp in KAPPAS:
            for rho in (0.9, 0.5, 0.1, "final"):
                tl, ts, rt, rs_ = [], [], [], []
                for seed in range(5):
                    v = cells.get((h, seed, kp))
                    if not v:
                        continue
                    for s_ in ("+", "-"):
                        if s_ not in v:
                            continue
                        c = ck_of(v[s_], rho)
                        if c is None or "travel" not in c["split_lam1"]:
                            continue
                        tl.append(c["split_lam1"]["travel"] / c["travel_pred"])
                        ts.append(c["split_lam1"]["travel"] / c["travel_pred_supp"])
                        rt.append(c["split_lam1"]["route"] / c["route_pred"])
                        if c.get("pred_route_meas"):
                            rs_.append(c["split_lam1"]["route"] / c["pred_route_meas"])
                rows.append([str(h), str(kp), str(rho), med(tl), med(ts), med(rt), med(rs_)])
    o.table(["h", "κ", "ρ", "travel/travel_pred (lit)", "travel/travel_pred_supp", "route/route_pred", "route/pred(C−C_SL)"], rows)


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
def analyse_C(db, part="C"):
    recs = [r for r in db["records"] if r["part"] == part]
    return recs


def c_rules(recs, o, W=16, label=""):
    rows = []
    for h in WIDTHS:
        R = [r for r in recs if r["h"] == h and r.get("status") == "OK" and r.get("eligible")]
        nel = len({r["seed"] for r in R})
        if not R:
            rows.append((f"H12 (all){label}", h, "no eligible cell at this width", None))
            continue
        win = {"PRE": [], "EDGE": []}
        for r in R:
            for w in r["windows"]:
                if w["W"] == W:
                    win[w["phase"]].append(w)
        for sup in (False, True):
            tag = " SUPP" if sup else ""
            pp = "pathpred_supp" if sup else "pathpred"
            tv = "travel_supp" if sup else "travel"
            parts, ok = [], True
            for ph in ("PRE", "EDGE"):
                ws = [w for w in win[ph] if "path" in w]
                if len(ws) < 3:
                    parts.append(f"{ph}: {len(ws)} windows (not testable)")
                    ok = False
                    continue
                x = [w["path"] for w in ws]
                y = [w[pp] for w in ws]
                pr = stats.pearsonr(x, y)[0]
                mr = med([a / b for a, b in zip(x, y) if b != 0])
                parts.append(f"{ph}: r={f(pr)}, median ratio={f(mr)} (n={len(ws)})")
                ok &= pr >= 0.9 and 0.67 <= mr <= 1.5
            rows.append((f"H12a{tag}{label}", h, f"cells(seed) {nel}; " + "; ".join(parts), ok))
            if not sup:
                ws = win["EDGE"]
                if ws:
                    fj = frac([w["jitter"] < 0 for w in ws])
                    rows.append((f"H12b{label}", h, f"EDGE jitter<0 in {fj:.3f} of {len(ws)}", fj >= 0.95))
                else:
                    rows.append((f"H12b{label}", h, "no EDGE windows", None))
            ws = [w for w in win["EDGE"] if "position" in w]
            if ws:
                a = np.mean([w["position"] + w[tv] + w["area"] for w in ws])
                j = np.mean([w["jitter"] for w in ws])
                tot = a + j
                ok = a > 0 and j < 0 and abs(tot) <= 0.3 * abs(j)
                rows.append((f"H12c{tag}{label}", h, f"EDGE mean(pos+travel+area)={f(a)}, mean(jitter)={f(j)}, |sum|/|jitter|={f(abs(tot) / abs(j) if j else float('nan'))}; "
                             f"mean measured total Δλ₁ per window={f(np.mean([w['lam_b'] - w['lam_a'] for w in ws]))}", ok))
            else:
                rows.append((f"H12c{tag}{label}", h, "no EDGE windows", None))
            ws = [w for w in win["PRE"] if "position" in w]
            if ws:
                j = np.mean([w["jitter"] for w in ws])
                pt = np.mean([w["position"] + w[tv] for w in ws])
                rows.append((f"H12d{tag}{label}", h, f"PRE |mean jitter|={f(abs(j))}, |mean(pos+travel)|={f(abs(pt))}, ratio={f(abs(j) / abs(pt) if pt else float('nan'))}",
                             abs(j) <= 0.2 * abs(pt)))
            else:
                rows.append((f"H12d{tag}{label}", h, "no PRE windows", None))
        # H12e
        ch = []
        for r in R:
            mf = os.path.join(MATDIR, r["matfile"])
            if not os.path.exists(mf):
                continue
            z = np.load(mf)
            cH = r["eta"] * z["sh"]
            for j in range(len(cH) - 1):
                if cH[j] >= 1.9 and cH[j + 1] >= 1.9 and z["sh_k"][j] < r["steps"]:
                    ch.append(float(cH[j]))
        if ch:
            m = float(np.median(ch))
            rows.append((f"H12e{label}", h, f"median c_H over EDGE samples = {f(m)} (n={len(ch)})", 1.9 <= m <= 2.1))
        else:
            rows.append((f"H12e{label}", h, "no EDGE phase", None))
    return rows


def report_C(recs, o):
    o("### R12 — per run")
    o()
    rows = []
    for r in sorted(recs, key=lambda r: (r["h"], r["seed"], r["f"] or 0)):
        if r.get("status") != "OK":
            rows.append([str(r["h"]), str(r["seed"]), str(r.get("f")), r.get("status"), r.get("error", "")[:60]] + [""] * 14)
            continue
        if not r.get("eligible"):
            rows.append([str(r["h"]), str(r["seed"]), str(r.get("f")), "not eligible",
                         f"λ_i={f(r['lam_i'])}, λ_f={f(r['lam_f'])} (ratio {f(r['lam_f'] / r['lam_i'])})"] + [""] * 14)
            continue
        means = {}
        for ph in ("PRE", "EDGE"):
            ws = [w for w in r["windows"] if w["W"] == 16 and w["phase"] == ph and "position" in w]
            for k in ("position", "travel", "jitter", "area", "topshare"):
                means[(ph, k)] = float(np.mean([w[k] for w in ws if w[k] is not None])) if ws else None
        rows.append([str(r["h"]), str(r["seed"]), str(r["f"]), r.get("edge_status"), f"{r['steps']}",
                     r["final_loss"], r.get("ceff_PRE_max"), r.get("ceff_EDGE_median"), r.get("cH_EDGE_median"), r.get("cH_max"),
                     r["frac_pre"], r["frac_edge"],
                     means[("PRE", "position")], means[("PRE", "travel")], means[("PRE", "jitter")],
                     means[("EDGE", "position")], means[("EDGE", "travel")], means[("EDGE", "jitter")], means[("EDGE", "area")]])
    o.table(["h", "seed", "f", "status", "steps", "final loss", "max c_eff PRE", "median c_eff EDGE", "median c_H EDGE", "max c_H",
             "frac PRE", "frac EDGE", "PRE pos", "PRE trav", "PRE jit", "EDGE pos", "EDGE trav", "EDGE jit", "EDGE area"], rows)
    o("Window means are per window, W = 16. 'travel' is the literal pred(½ΔYΔYᵀ); the supplementary travel is exactly its negative.")
    o()
    rows = []
    for r in sorted(recs, key=lambda r: (r["h"], r["seed"], r["f"] or 0)):
        if r.get("status") != "OK" or not r.get("eligible"):
            continue
        ts = [w["topshare"] for w in r["windows"] if w["W"] == 16 and w["phase"] == "EDGE" and w["topshare"] is not None]
        tsp = [w["topshare"] for w in r["windows"] if w["W"] == 16 and w["phase"] == "PRE" and w["topshare"] is not None]
        rows.append([str(r["h"]), str(r["seed"]), str(r["f"]), r["lam_i"], r["lam_star"], r["lam_f"], r["final_lam1"],
                     r["final_lam1"] / r["lam_f"] - 1, r["final_lam1"] / r["lam_star"], med(tsp), med(ts),
                     r.get("nwin_PRE_W16_candidates"), r.get("nwin_EDGE_W16_candidates")])
    o.table(["h", "seed", "f", "λ_i", "λ*", "λ_f (flow)", "final λ₁ (edge run)", "final/λ_f − 1", "final/λ*",
             "median top-share jitter PRE", "EDGE", "#PRE windows W16", "#EDGE windows W16"], rows)


# ---------------------------------------------------------------------------
# SUPPLEMENTARY: GN as the reference point (coordinator request)
# ---------------------------------------------------------------------------
ADIST = {}
ORD_LS = ["GN", "SL", "GD0.25", "NP"]


def order_str(vals):
    """vals: {name: lam1}; returns 'A<B<C<D' ascending."""
    return "<".join(k for k, _ in sorted(vals.items(), key=lambda kv: kv[1]))


def gnref_A(A, o):
    o("### SUPP-GN (Part A): λ₁ differences with GN (the C = 0 reference) as the base")
    o()
    o("Δ_X = λ₁(X) − λ₁(GN), even part over σ, divided by s². 'GN−start' is λ₁(GN) − λ₁(θ_A). "
      "Predicted (second-order path law): SL−GN ≈ s²·½(1−1/64)·G_e, GD(c)−GN ≈ s²·φ(c_i)·G_i (eigen-directions), NP−GN ≈ s²·G_e.")
    o()
    rows = []
    for h, W in A.items():
        for s in SIZES:
            R = [r for r in W["rows"] if r["s"] == s and r["GN"] is not None]
            line = [str(h), str(s)]
            gn = [r["GN"]["E"] / s ** 2 for r in R]
            line += [med(gn), frac([x > 0 for x in gn])]
            for L in ("SL", "GD0.25", "GD1.9", "NP"):
                v = [(r[L]["E"] - r["GN"]["E"]) / s ** 2 for r in R if r[L] is not None]
                line += [med(v), frac([x > 0 for x in v])]
            sl = [(r["SL"]["E"] - r["GN"]["E"]) / (s * s * 0.5 * (1 - 1 / M_SL) * r["G"]) for r in R if r["SL"] and r["G"]]
            npv = [(r["NP"]["E"] - r["GN"]["E"]) / (s * s * r["G"]) for r in R if r["NP"] and r["G"]]
            line += [med(sl), med(npv)]
            rows.append(line)
    o.table(["h", "s", "GN−start /s² (med)", "frac>0", "SL−GN /s²", "frac>0", "GD.25−GN /s²", "frac>0",
             "GD1.9−GN /s²", "frac>0", "NP−GN /s²", "frac>0", "med (SL−GN)/pred", "med (NP−GN)/(s²G_e)"], rows)
    o("Ordering of λ₁ among GN / SL / GD(0.25) / NP (ascending), frequency over (seed, direction, σ):")
    o()
    rows = []
    for h, W in A.items():
        for s in SIZES:
            cnt = defaultdict(int)
            tot = 0
            for r in W["rows"]:
                if r["s"] != s or any(r[L] is None for L in ORD_LS):
                    continue
                for sg in ("p", "m"):
                    cnt[order_str({L: r[L]["D" + sg] for L in ORD_LS})] += 1
                    tot += 1
            top = sorted(cnt.items(), key=lambda kv: -kv[1])[:5]
            rows.append([str(h), str(s), tot, "; ".join(f"{k}: {v / tot:.3f}" for k, v in top)])
    o.table(["h", "s", "n", "most frequent orderings (fraction)"], rows)
    o("Distance ‖θ − θ_A‖ per learner (median over (seed, direction, σ)) and relative to NP (median of dist/dist_NP − 1):")
    o()
    rows = []
    for h, W in A.items():
        for s in SIZES:
            line = [str(h), str(s)]
            for L in ("NP", "GN", "SL", "GD0.25", "GD1.9"):
                ds = []
                rel = []
                for r in W["rows"]:
                    if r["s"] != s or r[L] is None:
                        continue
                    for sg in ("p", "m"):
                        dv = r[L].get("dist_" + sg)
                        if dv is not None:
                            ds.append(dv)
                            dn = r["NP"].get("dist_" + sg) if r["NP"] else None
                            if dn:
                                rel.append(dv / dn - 1)
                line += [med(ds) if ds else None, (f"{f(med(rel))} (n={len(rel)})" if rel else "not recorded")]
            rows.append(line)
    o.table(["h", "s", "NP", "", "GN", "rel", "SL", "rel", "GD.25", "rel", "GD1.9", "rel"], rows)


def gnref_B(cells, db, o):
    o("### SUPP-GN (Part B): GN solve from θ_init to F_t at every checkpoint as the base")
    o()
    G = {}
    fails = []
    for r in db["records"]:
        if r["part"] != "BGN":
            continue
        if r["status"] != "OK":
            fails.append(f"BGN h={r['h']} seed={r['seed']} κ={r['kappa']} σ={r['sign']}: {r['status']} {r.get('error', '')}")
            continue
        for c in r["checkpoints"]:
            G[(r["h"], r["seed"], r["kappa"], r["sign"], c["rho"])] = c
            if c["status"] != "CONVERGED":
                fails.append(f"BGN h={r['h']} seed={r['seed']} κ={r['kappa']} σ={r['sign']} ρ={c['rho']}: GN {c['status']} {c.get('error') or ''}")
    if not G:
        o("(no BGN records)")
        o()
        return fails

    def lam(k, sign, rho, who):
        v = cells.get(k, {}).get(sign)
        if v is None:
            return None
        c = ck_of(v, rho)
        if c is None:
            return None
        if who == "GN":
            g = G.get(k + (sign, rho))
            return g.get("lam1") if g else None
        if who == "start":
            return v["init"]["lam1"]
        return c.get(who, {}).get("lam1")

    for kp_sel in (1.0, 4.0, "all"):
        rows = []
        for h in WIDTHS:
            for rho in RHOS:
                line = [str(h), str(kp_sel), str(rho)]
                comps = {"GN−start": ("GN", "start"), "SL−GN": ("SL", "GN"), "GD−GN": ("GD", "GN"), "NP−GN": ("NP", "GN")}
                for nm, (a, b) in comps.items():
                    ev = []
                    for k in cells:
                        if k[0] != h or (kp_sel != "all" and k[2] != kp_sel):
                            continue
                        vals = []
                        for sg in ("+", "-"):
                            la, lb = lam(k, sg, rho, a), lam(k, sg, rho, b)
                            vals.append(None if la is None or lb is None else la - lb)
                        if None not in vals:
                            ev.append(0.5 * (vals[0] + vals[1]))
                    line += [med(ev), frac([x > 0 for x in ev]) if ev else None]
                rows.append(line)
        o.table(["h", "κ", "ρ", "GN−start even (med)", "frac>0", "SL−GN even", "frac>0", "GD−GN even", "frac>0", "NP−GN even", "frac>0"], rows)
    o("Ordering of λ₁ among GN / SL / GD / NP at the final point (all cells and both σ), and distances ‖θ − θ_init‖ (median):")
    o()
    rows = []
    for h in WIDTHS:
        cnt = defaultdict(int)
        tot = 0
        dist = defaultdict(list)
        for k, v in cells.items():
            if k[0] != h:
                continue
            for sg in v:
                vals = {w: lam(k, sg, "final", w) for w in ("GN", "SL", "GD", "NP")}
                c = ck_of(v[sg], "final")
                g = G.get(k + (sg, "final"))
                for w in ("SL", "GD", "NP"):
                    if w in c:
                        dist[w].append(c[w]["dist"])
                if g and g.get("dist") is not None:
                    dist["GN"].append(g["dist"])
                if None in vals.values():
                    continue
                cnt[order_str(vals)] += 1
                tot += 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:5]
        rows.append([str(h), tot, "; ".join(f"{k}: {v / tot:.3f}" for k, v in top) if tot else "—",
                     med(dist["NP"]), med(dist["GN"]), med(dist["SL"]), med(dist["GD"])])
    o.table(["h", "n", "orderings (fraction)", "‖NP−init‖", "‖GN−init‖", "‖SL−init‖", "‖GD−init‖"], rows)
    return fails


def linear_control(db, o):
    o("## Linear control (SPEC7 §6)")
    o()
    mx = {}
    for r in db["records"]:
        if r["part"] == "LIN-A" and r["status"] == "OK":
            lamA, trA = r["kA"]["lam1"], r["kA"]["trK"]
            vals = [abs(x["lam1"] - lamA) for x in r["runs"] if "lam1" in x]
            mx["A: max |Δλ₁| over all runs"] = max(vals)
            mx["A: max |pred| (lit and supp)"] = max(max(abs(x.get("pred_lit", 0)), abs(x.get("pred_supp", 0))) for x in r["runs"])
            mx["A: max |G_e|, |Dλ₁[ξ_e]|"] = max(max(abs(d["G"]), abs(d["Dlam_xi"])) for d in r["dirs"].values())
            mx["A: runs not converged"] = sum(1 for x in r["runs"] if x["status"] not in ("CONVERGED", "CONVERGED-CONT"))
        if r["part"] == "LIN-B" and r["status"] == "OK":
            m = 0.0
            mp = 0.0
            for c in r["checkpoints"]:
                for qty in ("lam1",):
                    for v in c["split_" + qty].values():
                        m = max(m, abs(v))
                for k in ("travel_pred", "route_pred", "travel_pred_supp", "pathpred_lit", "pathpred_supp", "pred_SL_lit", "pred_SL_supp", "pred_route_meas"):
                    if c.get(k) is not None:
                        mp = max(mp, abs(c[k]))
            mx["B: max |piece| (λ₁ total/position/travel/route, all checkpoints)"] = m
            mx["B: max |prediction|"] = mp
            mx["B: GD status/steps"] = f"{r['gd']['status']} / {r['gd']['steps']}"
    rows = [[k, v if isinstance(v, str) else v, ("PASS" if (isinstance(v, (int, float)) and "converged" not in k and v <= 1e-9) else "")] for k, v in mx.items()]
    o.table(["quantity", "value", "≤ 1e-9"], rows)


def main(argv):
    tests_file = None
    if "--tests" in argv:
        tests_file = argv[argv.index("--tests") + 1]
    db = load()
    o = Out()
    o("# REPORT7 — Tests 10–12: where does the rise in sharpness during training come from?")
    o()
    o("Generated by `report7.py` from `results7.json` (scalars, per cell) and `results7_mats/*.npz` "
      "(C, Q matrices of Parts A/B; per-step series of Part C). Pass rules exactly as pre-registered in SPEC7.md; "
      "rows marked SUPP are supplementary variants motivated by the spec error documented in §5 and are never "
      "used to change a literal verdict.")
    o()
    o("## 1. Unit tests (test_part7.py output, verbatim)")
    o()
    if tests_file and os.path.exists(tests_file):
        o("```")
        with open(tests_file, encoding="utf-8", errors="replace") as fh:
            for line in fh.read().splitlines():
                o(line)
        o("```")
    else:
        o("(test output file not given)")
    o()
    for r in db["records"]:
        if r["part"] == "ADIST" and r["status"] == "OK":
            for x in r.get("dists", []):
                ADIST[(r["h"], r["seed"], x["dir"], x["s"], x["sign"], x["learner"])] = x["dist"]
    A, failsA = analyse_A(db)
    cells, failsB = analyse_B(db)
    recC = analyse_C(db)
    recC8 = analyse_C(db, "C-SUPP-k8")
    o("## 2. H-table (literal rules; SUPP rows are supplementary)")
    o()
    hrows = rules_A(A, o) + rules_A(A, o, sup=True) + rules_B(cells, o)
    elig = sum(1 for r in recC if r.get("status") == "OK" and r.get("eligible"))
    elig_cells = {(r["h"], r["seed"]) for r in recC if r.get("status") == "OK" and r.get("eligible")}
    if len(elig_cells) < 3:
        hrows.append(("H12 (all)", "all", f"only {len(elig_cells)} eligible cells (<3): Part C rules NOT TESTABLE", None))
    else:
        hrows += c_rules(recC, o, 16)
        hrows += c_rules(recC, o, 64, label=" [W=64, reported alongside]")
    o.table(["rule", "width", "measured", "verdict"],
            [[r[0], str(r[1]), r[2], ("NOT TESTABLE" if r[3] is None else verdict(r[3]))] for r in hrows])
    o(f"Part C eligibility: {len(elig_cells)} eligible (h, seed) cells of "
      f"{len({(r['h'], r['seed']) for r in recC})} examined; eligible cells: {sorted(elig_cells)}.")
    o()
    if recC8:
        o("### SUPPLEMENTARY Part C at κ = 8 (run only because fewer than 3 cells were eligible at κ = 4)")
        o()
        el8 = {(r["h"], r["seed"]) for r in recC8 if r.get("status") == "OK" and r.get("eligible")}
        o(f"eligible at κ = 8: {sorted(el8)}")
        o()
        if el8:
            rows8 = c_rules(recC8, o, 16, label=" [κ=8 SUPP]")
            o.table(["rule", "width", "measured", "verdict"],
                    [[r[0], str(r[1]), r[2], ("NOT TESTABLE" if r[3] is None else verdict(r[3]))] for r in rows8])
    o("## 3. R tables")
    o()
    o("### Part A (Test 10)")
    o()
    report_A(o, A)
    gnref_A(A, o)
    o("### Part B (Test 11)")
    o()
    report_B(cells, o)
    failsBGN = gnref_B(cells, db, o)
    o("### Part C eligibility from the Part-B κ = 4, σ = + flow runs (all widths, incl. the cut h = 128)")
    o()
    rows = []
    for k in sorted(cells):
        if k[2] == 4.0 and "+" in cells[k]:
            v = cells[k]["+"]
            li, lf = v["init"]["lam1"], ck_of(v, "final")["GD"]["lam1"]
            rows.append([str(k[0]), str(k[1]), li, lf, lf / li, "eligible" if lf >= 1.1 * li else "not eligible"])
    o.table(["h", "seed", "λ_i", "λ_f", "λ_f/λ_i", "eligibility"], rows)
    o("### Part C (Test 12)")
    o()
    report_C(recC, o)
    if recC8:
        o("### Part C SUPPLEMENTARY κ = 8")
        o()
        report_C(recC8, o)
    linear_control(db, o)
    o("## 4. Skipped / failed / non-converged cells")
    o()
    fl = failsA + failsB + failsBGN
    for r in recC + recC8:
        if r.get("status") != "OK":
            fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r.get('f')}: {r.get('status')} {r.get('error', '')}")
        elif not r.get("eligible"):
            fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r.get('f')}: not eligible — no threshold crossing possible "
                      f"(λ_f/λ_i = {f(r['lam_f'] / r['lam_i'])})")
        else:
            if r.get("edge_status") != "CONVERGED":
                fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r['f']}: edge run status {r.get('edge_status')} after {r['steps']} steps")
            if not r.get("edge_exists"):
                fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r['f']}: no EDGE phase (c_H never ≥ 1.9 at two consecutive samples; max c_H {f(r['cH_max'])})")
            bad = [w for w in r["windows"] if w["np_status"] not in ("CONVERGED", "CONVERGED-CONT")]
            if bad:
                fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r['f']}: {len(bad)} windows with NP-FAILED")
            cont = [w for w in r["windows"] if w["np_status"] == "CONVERGED-CONT"]
            if cont:
                fl.append(f"Part {r['part']} h={r['h']} seed={r['seed']} f={r['f']}: {len(cont)} windows NP converged only via continuation")
    for r in db["records"]:
        if r.get("status") == "NOT RUN":
            fl.append(f"NOT RUN: {r['part']} h={r['h']} seed={r['seed']} kappa={r.get('kappa')} sign={r.get('sign')} f={r.get('f')} — {r.get('error')}")
    if fl:
        for x in fl:
            o("- " + x)
    else:
        o("- none")
    o()
    o(NOTES_DEVIATIONS)
    o("## 6. Timing")
    o()
    tw = defaultdict(float)
    cnt = defaultdict(int)
    for r in db["records"]:
        if "wall" in r:
            tw[(r["part"], r["h"])] += r["wall"]
            cnt[(r["part"], r["h"])] += 1
    o.table(["part", "h", "jobs", "sum of job wall (s)", "mean per job (s)"],
            [[k[0], str(k[1]), cnt[k], tw[k], tw[k] / cnt[k]] for k in sorted(tw)])
    o(NOTES_TIMING)
    o(NOTES_SUMMARY)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o.lines) + "\n")
    print("wrote", OUT, len(o.lines), "lines")


NOTES_DEVIATIONS = """## 5. Deviations and spec errors

(filled in below)
"""
NOTES_TIMING = ""
NOTES_SUMMARY = """## 7. What the data say

(filled in below)
"""

if __name__ == "__main__":
    main(sys.argv[1:])
