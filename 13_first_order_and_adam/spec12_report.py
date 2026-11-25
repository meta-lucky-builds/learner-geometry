"""SPEC12 report (section 9): reads stored units only (no network), prints the report and writes
<out>/spec12_report.txt and <out>/spec12_report.json.

Usage: python spec12_report.py --out <dir> [--bundle ...] [--combos ...] [--quick] [--break ...]
(--bundle/--quick/--break only select the signature of the units to read: they must match the run.)
"""
import os
import sys
import math
import argparse
import importlib

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spec12_common as C          # noqa: E402
S = C.S

PARTS = ["V", "R", "M", "G", "A", "T", "H", "S", "D"]


def num(x):
    return C.num(x)


def fm(x, fmt="%+.4f"):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return fmt % x if math.isfinite(x) else str(x)
    return str(x)


def ratio(a, b):
    return a / b if num(a) and num(b) and b != 0 else float("nan")


def log2r(a, b):
    return math.log2(a / b) if num(a) and num(b) and a > 0 and b > 0 else float("nan")


def linfit(xs, ys):
    """Least squares y = A x + B. Returns (A, B, R^2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, the = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - the) for x, y in zip(xs, ys))
    syy = sum((y - the) ** 2 for y in ys)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    A = sxy / sxx
    B = the - A * mx
    ssr = sum((y - A * x - B) ** 2 for x, y in zip(xs, ys))
    return A, B, (1.0 - ssr / syy) if syy > 0 else float("nan")


class Report:
    def __init__(self, out, settings, sig, combos, safe_name):
        self.out, self.settings, self.sig, self.combos, self.safe = out, settings, sig, combos, safe_name
        self.items = C.plan12(settings, combos, PARTS, safe_name)
        self.recs = {}
        self.other_sig = 0
        for it in self.items:
            r = S.read_unit(out, it["name"])
            if r is None:
                continue
            if r.get("signature") != sig:
                self.other_sig += 1
                continue
            self.recs[it["name"]] = r
        self.lines = []
        self.res = dict(validations={}, rules={}, tables={}, not_ok=[], missing=[])
        self.rules = []                                 # (combo, rule, verdict, text)
        self.invalid = {}

    # ------------------------------------------------------------------ helpers
    def p(self, s=""):
        self.lines.append(s)

    def get(self, name):
        return self.recs.get(name)

    def unit(self, part, c, dirs, size, Ld, suffix=""):
        return self.get(C.pair_name(part, c, dirs, size, Ld, self.safe, suffix))

    def rule(self, c, name, verdict, text):
        if self.invalid.get(c):
            verdict = "INVALID"
        self.rules.append((c, name, verdict, text))
        self.res["rules"].setdefault(c, {})[name] = dict(verdict=verdict, text=text)
        self.p(f"  {name}: {verdict}  {text}")

    def ok(self, r):
        return r is not None and r.get("status") == "OK"

    def gam(self, r, c, part_ref="A"):
        """Gam = dGe(unit) / dGe(SL1 unit of the same direction(s) and size; Part A0, or Part T for triangles)."""
        if r is None or not num(r.get("dGe")):
            return float("nan")
        part = "T" if len(r["dirs"]) > 1 else part_ref
        base = self.unit(part, c, r["dirs"], r["size"], C.SL1())
        if not self.ok(base):
            return float("nan")
        return ratio(r["dGe"], base["dGe"])

    def pair_items(self, c, part):
        return [it for it in self.items if it["combo"] == c and it["part"] == part and it["kind"] in ("pair", "mode")]

    # ------------------------------------------------------------------ validations (section 11)
    def validations(self, c):
        self.p(f"=== VALIDATIONS {c} ===")
        env = self.get(f"V__{c}__env")
        if env is None or env.get("status") != "OK":
            v1 = "NO DATA"
            self.p(f"  V1 environment: NO DATA (status {None if env is None else env.get('status')})")
        else:
            v1 = "PASS" if env.get("pass") else "FAIL"
            self.p(f"  V1 environment: {v1} rel={env.get('rel')}")
        if v1 == "FAIL":
            self.invalid[c] = True
            self.lines.insert(0, f"ENVIRONMENT MISMATCH ({c}): every rule of this combination is INVALID")
        chk = self.get(f"V__{c}__chk")
        if chk is None or "rel" not in chk:
            v2 = "NO DATA"
            self.p("  V2 library equivalence: NO DATA")
        else:
            v2 = "PASS" if all(x <= 1e-9 for x in chk["rel"].values()) else "FAIL"
            self.p(f"  V2 library equivalence: {v2} " + " ".join(f"{k} {v:.2e}" for k, v in chk["rel"].items()))
        # V3 closure
        pairs = [self.get(it["name"]) for it in self.items if it["combo"] == c and it["kind"] in ("pair", "mode")]
        okp = [r for r in pairs if self.ok(r)]
        closes = [r["close_max"] for r in okp if num(r.get("close_max"))]
        # spec note: the star checkpoints (closed after every OK polygon) are included in "all OK polygons"
        for it in self.items:
            if it["combo"] == c and it["kind"] == "star":
                r = self.get(it["name"])
                if r is not None:
                    closes += [cp["close"] for cp in r.get("checkpoints", [])]
        cws = [r["close_w_max"] for r in okp if num(r.get("close_w_max"))]
        hz = [r["horiz_max"] for r in okp if r.get("excursion") and num(r.get("horiz_max"))
              and num(ratio(r.get("nve"), r.get("nvNPe"))) and ratio(r["nve"], r["nvNPe"]) >= 0.1]
        mc = max(closes) if closes else float("nan")
        mw = max(cws) if cws else float("nan")
        mh = max(hz) if hz else float("nan")
        if not closes:
            v3 = "NO DATA"
        else:
            v3 = "PASS" if mc <= 1e-6 and (not hz or mh <= 1e-2) else "FAIL"
        self.p(f"  V3 closure: {v3} max close {fm(mc, '%.2e')} over {len(closes)} polygons/checkpoints; max close_w "
               f"{fm(mw, '%.3e')}{' (FLAG > 0.05)' if num(mw) and mw > 0.05 else ''}; max horiz {fm(mh, '%.2e')} "
               f"over {len(hz)} excursion units with nve/nvNPe >= 0.1")
        # V4 universal identity
        groups = {"M,G": [], "A keep none/m": [], "A keep v/mv": []}
        for it in self.items:
            if it["combo"] != c or it["kind"] not in ("pair", "mode") or len(it["dirs"]) != 1:
                continue
            r = self.get(it["name"])
            if r is None:
                continue
            if it["part"] == "M":
                use = C.usable_M(r)[0]
                g = "M,G"
            elif it["part"] == "G":
                use = C.usable_G(r)
                g = "M,G"
            elif it["part"] == "A":
                use = self.ok(r)
                # spec note: Part A units without a keep policy (SL1, EM, PRE, PREM) are counted in the "none or m" group
                g = "A keep v/mv" if it["Ld"].get("keep") in ("v", "mv") else "A keep none/m"
            else:
                continue
            if use:
                groups[g].append(C.v4_ok(r))
        txt = "; ".join(f"{g}: {sum(v)}/{len(v)}" for g, v in groups.items())
        self.p(f"  V4 universal identity |a_u + rho| <= 0.03 max(1,|rho|), pass shares: {txt}")
        # V5 controls
        bad5, n5 = [], 0
        for it in self.items:
            if it["combo"] != c or it["kind"] != "pair" or len(it["dirs"]) != 1 or it["Ld"]["kind"] != "X":
                continue
            r = self.get(it["name"])
            if not self.ok(r):
                continue
            n5 += 1
            if it["Ld"]["name"] == "SL1":
                good = (num(r.get("rho")) and abs(r["rho"] + 1) <= 0.03 and num(r.get("cos")) and r["cos"] <= -0.99
                        and num(r.get("a_u")) and abs(r["a_u"] - 1) <= 0.03)
            else:
                good = num(r.get("nr")) and r["nr"] <= 0.05 and num(r.get("a_u")) and abs(r["a_u"]) <= 0.03
            if not good:
                bad5.append(f"{it['name']} rho {fm(r.get('rho'))} cos {fm(r.get('cos'))} nr {fm(r.get('nr'))} "
                            f"a_u {fm(r.get('a_u'))}")
        v5 = "NO DATA" if n5 == 0 else ("PASS" if not bad5 else "FAIL")
        self.p(f"  V5 controls (SL1, EM excursions): {v5} {n5} units; failing: {bad5}")
        # V6 references
        refs = [it for it in self.items if it["combo"] == c and it["kind"] == "ref"]
        bad6, n6 = [], 0
        for it in refs:
            r = self.get(it["name"])
            if r is None:
                continue
            n6 += 1
            if not (r.get("status") == "OK" and num(r.get("agree")) and r["agree"] >= 0.99):
                bad6.append(f"{it['name']} status {r.get('status')} agree {fm(r.get('agree'))}")
        v6 = "NO DATA" if n6 == 0 else ("PASS" if not bad6 else "FAIL")
        self.p(f"  V6 references: {v6} {n6} references; failing: {bad6}")
        # V7 tolerance independence
        r4 = self.unit("A", c, ["E1"], "s1", C.ADAM(0.1), "__tau1e-4")
        r3 = self.unit("A", c, ["E1"], "s1", C.ADAM(0.1))
        if self.ok(r4) and self.ok(r3) and num(r4.get("a_u")) and num(r3.get("a_u")):
            d7 = abs(r4["a_u"] - r3["a_u"])
            v7 = "PASS" if d7 <= 0.02 else "FAIL"
            self.p(f"  V7 tolerance independence: {v7} a_u(1e-4) {fm(r4['a_u'])} a_u(1e-3) {fm(r3['a_u'])} |diff| "
                   f"{d7:.4f}; nvo ratio {fm(ratio(r4.get('nvo'), r3.get('nvo')))}")
        else:
            v7 = "NO DATA"
            self.p("  V7 tolerance independence: NO DATA")
        self.res["validations"][c] = dict(V1=v1, V2=v2, V3=v3, V3_numbers=dict(max_close=mc, max_close_w=mw,
                                          max_horiz=mh), V4={g: [sum(v), len(v)] for g, v in groups.items()},
                                          V5=v5, V6=v6, V7=v7)

    # ------------------------------------------------------------------ Part M
    def part_m(self, c):
        self.p(f"=== PART M {c} ===")
        self.p("  learner              dir   c_i      f_pred   g_pred   status         steps res      close_w  "
               "rho      cos      nr       a_u      rho1     phi_u    uCu")
        rows = []
        for it in self.pair_items(c, "M"):
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['lname']:<20} {it['dirs'][0]:<5} (missing)")
                continue
            use, star = C.usable_M(r)
            if r.get("status") == "SKIPPED-UNSTABLE":
                use = False
            rows.append((it, r, use))
            self.p(f"  {it['lname']:<20} {it['dirs'][0]:<5} {fm(r.get('c_i'), '%.5f')} {fm(r.get('f_pred'))} "
                   f"{fm(r.get('g_pred'))} {str(r.get('status')) + ('*' if star else ''):<14} "
                   f"{fm(r.get('steps_max'), '%5d')} {fm(r.get('res_max'), '%.1e')} {fm(r.get('close_w_max'), '%.1e')} "
                   f"{fm(r.get('rho'))} {fm(r.get('cos'))} {fm(r.get('nr'))} {fm(r.get('a_u'))} {fm(r.get('rho1'))} "
                   f"{fm(r.get('phi_u'))} {fm(r.get('uCu'))}")
        us = [(it, r) for it, r, u in rows if u]
        for nm, fn in (("HM1", C.hm1_ok), ("HM2", C.hm2_ok), ("HM3", C.hm3_ok), ("HM4", C.hm4_ok)):
            npass = sum(1 for it, r in us if fn(r))
            fails = [it["name"] for it, r in us if not fn(r)]
            if len(us) < 14:
                v = "NO DATA"
            else:
                v = "PASS" if npass >= 0.9 * len(us) else "FAIL"
            self.rule(c, nm, v, f"{npass}/{len(us)} usable units pass (need >= 90%, >= 14 usable); failing {fails}")
        # eigA extras
        self.p("  eigA: measured a = -rho against the closed forms of section 1.4")
        ea = {}
        for it, r, u in rows:
            if it["dirs"][0] != "eigA":
                continue
            ea[it["lname"]] = r
            Ld = it["Ld"]
            if Ld["kind"] == "E" and Ld["gamma"] == 0 and not Ld["mid"] and num(r.get("c_i")):
                closed = C.a_gd(r["c_i"]) if Ld["mu"] == 0 else C.a_hb(r["c_i"], Ld["mu"])
                self.p(f"    {it['lname']:<16} a measured {fm(-r['rho'] if num(r.get('rho')) else None)} closed form "
                       f"{fm(closed)} (usable {u})")
        hb = ea.get("HB(0.02,0.9)")
        amp = ratio(-hb["rho"], 0.2 / (2 - 0.2)) if hb is not None and num(hb.get("rho")) else float("nan")
        self.p(f"    momentum amplification a(HB(0.02,0.9)) / [c/(2-c) at c=0.2] = {fm(amp)} (predicted 1.910/0.1111 = "
               f"17.19)")
        series = []
        for gname, nm in ((0.0, "HB(0.02,0.9)"), (0.9, "HBLA(0.02,0.9,0.9)"), (4.75, "HBLA(0.02,0.9,4.75)"),
                          (9.5, "HBLA(0.02,0.9,9.5)"), (14.25, "HBLA(0.02,0.9,14.25)")):
            r = ea.get(nm)
            a = -r["rho"] if r is not None and num(r.get("rho")) else float("nan")
            ap = -r["g_pred"] if r is not None and num(r.get("g_pred")) else float("nan")
            series.append([gname, a, ap])
            self.p(f"    look-ahead HB(0.02,0.9) gamma {gname:<5} a measured {fm(a)} predicted {fm(ap)}")
        self.res["tables"].setdefault(c, {})["M"] = dict(amplification=amp, lookahead=series)

    # ------------------------------------------------------------------ Part G
    def part_g(self, c):
        self.p(f"=== PART G {c} ===")
        self.p("  learner              dir status       steps res      close_w  nve/nvNPe rho      cos      law_ratio "
               "law_cos  n_law/nvNPe kept frac     uCu      asym/sym a_u      Gam")
        us = []
        for it in self.pair_items(c, "G"):
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['lname']:<20} {it['dirs'][0]} (missing)")
                continue
            if C.usable_G(r):
                us.append((it, r))
            nv = r.get("nvNPe")
            self.p(f"  {it['lname']:<20} {it['dirs'][0]:<3} {str(r.get('status')):<12} {fm(r.get('steps_max'), '%5d')} "
                   f"{fm(r.get('res_max'), '%.1e')} {fm(r.get('close_w_max'), '%.1e')} {fm(ratio(r.get('nve'), nv))} "
                   f"{fm(r.get('rho'))} {fm(r.get('cos'))} {fm(r.get('law_ratio'))} {fm(r.get('law_cos'))} "
                   f"{fm(ratio(r.get('n_law'), nv))} {fm(r.get('kept'), '%4d')} {fm(r.get('kept_frac'), '%.6f')} "
                   f"{fm(r.get('uCu'))} {fm(ratio(r.get('n_asym'), r.get('n_sym')))} {fm(r.get('a_u'))} "
                   f"{fm(self.gam(r, c))}")
        fails = [it["name"] for it, r in us if not C.hg1_ok(r)]
        v = "NO DATA" if len(us) < 6 else ("PASS" if not fails else "FAIL")
        self.rule(c, "HG1", v, f"{len(us) - len(fails)}/{len(us)} usable units pass (need all, >= 6); failing {fails}")
        # NOTE (review, A1.6): HG2 compares the SYMMETRIC (wear) part of the look matrix; HG3 = spectral area
        self.p("  area: learner dir n_sym n_asym n_asym_pred area_ratio area_cos")
        for it in self.pair_items(c, "G"):
            r = self.get(it["name"])
            if r is not None:
                self.p(f"    {it['lname']:<20} {it['dirs'][0]:<3} {fm(r.get('n_sym'), '%.3e')} {fm(r.get('n_asym'), '%.3e')} "
                       f"{fm(r.get('n_asym_pred'), '%.3e')} {fm(r.get('area_ratio'))} {fm(r.get('area_cos'))}")
        parts, oks = [], []
        for d in ("E1", "E2"):
            la = self.unit("G", c, [d], "s1", C.HBLA(0.1, 0.9, 9.5))
            hb = self.unit("G", c, [d], "s1", C.HB(0.1, 0.9))
            if la is not None and hb is not None and C.usable_G(la) and C.usable_G(hb):
                rs = ratio(la.get("n_sym"), hb.get("n_sym"))
                oks.append(num(rs) and rs <= 1.0 / 3.0)
                parts.append(f"{d}: n_sym ratio {fm(rs)} nve ratio {fm(ratio(la.get('nve'), hb.get('nve')))} a_u "
                             f"{fm(la.get('a_u'))} / {fm(hb.get('a_u'))} Gam {fm(self.gam(la, c))} / {fm(self.gam(hb, c))}")
            else:
                parts.append(f"{d}: units not both usable")
        v = "NO DATA" if len(oks) < 2 else ("PASS" if all(oks) else "FAIL")
        self.rule(c, "HG2", v, "; ".join(parts))
        res3 = []
        for d, Ld in (("E1", C.GD(1)), ("E1", C.HBLA(0.1, 0.9, 9.5)), ("E2", C.HBLA(0.1, 0.9, 9.5))):
            r = self.unit("G", c, [d], "s1", Ld)
            if r is not None and C.usable_G(r) and "area_cos" in r:
                res3.append((d, C.learner_name(Ld), round(r["area_ratio"], 4), round(r["area_cos"], 4), C.hg3_ok(r)))
        noarea = []
        for Ld in (C.SL1(), C.EM()):
            r = self.unit("G", c, ["E1"], "s1", Ld)
            if r is not None and C.usable_G(r) and num(r.get("n_asym_pred")):
                noarea.append((C.learner_name(Ld), round(ratio(r.get("n_asym"), r["n_asym_pred"]), 5)))
        v = "NO DATA" if len(res3) < 2 else ("PASS" if all(x[-1] for x in res3) and all(q <= 0.05 for _, q in noarea)
                                             else "FAIL")
        self.rule(c, "HG3", v, f"(dir, learner, area_ratio, area_cos, ok) {res3}; n_asym/n_asym_pred of SL1, EM {noarea}")

    # ------------------------------------------------------------------ Part A
    def part_a(self, c):
        self.p(f"=== PART A {c} ===")
        self.p("  group dir size learner                          status       steps res      close_w  nve/nvNPe "
               "nvo/nve  rho      cos      a_u      Gam      lamNe/SL1 rho1     phi_u    nv1o/dJ  Ho")
        its = self.pair_items(c, "A")
        for it in its:
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['group']:<5} {it['dirs'][0]} {it['size']} {it['lname'] + it['suffix']} (missing)")
                continue
            base = self.unit("A", c, it["dirs"], it["size"], C.SL1())
            lr_ = ratio(r.get("lamN_e"), base.get("lamN_e")) if self.ok(base) else float("nan")
            self.p(f"  {it['group']:<5} {it['dirs'][0]:<3} {it['size']:<4} {(it['lname'] + it['suffix']):<32} "
                   f"{str(r.get('status')):<12} {fm(r.get('steps_max'), '%5d')} {fm(r.get('res_max'), '%.1e')} "
                   f"{fm(r.get('close_w_max'), '%.1e')} {fm(ratio(r.get('nve'), r.get('nvNPe')))} "
                   f"{fm(ratio(r.get('nvo'), r.get('nve')))} {fm(r.get('rho'))} {fm(r.get('cos'))} {fm(r.get('a_u'))} "
                   f"{fm(self.gam(r, c))} {fm(lr_)} {fm(r.get('rho1'))} {fm(r.get('phi_u'))} "
                   f"{fm(ratio(r.get('nv1o'), r.get('dJ')))} {fm(r.get('Ho'))}")
        # exponents for every learner run at both sizes
        self.p("  exponents (same learner and direction at s1 and s2): x_even x_odd x1_odd a_u(s1) a_u(s2)")
        seen = set()
        expo = {}
        for it in its:
            key = (it["dirs"][0], it["lname"], it["suffix"])
            if key in seen:
                continue
            seen.add(key)
            r1 = self.unit("A", c, it["dirs"], "s1", it["Ld"], it["suffix"])
            r2 = self.unit("A", c, it["dirs"], "s2", it["Ld"], it["suffix"])
            if r1 is None or r2 is None:
                continue
            e = dict(x_even=log2r(r1.get("nve"), r2.get("nve")), x_odd=log2r(r1.get("nvo"), r2.get("nvo")),
                     x1_odd=log2r(r1.get("nv1o"), r2.get("nv1o")), a1=r1.get("a_u"), a2=r2.get("a_u"),
                     usable=self.ok(r1) and self.ok(r2))
            expo[key] = e
            self.p(f"    {key[0]:<3} {key[1] + key[2]:<32} {fm(e['x_even'])} {fm(e['x_odd'])} {fm(e['x1_odd'])} "
                   f"{fm(e['a1'])} {fm(e['a2'])} (both OK {e['usable']})")
        # HA0
        res0 = []
        for d in ("E1", "E2"):
            for k in (0.01, 0.03, 0.1, 0.3, 1.0):
                e = expo.get((d, C.learner_name(C.ADAM(k)), ""))
                if e is None or not e["usable"]:
                    continue
                good = (num(e["x_even"]) and 1.8 <= e["x_even"] <= 2.2 and num(e["a1"]) and num(e["a2"])
                        and abs(e["a1"] - e["a2"]) <= 0.05 * max(1.0, abs(e["a2"])))
                res0.append((d, k, good))
        n0 = sum(1 for x in res0 if x[2])
        v = "NO DATA" if len(res0) < 6 else ("PASS" if n0 >= 0.9 * len(res0) else "FAIL")
        self.rule(c, "HA0", v, f"{n0}/{len(res0)} (direction, k) pairs pass; failing {[x[:2] for x in res0 if not x[2]]}")
        # HA1
        bad, n1 = [], 0
        for it in its:
            r = self.get(it["name"])
            if not self.ok(r):
                continue
            q = ratio(r.get("nv1o"), r.get("dJ"))
            if it["group"] == "A1":
                n1 += 1
                if not (num(q) and q >= 0.3):
                    bad.append(f"{it['name']} nv1o/dJ {fm(q)}")
            if it["group"] == "A0":
                if not (num(q) and q <= 0.02):
                    bad.append(f"{it['name']} nv1o/dJ {fm(q)}")
        for d in ("E1", "E2"):
            for k in (0.01, 0.03, 0.1, 0.3, 1.0):
                e = expo.get((d, C.learner_name(C.ADAM(k)), ""))
                if e is None or not e["usable"]:
                    continue
                if not (num(e["x1_odd"]) and 0.9 <= e["x1_odd"] <= 1.1):
                    bad.append(f"{d} ADAM({k}) x1_odd {fm(e['x1_odd'])}")
        v = "NO DATA" if n1 == 0 else ("PASS" if not bad else "FAIL")
        self.rule(c, "HA1", v, f"{n1} usable A1 units; failing {bad}")
        # HA2 (each size separately)
        # spec note: "each size separately" (HA2) and "each k separately" (HA3) are given separate verdicts
        for kk, size in ((0.1, "s1"), (0.1, "s2"), (1.0, "s1")):      # NOTE (review, A1.1, A1.3): verdict per (k, size)
            pts = []
            for b1 in (0.5, 0.8, 0.9, 0.95):
                r = self.unit("A", c, ["E1"], size, C.ADAM(kk, b1=b1))
                if self.ok(r) and num(r.get("a_u")):
                    pts.append((b1, r["a_u"], self.gam(r, c)))
            b0 = self.unit("A", c, ["E1"], size, C.ADAM(kk, b1=0.0))
            txt0 = f"b1=0 reported only: status {None if b0 is None else b0.get('status')} a_u " \
                   f"{fm(None if b0 is None else b0.get('a_u'))}"
            if len(pts) < 3:
                self.rule(c, f"HA2[k={kk:g},{size}]", "NO DATA", f"{len(pts)} usable units; {txt0}")
                continue
            xs = [(1 + b) / (1 - b) for b, _, _ in pts]
            sp = S.spearman([b for b, _, _ in pts], [a for _, a, _ in pts])
            A, B, R2 = linfit(xs, [a for _, a, _ in pts])
            gp = [(x, g) for x, (_, _, g) in zip(xs, pts) if num(g)]
            Ag, Bg, R2g = linfit([x for x, _ in gp], [g for _, g in gp])
            # spec note: "Spearman = 1" is tested as >= 1 - 1e-12 (floating point)
            v = "PASS" if (num(sp) and sp >= 1 - 1e-12 and num(A) and A > 0 and num(R2) and R2 >= 0.9) else "FAIL"
            self.rule(c, f"HA2[k={kk:g},{size}]", v, f"points (b1, a_u, Gam) {[(b, round(a, 5), round(g, 5) if num(g) else g) for b, a, g in pts]}; "
                      f"Spearman {fm(sp)}; a_u = A (1+b1)/(1-b1) + B: A {fm(A)} B {fm(B)} R^2 {fm(R2)}; "
                      f"Gam: A {fm(Ag)} B {fm(Bg)} R^2 {fm(R2g)}; {txt0}")
        # HA3 (each k separately)
        for k in (0.1, 1.0):                                          # NOTE (review, A1.1): was (0.03, 0.1)
            pts = []
            for b2 in (0.5, 0.9, 0.99, 0.999):
                r = self.unit("A", c, ["E1"], "s1", C.ADAM(k, b2=b2))
                if self.ok(r) and num(r.get("a_u")):
                    pts.append((b2, r["a_u"], self.gam(r, c)))
            if len(pts) < 3:
                self.rule(c, f"HA3[k={k:g}]", "NO DATA", f"{len(pts)} usable units {pts}")
                continue
            sp = S.spearman([b for b, _, _ in pts], [a for _, a, _ in pts])
            v = "PASS" if num(sp) and sp <= -0.8 else "FAIL"
            self.rule(c, f"HA3[k={k:g}]", v, f"(b2, a_u, Gam) {[(b, round(a, 5), round(g, 5) if num(g) else g) for b, a, g in pts]}; "
                      f"Spearman {fm(sp)}")
        # HA4
        lines4 = []
        verdict4 = "NO DATA"
        for k, gams in ((0.1, (0.0, 0.5, 1.0, 2.0, 4.0, 9.5)), (0.03, (0.0, 1.0, 4.0, 9.5))):
            pts = []
            for g in gams:
                r = self.unit("A", c, ["E1"], "s1", C.ADAM(k, gamma=g))
                if r is None:
                    lines4.append(f"k {k:g} gamma {g:g}: missing")
                    continue
                lines4.append(f"k {k:g} gamma {g:g}: {r.get('status')} steps {r.get('steps_max')} res "
                              f"{fm(r.get('res_max'), '%.1e')} a_u {fm(r.get('a_u'))} Gam {fm(self.gam(r, c))}")
                if self.ok(r) and num(r.get("a_u")):
                    pts.append((g, r["a_u"]))
            if k == 0.1:
                if len(pts) >= 4:
                    sp = S.spearman([g for g, _ in pts], [a for _, a in pts])
                    verdict4 = "PASS" if num(sp) and sp <= -0.9 else "FAIL"
                    lines4.append(f"Spearman(gamma, a_u) at k 0.1 over {len(pts)} usable: {fm(sp)}")
                else:
                    lines4.append(f"k 0.1: only {len(pts)} usable units")
        self.rule(c, "HA4", verdict4, "; ".join(lines4))
        # HA5
        parts, okall, have = [], True, 0
        for k in (0.03, 0.1):
            pre, prem, ad = (self.unit("A", c, ["E1"], "s1", C.PRE(k)), self.unit("A", c, ["E1"], "s1", C.PREM(k)),
                             self.unit("A", c, ["E1"], "s1", C.ADAM(k)))
            if not (self.ok(pre) and self.ok(prem) and self.ok(ad)):
                parts.append(f"k {k:g}: units not all OK")
                okall = False
                continue
            have += 1
            i_ok = (num(prem.get("a_u")) and num(pre.get("a_u")) and prem["a_u"] > pre["a_u"]
                    and self.gam(prem, c) > self.gam(pre, c))
            ii_ok = num(ad.get("a_u")) and ad["a_u"] < prem["a_u"]
            okall = okall and i_ok and ii_ok
            parts.append(f"k {k:g}: a_u PRE {fm(pre.get('a_u'))} PREM {fm(prem.get('a_u'))} ADAM {fm(ad.get('a_u'))}; "
                         f"Gam PRE {fm(self.gam(pre, c))} PREM {fm(self.gam(prem, c))} ADAM {fm(self.gam(ad, c))}; "
                         f"(i) {i_ok} (ii) {ii_ok}")
        v = "NO DATA" if have < 2 else ("PASS" if okall else "FAIL")
        self.rule(c, "HA5", v, "; ".join(parts))
        # HA6
        g_vmv, g_mn = [], []
        cases = [("E1", 0.1), ("E2", 0.1), ("E1", 0.3)]
        for d, k in cases:
            for keep in ("none", "m", "v", "mv"):
                if k == 0.3 and keep in ("m", "v"):
                    continue
                Ld = C.ADAM(k, keep=keep)
                r1 = self.unit("A", c, [d], "s1", Ld)
                r2 = self.unit("A", c, [d], "s2", Ld)
                if keep in ("v", "mv"):
                    if not (self.ok(r1) and self.ok(r2)):
                        continue
                    xo = log2r(r1.get("nvo"), r2.get("nvo"))
                    q = ratio(r1.get("nvo"), r1.get("nve"))
                    good = num(xo) and 0.8 <= xo <= 1.2 and num(q) and q >= 1
                    g_vmv.append((d, k, keep, round(xo, 4) if num(xo) else xo, round(q, 4) if num(q) else q, good))
                else:
                    if not self.ok(r1):
                        continue
                    q = ratio(r1.get("nvo"), r1.get("nve"))
                    good = num(q) and q <= 0.5
                    g_mn.append((d, k, keep, round(q, 4) if num(q) else q, good))
        if len(g_vmv) < 4 or len(g_mn) < 4:
            v = "NO DATA"
        else:
            v = "PASS" if all(x[-1] for x in g_vmv + g_mn) else "FAIL"
        self.rule(c, "HA6", v, f"keep v/mv (dir, k, keep, x_odd, nvo/nve, ok): {g_vmv}; keep m/none (dir, k, keep, "
                               f"nvo/nve, ok): {g_mn}")
        # reported without verdict
        self.p("  k series of A1 (no verdict): dir size k status steps a_u Gam nv1o/dJ nve/nvNPe")
        for d in ("E1", "E2"):
            for size in ("s1", "s2"):
                for k in (0.01, 0.03, 0.1, 0.3, 1.0):
                    r = self.unit("A", c, [d], size, C.ADAM(k))
                    if r is not None:
                        self.p(f"    {d} {size} {k:<5g} {r.get('status')} {r.get('steps_max')} {fm(r.get('a_u'))} "
                               f"{fm(self.gam(r, c))} {fm(ratio(r.get('nv1o'), r.get('dJ')))} "
                               f"{fm(ratio(r.get('nve'), r.get('nvNPe')))}")
        # NOTE (review, A1.3): HA7 sign pattern of the rate series at s1 (+, +, - at k = 0.1, 0.3, 1.0), per direction
        cases7 = []
        for d in ("E1", "E2"):
            rs = [self.unit("A", c, [d], "s1", C.ADAM(k)) for k in (0.1, 0.3, 1.0)]
            if not all(self.ok(r) and num(r.get("a_u")) for r in rs):
                continue
            gs = [self.gam(r, c) for r in rs]
            good = (rs[0]["a_u"] > 0 and rs[1]["a_u"] > 0 and rs[2]["a_u"] < 0
                    and num(gs[0]) and num(gs[1]) and num(gs[2]) and gs[0] > 0 and gs[1] > 0 and gs[2] < 0)
            cases7.append((d, [round(r["a_u"], 4) for r in rs], [round(g, 4) if num(g) else g for g in gs], good))
        v = "NO DATA" if not cases7 else ("PASS" if all(x[-1] for x in cases7) else "FAIL")
        self.rule(c, "HA7", v, f"per direction (a_u, Gam at k = 0.1, 0.3, 1.0; ok): {cases7} "
                               f"(the full verdict needs all (combination, direction) cases: read both combinations)")
        self.p("  learners that never settle (UNCONVERGED Part A units): name steps res close_w a_u Gam nve/nvNPe")
        for it in its:
            r = self.get(it["name"])
            if r is not None and r.get("status") == "UNCONVERGED":
                self.p(f"    {it['name']} {r.get('steps_max')} {fm(r.get('res_max'), '%.2e')} "
                       f"{fm(r.get('close_w_max'), '%.2e')} {fm(r.get('a_u'))} {fm(self.gam(r, c))} "
                       f"{fm(ratio(r.get('nve'), r.get('nvNPe')))}")

    # ------------------------------------------------------------------ Part T
    def part_t(self, c):
        self.p(f"=== PART T {c} ===")
        self.p("  learner                          size status       nve        nvo        nvo/nve  dGe        dGo        Gam")
        its = self.pair_items(c, "T")
        for it in its:
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['lname']} {it['size']} (missing)")
                continue
            self.p(f"  {it['lname']:<32} {it['size']:<4} {str(r.get('status')):<12} {fm(r.get('nve'), '%.3e')} "
                   f"{fm(r.get('nvo'), '%.3e')} {fm(ratio(r.get('nvo'), r.get('nve')))} {fm(r.get('dGe'), '%.3e')} "
                   f"{fm(r.get('dGo'), '%.3e')} {fm(self.gam(r, c))}")
        res, nuse = [], 0
        seen = []
        for it in its:
            if it["lname"] in seen:
                continue
            seen.append(it["lname"])
            r1 = self.unit("T", c, ["E1", "E2"], "s1", it["Ld"])
            r2 = self.unit("T", c, ["E1", "E2"], "s2", it["Ld"])
            xe = log2r(r1.get("nve"), r2.get("nve")) if r1 and r2 else float("nan")
            xo = log2r(r1.get("nvo"), r2.get("nvo")) if r1 and r2 else float("nan")
            self.p(f"    {it['lname']:<32} x_even {fm(xe)} x_odd {fm(xo)}")
            if not self.ok(r1):
                continue
            q = ratio(r1.get("nvo"), r1.get("nve"))
            kind = it["Ld"]["kind"]
            if kind == "A":
                if not self.ok(r2):
                    continue
                good = num(xo) and 0.8 <= xo <= 1.2 and num(q) and q >= 1
            else:
                good = num(q) and q <= 0.2
                if it["lname"] in ("SL1", "GD(1)"):
                    if not self.ok(r2):
                        continue
                    good = good and num(xe) and 1.8 <= xe <= 2.2
            nuse += 1
            res.append((it["lname"], good))
        v = "NO DATA" if nuse < 5 else ("PASS" if all(g for _, g in res) else "FAIL")
        self.rule(c, "HT", v, f"{res}")

    # ------------------------------------------------------------------ Part H
    def part_h(self, c):
        self.p(f"=== PART H {c} ===")
        hs = self.settings["HS"]
        base = self.unit("A", c, ["E1"], "s1", C.SL1())
        dge = abs(base["dGe"]) if self.ok(base) and num(base.get("dGe")) else float("nan")
        rows = {}
        for it in self.items:
            if it["combo"] != c or it["kind"] != "hover":
                continue
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['lname']} (missing)")
                continue
            line = f"  {it['lname']:<28} {r.get('status')}"
            for sg in ("+1", "-1"):
                sr = r.get(f"sig{sg}", {})
                line += f" | sig{sg}: steps_stop {sr.get('steps_stop')} res_stop {fm(sr.get('res_stop'), '%.1e')} " \
                        f"Qh(end) {fm(sr.get('Qh'), '%.3e')}"
                for i, n in enumerate(hs):
                    recs = sr.get("records") or []
                    rm = recs[i]["res_mean"] if i < len(recs) and recs[i] else None
                    nvh = sr["nvh"][i] if "nvh" in sr else None
                    line += f" n{n}: res_mean {fm(rm, '%.1e')} nvh/nvNPe {fm(ratio(nvh, r.get('nvNPe')))}"
            Wg = [(-x / dge) if num(x) and num(dge) else float("nan") for x in r.get("we", [])]
            Wu = r.get("wue", [])
            line += f" | Wg {[fm(x) for x in Wg]} Wu {[fm(x) for x in Wu]}"
            self.p(line)
            if r.get("status") == "OK":
                rows[it["lname"]] = (it["Ld"], Wg)
        gd = rows.get("GD(1)")
        adam = [(nm, w) for nm, (Ld, w) in rows.items() if Ld["kind"] == "A"]
        if gd is None or not adam or not num(dge):
            v = "NO DATA"
        else:
            last, firstn = -1, 0
            sup = (abs(gd[1][last]) <= 0.05 and
                   any(num(w[last]) and w[last] >= 0.1 and w[last] >= 1.5 * w[firstn] for _, w in adam))
            nots = all(num(w[last]) and abs(w[last]) <= 0.05 for _, w in adam)
            v = "supported" if sup else ("not supported" if nots else "mixed")
        self.rule(c, "HH", v, f"|dGe(SL1,E1,s1)| {fm(dge, '%.3e')}; Wg[{hs}] per learner "
                              f"{ {nm: [fm(x) for x in w] for nm, (_, w) in rows.items()} }")

    # ------------------------------------------------------------------ Part S
    def part_s(self, c):
        self.p(f"=== PART S {c} ===")
        n_mm = self.settings["N_MM"]
        op0 = self.get(f"S__{c}__OP0")
        stars = {}
        for it in self.items:
            if it["combo"] != c or it["kind"] != "star":
                continue
            r = self.get(it["name"])
            if r is None:
                self.p(f"  {it['lname']} (missing)")
                continue
            self.p(f"  {it['lname']} status {r.get('status')} checkpoints {r.get('mm_done')}/{n_mm}")
            k0 = r["ksum0"]
            for cp in r.get("checkpoints", []):
                self.p(f"    mm {cp['mm']}: G {cp['G']:.6f} lamN/lamN0 {cp['ksum']['lamN'] / k0['lamN']:.6f} trK/trK0 "
                       f"{cp['ksum']['trK'] / k0['trK']:.6f} inv_mean ratio {cp['ksum']['inv_mean'] / k0['inv_mean']:.6f} "
                       f"nvc {cp['nvc']:.4e} close {cp['close']:.1e}")
            if r.get("status") == "OK" and r.get("mm_done") == n_mm:
                stars[it["lname"]] = (it["Ld"], r)
        sl = stars.get("SL1")
        G6sl = sl[1]["checkpoints"][-1]["G"] if sl else float("nan")
        rstar = {}
        for nm, (Ld, r) in stars.items():
            rstar[nm] = ratio(1 - r["checkpoints"][-1]["G"], 1 - G6sl)
        # Gbar from Part A single loops at s1
        gbar = {}
        for nm in stars:
            gs = []
            for it in self.pair_items(c, "A"):
                if it["lname"] == nm and it["size"] == "s1" and it["suffix"] == "":
                    r = self.get(it["name"])
                    if self.ok(r) and num(self.gam(r, c)):
                        gs.append(self.gam(r, c))
            gbar[nm] = sum(gs) / len(gs) if gs else float("nan")
        lnr = {}
        for nm, (Ld, r) in stars.items():
            op = r.get("op")
            if op and op0 and op0.get("op"):
                vals = [math.log(op[x]["r50"] / op0["op"][x]["r50"]) for x in ("W1", "W5")
                        if num(op[x]["r50"]) and num(op0["op"][x]["r50"]) and op[x]["r50"] > 0 and op0["op"][x]["r50"] > 0]
                # spec note: ln(r50_end / r50_OP0) averaged over the two probe directions W1, W5
                lnr[nm] = sum(vals) / len(vals) if vals else float("nan")
            self.p(f"  {nm:<28} Rstar {fm(rstar[nm])} Gbar {fm(gbar[nm])} op {op} ln(r50/r50_OP0) "
                   f"{fm(lnr.get(nm))}")
        self.p(f"  OP0: {None if op0 is None else op0.get('op')}")
        em = stars.get("EM")
        if sl is None or em is None:
            v = "NO DATA"
        else:
            v = "PASS" if (num(rstar["EM"]) and abs(rstar["EM"]) <= 0.05 and G6sl < 1) else "FAIL"
        self.rule(c, "HS1", v, f"Rstar(EM) {fm(rstar.get('EM'))} G_last(SL1) {fm(G6sl, '%.6f')}")
        reset = [C.learner_name(x) for x in (C.ADAM(0.03), C.ADAM(0.1), C.ADAM(0.3), C.ADAM(1.0),
                                             C.ADAM(0.1, b1=0.95), C.ADAM(0.1, b2=0.9))]   # NOTE (review, A1.4)
        use = [nm for nm in reset if nm in stars and num(rstar.get(nm)) and num(gbar.get(nm))]
        if len(use) < 4:
            self.rule(c, "HS2", "NO DATA", f"{len(use)} usable learners {use}")
            # spec note: HS3 uses "the same learners" as HS2, so it also needs >= 4
            self.rule(c, "HS3", "NO DATA", f"{len(use)} usable learners")
        else:
            sp = S.spearman([rstar[n] for n in use], [gbar[n] for n in use])
            self.rule(c, "HS2", "PASS" if num(sp) and sp >= 0.9 else "FAIL",
                      f"Spearman(Rstar, Gbar) {fm(sp)} over {[(n, round(rstar[n], 4), round(gbar[n], 4)) for n in use]}")
            hold = [abs(rstar[n] - gbar[n]) <= 0.1 + 0.3 * abs(gbar[n]) for n in use]
            self.rule(c, "HS3", "PASS" if sum(hold) >= 0.8 * len(hold) else "FAIL",
                      f"{sum(hold)}/{len(hold)} hold |Rstar - Gbar| <= 0.1 + 0.3|Gbar|")
        mv = C.learner_name(C.ADAM(0.1, keep="mv"))
        self.p(f"  reported: Rstar keep mv {fm(rstar.get(mv))} vs ADAM(0.1) {fm(rstar.get(C.learner_name(C.ADAM(0.1))))}")
        both = [n for n in rstar if num(rstar[n]) and num(lnr.get(n))]
        sp = S.spearman([rstar[n] for n in both], [lnr[n] for n in both]) if len(both) >= 2 else float("nan")
        self.p(f"  reported: Spearman(Rstar, ln(r50_end/r50_OP0)) over {len(both)} learners: {fm(sp)}")
        self.res["tables"].setdefault(c, {})["S"] = dict(Rstar=rstar, Gbar=gbar, ln_r50=lnr)

    # ------------------------------------------------------------------ Part D
    def part_d(self, c):
        self.p(f"=== PART D {c} ===")
        usable = []
        for it in self.items:
            if it["combo"] != c or it["kind"] != "own":
                continue
            r = self.get(it["name"])
            if r is None:
                self.p(f"  k {it['kappa']:g} (missing)")
                continue
            self.p(f"  k {it['kappa']:<5g} {r.get('status')} D {r.get('D_summary')} K_D lam1 {fm(r.get('lamD1'), '%.3e')} "
                   f"lamN {fm(r.get('lamDN'), '%.3e')} ctrl {r.get('ctrl')} own {r.get('own')} eucl {r.get('eucl')}")
            if r.get("status") == "OK" and r.get("own"):
                usable.append(r)
        if not usable:
            self.rule(c, "HD0", "NO DATA", "no usable unit")
            self.rule(c, "HD", "NO DATA", "no usable unit")
            return
        ok0 = all(num(r["ctrl"]["rho"]) and abs(r["ctrl"]["rho"] + 1) <= 0.03 and num(r["ctrl"]["cos"])
                  and r["ctrl"]["cos"] <= -0.99 for r in usable)
        self.rule(c, "HD0", "PASS" if ok0 else "FAIL", f"{[(r['kappa'], r['ctrl']) for r in usable]}")
        if not ok0:
            self.rule(c, "HD", "NOT READ", "HD0 failed")
            return
        cs = [abs(r["own"]["cos"]) if num(r["own"]["cos"]) else float("nan") for r in usable]
        if all(num(x) and x >= 0.8 for x in cs):
            v = "supported"
        elif all(num(x) and x <= 0.3 for x in cs):
            v = "not supported"
        else:
            v = "mixed"
        self.rule(c, "HD", v, f"|own cos| per k {[(r['kappa'], round(x, 4) if num(x) else x) for r, x in zip(usable, cs)]}")

    # ------------------------------------------------------------------ run
    def run(self):
        for c in self.combos:
            self.validations(c)
        for c in self.combos:
            self.part_m(c)
            self.part_g(c)
            self.part_a(c)
            self.part_t(c)
            self.part_h(c)
            self.part_s(c)
            self.part_d(c)
        self.p("=== SUMMARY: one line per rule and combination ===")
        for c, nm, v, txt in self.rules:
            self.p(f"  {c:<14} {nm:<10} {v:<14} {txt[:300]}")
        self.p("=== UNITS NOT OK (V8) ===")
        for it in self.items:
            r = self.recs.get(it["name"])
            if r is None:
                self.res["missing"].append(it["name"])
                continue
            if r.get("status") != "OK":
                self.res["not_ok"].append([it["name"], r.get("status")])
                self.p(f"  {it['name']}: {r.get('status')}")
        self.p(f"  planned units missing (or with another signature): {len(self.res['missing'])} "
               f"{self.res['missing'][:50]}{' ...' if len(self.res['missing']) > 50 else ''}")
        if self.other_sig:
            self.p(f"  {self.other_sig} stored units carry another signature and were ignored")
        return self.lines, self.res


def write_outputs(L, out, lines, results):
    with open(os.path.join(out, "spec12_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    L.write_json_atomic(os.path.join(out, "spec12_report.json"), results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    # spec note: section 9 names only --out; --bundle/--combos/--quick/--break are added to select the signature of the
    #       units to read (QUICK, BREAK and full results never mix) and the combinations
    ap.add_argument("--bundle", default=os.path.normpath(os.path.join(HERE, "..", "spec11", "bundle")))
    ap.add_argument("--combos", default="tanh-init-s0,gelu-init-s0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--break", dest="brk", default="")
    a = ap.parse_args()
    combos = [c.strip() for c in a.combos.split(",") if c.strip()]
    if a.quick:
        combos = ["tanh-init-s0"]
    sys.path.insert(0, a.bundle)
    L = importlib.import_module("part9lib")             # names and json writing only; no network
    settings = C.settings12(a.quick, C.check_break(a.brk))
    sig = S.signature(settings, os.path.join(a.bundle, "part9lib.py"), os.path.join(a.bundle, "cfg11.json"))
    rep = Report(os.path.abspath(a.out), settings, sig, combos, L.safe_name)
    lines, results = rep.run()
    print("\n".join(lines))
    write_outputs(L, os.path.abspath(a.out), lines, results)


if __name__ == "__main__":
    main()
