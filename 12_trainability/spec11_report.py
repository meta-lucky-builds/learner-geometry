"""SPEC11 report (section 9): reads SPEC11_OUT, prints everything and writes spec11_report.txt and spec11_results.json
into SPEC11_OUT. Never recomputes vectors: only stored scalars, dot products and Gram matrices are used.

Environment: as spec11_run.py (SPEC11_BUNDLE, SPEC11_OUT, SPEC11_PARTS, SPEC11_COMBOS, SPEC11_QUICK, SPEC11_BREAK).
The signature (QUICK flag, BREAK value, md5 of part9lib.py and cfg11.json) must match the stored units; units with
another signature are treated as NOT RUN.
"""
import os
import sys
import json
import math
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spec11_common as S         # noqa: E402

RULE_FRAC = 0.90


class Rep:
    def __init__(self):
        self.lines = []

    def p(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.lines.append(line)


def fmt(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "nan"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:+.{p - 1}e}"
        return f"{x:+.{p}f}"
    return str(x)


def combine(vs):
    """Verdict over several combinations."""
    if any(v == "INVALID" for v in vs):
        return "INVALID"
    if vs and all(v == "PASS" for v in vs):
        return "PASS"
    if any(v == "FAIL" for v in vs):
        return "FAIL"
    return "NO DATA"


class Report:
    def __init__(self, L, out, settings, sig, combos, parts, brk):
        self.L, self.out, self.st, self.sig = L, out, settings, sig
        self.combos, self.parts, self.brk = combos, parts, brk
        self.rep = Rep()
        self.items = S.plan(L, settings, combos, parts)
        self.U, self.why = {}, {}
        for it in self.items:
            rec = S.read_unit(out, it["name"])
            if rec is None:
                self.why[it["name"]] = "NOT RUN"
            elif rec.get("signature") != sig:
                self.why[it["name"]] = "NOT RUN (stored with another signature)"
            else:
                self.U[it["name"]] = rec
        self.f = {}
        for key in ("LA", "LB", "LC", "LIN"):
            for spec in settings[key]:
                Ld = S.make_learner(L, spec)
                self.f[L.learner_name9(Ld)] = S.report_f(L, Ld, brk)
        self.lname = {key: [L.learner_name9(S.make_learner(L, sp)) for sp in settings[key]] for key in ("LA", "LB", "LC")}
        self.results = dict(signature=sig, quick=settings["QUICK"], brk=brk, validations={}, rules={}, tables={})
        self.env_bad = False
        self.part_nodata = {}

    # ---------------- access to stored units ----------------
    def unit(self, name):
        return self.U.get(name), self.why.get(name, "")

    def G(self, c):
        return self.U.get(f"G__{c}")

    def a_unit(self, c, u, size, ln):
        """(record, '') if the Part A unit at this size is usable, else (None, reason)."""
        name = S.a_name(c, u, size, ln)
        rec, why = self.unit(name)
        if rec is None:
            return None, why
        if rec.get("status") != "OK":
            return None, f"status {rec.get('status')}" + (f" ({rec.get('error')})" if rec.get("error") else "")
        if not rec.get("refs_usable"):
            return None, "references unusable"
        return rec, ""

    def a_pair(self, c, u, ln):
        recs = {}
        for size in ("s1", "s2"):
            if size not in self.st["A_SIZES"]:
                return None, f"size {size} not planned"
            r, why = self.a_unit(c, u, size, ln)
            if r is None:
                return None, f"{size}: {why}"
            recs[size] = r
        return recs, ""

    def nvNPe_sq(self, c, u, size):
        g = self.G(c)
        if g is None:
            return None
        return S.gram_lookup(g[f"G_{size}"], g[f"names_{size}"], u, u)

    # ---------------- rule printing ----------------
    def rule(self, name, threshold, used, excluded, n_planned, n_pass, need, part, extra=None):
        """used: list of (key, ok, text); excluded: list of (key, reason). need: number of passes required."""
        p = self.rep.p
        p(f"--- {name}: {threshold}")
        for key, ok, text in used:
            p(f"    used  {key}: {text} -> {'ok' if ok else 'FAIL'}")
        for key, reason in excluded:
            p(f"    excl  {key}: {reason}")
        n_used = len(used)
        if self.env_bad:
            v = "INVALID"
        elif self.part_nodata.get(part):
            v = "NO DATA"
        elif n_planned == 0 or n_used < 0.5 * n_planned:
            v = "NO DATA"
        else:
            v = "PASS" if n_pass >= need else "FAIL"
        p(f"    {name}: passed {n_pass} of {n_used} usable ({n_planned} planned); required {need} -> {v}")
        if extra:
            p(f"    {extra}")
        self.results["rules"][name] = dict(verdict=v, passed=n_pass, usable=n_used, planned=n_planned, required=need,
                                           used=[dict(key=k, ok=o, text=t) for k, o, t in used],
                                           excluded=[dict(key=k, reason=r) for k, r in excluded])
        return v

    def points_check(self, name, v, n_points, n_planned):
        """Regression rules: NO DATA if fewer than half of the planned probe points are usable.
        # spec note: for HP4/HB2 the "planned units" of the NO DATA rule are read as probe points"""
        if v in ("PASS", "FAIL") and n_points < 0.5 * n_planned:
            v = "NO DATA"
            self.rep.p(f"    {name}: only {n_points} of {n_planned} planned probe points usable -> NO DATA")
            self.results["rules"][name]["verdict"] = v
        return v

    # ---------------- validations (section 13) ----------------
    def validations(self):
        p = self.rep.p
        V = self.results["validations"]
        # V1
        env = [it for it in self.items if it["kind"] == "env"]
        v1 = []
        for it in env:
            rec, why = self.unit(it["name"])
            if rec is None:
                v1.append((it["combo"], "NOT RUN", why))
            elif rec.get("status") != "OK":
                v1.append((it["combo"], rec.get("status"), rec.get("error", "")))
            else:
                worst = max(rec["rel"], key=lambda k: rec["rel"][k] / rec["tol"][k])
                v1.append((it["combo"], "PASS" if rec["pass"] else "FAIL",
                           f"worst {worst}: rel {rec['rel'][worst]:.2e} (tol {rec['tol'][worst]:.0e})"))
                if not rec["pass"]:
                    self.env_bad = True
        if self.env_bad:
            p("!" * 100)
            p("!!! ENVIRONMENT MISMATCH: the recomputed start numbers differ from ref11.json. EVERY RULE IS INVALID.")
            p("!" * 100)
        p("=== VALIDATIONS (section 13)")
        p("V1 environment:" + ("" if env else " NOT RUN (part V not requested)"))
        for c, v, t in v1:
            p(f"    {c}: {v} {t}")
        V["V1"] = dict(verdict="INVALID-ENV" if self.env_bad else ("PASS" if v1 and all(x[1] == "PASS" for x in v1)
                                                                  else "NOT COMPLETE"),
                       items=[dict(combo=c, verdict=v, text=t) for c, v, t in v1])
        # V6 first (it decides NO DATA per part)
        # spec note: NOT RUN units and units that failed for physical reasons (e.g. a Part C learner that does not finish)
        #       count as excluded; > 20% excluded makes the whole part NO DATA
        planned = {}
        bad = {}
        for it in self.items:
            part = it["part"]
            planned[part] = planned.get(part, 0) + 1
            reason = self.unit_problem(it)
            if reason:
                bad.setdefault(part, []).append((it["name"], reason))
        # V2
        used, excl, npass = [], [], 0
        for c in self.combos:
            for u in self.st["A_DIRS"]:
                for ln in self.lname["LA"]:
                    key = f"{c}/{u}/{ln}"
                    recs, why = self.a_pair(c, u, ln)
                    if recs is None:
                        excl.append((key, why))
                        continue
                    f = self.f[ln]
                    rx = S.extrap(recs["s1"]["rho_leg"], recs["s2"]["rho_leg"])
                    ok = S.within(rx, f, 0.02)
                    npass += ok
                    used.append((key, ok, f"rho_leg_x {fmt(rx)} vs f {fmt(f)}"))
        n_pl = len(self.combos) * len(self.st["A_DIRS"]) * len(self.lname["LA"])
        p("V2 single-leg law (rho_leg extrapolated within 0.02 max(1,|f|) of f; >= 90%):")
        for key, ok, text in used:
            p(f"    {key}: {text} {'ok' if ok else 'FAIL'}")
        for key, why in excl:
            p(f"    excl {key}: {why}")
        v2 = "NO DATA" if len(used) < 0.5 * max(n_pl, 1) else ("PASS" if npass >= RULE_FRAC * len(used) else "FAIL")
        p(f"    V2: {npass}/{len(used)} -> {v2}")
        V["V2"] = dict(verdict=v2, passed=npass, usable=len(used), planned=n_pl)
        # V3 (amended A1.1): close for all OK excursions; horiz only for |2f-1| >= 0.1; nd/|vNP| <= 0.05 otherwise
        # spec note: "all OK excursions" read as the Part A excursions whose own status is OK (Part B closure not included)
        mc, mh, mr = 0.0, 0.0, 0.0
        n_exc = 0
        for it in self.items:
            if it["kind"] != "A":
                continue
            rec = self.U.get(it["name"])
            if rec is None or rec.get("status") == "ERROR":
                continue
            g = 2 * self.f[it["lname"]] - 1
            for sg in S.SIGNS:
                e = rec[f"sig{sg:+d}"]
                if e["status"] != "OK":
                    continue
                n_exc += 1
                mc = max(mc, e["close"])
                if abs(g) >= 0.1:
                    mh = max(mh, e["horiz"])
                else:
                    ref = self.U.get(S.ref_name(it["combo"], it["dir"], it["size"], sg))
                    if ref is None:
                        mr = float("inf")
                    else:
                        mr = max(mr, e["nd"] / ref["nvNP"])
        v3 = "NO DATA" if n_exc == 0 else ("PASS" if mc <= 1e-6 and mh <= 1e-2 and mr <= 0.05 else "FAIL")
        p(f"V3 closure over {n_exc} OK excursions: max close {mc:.2e} [<=1e-6], max horiz (|2f-1|>=0.1) {mh:.2e} "
          f"[<=1e-2], max nd/|vNP| (|2f-1|<0.1) {mr:.2e} [<=0.05] -> {v3}")
        V["V3"] = dict(verdict=v3, max_close=mc, max_horiz=mh, max_nd_over_vNP_small_g=mr, n=n_exc)
        # V4
        lin = [it for it in self.items if it["kind"] == "lin"]
        v4, oks = [], []
        for it in lin:
            rec, why = self.unit(it["name"])
            if rec is None:
                v4.append(f"{it['lname']}: {why}")
                oks.append(False)
            else:
                ok = rec.get("status") == "OK" and rec["nd_over_n1"] <= 1e-8
                oks.append(ok)
                v4.append(f"{it['lname']}: status {rec.get('status')} nd/n1 {rec.get('nd_over_n1', float('nan')):.2e} "
                          f"{'ok' if ok else 'FAIL'}")
        v4v = "NOT RUN" if not lin else ("PASS" if all(oks) else "FAIL")
        p("V4 linear control (nd/n1 <= 1e-8): " + "; ".join(v4) + f" -> {v4v}")
        V["V4"] = dict(verdict=v4v, items=v4)
        # V5
        v5 = []
        for c in self.combos:
            refs = [it for it in self.items if it["kind"] == "ref" and it["combo"] == c]
            if not refs:
                continue
            nu = sum(1 for it in refs if self.U.get(it["name"]) and S.ref_usable(self.U[it["name"]]))
            g = self.G(c)
            agree = g["sign_agreement"] if g else {}
            ok = nu == len(refs) and bool(agree) and all(a >= 0.99 for a in agree.values())
            v5.append(ok)
            p(f"V5 references {c}: usable {nu}/{len(refs)}; sign agreement "
              f"{ {k: round(v, 5) for k, v in agree.items()} } [>=0.99] -> {'PASS' if ok else 'FAIL'}")
        V["V5"] = dict(verdict="NOT RUN" if not v5 else ("PASS" if all(v5) else "FAIL"))
        # V6
        p("V6 excluded units (not OK), per part:")
        V["V6"] = {}
        for part in ("V", "R", "A", "B", "C"):
            if part not in planned:
                continue
            lst = bad.get(part, [])
            frac = len(lst) / planned[part]
            nod = frac > 0.20 and part != "C"       # NOTE (review, A2.1): Part C learners may fail for physical reasons;
            #                                         HC1 has its own minimum (>= 5 completed learners)
            self.part_nodata[part] = nod
            p(f"    part {part}: {len(lst)} of {planned[part]} excluded ({100 * frac:.0f}%)"
              f"{' -> PART IS NO DATA (> 20%)' if nod else ''}")
            for n, r in lst:
                p(f"        {n}: {r}")
            V["V6"][part] = dict(excluded=[dict(name=n, reason=r) for n, r in lst], planned=planned[part], no_data=nod)

    def unit_problem(self, it):
        """'' if the unit is OK for its part, else the reason (V6)."""
        rec, why = self.unit(it["name"])
        if rec is None:
            return why
        st = rec.get("status")
        if st == "ERROR":
            return f"ERROR {rec.get('error')}"
        if it["kind"] == "ref":
            return "" if S.ref_usable(rec) else f"unusable (np {rec.get('np_status')}, th {rec.get('th_status')}, " \
                                                f"jump {rec.get('jump_status')}, invalid {rec.get('invalid')}, " \
                                                f"below_resolution {rec.get('below_resolution')})"
        if st != "OK":
            return f"status {st}"
        if it["kind"] == "env" and not rec.get("pass"):
            return "environment mismatch"
        if it["kind"] == "A" and not rec.get("refs_usable"):
            return "references unusable"
        if it["kind"] == "ID" and not all(rec.get("refs_usable", {}).values()):
            return "references unusable"
        return ""

    # ---------------- Part A rules ----------------
    def part_a(self):
        p = self.rep.p
        p("=== PART A rules")
        A_units = [(c, u, ln) for c in self.combos for u in self.st["A_DIRS"] for ln in self.lname["LA"]]
        # HP1
        used, excl, npass = [], [], 0
        for c, u, ln in A_units:
            recs, why = self.a_pair(c, u, ln)
            key = f"{c}/{u}/{ln}"
            if recs is None:
                excl.append((key, why))
                continue
            g = 2 * self.f[ln] - 1
            r1, r2 = recs["s1"], recs["s2"]
            rx = S.extrap(r1["rho_loop"], r2["rho_loop"])
            ok = S.loop_law_ok(rx, r2["cos_loop"], r2["nr_loop"], g)
            npass += ok
            shape = f"cos(s2) {fmt(r2['cos_loop'])}" if abs(g) >= 0.1 else f"nr(s2) {fmt(r2['nr_loop'])}"
            used.append((key, ok, f"rho(s1) {fmt(r1['rho_loop'])} rho(s2) {fmt(r2['rho_loop'])} rho_x {fmt(rx)} "
                                  f"g {fmt(g)} {shape}"))
        hp1 = self.rule("HP1", "|rho_x - g| <= 0.03 max(1,|g|) and (|g|>=0.1: cos(s2) sign(g) >= 0.99; else nr(s2) <= "
                        "0.05); >= 90% of usable", used, excl, len(A_units), npass,
                        math.ceil(RULE_FRAC * len(used) - 1e-12), "A")
        # HP2
        used, excl, npass, npl = [], [], 0, 0
        for c, u, ln in A_units:
            g = 2 * self.f[ln] - 1
            if abs(g) < 0.1:
                continue
            npl += 1
            recs, why = self.a_pair(c, u, ln)
            key = f"{c}/{u}/{ln}"
            if recs is None:
                excl.append((key, why))
                continue
            ok = S.sign(recs["s2"]["rho_loop"]) == S.sign(g)
            npass += ok
            used.append((key, ok, f"rho_loop(s2) {fmt(recs['s2']['rho_loop'])} g {fmt(g)}"))
        hp2 = self.rule("HP2", "sign(rho_loop(s2)) = sign(g) for |g| >= 0.1; 100%", used, excl, npl, npass,
                        len(used), "A")
        # HP3
        used, excl, npass = [], [], 0
        for c, u, ln in A_units:
            recs, why = self.a_pair(c, u, ln)
            key = f"{c}/{u}/{ln}"
            if recs is None:
                excl.append((key, why))
                continue
            n1, n2 = self.nvNPe_sq(c, u, "s1"), self.nvNPe_sq(c, u, "s2")
            if n1 is None or n2 is None:
                excl.append((key, "no Gram matrix entry"))
                continue
            k1 = S.kappa(recs["s1"]["dq"]["u"], n1, recs["s1"]["s"])
            k2 = S.kappa(recs["s2"]["dq"]["u"], n2, recs["s2"]["s"])
            kx = S.extrap(k1, k2)
            target = 1 - 2 * self.f[ln]
            ok = S.within(kx, target, 0.05)
            npass += ok
            used.append((key, ok, f"kappa(s1) {fmt(k1)} kappa(s2) {fmt(k2)} kappa_x {fmt(kx)} vs 1-2f {fmt(target)}"))
        hp3 = self.rule("HP3", "|kappa_x - (1-2f)| <= 0.05 max(1,|1-2f|); >= 90% of usable", used, excl, len(A_units),
                        npass, math.ceil(RULE_FRAC * len(used) - 1e-12), "A")
        # HP4 (size s1), separately per combination
        vs = []
        self.results["tables"]["HP4_points"] = {}
        for c in self.combos:
            used, excl = [], []
            xs, ys = [], []
            npl = len(self.st["A_DIRS"]) * len(self.lname["LA"]) * len(S.W_NAMES)
            g = self.G(c)
            pts = []
            for u in self.st["A_DIRS"]:
                for ln in self.lname["LA"]:
                    rec, why = self.a_unit(c, u, "s1", ln)
                    key = f"{c}/{u}/{ln}"
                    if rec is None or g is None:
                        excl.append((key, why or "no Gram matrix"))
                        continue
                    n_ok = 0
                    for w in S.W_NAMES:
                        gw = S.gram_lookup(g["G_s1"], g["names_s1"], w, u)
                        if gw is None or not g["even_usable_s1"].get(w) or not g["even_usable_s1"].get(u):
                            excl.append((f"{key}/{w}", "reference of w or u missing or unusable"))
                            continue
                        x = (1 - 2 * self.f[ln]) * 2 * gw / rec["s"] ** 2
                        y = rec["dq"][w]
                        xs.append(x)
                        ys.append(y)
                        pts.append(dict(u=u, learner=ln, w=w, pred=x, meas=y))
                        n_ok += 1
                    used.append((key, True, f"{n_ok} probe points"))
            slope, r2 = S.through_origin(xs, ys) if xs else (float("nan"), float("nan"))
            ok = 0.9 <= slope <= 1.1 and r2 >= 0.95
            # the rule unit is the probe point; the verdict is the regression
            n_groups = len(self.st["A_DIRS"]) * len(self.lname["LA"])
            v = self.rule(f"HP4({c})", "through-origin fit of dq_w on (1-2f) 2 G1[w][u]/s1^2: slope in [0.9,1.1], "
                          "R^2 >= 0.95", used, excl, n_groups, int(ok), 1, "A",
                          extra=f"points {len(xs)}, slope {fmt(slope)}, R^2 {fmt(r2)}")
            v = self.points_check(f"HP4({c})", v, len(xs), npl)
            self.results["rules"][f"HP4({c})"].update(slope=slope, R2=r2, points=len(xs))
            self.results["tables"]["HP4_points"][c] = pts
            vs.append(v)
        hp4 = combine(vs)
        p(f"--- HP4 overall (both combinations): {hp4}")
        self.results["rules"]["HP4"] = dict(verdict=hp4)
        # HP5
        used, excl, npass, npl = [], [], 0, 0
        for c, u, ln in A_units:
            g = 2 * self.f[ln] - 1
            if abs(g) < 0.1:
                continue
            npl += 1
            rec, why = self.a_unit(c, u, "s1", ln)
            key = f"{c}/{u}/{ln}"
            if rec is None:
                excl.append((key, why))
                continue
            m = sum(rec["dq"][w] / rec["q0"][w] for w in S.W_NAMES) / len(S.W_NAMES)
            ok = S.sign(m) == S.sign(1 - 2 * self.f[ln])
            npass += ok
            used.append((key, ok, f"mean dq_w/q_w {fmt(m)} vs sign(1-2f) {S.sign(1 - 2 * self.f[ln]):+d}"))
        hp5 = self.rule("HP5", "sign(mean_w dq_w/q_w(theta0)) = sign(1-2f) for |g| >= 0.1 at s1; >= 90%", used, excl,
                        npl, npass, math.ceil(RULE_FRAC * len(used) - 1e-12), "A")
        # HID
        used, excl, npass, npl = [], [], 0, 0
        for c in self.combos:
            for u in self.st["A_DIRS"]:
                npl += 1
                rec, why = self.unit(f"A__{c}__ID__{u}")
                key = f"{c}/{u}"
                if rec is None or rec.get("status") != "OK":
                    excl.append((key, why or f"status {rec.get('status')}"))
                    continue
                if not all(rec["refs_usable"].values()) or "s2" not in rec["iota"]:
                    excl.append((key, "references unusable or size s2 not available"))
                    continue
                ix = S.extrap(rec["iota"]["s1"], rec["iota"]["s2"])
                ok = math.isfinite(ix) and abs(ix - 1) <= 0.03
                npass += ok
                used.append((key, ok, f"iota(s1) {fmt(rec['iota']['s1'])} iota(s2) {fmt(rec['iota']['s2'])} "
                                      f"iota_x {fmt(ix)}"))
        # spec note: "PASS if 4/4": every planned (combination, direction) unit must be usable and pass
        hid = self.rule("HID", "|iota_x - 1| <= 0.03; all planned", used, excl, npl, npass, npl, "A")
        return dict(HP1=hp1, HP2=hp2, HP3=hp3, HP4=hp4, HP5=hp5, HID=hid)

    # ---------------- Part B rules ----------------
    def b_rec(self, c, ln):
        rec, why = self.unit(f"B__{c}__{ln}")
        if rec is None:
            return None, why
        if rec.get("status") != "OK":
            return None, f"status {rec.get('status')} after {rec.get('m_done')} checkpoints"
        return rec, ""

    def part_b(self):
        p = self.rep.p
        p("=== PART B rules")
        M = self.st["B_M"]
        out = {}
        need_frac = 5.0 / 6.0
        tab = {}
        for c in self.combos:
            g = self.G(c)
            # HB1
            used, excl, npass = [], [], 0
            for ln in self.lname["LB"]:
                rec, why = self.b_rec(c, ln)
                if rec is None or g is None:
                    excl.append((ln, why or "no Gram matrix"))
                    continue
                cp = rec["checkpoints"][M - 1]
                r = S.star_rho(cp["dots"], cp["nvc"], g["G_s1"], g["names_s1"], M)
                if r is None:
                    excl.append((ln, "reference missing"))
                    continue
                rho, cos, nr, _ = r
                gg = 2 * self.f[ln] - 1
                ok = S.within(rho, gg, 0.05) and ((cos * S.sign(gg) >= 0.98) if abs(gg) >= 0.1 else nr <= 0.05)
                npass += ok
                shape = f"cos {fmt(cos)}" if abs(gg) >= 0.1 else f"|vc|/|Vpred| {fmt(nr)}"
                used.append((ln, ok, f"rho_B(m={M}) {fmt(rho)} g {fmt(gg)} {shape}"))
            n_l = len(self.lname["LB"])
            out[f"HB1({c})"] = self.rule(f"HB1({c})", f"rho_B within 0.05 max(1,|g|) of g, cos sign(g) >= 0.98 "
                                         f"(|g|<0.1: |vc|/|Vpred| <= 0.05) at m={M}; >= 5 of 6 learners", used, excl,
                                         n_l, npass, math.ceil(need_frac * n_l - 1e-12), "B")
            # HB2
            used, excl, xs, ys = [], [], [], []
            for ln in self.lname["LB"]:
                rec, why = self.b_rec(c, ln)
                if rec is None or g is None:
                    excl.append((ln, why or "no Gram matrix"))
                    continue
                cp = rec["checkpoints"][M - 1]
                n_ok = 0
                for w in S.W_NAMES:
                    gs = [S.gram_lookup(g["G_s1"], g["names_s1"], w, f"E{k}") for k in range(1, M + 1)]
                    if any(x is None for x in gs):
                        excl.append((f"{ln}/{w}", "Gram entry missing"))
                        continue
                    xs.append((1 - 2 * self.f[ln]) * 2 * sum(2 * x for x in gs) / rec["s1"] ** 2)
                    ys.append(cp["q"][w] - rec["q0"][w])
                    n_ok += 1
                used.append((ln, True, f"{n_ok} probe points"))
            slope, r2 = S.through_origin(xs, ys) if xs else (float("nan"), float("nan"))
            ok = 0.85 <= slope <= 1.15 and r2 >= 0.9
            v = self.rule(f"HB2({c})", "through-origin fit of dq_w(m=last) on (1-2f) 2 sum_k 2 G1[w][E_k]/s1^2: "
                          "slope in [0.85,1.15], R^2 >= 0.9", used, excl, len(self.lname["LB"]), int(ok), 1, "B",
                          extra=f"points {len(xs)}, slope {fmt(slope)}, R^2 {fmt(r2)}")
            self.results["rules"][f"HB2({c})"].update(slope=slope, R2=r2, points=len(xs))
            v = self.points_check(f"HB2({c})", v, len(xs), len(self.lname["LB"]) * len(S.W_NAMES))
            out[f"HB2({c})"] = v
            # HB3
            used, excl, npass, npl = [], [], 0, 0
            for ln in self.lname["LB"]:
                gg = 2 * self.f[ln] - 1
                rec, why = self.b_rec(c, ln)
                rows = []
                if rec is not None and g is not None:
                    for cp in rec["checkpoints"]:
                        r = S.star_rho(cp["dots"], cp["nvc"], g["G_s1"], g["names_s1"], cp["m"])
                        rows.append(None if r is None else r[0])
                    tab[f"{c}/{ln}"] = dict(g=gg, rho_B=rows, rho_B_minus_g=[None if x is None else x - gg for x in rows])
                if abs(gg) < 0.1:
                    continue
                npl += 1
                if rec is None or g is None:
                    excl.append((ln, why or "no Gram matrix"))
                    continue
                if any(x is None for x in rows):
                    excl.append((ln, "reference missing"))
                    continue
                ok = all(S.within(x, gg, 0.1) for x in rows)
                npass += ok
                used.append((ln, ok, "rho_B(m) - g: " + " ".join(fmt(x - gg) for x in rows)))
            # spec note: "PASS if >= 5 of 6 learners" while only 5 learners have |g| >= 0.1: required = ceil(5/6 * |LB|),
            #       capped at the number of evaluated learners (5 in the full run: all of them must pass).
            need = min(math.ceil(need_frac * len(self.lname["LB"]) - 1e-12), npl)
            out[f"HB3({c})"] = self.rule(f"HB3({c})", "rho_B(m) within 0.1 max(1,|g|) of g for every m, |g| >= 0.1",
                                         used, excl, npl, npass, need, "B")
        self.results["tables"]["B_rho_trend"] = tab
        for k in ("HB1", "HB2", "HB3"):
            out[k] = combine([out[f"{k}({c})"] for c in self.combos])
            self.results["rules"][k] = dict(verdict=out[k])
            p(f"--- {k} overall: {out[k]}")
        return out

    # ---------------- Part C rules ----------------
    def c_Q(self, c):
        """{learner: Q} for completed learners, and the exclusion list."""
        Q, excl = {}, []
        n_t = self.st["C_TASKS"]
        for ln in self.lname["LC"]:
            rec, why = self.unit(f"C__{c}__{ln}")
            if rec is None:
                excl.append((ln, why))
            elif rec.get("status") != "OK" or rec.get("tasks_done") != n_t:
                excl.append((ln, f"status {rec.get('status')} after {rec.get('tasks_done')} tasks"))
            else:
                cp = rec["tasks"][n_t - 1]
                Q[ln] = sum(math.log(cp["q"][w]) for w in S.W_NAMES) / len(S.W_NAMES)
        return Q, excl

    def part_c(self):
        p = self.rep.p
        p("=== PART C rules")
        out = {}
        for c in self.combos:
            Q, excl = self.c_Q(c)
            n_l = len(self.lname["LC"])
            # HC1
            used = [(ln, True, f"f {fmt(self.f[ln])} Q {fmt(Q[ln])}") for ln in Q]
            if len(Q) >= 5:
                rs = S.spearman([self.f[ln] for ln in Q], [Q[ln] for ln in Q])
                ok = rs <= -0.8
            else:
                rs, ok = float("nan"), False
            v = self.rule(f"HC1({c})", "Spearman(f, Q) <= -0.8 over completed learners (needs >= 5)", used, excl, n_l,
                          int(ok), 1, "C", extra=f"Spearman {fmt(rs)}")
            if len(Q) < 5 and v in ("PASS", "FAIL"):
                v = "NO DATA"
                p(f"    HC1({c}): fewer than 5 completed learners -> NO DATA")
                self.results["rules"][f"HC1({c})"]["verdict"] = v
            self.results["rules"][f"HC1({c})"]["spearman"] = rs
            out[f"HC1({c})"] = v
            # HC2
            ladder = ["DJ(1.5)", "DJ(1)", "SL(2)", "SL(4)", "SL(8)"]
            have = all(x in Q for x in ladder)
            ok = have and all(Q[a] > Q[b] for a, b in zip(ladder, ladder[1:]))
            p(f"--- HC2({c}): Q(DJ(1.5)) > Q(DJ(1)) > Q(SL(2)) > Q(SL(4)) > Q(SL(8)): "
              + (" > ".join(f"{x} {fmt(Q[x])}" for x in ladder) if have else "ladder learners missing"))
            v = "INVALID" if self.env_bad else ("NO DATA" if (not have or self.part_nodata.get("C"))
                                                else ("PASS" if ok else "FAIL"))
            p(f"    HC2({c}) -> {v}")
            self.results["rules"][f"HC2({c})"] = dict(verdict=v, Q={x: Q.get(x) for x in ladder})
            out[f"HC2({c})"] = v
            # HC3
            imp = [ln for ln in Q if self.f[ln] > 0.5]
            have = "SL(8)" in Q and len(imp) > 0
            ok = have and all(Q[ln] < Q["SL(8)"] for ln in imp)
            p(f"--- HC3({c}): every completed learner with f > 1/2 has Q < Q(SL(8)): "
              + (", ".join(f"{ln} {fmt(Q[ln])}" for ln in imp) + f" vs SL(8) {fmt(Q['SL(8)'])}" if have
                 else "SL(8) or implicit learners missing"))
            v = "INVALID" if self.env_bad else ("NO DATA" if (not have or self.part_nodata.get("C"))
                                                else ("PASS" if ok else "FAIL"))
            p(f"    HC3({c}) -> {v}")
            self.results["rules"][f"HC3({c})"] = dict(verdict=v)
            out[f"HC3({c})"] = v
        return out

    # ---------------- exploratory tables ----------------
    def tables(self):
        p = self.rep.p
        T = self.results["tables"]
        p("=== EXPLORATORY TABLES (no rules)")
        p("Part A kernel summary changes (sign-averaged, th2 - theta0): trK, lam1, lamN, A")
        T["A_kernel"] = {}
        for it in self.items:
            if it["kind"] != "A":
                continue
            rec = self.U.get(it["name"])
            if rec is None or rec.get("status") != "OK":
                continue
            d = rec["dksum"]
            T["A_kernel"][it["name"]] = d
            p(f"    {it['combo']} {it['dir']} {it['size']} {it['lname']:>10}: dtrK {fmt(d['trK'])} dlam1 {fmt(d['lam1'])} "
              f"dlamN {fmt(d['lamN'])} dA {fmt(d['A'])}")
        T["cos_even_refs"] = {}
        for c in self.combos:
            g = self.G(c)
            if g is None:
                continue
            names, G = g["names_s1"], g["G_s1"]
            cosm = [[G[i][j] / math.sqrt(G[i][i] * G[j][j]) for j in range(len(names))] for i in range(len(names))]
            T["cos_even_refs"][c] = dict(names=names, cos=cosm)
            p(f"Cosine matrix of even references, size s1, {c}:")
            p("        " + " ".join(f"{n:>6}" for n in names))
            for i, n in enumerate(names):
                p(f"    {n:>3} " + " ".join(f"{x:+6.3f}" for x in cosm[i]))
        p("Part B dq_w(m) = q_w(theta_m) - q_w(theta0):")
        T["B_dq"] = {}
        for c in self.combos:
            for ln in self.lname["LB"]:
                rec, _ = self.unit(f"B__{c}__{ln}")
                if rec is None or "checkpoints" not in rec:
                    continue
                rows = {}
                for cp in rec["checkpoints"]:
                    rows[cp["m"]] = {w: cp["q"][w] - rec["q0"][w] for w in S.W_NAMES}
                    p(f"    {c} {ln:>8} m={cp['m']}: " + " ".join(fmt(rows[cp['m']][w], 3) for w in S.W_NAMES))
                T["B_dq"][f"{c}/{ln}"] = rows
        p("Part B rho_B(m) - g (trend; kernel feedback):")
        for k, v in T.get("B_rho_trend", {}).items():
            p(f"    {k}: " + " ".join(fmt(x) for x in v["rho_B_minus_g"]))
        p("Part C per task: Q = mean_w log q_w, A, trK, lam1, lamN, |theta - theta0|; final true-label test loss/acc")
        T["C"] = {}
        for c in self.combos:
            for ln in self.lname["LC"]:
                rec, _ = self.unit(f"C__{c}__{ln}")
                if rec is None or "tasks" not in rec:
                    continue
                rows = []
                for cp in rec["tasks"]:
                    if "q" not in cp:
                        p(f"    {c} {ln:>10} task {cp['m']}: status {cp['status']} fit {fmt(cp['fit'])}")
                        continue
                    Qm = sum(math.log(cp["q"][w]) for w in S.W_NAMES) / len(S.W_NAMES)
                    k = cp["ksum"]
                    rows.append(dict(m=cp["m"], Q=Qm, A=k["A"], trK=k["trK"], lam1=k["lam1"], lamN=k["lamN"],
                                     dist=cp["dist"]))
                    p(f"    {c} {ln:>10} task {cp['m']}: Q {fmt(Qm)} A {fmt(k['A'])} trK {fmt(k['trK'])} "
                      f"lam1 {fmt(k['lam1'])} lamN {fmt(k['lamN'])} |th-th0| {fmt(cp['dist'])} {cp['seconds']:.0f}s")
                fin = rec.get("final")
                if fin:
                    p(f"    {c} {ln:>10} FINAL (SL(8) to true labels): status {fin['status']} fit {fmt(fin['fit'])} "
                      f"test loss {fmt(fin.get('test_loss'))} test acc {fmt(fin.get('test_acc'))}")
                T["C"][f"{c}/{ln}"] = dict(rows=rows, final=fin)

    def repro(self):
        if self.brk != "repro":
            return
        p = self.rep.p
        c, u, size, ln = S.REPRO_UNIT
        a = self.U.get(S.a_name(c, u, size, ln))
        b = self.U.get(S.a_name(c, u, size, ln) + "__repro")
        if a is None or b is None:
            v, diffs = "NO DATA", []
        else:
            diffs = S.compare_repro(a, b)
            v = "PASS (bit-identical)" if not diffs else "FAIL"
        p(f"=== BREAK repro: {S.a_name(c, u, size, ln)} vs its __repro twin: {v}")
        for d in diffs[:50]:
            p(f"    {d}")
        self.results["validations"]["REPRO"] = dict(verdict=v, diffs=diffs)

    def run(self):
        p = self.rep.p
        if self.brk:
            p("!" * 100)
            p(f"!!! SPEC11_BREAK = {self.brk!r}: NEGATIVE-CONTROL REPORT (deliberately wrong inputs/f)")
            p("!" * 100)
        p(f"SPEC11 report: out={self.out} signature={self.sig} quick={self.st['QUICK']} combos={self.combos} "
          f"parts={self.parts}")
        self.validations()
        self.repro()
        verd = {}
        if "A" in self.parts:
            verd.update(self.part_a())
        if "B" in self.parts:
            verd.update(self.part_b())
        if "C" in self.parts:
            verd.update(self.part_c())
        self.tables()
        keys = ["HP1", "HP2", "HP3", "HP4", "HP5", "HID", "HB1", "HB2", "HB3"] + \
               [f"{h}({c})" for h in ("HC1", "HC2", "HC3") for c in self.combos]
        V = self.results["validations"]
        p("DECISION INPUT: " + " ".join(f"{k} {verd[k]}" for k in keys if k in verd)
          + " | V1 " + V["V1"]["verdict"] + " V2 " + V["V2"]["verdict"] + " V3 " + V["V3"]["verdict"]
          + " V4 " + V["V4"]["verdict"] + " V5 " + V["V5"]["verdict"])
        self.results["decision_input"] = {k: verd[k] for k in keys if k in verd}
        return self.rep.lines, self.results


def write_outputs(out, lines, results):
    # spec note: section 9 does not say where: both files go into SPEC11_OUT
    with open(os.path.join(out, "spec11_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tmp = os.path.join(out, "spec11_results.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    os.replace(tmp, os.path.join(out, "spec11_results.json"))


def main():
    bundle = os.environ.get("SPEC11_BUNDLE") or HERE
    out = os.environ.get("SPEC11_OUT") or "./spec11_out"
    parts = [x.strip() for x in os.environ.get("SPEC11_PARTS", "V,R,A,B,C").split(",") if x.strip()]
    combos = [x.strip() for x in os.environ.get("SPEC11_COMBOS", "tanh-init-s0,gelu-init-s0").split(",") if x.strip()]
    qv = os.environ.get("SPEC11_QUICK", "")
    if qv not in ("", "0", "1"):
        raise ValueError(f"SPEC11_QUICK={qv!r}")
    quick = qv == "1"
    brk = S.check_break(os.environ.get("SPEC11_BREAK", ""))
    if quick:
        combos = ["tanh-init-s0"]
    S.break_banner(brk)
    sys.path.insert(0, bundle)
    L = importlib.import_module("part9lib")          # only for learner names and predict_f9
    with open(os.path.join(bundle, "cfg11.json"), encoding="utf-8") as f:
        CFG = json.load(f)
    settings = S.science_settings(quick, brk, CFG)
    sig = S.signature(settings, os.path.join(bundle, "part9lib.py"), os.path.join(bundle, "cfg11.json"))
    lines, results = Report(L, out, settings, sig, combos, parts, brk).run()
    write_outputs(out, lines, results)
    print(f"wrote {os.path.join(out, 'spec11_report.txt')} and {os.path.join(out, 'spec11_results.json')}", flush=True)


if __name__ == "__main__":
    main()
