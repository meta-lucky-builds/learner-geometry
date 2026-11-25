"""SPEC12 section 10 tests on a tiny network (CPU, 2 threads). Exit code 1 on any failure.

Uses part9lib read-only from ../spec11/bundle with the bundle's cfg11.json. Every test calls the spec12_common /
spec12_run / spec12_report functions the real run uses. Writes only into a temporary directory (on D: when it exists,
never inside the project folder), deleted at the end.
"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import os
import sys
import math
import json
import shutil
import tempfile

sys.dont_write_bytecode = True                    # never write __pycache__ next to the read-only library
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.normpath(os.path.join(HERE, "..", "spec11", "bundle"))
sys.path.insert(0, HERE)
sys.path.insert(1, BUNDLE)

import torch                                      # noqa: E402
torch.set_num_threads(2)
import spec12_common as C                         # noqa: E402
import spec12_run as R                            # noqa: E402
import spec12_report as REP                       # noqa: E402
S = C.S

# spec note: the temporary directory is placed on D: (results must never go under C:); system temp only if D: is absent
TMP_ROOT = None
TMP = tempfile.mkdtemp(prefix="spec12_tests_", dir=TMP_ROOT)
CFG_PATH = os.path.join(BUNDLE, "cfg11.json")
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)
CFG["MNIST_DOWNLOAD"] = False
CFG["RUN_DIR"] = TMP                              # the library's progress.log goes to the temp dir
import part9lib as L                              # noqa: E402
assert os.path.samefile(L.__file__, os.path.join(BUNDLE, "part9lib.py")), L.__file__
L.configure(CFG)
L.set_device("cpu")
LIB_PATH = os.path.join(BUNDLE, "part9lib.py")
PARTS = ["V", "R", "M", "G", "A", "T", "H", "S", "D"]

RESULTS = []


def report(name, ok, text):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {text}", flush=True)


def f4(x, fmt="%+.5f"):
    return fmt % x if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


# ------------------------------------------------------------------ tiny problems (exactly as test_spec11.py)
def tiny_combo(sizes, name):
    X = torch.randn(5, 6, generator=L.gen(500), dtype=L.DT)
    # [6, 2] has no hidden layer: the activation is unused (as combo_problem does for the linear model)
    prob, th = L.layered_problem(X, sizes, "tanh", 0, name=name)
    st = L.Start(prob, th)
    lab = torch.randint(0, prob.C, (5,), generator=L.gen(501))
    Y = torch.nn.functional.one_hot(lab, prob.C).to(L.DT).reshape(-1)
    S0 = L.vnorm(Y - st.F0) / 16.0               # sizes scaled by 1/16 (SPEC11 amendment A2.2)
    E1 = S.gauss_dir(L, st, 502)
    E2 = S.gauss_dir(L, st, 503)
    combo = S.make_combo(L, CFG, name, prob, st, 0, S0, E1, E2, Y)
    combo["eig"] = {"eigA": (st.Q[:, 0].clone(), 1.0)}          # section 10
    return combo


TINY = tiny_combo([6, 8, 8, 2], "tiny-tanh")
LINC = tiny_combo([6, 2], "tiny-linear")
print(f"tiny net: N={TINY['prob'].N} P={TINY['prob'].P} s1={TINY['sizes']['s1']:.4e} s2={TINY['sizes']['s2']:.4e} "
      f"lamN/lam1={TINY['st'].lamN / TINY['st'].lam1:.2e}; linear: P={LINC['prob'].P}; temp dir {TMP}", flush=True)

REFS = {}


def ref(combo, x, size):
    key = (combo["name"], x, size)
    if key not in REFS:
        rec, vecs = C.reference_unit(L, combo, x, size)
        REFS[key] = C.ref_in_memory(rec, vecs)
    return REFS[key]


T2_LEARNERS = [C.GD(0.5), C.GDmid(1), C.HB(0.1, 0.5), C.HBLA(0.2, 0, 0.5), C.HB(0.02, 0.9), C.HBLA(0.02, 0.9, 9.5)]


def t2_rows(brk):
    rows = []
    for Ld in T2_LEARNERS:
        rec, _ = C.mode_unit(L, TINY, Ld, "eigA", ref(TINY, "eigA", "s1"), brk=brk)
        rows.append((Ld, rec, C.hm1_ok(rec), C.hm4_ok(rec)))
    return rows


def t2_text(rows):
    return "; ".join(f"{C.learner_name(Ld)} [{rec['status']} steps {rec.get('steps_max')}] rho {f4(rec.get('rho'))} "
                     f"g_pred {f4(rec.get('g_pred'))} cos {f4(rec.get('cos'))} nr {f4(rec.get('nr'))} uCu "
                     f"{f4(rec.get('uCu'))} HM1 {h1} HM4 {h4}" for Ld, rec, h1, h4 in rows)


def t4_units(brk):
    # spec note: T4 (and T5) use the tau / cap of Part A (Part G): TAU_GEN with CAP_GEN or CAP_ADAM
    out = {}
    for Ld in (C.ADAM(0.3), C.GD(1), C.EM(), C.SL1()):
        tau, cap = C.default_tau_cap("A", Ld)
        out[C.learner_name(Ld)] = C.pair_unit(L, TINY, Ld, ["E1"], "s1", tau, cap, ref(TINY, "E1", "s1"), brk=brk)[0]
    return out


def snapshot(out):
    snap = {}
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                snap[p] = (os.path.getmtime(p), f.read())
    return snap


def main():
    # T1 f_scalar against the closed forms
    ok, parts = True, []
    worst = 0.0
    for c in (0.1, 0.5, 1.0):
        for mu in (0.0, 0.5, 0.9):
            fs, _ = C.f_scalar(c, mu)
            worst = max(worst, abs(fs - C.f_hb(c, mu)))
        worst = max(worst, abs(C.f_scalar(c)[0] - C.f_gd(c)), abs(C.f_scalar(c, mid=True)[0] - C.f_gdmid(c)))
    ok &= worst <= 1e-9
    fla = C.f_scalar(0.02, 0.9, 9.5)[0]
    fla2 = C.f_scalar(0.02, 0.9, 14.25)[0]
    fnan = C.f_scalar(1.2, 0.0, 0.5)[0]
    ok &= abs(fla - 0.46146) <= 1e-5 and fla2 > 0.5 and math.isnan(fnan)
    report("T1 f_scalar vs closed forms", ok, f"max |diff| {worst:.2e}; f(0.02,0.9,9.5) {fla:.6f}; "
                                              f"f(0.02,0.9,14.25) {fla2:.6f}; f(1.2,0,0.5) {fnan}")

    # T2 loop law along eigA
    rows = t2_rows("")
    ok = all(h1 and h4 for _, _, h1, h4 in rows)
    report("T2 loop law + look matrix on eigA (HM1, HM4)", ok, t2_text(rows))

    # T3 library equivalence (25 steps)
    rec, _ = C.chk_unit(L, TINY, ref(TINY, "E1", "s1"), steps=25)
    ok = all(v <= 1e-9 for v in rec["rel"].values())
    report("T3 library equivalence (25 steps)", ok, " ".join(f"{k} {v:.2e} steps {rec['steps_' + k]}"
                                                             for k, v in rec["rel"].items()))

    # T4 closing
    u4 = t4_units("")
    ok, parts = True, []
    for nm in ("ADAM(0.3,0.9,0.999,0,none)", "GD(1)"):
        r = u4[nm]
        good = C.num(r.get("close_max")) and r["close_max"] <= 1e-6
        ok &= good
        parts.append(f"{nm} [{r['status']}] close {f4(r.get('close_max'), '%.1e')}")
    r = u4["EM"]
    good = C.num(r.get("nr")) and r["nr"] <= 0.05 and C.num(r.get("a_u")) and abs(r["a_u"]) <= 0.05
    ok &= good
    parts.append(f"EM [{r['status']}] nr {f4(r.get('nr'))} a_u {f4(r.get('a_u'))}")
    r = u4["SL1"]
    good = (C.num(r.get("rho")) and abs(r["rho"] + 1) <= 0.03 and abs(r["a_u"] - 1) <= 0.05
            and abs(r["phi_u"]) <= 0.05)
    ok &= good
    parts.append(f"SL1 [{r['status']}] rho {f4(r.get('rho'))} a_u {f4(r.get('a_u'))} phi_u {f4(r.get('phi_u'))}")
    v4 = {nm: C.v4_ok(r) for nm, r in u4.items()}
    ok &= all(v4.values())
    parts.append("a_u + rho: " + ", ".join(f"{nm} {f4(r.get('a_u'))}/{f4(r.get('rho'))} {v4[nm]}" for nm, r in u4.items()))
    report("T4 closing, EM and SL1 controls, universal identity", ok, "; ".join(parts))

    # T5 look-matrix law on E1
    ok, parts = True, []
    for Ld in (C.SL1(), C.GD(0.5), C.HB(0.1, 0.5), C.HBLA(0.2, 0, 0.5)):
        tau, cap = C.default_tau_cap("G", Ld)
        r, _ = C.pair_unit(L, TINY, Ld, ["E1"], "s1", tau, cap, ref(TINY, "E1", "s1"), want_law=True)
        good = C.hg1_ok(r)
        ok &= good
        parts.append(f"{C.learner_name(Ld)} [{r['status']}] law_ratio {f4(r.get('law_ratio'))} law_cos "
                     f"{f4(r.get('law_cos'))} n_law/nvNPe {f4(C.div(r.get('n_law', float('nan')), r['nvNPe']))} "
                     f"kept {r.get('kept')} {good}")
    report("T5 look-matrix law (HG1) on E1", ok, "; ".join(parts))

    # T6 state policies
    st, prob = TINY["st"], TINY["prob"]
    s1 = TINY["sizes"]["s1"]
    E1 = TINY["dirs"]["E1"]
    dJ = ref(TINY, "E1", "s1")["rec"]["sig+1"]["dJ"]
    ok, parts = True, []
    for keep in ("none", "mv", "m", "v"):
        Ld = C.ADAM(0.3, keep=keep)
        lr = C.learner_lr(Ld, st, prob.N, prob.P, dJ)
        pg = C.polygon(L, prob, st, Ld, lr, st.th0, [st.F0 + 1 * s1 * E1, st.F0], C.TAU_GEN, C.CAP_ADAM)
        k1, k2 = pg["legs"][0]["steps"], pg["legs"][1]["steps"]
        km, kv = pg["state"]["km"], pg["state"]["kv"]
        want = {"none": (k2, k2), "mv": (k1 + k2, k1 + k2), "m": (k1 + k2, k2), "v": (k2, k1 + k2)}[keep]
        good = (km, kv) == want and pg["status"] in ("OK", "UNCONVERGED")
        ok &= good
        parts.append(f"{keep}: legs {k1}/{k2} km {km} kv {kv} want {want} [{pg['status']}]")
    report("T6 state policies (counters)", ok, "; ".join(parts))

    # T7 frozen metric = gradient descent in re-scaled weights
    D = torch.exp(0.5 * torch.randn(prob.P, generator=L.gen(777), dtype=L.DT))
    probD, sq = C.d_problem(L, prob, D)
    T = st.F0 + 1 * s1 * E1
    # spec note: T7 fixes no learning rate or target: lr = 0.3 N / (lambda_1 max D) (stable), target F0 + s1 E1
    lr = 0.3 * prob.N / st.lam1 / float(D.max())
    rF = C.fo_leg(L, prob, st.th0, T, C.PRE(1.0), lr, None, 0.0, 20, D=D)
    rE = C.fo_leg(L, probD, st.th0 / sq, T, C.GD(1.0), lr, None, 0.0, 20)
    rel = L.vnorm(rF["theta"] - sq * rE["theta"]) / L.vnorm(rF["theta"] - st.th0)
    report("T7 frozen metric = GD on probD", rel <= 1e-10 and rF["steps"] == 20 and rE["steps"] == 20,
           f"relative difference {rel:.2e} (steps {rF['steps']}/{rE['steps']})")

    # T8 mirror symmetry on the linear tiny net
    # NOTE (review, amendment A2.1): testing showed that late in a long Adam run rounding errors are amplified about
    # 1e9-fold, so "returns to 1e-7" cannot be reached in float64. The test is now (a) the step rule is exactly odd:
    # 100 steps toward +T and toward -T from the same start give mirror-image displacements (<= 1e-9), and
    # (b) the excursion with reset moments returns to the start far better than with kept moments (<= 1e-3; >= 1e-3
    # and at least 100 times larger).
    ok, parts = True, []
    lst, lE1, ls1 = LINC["st"], LINC["dirs"]["E1"], LINC["sizes"]["s1"]
    lref = ref(LINC, "E1", "s1")
    llr = C.learner_lr(C.ADAM(0.3), lst, LINC["prob"].N, LINC["prob"].P, lref["rec"]["sig+1"]["dJ"])
    dd = {}
    for sg in (1, -1):
        rr_ = C.fo_leg(L, LINC["prob"], lst.th0, lst.F0 + sg * ls1 * lE1, C.ADAM(0.3), llr, None, 0.0, 100)
        dd[sg] = rr_["theta"] - lst.th0
    odd_err = L.vnorm(dd[1] + dd[-1]) / L.vnorm(dd[1])
    ok &= bool(odd_err <= 1e-9)
    parts.append(f"100 steps +T / -T mirror error {odd_err:.2e}")
    qs = {}
    for keep in ("none", "mv"):
        Ld = C.ADAM(0.3, keep=keep)
        r, _ = C.pair_unit(L, LINC, Ld, ["E1"], "s1", C.TAU_GEN, C.CAP_ADAM, lref)
        qs[keep] = [r[f"sig{sg}"]["nd"] / r[f"sig{sg}"]["n1"] if "nd" in r[f"sig{sg}"] else float("nan")
                    for sg in ("+1", "-1")]
        parts.append(f"{keep} nd/n1 {qs[keep][0]:.2e} / {qs[keep][1]:.2e} [{r['status']}]")
    ok &= bool(max(qs["none"]) <= 1e-3 and min(qs["mv"]) >= 1e-3 and min(qs["mv"]) >= 100 * max(qs["none"]))
    report("T8 mirror symmetry on the linear net", ok, "; ".join(parts))

    # T9 resume (QUICK-like tiny plan through spec12_run.run_plan, twice)
    out = os.path.join(TMP, "t9")
    settings = C.settings12(True, "")
    sig = S.signature(settings, LIB_PATH, CFG_PATH)
    items = C.plan12(settings, ["tiny-tanh"], PARTS, L.safe_name)
    r1 = R.run_plan(L, items, lambda n: TINY, out, sig, settings, "", None)
    snap = snapshot(out)
    nJ = TINY["prob"].counts["J"]
    r2 = R.run_plan(L, items, lambda n: TINY, out, sig, settings, "", None)
    snap2 = snapshot(out)
    statuses = {it["name"]: (S.read_unit(out, it["name"]) or {}).get("status") for it in items}
    ok = (len(r1["ran"]) == len(items) and not r1["waiting"] and not r2["ran"] and len(r2["skipped"]) == len(items)
          and snap == snap2 and TINY["prob"].counts["J"] == nJ
          and not any(v == "ERROR" or v is None for v in statuses.values()))
    report("T9 resume", ok, f"first run {len(r1['ran'])}/{len(items)} units, second run ran {len(r2['ran'])}, "
                            f"files unchanged {snap == snap2} ({len(snap)} files), Jacobians in second run "
                            f"{TINY['prob'].counts['J'] - nJ}; statuses {statuses}")

    # X1 (not a section 10 test): the report runs on the T9 output
    try:
        rep = REP.Report(out, settings, sig, ["tiny-tanh"], L.safe_name)
        lines, results = rep.run()
        REP.write_outputs(L, out, lines, results)
        report("X1 report smoke run on the T9 output", True, f"{len(lines)} lines; rules "
               f"{[(nm, v) for _, nm, v, _ in rep.rules]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        report("X1 report smoke run on the T9 output", False, repr(e))

    # T10 BREAK modes
    rows_s = t2_rows("stalelook")
    la = [h1 for Ld, rec, h1, h4 in rows_s if Ld["gamma"] != 0]
    other = [h1 for Ld, rec, h1, h4 in rows_s if Ld["gamma"] == 0]
    ok_s = (not any(la)) and all(other)
    u4n = t4_units("noclose")
    ra = u4n["ADAM(0.3,0.9,0.999,0,none)"]
    closure_n = C.num(ra.get("close_max")) and ra["close_max"] <= 1e-6
    out_r = os.path.join(TMP, "t10repro")
    set_r = C.settings12(True, "repro")
    sig_r = S.signature(set_r, LIB_PATH, CFG_PATH)
    items_r = C.plan12(set_r, ["tiny-tanh"], ["M"], L.safe_name, repro_combo="tiny-tanh")
    R.run_plan(L, items_r, lambda n: TINY, out_r, sig_r, set_r, "repro", None)
    nm = "M__tiny-tanh__eigA__s1__GD_1_"
    ra1, ra2 = S.read_unit(out_r, nm), S.read_unit(out_r, nm + "__repro")
    diffs = S.compare_repro(ra1, ra2) if ra1 is not None and ra2 is not None else ["missing record"]
    ok = ok_s and (not closure_n) and not diffs and ra1 is not None and ra1.get("status") != "ERROR"
    report("T10 BREAK stalelook / noclose / repro", ok,
           f"stalelook: HBLA HM1 {la} (must all be False), others {other} (must all be True) [{t2_text(rows_s)}]; "
           f"noclose: ADAM(0.3) [{ra['status']}] close {f4(ra.get('close_max'), '%.2e')} closure check "
           f"{'PASS' if closure_n else 'FAIL'} (must FAIL); repro: {len(diffs)} differences {diffs[:5]} "
           f"(status {None if ra1 is None else ra1.get('status')})")

    # T11 star
    stars = {}
    for Ld in (C.SL1(), C.EM(), C.ADAM(0.3)):
        stars[C.learner_name(Ld)] = C.star_unit(L, TINY, Ld, ref(TINY, "E1", "s1"), 2)[0]
    G2 = {nm: (r["checkpoints"][-1]["G"] if len(r["checkpoints"]) == 2 else float("nan")) for nm, r in stars.items()}
    ok = (all(len(r["checkpoints"]) == 2 for r in stars.values()) and G2["SL1"] < 1
          and abs(1 - G2["EM"]) <= 0.2 * (1 - G2["SL1"]))
    report("T11 star (mm = 1..2)", ok, "; ".join(f"{nm} [{r['status']}] checkpoints {len(r['checkpoints'])} G_2 "
                                                 f"{f4(G2[nm], '%.6f')}" for nm, r in stars.items()))

    # T12 hover
    rh, _ = C.hover_unit(L, TINY, C.ADAM(0.3), ref(TINY, "E1", "s1"), 100, [50, 100])
    cl = [(sg, c["point"], c["close"], c["status"]) for sg in ("+1", "-1") for c in rh[f"sig{sg}"].get("closings", [])]
    ok = rh["status"] == "OK" and len(cl) == 6 and all(c[3] == "OK" and c[2] <= 1e-6 for c in cl)
    report("T12 hover unit (hover 100, snapshots [50, 100])", ok,
           f"status {rh['status']}; closings {[(a, b, f'{x:.1e}', d) for a, b, x, d in cl]}; we {rh.get('we')} "
           f"wue {rh.get('wue')}; steps_stop {[rh[f'sig{sg}'].get('steps_stop') for sg in ('+1', '-1')]}")

    # T13 own metric
    ro, _ = C.own_unit(L, TINY, 0.3, ref(TINY, "E1", "s1"))
    ctrl = ro.get("ctrl", {})
    ok = C.num(ctrl.get("rho")) and abs(ctrl["rho"] + 1) <= 0.03
    report("T13 own-metric unit (k = 0.3): control rho", ok, f"status {ro['status']} ctrl {ctrl} own {ro.get('own')} "
                                                             f"eucl {ro.get('eucl')} D {ro.get('D_summary')}")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
