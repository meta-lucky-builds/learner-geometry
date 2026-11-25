"""SPEC11 section 12 tests on a tiny network (CPU, 2 threads). Exit code 1 on any failure.

Uses part9lib read-only (from the bundle if it exists, else from $TCR_DATA/keep) with a CFG taken from
bundle/cfg11.json if it exists, else built from the notebook's CONFIG cell exactly as loc.py does. Every test calls the
spec11_common / spec11_run functions the real run uses. Writes only into a temporary directory.
"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import os
import sys
import json
import math
import shutil
import tempfile

sys.dont_write_bytecode = True                    # never write __pycache__ next to the read-only library
HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = DATA + "/keep"
BUNDLE = os.path.join(HERE, "bundle")
LIBDIR = BUNDLE if os.path.exists(os.path.join(BUNDLE, "part9lib.py")) else KEEP
sys.path.insert(0, HERE)
sys.path.insert(1, LIBDIR)

import torch                                      # noqa: E402
torch.set_num_threads(2)
import spec11_common as S                         # noqa: E402
import spec11_run as R                            # noqa: E402
import spec11_report as REP                       # noqa: E402

TMP = tempfile.mkdtemp(prefix="spec11_tests_")
if os.path.exists(os.path.join(BUNDLE, "cfg11.json")):
    CFG_PATH = os.path.join(BUNDLE, "cfg11.json")
else:
    CFG_PATH = os.path.join(TMP, "cfg11.json")
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(S.config_from_notebook(KEEP + "/evaluation_point_tests.ipynb"), f, sort_keys=True)
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)
CFG["MNIST_DOWNLOAD"] = False
CFG["RUN_DIR"] = TMP                              # the library's progress.log goes to the temp dir
import part9lib as L                              # noqa: E402
L.configure(CFG)
L.set_device("cpu")
LIB_PATH = os.path.join(LIBDIR, "part9lib.py")

RESULTS = []


def report(name, ok, text):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {text}", flush=True)


# ------------------------------------------------------------------ tiny problems
def tiny_combo(sizes, name):
    X = torch.randn(5, 6, generator=L.gen(500), dtype=L.DT)
    # [6, 2] has no hidden layer: the activation is unused (as combo_problem does for the linear model)
    prob, th = L.layered_problem(X, sizes, "tanh", 0, name=name)
    st = L.Start(prob, th)
    lab = torch.randint(0, prob.C, (5,), generator=L.gen(501))
    Y = torch.nn.functional.one_hot(lab, prob.C).to(L.DT).reshape(-1)
    # NOTE (review, A2.2): the tiny net is far more curved than the MNIST net; at 0.02 x |onehot - F0| one excursion
    # changes q by 36% (outside second order). Sizes are scaled by 1/16 so the tests check the code, not the regime.
    S0 = L.vnorm(Y - st.F0) / 16.0               # s1 = 0.02 S0, s2 = 0.01 S0 (CFG S1_FRAC, S2_FRAC)
    E1 = S.gauss_dir(L, st, 502)                 # unit directions: Gaussian
    E2 = S.gauss_dir(L, st, 503)
    return S.make_combo(L, CFG, name, prob, st, 0, S0, E1, E2, Y)


TINY = tiny_combo([6, 8, 8, 2], "tiny-tanh")
LINC = tiny_combo([6, 2], "tiny-linear")
print(f"tiny net: N={TINY['prob'].N} P={TINY['prob'].P} s1={TINY['sizes']['s1']:.4e} s2={TINY['sizes']['s2']:.4e} "
      f"lamN/lam1={TINY['st'].lamN / TINY['st'].lam1:.2e}; linear: P={LINC['prob'].P}", flush=True)

REFS = {}                                        # (name, size) -> dict(p, m, usable)


def refs(x, size):
    if (x, size) not in REFS:
        out = {}
        for sg in S.SIGNS:
            rec, v = S.reference_unit(L, TINY, x, size, sg)
            out[sg] = (v["vNP"], S.ref_usable(rec))
        REFS[(x, size)] = dict(p=out[1][0], m=out[-1][0], usable=out[1][1] and out[-1][1])
    return REFS[(x, size)]


def a_run(spec, size, brk="", u="E1"):
    r = refs(u, size)
    rec, vecs = S.part_a_unit(L, TINY, S.make_learner(L, spec), u, size, r["p"], r["m"], r["usable"], brk)
    return rec, vecs


LOOP = [("DJ", 1.0), ("SL", 2), ("SL", 8), ("AJ", 0.5), ("AJ", 1.0), ("DJ", 1.5)]


def loop_check(results, brk):
    """T2 rule on stored Part A records: |rho_x - (2f-1)| <= 0.03 max(1,|g|), f as the report uses it."""
    all_ok, parts = True, []
    for spec in LOOP:
        r1, r2 = results[(spec, "s1")], results[(spec, "s2")]
        Ld = S.make_learner(L, spec)
        g = 2 * S.report_f(L, Ld, brk) - 1
        if r1["status"] != "OK" or r2["status"] != "OK":
            all_ok = False
            parts.append(f"{L.learner_name9(Ld)} status {r1['status']}/{r2['status']}")
            continue
        rx = S.extrap(r1["rho_loop"], r2["rho_loop"])
        ok = S.within(rx, g, 0.03)
        all_ok &= ok
        parts.append(f"{L.learner_name9(Ld)} rho_x {rx:+.4f} g {g:+.4f} cos(s2) {r2['cos_loop']:+.4f} {'ok' if ok else 'X'}")
    return all_ok, "; ".join(parts)


def main():
    # T1
    want = {"DJ(1)": 0.0, "DJ(1.5)": -1.0, "DJ(0.5)": 1 / 3, "SL(2)": 0.25, "SL(8)": 0.4375, "AJ(0.5)": 0.5,
            "AJ(1)": 1.0, "AJ(1.5)": 1.5, "SLa(2,1)": 0.75, "SL(4)": 0.375, "SLa(8,1)": 0.5625, "TJ(1.5,1)": 5 / 7}
    ok, parts = True, []
    for spec in S.LA_FULL + S.LB_FULL + S.LC_FULL:
        Ld = S.make_learner(L, spec)
        nm, f = L.learner_name9(Ld), L.predict_f9(Ld)
        good = nm in want and abs(f - want[nm]) <= 1e-12
        ok &= good
        parts.append(f"{nm}={f:.6g}")
    report("T1 predict_f9 of LA, LB, LC", ok, " ".join(sorted(set(parts))))

    # T2 loop law
    res = {}
    for spec in LOOP:
        for size in ("s1", "s2"):
            res[(spec, size)] = a_run(spec, size)[0]
    ok, text = loop_check(res, "")
    ru = all(refs("E1", sz)["usable"] for sz in ("s1", "s2"))
    report("T2 loop law (rho_x within 0.03 max(1,|g|) of 2f-1)", ok and ru, text + f"; references usable {ru}")

    # T3 identity
    ev = {sz: S.even(refs("E1", sz)["p"], refs("E1", sz)["m"]) for sz in ("s1", "s2")}
    idr, _ = S.identity_unit(L, TINY, "E1", ev)
    ix = S.extrap(idr["iota"]["s1"], idr["iota"]["s2"])
    report("T3 identity iota_x within 0.03 of 1", abs(ix - 1) <= 0.03,
           f"iota(s1) {idr['iota']['s1']:+.5f} iota(s2) {idr['iota']['s2']:+.5f} iota_x {ix:+.5f}")

    # T4 own cost
    ok, parts = True, []
    for spec in (("DJ", 1.0), ("AJ", 1.0)):
        f = L.predict_f9(S.make_learner(L, spec))
        ks = {sz: S.kappa(res[(spec, sz)]["dq"]["u"], float(ev[sz] @ ev[sz]), TINY["sizes"][sz]) for sz in ("s1", "s2")}
        kx = S.extrap(ks["s1"], ks["s2"])
        good = S.within(kx, 1 - 2 * f, 0.05)
        ok &= good
        parts.append(f"{spec}: kappa(s1) {ks['s1']:+.4f} kappa(s2) {ks['s2']:+.4f} kappa_x {kx:+.4f} vs {1 - 2 * f:+.4f}")
    report("T4 own cost kappa_x within 0.05 max(1,|1-2f|) of 1-2f", ok, "; ".join(parts))

    # T5 cross cost, direction W1
    ew = S.even(refs("W1", "s1")["p"], refs("W1", "s1")["m"])
    eu = ev["s1"]
    cs = L.cosine(ew, eu)
    if abs(cs) < 0.1:
        report("T5 cross cost (W1)", True, f"SKIPPED: |cos(vNPe_W1, vNPe_E1)| = {abs(cs):.3f} < 0.1")
    else:
        ok, parts = True, []
        s1 = TINY["sizes"]["s1"]
        for spec in (("DJ", 1.0), ("AJ", 1.0)):
            f = L.predict_f9(S.make_learner(L, spec))
            pred = (1 - 2 * f) * 2 * float(ew @ eu) / s1 ** 2
            meas = res[(spec, "s1")]["dq"]["W1"]
            good = abs(meas - pred) <= 0.1 * abs(pred)
            ok &= good
            parts.append(f"{spec}: dq_W1 {meas:+.5e} pred {pred:+.5e} ratio {meas / pred:+.4f}")
        report("T5 cross cost within 10%", ok, f"cos {cs:+.3f}; " + "; ".join(parts))

    # T6 linear model
    ok, parts = True, []
    for spec in S.LIN_LEARNERS:
        rec, _ = S.linear_unit(L, LINC, S.make_learner(L, spec), "")
        good = rec["status"] == "OK" and rec["nd_over_n1"] <= 1e-8
        ok &= good
        parts.append(f"{rec['learner']} nd/n1 {rec['nd_over_n1']:.2e} ({rec['status']})")
    report("T6 linear control nd/n1 <= 1e-8", ok, "; ".join(parts))

    # T7 resume (QUICK-like tiny run through spec11_run.run_plan, twice)
    out = os.path.join(TMP, "t7")
    settings = S.science_settings(True, "", CFG)
    sig = S.signature(settings, LIB_PATH, CFG_PATH)
    items = S.plan(L, settings, ["tiny-tanh"], ["R", "A", "B", "C"])
    r1 = R.run_plan(L, items, lambda n: TINY, out, sig, settings, "", None)
    snap = {}
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn)
            snap[p] = (os.path.getmtime(p), open(p, "rb").read())
    nJ = TINY["prob"].counts["J"]
    r2 = R.run_plan(L, items, lambda n: TINY, out, sig, settings, "", None)
    snap2 = {}
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn)
            snap2[p] = (os.path.getmtime(p), open(p, "rb").read())
    statuses = {it["name"]: S.read_unit(out, it["name"])["status"] for it in items}
    # T7 is about resume only; unit statuses are printed (an ERROR unit must also be skipped, literal section 8)
    ok = (len(r1["ran"]) == len(items) and not r2["ran"] and len(r2["skipped"]) == len(items)
          and snap == snap2 and TINY["prob"].counts["J"] == nJ)
    not_ok = {n: v for n, v in statuses.items() if v != "OK"}
    report("T7 resume", ok, f"first run {len(r1['ran'])}/{len(items)} units, second run ran {len(r2['ran'])}, "
                            f"files unchanged {snap == snap2} ({len(snap)} files), Jacobians in second run "
                            f"{TINY['prob'].counts['J'] - nJ}; units not OK: {not_ok}")

    # X1 (not a section 12 test): the report runs on the T7 output
    try:
        lines, results = REP.Report(L, out, settings, sig, ["tiny-tanh"], ["R", "A", "B", "C"], "").run()
        REP.write_outputs(out, lines, results)
        report("X1 report smoke run on the T7 output", True, f"{len(lines)} lines; decision {results['decision_input']}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        report("X1 report smoke run on the T7 output", False, repr(e))

    # T8 BREAK modes
    resn = {}
    for spec in LOOP:
        for size in ("s1", "s2"):
            resn[(spec, size)] = a_run(spec, size, brk="noreturn")[0]
    okn, textn = loop_check(resn, "")
    oks, texts = loop_check(res, "swap")
    ra, va = a_run(("SL", 2), "s1")
    rb, vb = a_run(("SL", 2), "s1")
    diffs = S.compare_repro(ra, rb)
    same_vec = set(va) == set(vb) and all(torch.equal(va[k], vb[k]) for k in va)
    report("T8 BREAK noreturn/swap fail the loop-law check, repro bit-identical",
           (not okn) and (not oks) and not diffs and same_vec,
           f"noreturn check {'PASS' if okn else 'FAIL'} [{textn}]; swap check {'PASS' if oks else 'FAIL'} [{texts}]; "
           f"repro: {len(diffs)} scalar differences, vectors identical {same_vec}")

    # T9 leg-2 target is the original F0 object (spy on L.run_learner9), closure
    calls = []
    orig = L.run_learner9

    def spy(prob, st, t, s, Ld):
        calls.append((st, t))
        return orig(prob, st, t, s, Ld)

    ok, parts = True, []
    st0 = TINY["st"]
    try:
        L.run_learner9 = spy
        for spec in (("SL", 2), ("AJ", 1.0)):
            calls.clear()
            rec, vec = S.excursion(L, TINY["prob"], st0, S.make_learner(L, spec), TINY["dirs"]["E1"],
                                   TINY["sizes"]["s1"], 1, "")
            leg2_st, leg2_t = calls[1]
            # leg 2 starts from a Start at the learner's own end point th1: V0(th1 - th0) computed from it equals v1
            good = (len(calls) == 2 and calls[0][0] is st0 and leg2_t is st0.F0 and leg2_st is not st0
                    and torch.equal(st0.V0(leg2_st.th0 - st0.th0), vec["v1"])
                    and rec["close"] <= 1e-6 and rec["status"] == "OK")
            ok &= good
            parts.append(f"{spec}: leg-2 target is st.F0 {leg2_t is st0.F0}, close {rec['close']:.1e}")
        # Part B: leg-1 from a NEW Start at the current weights, target from the original F0; leg 2 to st.F0 itself
        calls.clear()
        evens = {x: S.even(refs(x, "s1")["p"], refs(x, "s1")["m"]) for x in ("E1",)}
        recb, _ = S.part_b_unit(L, TINY, S.make_learner(L, ("DJ", 1.0)), 1, evens, "")
        s1 = TINY["sizes"]["s1"]
        E1 = TINY["dirs"]["E1"]
        gb = (len(calls) == 4 and recb["status"] == "OK"
              and calls[0][0] is not st0 and torch.equal(calls[0][0].th0, st0.th0)
              and torch.equal(calls[0][1], st0.F0 + 1 * s1 * E1) and calls[1][1] is st0.F0
              and calls[2][0] is not st0 and calls[2][0] is not calls[0][0]
              and not torch.equal(calls[2][0].th0, st0.th0)          # second excursion starts where the first ended
              and torch.equal(calls[2][1], st0.F0 + (-1) * s1 * E1)  # built from the ORIGINAL F0, not stc.F0
              and not torch.equal(calls[2][0].F0, st0.F0)
              and calls[3][1] is st0.F0)
        ok &= gb
        parts.append(f"Part B call pattern ok {gb}, close {recb['checkpoints'][0]['close']:.1e}")
    finally:
        L.run_learner9 = orig
    report("T9 leg-2 target is the original F0 object; closure <= 1e-6", ok, "; ".join(parts))

    # T10 star additivity
    names = ["E1", "E2"]
    evens = {x: S.even(refs(x, "s1")["p"], refs(x, "s1")["m"]) for x in names}
    recb, _ = S.part_b_unit(L, TINY, S.make_learner(L, ("DJ", 1.0)), 2, evens, "")
    G = [[float(evens[a] @ evens[b]) for b in names] for a in names]
    cp = recb["checkpoints"][-1] if recb["checkpoints"] else None
    if recb["status"] == "OK" and cp is not None and cp["m"] == 2:
        rho, cos, nr, _ = S.star_rho(cp["dots"], cp["nvc"], G, names, 2)
        report("T10 star additivity (+E1,-E1,+E2,-E2, DJ(1)): rho_B within 0.1 of -1", abs(rho + 1) <= 0.1,
               f"rho_B {rho:+.4f} cos {cos:+.4f}")
    else:
        report("T10 star additivity", False, f"status {recb['status']}")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
