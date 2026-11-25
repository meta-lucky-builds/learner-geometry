"""SPEC11 run (section 8): computes and stores every unit; resumable by signature.

Environment: SPEC11_BUNDLE (default: this script's directory), SPEC11_OUT (default ./spec11_out), SPEC11_DEVICE
(default cuda:0 if available else cpu), SPEC11_THREADS, SPEC11_PARTS (default V,R,A,B,C), SPEC11_COMBOS
(default tanh-init-s0,gelu-init-s0), SPEC11_QUICK ("1" = smoke run), SPEC11_BREAK (negative controls; default "").
"""
import os
import sys
import math
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                      # noqa: E402
import spec11_common as S         # noqa: E402


def _refs_loaded(L, out, sig, c, size, names):
    """{name: dict(p, m, usable_p, usable_m)} for the references of combination c at this size that are done (both signs)."""
    res = {}
    for x in names:
        recs = [S.read_unit(out, S.ref_name(c, x, size, sg)) for sg in S.SIGNS]
        if not all(S.unit_done(out, S.ref_name(c, x, size, sg), sig) for sg in S.SIGNS):
            continue
        if not all(any(fn.endswith("__vNP.pt") for fn in r.get("vec_files", [])) for r in recs):
            continue                                    # a reference stored without a vector (ERROR)
        res[x] = dict(p=S.load_vec(L, out, S.ref_name(c, x, size, 1), "vNP"),
                      m=S.load_vec(L, out, S.ref_name(c, x, size, -1), "vNP"),
                      usable_p=S.ref_usable(recs[0]), usable_m=S.ref_usable(recs[1]))
    return res


def make_fn(L, it, combo, out, sig, settings, brk, ref11):
    """The function that computes one planned unit, or (None, reason) if it must wait for references."""
    kind = it["kind"]
    if kind == "env":
        ref = None if ref11 is None else ref11.get(it["combo"])
        return (lambda: S.env_unit(L, combo, ref, it["combo"] != S.LIN_NAME)), ""
    if kind == "lin":
        Ld = S.make_learner(L, it["learner"])
        return (lambda: S.linear_unit(L, combo, Ld, brk)), ""
    if kind == "ref":
        return (lambda: S.reference_unit(L, combo, it["dir"], it["size"], it["sign"])), ""
    if kind == "gram":
        need = [S.ref_name(it["combo"], x, size, sg) for size in ("s1", "s2") for x in settings["REFS"][size]
                for sg in S.SIGNS]
        if not all(S.unit_done(out, n, sig) for n in need):
            return None, "waiting for references"
        refs = {size: _refs_loaded(L, out, sig, it["combo"], size, settings["REFS"][size]) for size in ("s1", "s2")}
        return (lambda: S.gram_unit(it["combo"], combo["sizes"], refs)), ""
    if kind == "A":
        refs = _refs_loaded(L, out, sig, it["combo"], it["size"], [it["dir"]])
        if it["dir"] not in refs:
            return None, "waiting for references"
        r = refs[it["dir"]]
        Ld = S.make_learner(L, it["learner"])
        return (lambda: S.part_a_unit(L, combo, Ld, it["dir"], it["size"], r["p"], r["m"],
                                      r["usable_p"] and r["usable_m"], brk)), ""
    if kind == "ID":
        evens = {}
        usable = {}
        for size in settings["A_SIZES"]:
            refs = _refs_loaded(L, out, sig, it["combo"], size, [it["dir"]])
            if it["dir"] not in refs:
                return None, "waiting for references"
            r = refs[it["dir"]]
            evens[size] = S.even(r["p"], r["m"])
            usable[size] = bool(r["usable_p"] and r["usable_m"])

        def fn():
            rec, v = S.identity_unit(L, combo, it["dir"], evens)
            rec["refs_usable"] = usable
            return rec, v
        return fn, ""
    if kind == "B":
        names = settings["REFS"]["s1"]
        if not all(S.unit_done(out, S.ref_name(it["combo"], x, "s1", sg), sig) for x in names for sg in S.SIGNS):
            return None, "waiting for references"
        refs = _refs_loaded(L, out, sig, it["combo"], "s1", names)
        evens = {x: S.even(r["p"], r["m"]) for x, r in refs.items()}
        Ld = S.make_learner(L, it["learner"])
        return (lambda: S.part_b_unit(L, combo, Ld, settings["B_M"], evens, brk)), ""
    if kind == "C":
        Ld = S.make_learner(L, it["learner"])
        return (lambda: S.part_c_unit(L, combo, Ld, settings["C_TASKS"], settings["C_FINAL"])), ""
    raise ValueError(kind)


def summary(rec):
    """One line of key numbers for a finished unit."""
    st = rec.get("status")
    k = rec.get("kind", rec.get("part"))
    if st == "ERROR":
        return f"ERROR {rec.get('error')}"
    if rec.get("part") == "R" and k != "gram":
        return f"{st} |vNP|={rec.get('nvNP', float('nan')):.3e} np_it={rec.get('np_it')} jump={rec.get('jump_status')}"
    if k == "gram":
        return f"{st} names s1={len(rec['names_s1'])} s2={len(rec['names_s2'])} sign agreement {rec['sign_agreement']}"
    if k == "env":
        return f"{st} pass={rec.get('pass')} rel={rec.get('rel')}"
    if k == "lin":
        return f"{st} nd/n1={rec['nd_over_n1']:.2e}"
    if k == "ID":
        return f"{st} Dq={rec['Dq']:.4e} iota={rec['iota']}"
    if rec.get("part") == "A":
        if st != "OK":
            return f"{st} (f={rec['f']:.4f})"
        return (f"OK f={rec['f']:+.4f} rho_loop={rec['rho_loop']:+.4f} cos={rec['cos_loop']:+.4f} "
                f"rho_leg={rec['rho_leg']:+.4f} dq_u={rec['dq']['u']:+.4e}")
    if rec.get("part") == "B":
        return f"{st} f={rec['f']:+.4f} checkpoints={rec['m_done']}/{rec['n_m']}"
    if rec.get("part") == "C":
        fin = rec.get("final", {})
        return f"{st} f={rec['f']:+.4f} tasks={rec['tasks_done']}/{rec['n_tasks']} final={fin.get('status')} " \
               f"test_acc={fin.get('test_acc')}"
    return str(st)


def run_plan(L, items, get_combo, out, sig, settings, brk, ref11):
    """Runs every planned unit not yet done. get_combo(name) builds (or returns) a combination dict (spec11_common
    make_combo). Returns dict(ran=[names], skipped=[names], waiting=[names])."""
    done = dict(ran=[], skipped=[], waiting=[])
    current = dict(name=None, combo=None)
    for it in items:
        name = it["name"]
        if S.unit_done(out, name, sig):
            done["skipped"].append(name)
            continue
        try:
            if current["name"] != it["combo"]:
                current["combo"] = None
                L.free_memory()
                current["combo"] = get_combo(it["combo"])
                current["name"] = it["combo"]
            fn, why = make_fn(L, it, current["combo"], out, sig, settings, brk, ref11)
            if fn is None:
                print(f"{name}: NOT RUN ({why})", flush=True)
                done["waiting"].append(name)
                continue
            rec = S.run_unit(L, out, name, sig, fn)
            print(f"{name}: {summary(rec)} [{rec['seconds']:.1f}s]", flush=True)
            done["ran"].append(name)
        except Exception as e:                            # e.g. the combination could not be built, or a save failed
            print(f"{name}: FAILED OUTSIDE THE UNIT ({e!r}); nothing stored", flush=True)
            done["waiting"].append(name)
    return done


def main():
    bundle = os.environ.get("SPEC11_BUNDLE") or HERE
    out = os.environ.get("SPEC11_OUT") or "./spec11_out"
    device = os.environ.get("SPEC11_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    threads = os.environ.get("SPEC11_THREADS")
    threads = int(threads) if threads else None
    parts = [p.strip() for p in os.environ.get("SPEC11_PARTS", "V,R,A,B,C").split(",") if p.strip()]
    combos = [c.strip() for c in os.environ.get("SPEC11_COMBOS", "tanh-init-s0,gelu-init-s0").split(",") if c.strip()]
    qv = os.environ.get("SPEC11_QUICK", "")
    if qv not in ("", "0", "1"):
        raise ValueError(f"SPEC11_QUICK={qv!r}: use 1 or leave unset")
    quick = qv == "1"
    brk = S.check_break(os.environ.get("SPEC11_BREAK", ""))
    assert set(parts) <= {"V", "R", "A", "B", "C"}, parts
    assert set(combos) <= set(S.MAIN_COMBOS), combos
    if quick:
        combos = ["tanh-init-s0"]
    S.break_banner(brk)
    L, CFG = S.load(bundle, device, threads, out)
    settings = S.science_settings(quick, brk, CFG)
    sig = S.signature(settings, os.path.join(bundle, "part9lib.py"), os.path.join(bundle, "cfg11.json"))
    p11 = os.path.join(bundle, "ref11.json")
    ref11 = S.read_json(p11) if os.path.exists(p11) else None
    print(f"SPEC11 run: bundle={bundle} out={out} device={device} threads={torch.get_num_threads()} parts={parts} "
          f"combos={combos} quick={quick} break={brk!r} signature={sig}", flush=True)
    items = S.plan(L, settings, combos, parts)
    run_dir = os.path.join(bundle, "run")

    def get_combo(name):
        ctx = L.Ctx9(S.COMBOS[name], run_dir)
        return S.combo_from_ctx(L, CFG, ctx)

    t0 = time.time()
    res = run_plan(L, items, get_combo, out, sig, settings, brk, ref11)
    # second pass: units that waited for references made in the same run (e.g. G written after the last reference)
    if res["waiting"]:
        res = run_plan(L, items, get_combo, out, sig, settings, brk, ref11)
    errors = [it["name"] for it in items if (S.read_unit(out, it["name"]) or {}).get("status") == "ERROR"]
    print(f"ran {len(res['ran'])}, already done {len(res['skipped'])}, not run {len(res['waiting'])}, "
          f"stored with status ERROR {len(errors)} {errors} (delete their json files to retry); "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if all(S.unit_done(out, it["name"], sig) for it in items):
        print("SPEC11 RUN COMPLETE", flush=True)
    S.break_banner(brk)


if __name__ == "__main__":
    main()
