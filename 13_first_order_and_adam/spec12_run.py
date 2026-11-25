"""SPEC12 run (section 8): computes and stores every unit; resumable by signature.

Usage: python spec12_run.py --out D:/some/dir [--bundle ...] [--combos ...] [--parts V,R,M,G,A,T,H,S,D] [--threads 4]
       [--device cpu] [--quick] [--break stalelook|noclose|repro] [--only SUBSTRING] [--list]
"""
import os
import sys
import time
import math
import argparse
import importlib

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                       # noqa: E402
import spec12_common as C          # noqa: E402
S = C.S

PARTS = ["V", "R", "M", "G", "A", "T", "H", "S", "D"]
MAIN_COMBOS = ["tanh-init-s0", "gelu-init-s0"]


def get_combo_rig(L, CFG, bundle, cname):
    """Section 3 loading of one combination."""
    ctx = L.Ctx9(S.COMBOS[cname], bundle + "/run")
    combo = S.combo_from_ctx(L, CFG, ctx)
    combo["eig"] = {nm: (ctx.dirs[nm], float(ctx.dinfo[nm]["lam_ratio"])) for nm in ("eigA", "eigB", "eigC")}
    return combo


def make_fn(L, it, combo, refs, settings, brk, ref11):
    """The function computing one planned unit; refs[(x, size)] = load_ref(...) of every reference it needs."""
    kind = it["kind"]
    meta = C.item_meta(it)
    if kind == "env":
        ref = None if ref11 is None else ref11.get(it["combo"])
        return lambda: S.env_unit(L, combo, ref, with_test=False)
    if kind == "chk":
        return lambda: C.chk_unit(L, combo, refs[("E1", "s1")], meta=meta)
    if kind == "ref":
        return lambda: C.reference_unit(L, combo, it["dir"], it["size"])
    if kind == "mode":
        return lambda: C.mode_unit(L, combo, it["Ld"], it["dirs"][0], refs[(it["dirs"][0], "s1")], brk=brk,
                                   meta=meta)
    if kind == "pair":
        return lambda: C.pair_unit(L, combo, it["Ld"], it["dirs"], it["size"], it["tau"], it["cap"],
                                   refs[(it["dirs"][0], it["size"])], brk=brk, want_law=it["want_law"],
                                   want_uCu=it["want_uCu"], meta=meta)
    if kind == "hover":
        return lambda: C.hover_unit(L, combo, it["Ld"], refs[("E1", "s1")], settings["HOVER"], settings["HS"],
                                    brk=brk, meta=meta)
    if kind == "star":
        return lambda: C.star_unit(L, combo, it["Ld"], refs[("E1", "s1")], settings["N_MM"], brk=brk, meta=meta)
    if kind == "op0":
        return lambda: C.op0_unit(L, combo, refs[("E1", "s1")], meta=meta)
    if kind == "own":
        return lambda: C.own_unit(L, combo, it["kappa"], refs[("E1", "s1")], brk=brk, meta=meta)
    raise ValueError(kind)


def _f(x, fmt="%+.4f"):
    return fmt % x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else str(x)


def summary(rec):
    """One line of key numbers for a finished unit."""
    st = rec.get("status")
    k = rec.get("kind")
    if st == "ERROR":
        return f"ERROR {rec.get('error')}"
    if rec.get("part") == "R" or k == "ref":
        return f"{st} nvNPe={_f(rec.get('nvNPe'), '%.4e')} Muu={_f(rec.get('Muu'), '%.4e')} " \
               f"agree={_f(rec.get('agree'))} dJ(+1)={_f(rec.get('sig+1', {}).get('dJ'), '%.4e')}"
    if k == "env":
        return f"{st} pass={rec.get('pass')} rel={rec.get('rel')}"
    if k == "chk":
        return f"{st} rel={ {n: '%.2e' % v for n, v in rec['rel'].items()} }"
    if k == "pair":
        s = f"{st} steps={rec.get('steps_max')} res={_f(rec.get('res_max'), '%.1e')} " \
            f"close_w={_f(rec.get('close_w_max'), '%.1e')}"
        if "g_pred" in rec:
            s += f" g_pred={_f(rec['g_pred'])}"
        for key in ("rho", "cos", "a_u", "rho1", "phi_u", "uCu", "law_ratio", "law_cos", "nve", "nvo", "dGe"):
            if key in rec:
                s += f" {key}={_f(rec[key]) if key not in ('nve', 'nvo', 'dGe') else _f(rec[key], '%.3e')}"
        return s
    if k == "hover":
        return f"{st} we={rec.get('we')} wue={rec.get('wue')}"
    if k == "star":
        G = [cp["G"] for cp in rec.get("checkpoints", [])]
        return f"{st} checkpoints={rec.get('mm_done')}/{rec.get('n_mm')} G={[round(x, 6) for x in G]} op={rec.get('op')}"
    if k == "op0":
        return f"{st} op={rec.get('op')}"
    if k == "own":
        return f"{st} ctrl={rec.get('ctrl')} own={rec.get('own')} eucl={rec.get('eucl')}"
    return str(st)


def run_plan(L, items, get_combo, out, sig, settings, brk, ref11, only=""):
    """Runs every planned unit not yet done (resume: S.unit_done). References a unit needs are computed first if
    missing. Returns dict(ran=[names], skipped=[names], waiting=[names])."""
    done = dict(ran=[], skipped=[], waiting=[])
    current = dict(name=None, combo=None)
    for it in items:
        name = it["name"]
        if only and only not in name:
            continue
        if S.unit_done(out, name, sig):
            done["skipped"].append(name)
            continue
        try:
            if current["name"] != it["combo"]:
                current["combo"] = None
                L.free_memory()
                current["combo"] = get_combo(it["combo"])
                current["name"] = it["combo"]
            combo = current["combo"]
            refs = {}
            for x, size in it["needs"]:
                rname = C.ref_name(it["combo"], x, size)
                if not S.unit_done(out, rname, sig):
                    rrec = S.run_unit(L, out, rname, sig, lambda: C.reference_unit(L, combo, x, size))
                    print(f"{rname}: {summary(rrec)} [{rrec['seconds']:.1f}s] (needed by {name})", flush=True)
                    done["ran"].append(rname)
                refs[(x, size)] = C.load_ref(L, out, rname)
            missing = [k for k, v in refs.items() if v is None]
            if missing:
                print(f"{name}: NOT RUN (reference missing or failed: {missing})", flush=True)
                done["waiting"].append(name)
                continue
            fn = make_fn(L, it, combo, refs, settings, brk, ref11)
            rec = S.run_unit(L, out, name, sig, fn)
            print(f"{name}: {summary(rec)} [{rec['seconds']:.1f}s]", flush=True)
            done["ran"].append(name)
            L.free_memory()
        except Exception as e:                          # e.g. the combination could not be built, or a save failed
            print(f"{name}: FAILED OUTSIDE THE UNIT ({e!r}); nothing stored", flush=True)
            done["waiting"].append(name)
    return done


def print_list(items, settings):
    """--list: every unit name; for Part M c_i, f_pred, g_pred (lam_ratio of the note values, see below)."""
    print("PLAN (network not loaded). Part M predictions use the tanh lam_ratios quoted in SPEC12 section 3 "
          f"{C.LIST_NOTE_LAM_RATIOS} as a NOTE; the run uses the lam_ratio of the loaded combination.")
    counts = {}
    for it in items:
        key = (it["combo"], it["part"])
        counts[key] = counts.get(key, 0) + 1
        line = f"  {it['name']}"
        if it["kind"] == "mode":
            pr = C.mode_prediction(it["Ld"], C.LIST_NOTE_LAM_RATIOS[it["dirs"][0]])
            flag = "" if math.isfinite(pr["f_pred"]) else "   SKIPPED-UNSTABLE"
            line += f"   lam_ratio={pr['lam_ratio']:.4f} c_i={pr['c_i']:.6g} f_pred={pr['f_pred']:+.6f} " \
                    f"g_pred={pr['g_pred']:+.6f}{flag}"
        elif it["kind"] in ("pair",) and it.get("tau") is not None:
            line += f"   tau={it['tau']:g} cap={it['cap']}"
        print(line)
    print("COUNTS (combination, part): " + ", ".join(f"{c} {p}: {n}" for (c, p), n in counts.items()))
    print(f"TOTAL {len(items)} units")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=os.path.normpath(os.path.join(HERE, "..", "spec11", "bundle")))
    # spec note: --out is REQUIRED except with --list (which writes nothing)
    ap.add_argument("--out", default=None)
    ap.add_argument("--combos", default="tanh-init-s0,gelu-init-s0")
    ap.add_argument("--parts", default="V,R,M,G,A,T,H,S,D")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--break", dest="brk", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    brk = C.check_break(a.brk)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    combos = [c.strip() for c in a.combos.split(",") if c.strip()]
    assert set(parts) <= set(PARTS), parts
    assert set(combos) <= set(MAIN_COMBOS), combos
    if a.quick:
        combos = ["tanh-init-s0"]
    settings = C.settings12(a.quick, brk)
    if a.list:
        sys.path.insert(0, a.bundle)
        L = importlib.import_module("part9lib")        # names only (L.safe_name); nothing is built or loaded
        items = C.plan12(settings, combos, parts, L.safe_name)
        print_list(items, settings)
        return
    if not a.out:
        ap.error("--out is required (except with --list)")
    out = os.path.abspath(a.out)
    if brk:
        print("!" * 100 + f"\n!!! --break {brk!r}: NEGATIVE-CONTROL RUN. ITS RESULTS ARE DELIBERATELY WRONG.\n" + "!" * 100,
              flush=True)
    L, CFG = S.load(a.bundle, a.device, a.threads, out)
    sig = S.signature(settings, os.path.join(a.bundle, "part9lib.py"), os.path.join(a.bundle, "cfg11.json"))
    p11 = os.path.join(a.bundle, "ref11.json")
    ref11 = S.read_json(p11) if os.path.exists(p11) else None
    items = C.plan12(settings, combos, parts, L.safe_name)
    print(f"SPEC12 run: bundle={a.bundle} out={out} device={a.device} threads={torch.get_num_threads()} parts={parts} "
          f"combos={combos} quick={a.quick} break={brk!r} only={a.only!r} signature={sig} units={len(items)}",
          flush=True)
    t0 = time.time()
    res = run_plan(L, items, lambda nm: get_combo_rig(L, CFG, a.bundle, nm), out, sig, settings, brk, ref11,
                   only=a.only)
    errors = [it["name"] for it in items if (S.read_unit(out, it["name"]) or {}).get("status") == "ERROR"]
    print(f"ran {len(res['ran'])}, already done {len(res['skipped'])}, not run {len(res['waiting'])}, "
          f"stored with status ERROR {len(errors)} {errors} (retried on the next start); "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if all(S.unit_done(out, it["name"], sig) for it in items):
        print("SPEC12 RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
