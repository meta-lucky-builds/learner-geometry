"""SPEC10 worker: PLAIN model F(theta, x) against ZERO model g(theta, x) = F(theta, x) - F(theta0, x), same learners.

What it does (SPEC10 sec. 3-7), for each G4 combination assigned to this worker:
  small-move block: AJ(1) + measure_ext at tau = +-S2_FRAC, both kinds; b_kind and the split TH -> NP;
  large-move block: tau in G4_TAUS; PLAIN DJ(1), DJ(1.5) (reproduction of the stored G4 records); ZERO DJ(1.5), DJ(1), SL(4),
                    DJ(0.25) always and TJ(1.5,1), SLa(4,2), AJ(1) where the stored PLAIN G4 record is usable; split vs DJ(1);
  V7 (ZERO model at theta0). Worker 0 also does the linear control V5.
PLAIN and ZERO differ ONLY in target() (training target) and out() (kappa F0t subtracted before loss/accuracy).
It reads the run directory and writes only into spec10_out/ next to the notebook: records_K.json (atomically after every
combination) and bundle_K.pt (float32 F_test of every end point, F0t, labels). Judging is done by spec10_report.py.

usage on the rig (from the folder with the notebook and part9lib.py, AFTER the main run):
    python spec10_run.py K M          (K = 0..M-1; e.g. four workers: for k in 0 1 2 3; do python spec10_run.py $k 4 > spec10_$k.txt 2>&1 & done)
  With several visible GPUs and PART9_DEVICE unset, worker K uses cuda:(K mod number of GPUs).
  On cuda it refuses to start (exit code 2) while a process with 'part9lib.worker_main' in its command line is alive,
  unless SPEC10_FORCE=1.
environment (as g4_ends.py): PART9_NB, PART9_MODE, PART9LIB_DIR, PART9_RUN_DIR, PART9_NO_DOWNLOAD, PART9_DEVICE.
TEST HOOK SPEC10_BREAK in (target, kappa, repro): deliberately breaks one thing (negative controls of review10/test_spec10.py).
"""
import os, sys, json, re, time, math, subprocess

# ------------------------------------------------------------------ loading: the same code as g4_ends.py
nbp = os.path.abspath(os.environ.get("PART9_NB", "/workspace/evaluation_point_tests.ipynb"))
mode = os.environ.get("PART9_MODE")
k_w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m_w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
assert 0 <= k_w < m_w
os.chdir(os.path.dirname(nbp))
sys.path.insert(0, os.environ.get("PART9LIB_DIR", os.getcwd()))
nb = json.load(open(nbp, encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
if mode:
    src, k = re.subn(r'MODE = "\w+"', 'MODE = "%s"' % mode, src, count=1)
    assert k == 1
g = {}
exec(src, g)
CFG = g["CFG"]
if os.environ.get("PART9_NO_DOWNLOAD"):
    CFG["MNIST_DOWNLOAD"] = False
rd = os.environ.get("PART9_RUN_DIR", CFG["RUN_DIR"])          # the SPEC9 run directory (read only)
OUT = os.path.join(os.getcwd(), "spec10_out")                 # sec. 10.1: everything this script writes goes here
os.makedirs(OUT, exist_ok=True)
CFG["RUN_DIR"] = OUT        # only used by the library's progress()/heartbeat() log; keeps progress.log out of the run directory

# ------------------------------------------------------------------ test hook (sec. 10.3), read once
BREAK = os.environ.get("SPEC10_BREAK", "")
assert BREAK in ("", "target", "kappa", "repro"), "SPEC10_BREAK must be one of target, kappa, repro"
if BREAK:
    print("\n" + "#" * 100 + "\n### SPEC10_BREAK=%s IS SET: THIS RUN IS DELIBERATELY BROKEN (negative control). RESULTS ARE INVALID.\n"
          % BREAK + "#" * 100 + "\n", flush=True)

import torch
import part9lib as L
L.configure(CFG)

# ------------------------------------------------------------------ device and the busy guard (sec. 10.1)
dev = os.environ.get("PART9_DEVICE")
if dev is None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and torch.cuda.device_count() > 1:
        dev = "cuda:%d" % (k_w % torch.cuda.device_count())
if dev.startswith("cuda"):
    if os.environ.get("SPEC10_FORCE") != "1":
        try:
            pg = subprocess.run(["pgrep", "-f", "part9lib.worker_main"], capture_output=True, text=True)
        except FileNotFoundError:
            print("spec10_run: REFUSED: 'pgrep' is not available, so I cannot check that no part9lib.worker_main process is "
                  "alive. Set SPEC10_FORCE=1 to start anyway.", flush=True)
            sys.exit(2)
        if pg.returncode != 1:
            print("spec10_run: REFUSED: a part9lib.worker_main process is alive (pgrep -f part9lib.worker_main: exit %d, pids %s). "
                  "Wait until the main run has finished, or set SPEC10_FORCE=1." % (pg.returncode, pg.stdout.split()), flush=True)
            sys.exit(2)
    if ":" in dev:                                            # FIX: set_device needs an explicit index; plain "cuda"
        torch.cuda.set_device(int(dev.split(":")[1]))         # (one visible GPU, as the SPEC9 workers) needs no call at all
L.set_device(dev)
L._load_mnist_raw()
C = CFG["N_CLASSES"]

# ------------------------------------------------------------------ constants of SPEC10 (section in brackets)
KINDS = ("PLAIN", "ZERO")                                     # sec. 1
KAPPA = {"PLAIN": 0.0, "ZERO": 1.0}                           # sec. 4
SMALL_SPEC = ("AJ", 1.0)                                      # sec. 5.1
PLAIN_SPECS = [("DJ", 1.0), ("DJ", 1.5)]                      # sec. 5.2
ZERO_CHEAP = [("DJ", 1.5), ("DJ", 1.0), ("SL", 4), ("DJ", 0.25)]
ZERO_IMPLICIT = [("TJ", 1.5, 1.0), ("SLa", 4, 2.0), ("AJ", 1.0)]
REF_SPEC = ("DJ", 1.0)                                        # sec. 6: lo of the large block
BAD_STATUS = ("DIVERGED", "AJ-FAILED", "AJ-INCONSISTENT")     # sec. 5.2 USABLE (as the SPEC9 G4 report)
LIN_TAU, LIN_SPECS = 0.5, [("DJ", 1.5), ("DJ", 1.0), ("DJ", 0.25)]   # sec. 7 V5 (first entry after DJ(1) = hi)
REPRO_FACTOR = 1.001                                          # sec. 10.3 (iii)
V4_TOL = 1e-7                                                 # sec. 6 check (V4)


def same_spec(a, b):
    return a[0] == b[0] and len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a[1:], b[1:]))


FNOM = {}
for x, fnom in CFG["G4_LEARNERS"]:
    for sp in PLAIN_SPECS + ZERO_CHEAP + ZERO_IMPLICIT:
        if same_spec(x, sp):
            FNOM[sp] = fnom
assert all(sp in FNOM for sp in PLAIN_SPECS + ZERO_CHEAP + ZERO_IMPLICIT), "a SPEC10 learner is not one of the G4 learners"
assert FNOM[REF_SPEC] == 0.0
print("spec10_run: MODE %s data_source %s device %s run_dir %s out %s worker %d of %d break %r" % (
    CFG["MODE"], L.DATA_SOURCE, dev, os.path.abspath(rd), OUT, k_w, m_w, BREAK), flush=True)


# ------------------------------------------------------------------ the two places where PLAIN and ZERO differ
def target(ctx, kind, tau, small):
    """Training target t and size s handed to the learner (sec. 3). small=True: the Part A form for PLAIN (run_A_case)."""
    st = ctx.st
    if kind == "ZERO" and BREAK != "target":
        t = st.F0 + tau * ctx.Y
        return t, L.vnorm(t - st.F0)
    if small:                                                 # PLAIN, built exactly as run_A_case builds it
        sign = 1 if tau > 0 else -1
        s = CFG["S2_FRAC"] * ctx.S["true"]
        return st.F0 + sign * s * ctx.dirs["true"], s
    t = st.F0 + tau * (ctx.Y - st.F0)                         # PLAIN, exactly as run_G4_case
    return t, L.vnorm(t - st.F0)


def out(F, F0t, kind):
    """Model output on the test inputs (sec. 4): F_test(theta) - kappa F0t."""
    kappa = 0.0 if (kind == "ZERO" and BREAK == "kappa") else KAPPA[kind]
    return F - kappa * F0t


# ------------------------------------------------------------------ helpers
def rec_path(name):
    return os.path.join(rd, "done", L.safe_name(name) + ".json")


def stored(name):
    p = rec_path(name)
    return L.read_json(p) if os.path.exists(p) else None


def g4_name(cn, Ld, tau, fnom):
    return L._case9(cn, "G4", "G4", Ld, "true", f"tau{tau:g}", tau, 1, f_nominal=fnom)["name"]


def centre(x):
    """(centred per class over the test inputs, x - centred), both (n_test, C)."""
    X = x.reshape(-1, C)
    Xv = X - X.mean(0, keepdim=True)
    return Xv, X - Xv


def dot(a, b):
    return float((a * b).sum())


def fin(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def usable(e):
    """sec. 5.2 USABLE; ERROR is never usable."""
    return (e.get("learner_status") not in BAD_STATUS + ("ERROR",) and fin(e.get("train_res"))
            and e["train_res"] <= CFG["G4_RES_MAX"])


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else float("inf"))


def out_numbers(F, F0t, td, kind, prefix=""):
    """loss and accuracy of the model output (library and explicit formula, V6), a_star and loss_star (sec. 6)."""
    o = out(F, F0t, kind)
    Yt = td["Y"]
    N = o.numel()
    loss, acc = L.test_loss_acc(o, td)
    loss_x = float(((o - Yt) ** 2).sum()) / (2.0 * N)
    ov, oc = centre(o)
    n2 = dot(ov, ov)
    if n2 > 0:
        a_star = dot(Yt.reshape(-1, C), ov) / n2
        rs = (a_star * ov + oc) - Yt.reshape(-1, C)
        loss_star = dot(rs, rs) / (2.0 * N)
    else:
        a_star = loss_star = float("nan")
    return {prefix + "loss": loss, prefix + "acc": acc, prefix + "loss_explicit": loss_x, prefix + "a_star": a_star,
            prefix + "loss_star": loss_star}


def split(F_lo, F_hi, F0t, td, kind, loss_lo, loss_hi):
    """sec. 6: loss(hi) - loss(lo) = const + dN + dS + drest + quad (exact), with the 2x2 least-squares fit."""
    N = F0t.numel()
    Yt = td["Y"].reshape(-1, C)
    Nm = F0t.reshape(-1, C)
    Nv, _ = centre(Nm)
    S = F_lo.reshape(-1, C) - Nm
    Sv, _ = centre(S)
    r = out(F_lo, F0t, kind).reshape(-1, C) - Yt
    D = F_hi.reshape(-1, C) - F_lo.reshape(-1, C)
    Dv, _ = centre(D)
    Dc = D - Dv
    a11, a12, a22 = dot(Nv, Nv), dot(Nv, Sv), dot(Sv, Sv)
    det = a11 * a22 - a12 * a12
    g1, g2 = dot(Nv, Dv), dot(Sv, Dv)
    eN = (g1 * a22 - g2 * a12) / det if det != 0 else float("nan")
    eS = (g2 * a11 - g1 * a12) / det if det != 0 else float("nan")
    rest = Dv - eN * Nv - eS * Sv
    gN, gS = dot(r, Nv) / N, dot(r, Sv) / N
    nDv2 = dot(Dv, Dv)
    return dict(dl=loss_hi - loss_lo, const=dot(r, Dc) / N, dN=eN * gN, dS=eS * gS, drest=dot(r, rest) / N,
                quad=dot(D, D) / (2.0 * N), eN=eN, eS=eS, gN=gN, gS=gS,
                R2=1.0 - dot(rest, rest) / nDv2 if nDv2 > 0 else float("nan"))


def v4_ok(sp):
    terms = (sp["const"], sp["dN"], sp["dS"], sp["drest"], sp["quad"])
    lhs = sum(terms)
    return fin(lhs, sp["dl"]) and abs(lhs - sp["dl"]) <= V4_TOL * max([abs(x) for x in terms] + [abs(sp["dl"])])


def f32(F):
    return F.reshape(-1, C).float().cpu()


# ------------------------------------------------------------------ one end point of the large block / linear control
def end_point(ctx, F0t, td, kind, tau, sp):
    """Runs one learner on the target of (kind, tau) and returns (record, F_test or None)."""
    prob, st = ctx.prob, ctx.st
    Ld = L.L_from_spec(sp)
    e = dict(kind=kind, tau=tau, learner=L.learner_name9(Ld), spec=list(sp), f_nominal=FNOM[sp])
    t, s = target(ctx, kind, tau, small=False)
    if BREAK == "repro" and kind == "PLAIN" and same_spec(sp, REF_SPEC):
        t = t * REPRO_FACTOR                                  # negative control (iii)
        e["BROKEN"] = "target x %g" % REPRO_FACTOR
    t1 = time.time()
    F = None
    try:
        res = L.run_learner9(prob, st, t, s, Ld)
        thL = res["theta"]
        e["learner_status"] = res["status"]
        if not bool(torch.isfinite(thL).all()):
            e["learner_status"] = "DIVERGED"
        else:
            FLtr = prob.F(thL)
            e["train_res"] = L.vnorm(FLtr - t) / s
            if kind == "ZERO":                                # V3, explicit formula
                e["zero_res"] = L.vnorm((FLtr - st.F0) - tau * ctx.Y) / L.vnorm(tau * ctx.Y)
            F = L.test_outputs(prob, td["X"], thL)
            e.update(out_numbers(F, F0t, td, kind))
            e["dist"] = L.vnorm(thL - st.th0)
            e["lam1"], e["trK"] = L._lam1_tr(L.gram(prob.J(thL)))
    except Exception as ex:
        e["learner_status"] = "ERROR"
        e["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
        F = None
    e["wall_s"] = time.time() - t1
    L.free_memory()
    return e, F


def print_large(cn, tau, kind, ends, splits):
    print("\n%s tau=%g %s  (split against DJ(1) of the same kind)" % (cn, tau, kind), flush=True)
    print("   %-11s %7s %-15s %9s %9s %9s %7s %7s %9s %9s | %10s = %10s %10s %10s %10s %10s | %10s %10s %10s %6s" % (
        "learner", "f_nom", "status", "train res", "zero res", "loss", "acc", "a*", "loss*", "V1 repro", "loss-lo",
        "const", "dN", "dS", "drest", "quad", "eN", "eS", "gN", "R2"), flush=True)
    for e in ends:
        rp = max(e.get("v1_rel_loss", float("nan")), e.get("v1_rel_dist", float("nan"))) if "v1_rel_loss" in e else float("nan")
        head = "   %-11s %+7.4f %-15s %9.2e %9.2e %9.6f %7.4f %7.3f %9.6f %9.1e" % (
            e["learner"], e["f_nominal"], e["learner_status"], e.get("train_res", float("nan")),
            e.get("zero_res", float("nan")), e.get("loss", float("nan")), e.get("acc", float("nan")),
            e.get("a_star", float("nan")), e.get("loss_star", float("nan")), rp)
        sp = splits.get(e["learner"])
        if e["learner"] == L.learner_name9(L.L_from_spec(REF_SPEC)):
            print(head + " | (reference)", flush=True)
        elif sp is None:
            print(head + " | (no split: %s)" % ("not usable" if not usable(e) else "no usable reference"), flush=True)
        else:
            print(head + " | %+10.3e = %+10.2e %+10.2e %+10.2e %+10.2e %+10.2e | %+10.2e %+10.2e %+10.2e %6.3f%s" % (
                sp["dl"], sp["const"], sp["dN"], sp["dS"], sp["drest"], sp["quad"], sp["eN"], sp["eS"], sp["gN"], sp["R2"],
                "" if sp["v4_ok"] else "  <- SUM CHECK FAILED"), flush=True)
        if e.get("error"):
            print("      ERROR: " + e["error"], flush=True)


# ------------------------------------------------------------------ blocks
def small_block(ctx, F0t, td, cn, bundle_ends):
    """sec. 5.1: AJ(1) + measure_ext at tau = +-S2_FRAC for each kind; b_kind and the split TH -> NP (averaged over signs)."""
    prob, st = ctx.prob, ctx.st
    Ld = L.L_from_spec(SMALL_SPEC)
    blk = {}
    for kind in KINDS:
        signs = {}
        for sign in (1, -1):
            tau = sign * CFG["S2_FRAC"]
            key = "%+d" % sign
            t, s = target(ctx, kind, tau, small=True)
            e = dict(kind=kind, tau=tau, sign=sign, learner=L.learner_name9(Ld))
            t1 = time.time()
            try:
                res = L.run_learner9(prob, st, t, s, Ld)
                thL = res["theta"]
                e["learner_status"] = res["status"]
                if not bool(torch.isfinite(thL).all()):
                    e["learner_status"] = "DIVERGED"
                elif res["status"] != "AJ-FAILED":            # as run_A_case: measure unless failed or diverged
                    FLtr = prob.F(thL)
                    e["train_res"] = L.vnorm(FLtr - t) / s
                    if kind == "ZERO":
                        e["zero_res"] = L.vnorm((FLtr - st.F0) - tau * ctx.Y) / L.vnorm(tau * ctx.Y)
                    sc, vL, vNP, thNP, thTH = L.measure_ext(prob, st, thL)
                    e.update(rho=sc["rho"], np_status=sc["np_status"], th_status=sc["th_status"])
                    FL = L.test_outputs(prob, td["X"], thL)
                    FN = L.test_outputs(prob, td["X"], thNP)
                    FT = L.test_outputs(prob, td["X"], thTH)
                    e.update(out_numbers(FL, F0t, td, kind, "L_"))
                    e.update(out_numbers(FN, F0t, td, kind, "NP_"))
                    e.update(out_numbers(FT, F0t, td, kind, "TH_"))
                    e.update(loss_NP=e["NP_loss"], loss_TH=e["TH_loss"], acc_NP=e["NP_acc"], acc_TH=e["TH_acc"])
                    sp = split(FT, FN, F0t, td, kind, e["loss_TH"], e["loss_NP"])
                    sp["v4_ok"] = v4_ok(sp)
                    e["split"] = sp
                    for nm_, F_ in (("L", FL), ("NP", FN), ("TH", FT)):
                        bundle_ends["small|%s|%s|%s" % (kind, key, nm_)] = f32(F_)
            except Exception as ex:
                e["learner_status"] = "ERROR"
                e["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
            e["wall_s"] = time.time() - t1
            e["usable"] = bool(e.get("learner_status") == "CONVERGED" and e.get("np_status") == "CONVERGED"
                               and e.get("th_status") == "CONVERGED"
                               and fin(e.get("rho"), e.get("loss_NP"), e.get("loss_TH"), e.get("acc_NP"), e.get("acc_TH")))
            signs[key] = e
            L.free_memory()
        both = signs["+1"]["usable"] and signs["-1"]["usable"]
        b = 0.5 * sum(signs[k]["loss_NP"] - signs[k]["loss_TH"] for k in signs) if both else float("nan")
        avg = ({k: 0.5 * (signs["+1"]["split"][k] + signs["-1"]["split"][k]) for k in signs["+1"]["split"] if k != "v4_ok"}
               if both else None)
        blk[kind] = dict(signs=signs, b=b, split_avg=avg)
    # V2: PLAIN reproduction of the stored Part A records (n = N_TRAIN combinations where the record exists)
    v2 = []
    if ctx.n == CFG["N_TRAIN"]:
        for sign in (1, -1):
            nm = L._case9(cn, "A", "A", L.L_AJ(1.0), "true", "s2", CFG["S2_FRAC"], sign)["name"]
            r = stored(nm)
            if r is None:
                print("%s small block: no stored Part A record %s (V2 not applicable)" % (cn, nm), flush=True)
                continue
            e = blk["PLAIN"]["signs"]["%+d" % sign]
            v2.append(dict(sign=sign, name=nm, **{"rel_" + q: (rel(e[q], r[q]) if fin(e.get(q), r.get(q)) else float("nan"))
                                                   for q in ("loss_NP", "loss_TH", "rho")}))
    blk["V2"] = v2
    # table
    print("\n%s small-move block (AJ(1), |tau| = S2_FRAC = %g; split lo = TH, hi = NP)" % (cn, CFG["S2_FRAC"]), flush=True)
    print("   %-5s %4s %-15s %-10s %-10s %9s %9s %9s %10s %10s | %10s = %10s %10s %10s %10s %10s | %10s %10s %6s" % (
        "kind", "sign", "AJ status", "NP status", "TH status", "zero res", "rho", "loss_TH", "loss_NP", "acc_NP", "NP-TH",
        "const", "dN", "dS", "drest", "quad", "eN", "eS", "R2"), flush=True)
    for kind in KINDS:
        for key, e in blk[kind]["signs"].items():
            sp = e.get("split")
            tail = ("%+10.3e = %+10.2e %+10.2e %+10.2e %+10.2e %+10.2e | %+10.2e %+10.2e %6.3f%s" % (
                sp["dl"], sp["const"], sp["dN"], sp["dS"], sp["drest"], sp["quad"], sp["eN"], sp["eS"], sp["R2"],
                "" if sp["v4_ok"] else "  <- SUM CHECK FAILED") if sp else "-")
            print("   %-5s %4s %-15s %-10s %-10s %9.2e %9.5f %9.6f %10.6f %10.4f | %s" % (
                kind, key, e.get("learner_status"), e.get("np_status", "-"), e.get("th_status", "-"),
                e.get("zero_res", float("nan")), e.get("rho", float("nan")), e.get("loss_TH", float("nan")),
                e.get("loss_NP", float("nan")), e.get("acc_NP", float("nan")), tail), flush=True)
            if e.get("error"):
                print("      ERROR: " + e["error"], flush=True)
        print("   %-5s b = %+.4e" % (kind, blk[kind]["b"]), flush=True)
    for v in v2:
        print("   V2 PLAIN vs stored Part A (sign %+d): rel loss_NP %.1e, loss_TH %.1e, rho %.1e" % (
            v["sign"], v["rel_loss_NP"], v["rel_loss_TH"], v["rel_rho"]), flush=True)
    return blk


def large_block(ctx, F0t, td, cn, tau, bundle_ends):
    """sec. 5.2 for one tau: PLAIN DJ(1), DJ(1.5); ZERO cheap learners and the selected implicit learners; splits vs DJ(1)."""
    blk = dict(tau=tau, skips=[], stored={})
    # stored PLAIN G4 records of every G4 learner of this block (sec. 5.2: all other PLAIN numbers come from them)
    for x, fnom in CFG["G4_LEARNERS"]:
        Ld = L.L_from_spec(x)
        nm = g4_name(cn, Ld, tau, fnom)
        r = stored(nm)
        ln = L.learner_name9(Ld)
        if r is None:
            blk["stored"][ln] = dict(name=nm, missing=True, f_nominal=fnom)
        else:
            blk["stored"][ln] = dict(name=nm, missing=False, f_nominal=fnom, **{q: r.get(q) for q in (
                "learner_status", "train_res", "loss_L", "acc_L", "dist", "lam1_L", "trK_L")})
    specs = {"PLAIN": list(PLAIN_SPECS), "ZERO": list(ZERO_CHEAP)}
    for sp in ZERO_IMPLICIT:                                  # selection from OLD records only
        ln = L.learner_name9(L.L_from_spec(sp))
        r = blk["stored"][ln]
        ok = (not r["missing"]) and usable(dict(learner_status=r["learner_status"], train_res=r["train_res"]))
        if ok:
            specs["ZERO"].append(sp)
        else:
            why = "no stored G4 record" if r["missing"] else "stored G4 record not usable (status %s, train_res %s)" % (
                r["learner_status"], r["train_res"])
            blk["skips"].append(dict(learner=ln, reason=why))
            print("%s tau=%g ZERO %s SKIPPED: %s" % (cn, tau, ln, why), flush=True)
    blk["planned"] = {kind: [L.learner_name9(L.L_from_spec(sp)) for sp in specs[kind]] for kind in KINDS}
    ref_ln = L.learner_name9(L.L_from_spec(REF_SPEC))
    for kind in KINDS:
        ends, Fs = [], {}
        for sp in specs[kind]:
            e, F = end_point(ctx, F0t, td, kind, tau, sp)
            if kind == "PLAIN":                               # V1: against the stored G4 record
                r = blk["stored"][e["learner"]]
                if (not r["missing"]) and fin(e.get("loss"), e.get("dist"), r.get("loss_L"), r.get("dist")):
                    e["v1_rel_loss"], e["v1_rel_dist"] = rel(e["loss"], r["loss_L"]), rel(e["dist"], r["dist"])
                else:
                    e["v1_rel_loss"] = e["v1_rel_dist"] = float("nan")
            e["usable"] = usable(e)
            ends.append(e)
            if F is not None:
                Fs[e["learner"]] = F
                bundle_ends["large|%s|tau%g|%s" % (kind, tau, e["learner"])] = f32(F)
        splits = {}
        lo = [e for e in ends if e["learner"] == ref_ln][0]
        if lo["usable"]:
            for e in ends:
                if e is lo or not e["usable"]:
                    continue
                sp = split(Fs[ref_ln], Fs[e["learner"]], F0t, td, kind, lo["loss"], e["loss"])
                sp["v4_ok"] = v4_ok(sp)
                splits[e["learner"]] = sp
        blk[kind] = dict(ends=ends, splits=splits)
        print_large(cn, tau, kind, ends, splits)
    return blk


def run_combo(cb, bundle):
    cn = L.cname9(cb)
    t0 = time.time()
    ctx = L.Ctx9(cb, rd)
    prob, st = ctx.prob, ctx.st
    td = ctx.test()
    F0t, l0, a0 = ctx.test0()
    entry = dict(cname=cn, act=cb["act"], seed=cb["seed"], n=ctx.n, n_test=td["n_test"], loss0_plain=l0, acc0_plain=a0)
    # V7: ZERO model at theta0 (F_test(theta0) recomputed)
    o0 = out(L.test_outputs(prob, td["X"], st.th0), F0t, "ZERO")
    v7l, _ = L.test_loss_acc(o0, td)
    entry["V7"] = dict(loss=v7l, loss_explicit=float(((o0 - td["Y"]) ** 2).sum()) / (2.0 * o0.numel()), target=1.0 / (2.0 * C),
                       rel=rel(v7l, 1.0 / (2.0 * C)))
    print("\n========== %s (n = %d, n_test = %d): test loss at theta0 (PLAIN) %.6f, accuracy %.4f; V7 ZERO loss at theta0 %.15f "
          "(1/(2C) = %.15f)" % (cn, ctx.n, td["n_test"], l0, a0, v7l, 1.0 / (2.0 * C)), flush=True)
    bentry = dict(labels=td["labels"].cpu().to(torch.int16), F0=f32(F0t), ends={})
    entry["small"] = small_block(ctx, F0t, td, cn, bentry["ends"])
    entry["blocks"] = []
    for tau in sorted(CFG["G4_TAUS"]):
        entry["blocks"].append(large_block(ctx, F0t, td, cn, tau, bentry["ends"]))
    entry["wall_s"] = time.time() - t0
    bundle["combos"][cn] = bentry
    print("   (%s done in %.1f min)" % (cn, entry["wall_s"] / 60), flush=True)
    del ctx, prob, st
    L.free_memory()
    return entry


def run_linear(bundle):
    """V5 (worker 0): linear model, ZERO kind, tau = 0.5, DJ(1.5), DJ(1), DJ(0.25)."""
    cb = L.linear_combo()
    cn = L.cname9(cb)
    ctx = L.Ctx9(cb, rd)
    td = ctx.test()
    F0t, _, _ = ctx.test0()
    ends, Fs = [], {}
    bentry = dict(labels=td["labels"].cpu().to(torch.int16), F0=f32(F0t), ends={})
    for sp in LIN_SPECS:
        e, F = end_point(ctx, F0t, td, "ZERO", LIN_TAU, sp)
        e["usable"] = usable(e)
        ends.append(e)
        if F is not None:
            Fs[e["learner"]] = F
            bentry["ends"]["linear|ZERO|tau%g|%s" % (LIN_TAU, e["learner"])] = f32(F)
    ref_ln = L.learner_name9(L.L_from_spec(REF_SPEC))
    ratios = {}
    for e in ends:
        if e["learner"] != ref_ln:
            ok = ref_ln in Fs and e["learner"] in Fs
            ratios[e["learner"]] = (L.vnorm(Fs[e["learner"]] - Fs[ref_ln]) / L.vnorm(Fs[ref_ln] - F0t)) if ok else float("nan")
    vmax = max(ratios.values()) if ratios and all(fin(v) for v in ratios.values()) else float("nan")
    bundle["linear"] = bentry
    print("\nV5 linear control %s (ZERO, tau = %g): |F_test(L) - F_test(DJ(1))| / |F_test(DJ(1)) - F0t| = %s; max %.3e" % (
        cn, LIN_TAU, {k: "%.3e" % v for k, v in ratios.items()}, vmax), flush=True)
    for e in ends:
        print("   %-9s %-12s train res %.2e zero res %.2e loss %.6f" % (e["learner"], e["learner_status"], e.get("train_res", float("nan")),
                                                                         e.get("zero_res", float("nan")), e.get("loss", float("nan"))), flush=True)
    del ctx
    L.free_memory()
    return dict(cname=cn, tau=LIN_TAU, ends=ends, ratios=ratios, max_ratio=vmax)


def save(records, bundle):
    L.write_json_atomic(os.path.join(OUT, "records_%d.json" % k_w), records)
    p = os.path.join(OUT, "bundle_%d.pt" % k_w)
    torch.save(bundle, p + ".tmp")
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------------ main
cbs = sorted(L.combos("G4"), key=lambda cb: -int(cb["n"]))            # stable sort: the larger training sets first
mine = [cb for i, cb in enumerate(cbs) if i % m_w == k_w]
print("spec10_run: combinations of this worker %s" % [L.cname9(cb) for cb in mine], flush=True)
records = dict(spec="SPEC10", worker=k_w, n_workers=m_w, mode=CFG["MODE"], data_source=L.DATA_SOURCE, device=dev,
               BREAK=BREAK, run_dir=os.path.abspath(rd), notebook=nbp, torch=torch.__version__, cfg=CFG,
               combos_assigned=[L.cname9(cb) for cb in mine], combos={}, linear=None, complete=False,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
bundle = dict(mode=CFG["MODE"], data_source=L.DATA_SOURCE, device=dev, BREAK=BREAK, combos={}, linear=None)
if k_w == 0:
    records["linear"] = run_linear(bundle)
    save(records, bundle)
for cb in mine:
    records["combos"][L.cname9(cb)] = run_combo(cb, bundle)
    save(records, bundle)
records["complete"] = True
records["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save(records, bundle)
print("spec10_run: wrote %s and %s" % (os.path.join(OUT, "records_%d.json" % k_w), os.path.join(OUT, "bundle_%d.pt" % k_w)))
print("spec10_run: done%s" % (" (SPEC10_BREAK=%s: INVALID BY CONSTRUCTION)" % BREAK if BREAK else ""), flush=True)
