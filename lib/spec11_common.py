"""SPEC11 shared helpers (spec sections 2-7, 8 storage, 9 shared formulas).

Imported by spec11_run.py, spec11_report.py, make_bundle11.py and test_spec11.py. Every learner, jump, nearest-point
solve and measurement comes from part9lib, which is passed in explicitly as the argument L (no module-level state here).
All tensors are float64 (L.DT). One function per spec concept; the run, the report and the tests call these same
functions.
"""
import os
import sys
import json
import math
import time
import struct
import hashlib
import importlib
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------------------------
# Constants of the spec
# ---------------------------------------------------------------------------------------------------------------
CLOSE_TOL = 1e-6                     # section 5
FIT_TOL = 1e-6                       # section 7
BREAK_MODES = ("", "noreturn", "swap", "repro")   # section 8; "" = no break (default)

TANH = dict(act="tanh", start="init", seed=0)
GELU = dict(act="gelu", start="init", seed=0)
LIN = dict(act="linear", start="init", seed=0)
COMBOS = {"tanh-init-s0": TANH, "gelu-init-s0": GELU, "linear-init-s0": LIN}
LIN_NAME = "linear-init-s0"
MAIN_COMBOS = ["tanh-init-s0", "gelu-init-s0"]

E_NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
W_NAMES = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
REF14 = E_NAMES + W_NAMES            # order of the 14 x 14 Gram matrix and of the Part B q vectors
SIGNS = (1, -1)

# learner lists as (kind, args...) - built with make_learner(L, spec)
LA_FULL = [("DJ", 1.0), ("DJ", 1.5), ("DJ", 0.5), ("SL", 2), ("SL", 8), ("AJ", 0.5), ("AJ", 1.0), ("AJ", 1.5),
           ("SLa", 2, 1.0)]
LB_FULL = [("DJ", 1.0), ("DJ", 1.5), ("SL", 2), ("SL", 8), ("AJ", 0.5), ("AJ", 1.0)]
LC_FULL = [("DJ", 1.5), ("DJ", 1.0), ("SL", 2), ("SL", 4), ("SL", 8), ("SLa", 8, 1.0), ("TJ", 1.5, 1.0)]
LIN_LEARNERS = [("DJ", 1.0), ("SL", 2), ("AJ", 1.0)]
QUICK_A = [("DJ", 1.0), ("SL", 2), ("AJ", 1.0)]
QUICK_B = [("DJ", 1.0), ("SL", 2)]
QUICK_C = [("DJ", 1.0), ("SL", 2)]
FINAL_LEARNER = ("SL", 8)            # Part C final task, the same for every learner
REPRO_UNIT = ("tanh-init-s0", "E1", "s1", "SL(2)")   # BREAK "repro"


def check_break(brk):
    """BREAK must be one of the four allowed values; anything else stops the program (never active by accident)."""
    if brk not in BREAK_MODES:
        raise ValueError(f"SPEC11_BREAK={brk!r} is not one of {BREAK_MODES}")
    return brk


def break_banner(brk):
    if brk:
        bar = "!" * 100
        print(f"{bar}\n!!! SPEC11_BREAK = {brk!r} IS SET: THIS IS A NEGATIVE-CONTROL RUN. ITS RESULTS ARE DELIBERATELY WRONG.\n"
              f"{bar}", flush=True)


def science_settings(quick, brk, CFG):
    """Everything of this spec that changes the numbers (enters the signature, section 8)."""
    check_break(brk)
    if quick:
        s = dict(LA=QUICK_A, LB=QUICK_B, LC=QUICK_C, A_DIRS=["E1"], A_SIZES=["s1"], B_M=2, C_TASKS=2, C_FINAL=False,
                 REFS={"s1": ["E1", "E2", "W1", "W2"], "s2": []})
    else:
        s = dict(LA=LA_FULL, LB=LB_FULL, LC=LC_FULL, A_DIRS=["E1", "E2"], A_SIZES=["s1", "s2"], B_M=6, C_TASKS=6,
                 C_FINAL=True, REFS={"s1": list(REF14), "s2": ["E1", "E2"]})
    s.update(LIN=LIN_LEARNERS, LIN_DIR="E1", LIN_SIZE="s1", FINAL_LEARNER=FINAL_LEARNER,
             SEEDS=dict(E_label=8000, E_gauss=9000, W_label=7000, W_gauss=7500, task=6000, per_combo=100),
             S1_FRAC=CFG["S1_FRAC"], S2_FRAC=CFG["S2_FRAC"], CLOSE_TOL=CLOSE_TOL, FIT_TOL=FIT_TOL,
             QUICK=bool(quick), BREAK=brk)
    return s


def make_learner(L, spec):
    kind = spec[0]
    if kind == "DJ":
        return L.L_DJ(spec[1])
    if kind == "SL":
        return L.L_SL(spec[1])
    if kind == "AJ":
        return L.L_AJ(spec[1])
    if kind == "SLa":
        return L.L_SLa(spec[1], spec[2])
    if kind == "TJ":
        return L.L_TJ(spec[1], spec[2])
    raise ValueError(spec)


def report_f(L, Ld, brk):
    """The f the REPORT uses: L.predict_f9, replaced by 1 - f under BREAK "swap" (section 8)."""
    f = L.predict_f9(Ld)
    return 1.0 - f if brk == "swap" else f


# ---------------------------------------------------------------------------------------------------------------
# Section 2: loading
# ---------------------------------------------------------------------------------------------------------------
def read_idx(path):
    """One uncompressed MNIST idx file, exactly as loc.py reads it."""
    b = open(path, "rb").read()
    magic, n = struct.unpack(">II", b[:8])
    if magic == 2051:
        r, c = struct.unpack(">II", b[8:16])
        return torch.from_numpy(np.frombuffer(b, np.uint8, offset=16).reshape(n, r, c).copy())
    assert magic == 2049
    return torch.from_numpy(np.frombuffer(b, np.uint8, offset=8).astype(np.int64))


def config_from_notebook(nb_path):
    """The rig's CONFIG cell (the one containing 'S1_FRAC, S2_FRAC'), exec'd exactly as loc.py does; MODE must be full."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"]
           if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
    g = {}
    exec(src, g)
    CFG = g["CFG"]
    assert CFG["MODE"] == "full"
    return CFG


def load(bundle_dir, device, threads=None, out_dir="spec11_out"):
    """Section 2. Returns (L, CFG). out_dir becomes CFG["RUN_DIR"] (the library writes progress.log there).
    # spec note: section 2 sets CFG["RUN_DIR"] = <out dir> but load() has no out-dir argument: added as out_dir."""
    with open(os.path.join(bundle_dir, "cfg11.json"), encoding="utf-8") as f:
        CFG = json.load(f)
    assert CFG["MODE"] == "full"
    CFG["MNIST_DOWNLOAD"] = False
    CFG["RUN_DIR"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    if threads:
        torch.set_num_threads(int(threads))
    sys.path.insert(0, bundle_dir)
    L = importlib.import_module("part9lib")
    # the library must be the bundle's copy, not one imported earlier from somewhere else
    assert os.path.samefile(L.__file__, os.path.join(bundle_dir, "part9lib.py")), L.__file__
    L.configure(CFG)
    L.set_device(device)
    if ":" in device:                                   # only an explicit index; never "cuda" or a torch.device
        torch.cuda.set_device(int(device.split(":")[1]))
    raw = os.path.join(bundle_dir, "mnist")
    L._MNIST_RAW = (read_idx(os.path.join(raw, "train-images-idx3-ubyte")),
                    read_idx(os.path.join(raw, "train-labels-idx1-ubyte")))
    L._MNIST_TEST_RAW = (read_idx(os.path.join(raw, "t10k-images-idx3-ubyte")),
                         read_idx(os.path.join(raw, "t10k-labels-idx1-ubyte")))
    L.DATA_SOURCE = "mnist"
    return L, CFG


# ---------------------------------------------------------------------------------------------------------------
# Section 3: directions, sizes, cost q, kernel summary
# ---------------------------------------------------------------------------------------------------------------
def label_dir(L, st, seed):
    C = st.prob.C
    n = st.N // C
    lab = torch.randint(0, C, (n,), generator=L.gen(seed))
    y = torch.nn.functional.one_hot(lab, C).to(L.DT).reshape(-1).to(st.F0.device)
    return (y - st.F0) / L.vnorm(y - st.F0)


def gauss_dir(L, st, seed):
    v = torch.randn(st.N, generator=L.gen(seed), dtype=L.DT).to(st.F0.device)
    return v / L.vnorm(v)


def build_directions(L, st, sd, E1, E2):
    """E1..E6 and W1..W8 (section 3), built once from the START st (st.F0)."""
    d = {"E1": E1, "E2": E2,
         "E3": label_dir(L, st, 8000 + 100 * sd + 3), "E4": gauss_dir(L, st, 9000 + 100 * sd + 4),
         "E5": label_dir(L, st, 8000 + 100 * sd + 5), "E6": gauss_dir(L, st, 9000 + 100 * sd + 6)}
    for j in (1, 2, 3, 4):
        d[f"W{j}"] = label_dir(L, st, 7000 + 100 * sd + j)
    for j in (5, 6, 7, 8):
        d[f"W{j}"] = gauss_dir(L, st, 7500 + 100 * sd + j)
    return d


def make_combo(L, CFG, name, prob, st, sd, S0, E1, E2, Y, test=None, test0=None):
    """Everything a combination's units share. Y: true-label targets (Part C final task); test/test0: ctx.test,
    ctx.test0 (None on the tiny test networks)."""
    s1 = CFG["S1_FRAC"] * S0
    s2 = CFG["S2_FRAC"] * S0
    return dict(name=name, prob=prob, st=st, sd=int(sd), S0=float(S0), sizes={"s1": s1, "s2": s2},
                dirs=build_directions(L, st, int(sd), E1, E2), Y=Y, test=test, test0=test0)


def combo_from_ctx(L, CFG, ctx):
    """Section 2/3 on the rig: E1 = ctx.dirs["label"], E2 = ctx.dirs["rand"], S0 = ctx.S0."""
    return make_combo(L, CFG, ctx.cname, ctx.prob, ctx.st, int(ctx.cb["seed"]), ctx.S0, ctx.dirs["label"],
                      ctx.dirs["rand"], ctx.Y, test=ctx.test, test0=ctx.test0)


def kernel(L, prob, theta):
    """K(theta) = J J^T (one Jacobian)."""
    return L.gram(prob.J(theta))


def q_from_kernel(L, K, vecs):
    Lc = L.chol(K)
    return [float(x @ L.csolve(Lc, x)) for x in vecs]


def q(L, prob, theta, vecs):
    """Section 3: q_x(theta) = x^T K(theta)^-1 x for every x in vecs; one Jacobian per call."""
    return q_from_kernel(L, kernel(L, prob, theta), vecs)


def ksum(K):
    """Section 3 kernel summary from the eigenvalues of K (ascending, eigvalsh)."""
    lam = torch.linalg.eigvalsh(K)
    inv = 1.0 / lam
    return dict(trK=float(lam.sum()), lam1=float(lam[-1]), lamN=float(lam[0]), inv_mean=float(inv.mean()),
                A=math.log(float(lam.mean()) * float(inv.mean())))


def finite(x):
    return x is not None and bool(torch.isfinite(x).all())


# ---------------------------------------------------------------------------------------------------------------
# Section 4: references (nearest-point shift of one leg)
# ---------------------------------------------------------------------------------------------------------------
def reference(L, prob, st, x, s, sigma):
    """Returns (scalars, vNP). vNP = st.V0(thNP - st.th0) of the outputs reached by the exact jump from the start."""
    t = st.F0 + sigma * s * x
    jr = L.jump(prob, st.th0, t, Jb=st.J0)
    thT = jr["theta"]
    sc, vL, vNP, thNP, thTH = L.measure_ext(prob, st, thT)
    usable = (sc["np_status"] == "CONVERGED" and sc["th_status"] == "CONVERGED"
              and not sc["invalid"] and not sc["below_resolution"])
    rec = dict(sc)
    rec.update(usable_spec=bool(usable), jump_status=jr["status"], jump_it_chord=jr["it_chord"],
               jump_it_newton=jr["it_newton"], jump_res_inf=jr["res_inf"])
    return rec, vNP


def ref_usable(rec):
    """Section 4 usability, plus the jump status kept by the spec.
    # spec note: the spec's usable rule does not mention the status of the first jump (t -> thT); a failed jump would put
    #       the reference on the wrong outputs, so a reference is used only if the jump also CONVERGED."""
    return bool(rec.get("usable_spec")) and rec.get("jump_status") == "CONVERGED"


def even(vp, vm):
    """EVEN PART of a reference: 0.5 (vNP(+1) + vNP(-1))."""
    return 0.5 * (vp + vm)


def reference_unit(L, combo, x_name, size, sigma):
    s = combo["sizes"][size]
    rec, vNP = reference(L, combo["prob"], combo["st"], combo["dirs"][x_name], s, sigma)
    rec.update(part="R", combo=combo["name"], dir=x_name, size=size, s=s, sign=sigma,
               status="OK" if ref_usable(rec) else "UNUSABLE")
    return rec, {"vNP": vNP}


def gram_unit(combo_name, sizes, refs):
    """Section 4 Gram matrices. refs[size][name] = dict(p=vNP(+1), m=vNP(-1), usable_p, usable_m) for every reference
    that exists. G_s1 over the available names of (E1..E6, W1..W8); G_s2 over the available names of (E1, E2)."""
    rec = dict(part="R", kind="gram", combo=combo_name, s1=sizes["s1"], s2=sizes["s2"])
    for size, names in (("s1", REF14), ("s2", ["E1", "E2"])):
        avail = [x for x in names if x in refs.get(size, {})]
        ev = [even(refs[size][x]["p"], refs[size][x]["m"]) for x in avail]
        rec[f"names_{size}"] = avail
        rec[f"G_{size}"] = [[float(a @ b) for b in ev] for a in ev]
        rec[f"even_usable_{size}"] = {x: bool(refs[size][x]["usable_p"] and refs[size][x]["usable_m"]) for x in avail}
    agree = {}
    for x in ("E1", "E2"):
        for size in ("s1", "s2"):
            if x in refs.get(size, {}):
                p, m = refs[size][x]["p"], refs[size][x]["m"]
                agree[f"{x}@{size}"] = float(p @ m) / (float(torch.linalg.vector_norm(p)) * float(torch.linalg.vector_norm(m)))
    rec["sign_agreement"] = agree
    rec["status"] = "OK"
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 5: excursion (out and back)
# ---------------------------------------------------------------------------------------------------------------
def _leg_scalars(r):
    """Scalars the library reports for one learner run (theta and tensors left out)."""
    return {k: v for k, v in r.items() if k != "theta" and not torch.is_tensor(v)}


def two_legs(L, prob, stc, F0, Ld, u, s, sigma, brk):
    """Leg 1: learner Ld from the Start stc toward F0 + sigma s u. Leg 2: from L.Start(prob, th1) (the learner's own
    end point) back to the tensor F0 itself. F0 is always the ORIGINAL st.F0 of the reference start (callers pass it).
    BREAK "noreturn": leg 2 skipped, th2 := th1, closure computed but never applied.
    Returns dict(th1, th2, rec) with rec = statuses, close, leg scalars, seconds."""
    t0 = time.time()
    r1 = L.run_learner9(prob, stc, F0 + sigma * s * u, s, Ld)
    th1 = r1["theta"]
    rec = dict(sign=sigma, leg1_status=r1["status"], leg1=_leg_scalars(r1))
    th2 = None
    r2 = None
    if brk == "noreturn":
        th2 = th1
        rec["leg2_status"] = "SKIPPED-BREAK-noreturn"
    elif r1["status"] == "CONVERGED" and finite(th1):
        st1 = L.Start(prob, th1)                          # leg 2 starts from the learner's own end point
        r2 = L.run_learner9(prob, st1, F0, s, Ld)         # target: the ORIGINAL F0 tensor itself
        th2 = r2["theta"]
        rec["leg2_status"] = r2["status"]
        rec["leg2"] = _leg_scalars(r2)
        del st1
        L.free_memory()
    else:
        rec["leg2_status"] = "NOT-RUN"
    close = L.vnorm(prob.F(th2) - F0) / s if finite(th2) else float("nan")
    rec["close"] = close
    all_finite = finite(th1) and finite(th2) and math.isfinite(close)
    if r1["status"] != "CONVERGED":
        status = r1["status"]
    elif brk != "noreturn" and (r2 is None or r2["status"] != "CONVERGED"):
        status = r2["status"] if r2 is not None else "NONFINITE"
    elif not all_finite:
        status = "NONFINITE"
    elif brk != "noreturn" and not (close <= CLOSE_TOL):
        status = "NOT-CLOSED"
    else:
        status = "OK"
    rec["status"] = status
    rec["seconds"] = time.time() - t0
    return dict(th1=th1, th2=th2, rec=rec)


def excursion(L, prob, st, Ld, u, s, sigma, brk=""):
    """Section 5 EXCURSION from the start st. Returns (scalars, vectors v, v1, th2)."""
    leg = two_legs(L, prob, st, st.F0, Ld, u, s, sigma, brk)
    th1, th2, rec = leg["th1"], leg["th2"], leg["rec"]
    v1 = st.V0(th1 - st.th0) if finite(th1) else None       # single-leg vertical displacement (at the START)
    rec["n1"] = L.vnorm(th1 - st.th0) if finite(th1) else float("nan")
    v = None
    if finite(th2):
        d = th2 - st.th0
        v = st.V0(d)                                         # vertical part at the START (original J0, K0)
        nd = L.vnorm(d)
        rec.update(nd=nd, nv=L.vnorm(v), horiz=L.vnorm(d - v) / max(nd, 1e-300))   # A1.2
    return rec, dict(v=v, v1=v1, th2=th2)


def part_a_unit(L, combo, Ld, u_name, size, ref_p, ref_m, refs_usable, brk):
    """Section 5 UNIT: both signs of one (direction, size, learner). ref_p/ref_m: vNP(u, s, +1/-1)."""
    prob, st = combo["prob"], combo["st"]
    u = combo["dirs"][u_name]
    s = combo["sizes"][size]
    rec = dict(part="A", combo=combo["name"], learner=L.learner_name9(Ld), f=L.predict_f9(Ld), dir=u_name, size=size,
               s=s, refs_usable=bool(refs_usable))
    ex = {}
    for sigma in SIGNS:
        r, vecs = excursion(L, prob, st, Ld, u, s, sigma, brk)
        ex[sigma] = (r, vecs)
        rec[f"sig{sigma:+d}"] = r
    bad = [ex[sg][0]["status"] for sg in SIGNS if ex[sg][0]["status"] != "OK"]
    rec["status"] = bad[0] if bad else "OK"
    if bad:
        return rec, {}
    vP, vM = ex[1][1]["v"], ex[-1][1]["v"]
    es = L.even_stats(vP, ref_p, vM, ref_m)
    rec.update(rho_loop=es["rho"], cos_loop=es["cos"], nr_loop=es["nr"], nvLe_loop=es["nvLe"], nvNPe=es["nvNPe"])
    for sigma, v, vnp in ((1, vP, ref_p), (-1, vM, ref_m)):
        nn_ = L.vnorm(vnp)
        rec[f"rho_sig{sigma:+d}"] = float(v @ vnp) / (nn_ * nn_)
        rec[f"cos_sig{sigma:+d}"] = L.cosine(v, vnp)
    e1 = L.even_stats(ex[1][1]["v1"], ref_p, ex[-1][1]["v1"], ref_m)
    rec.update(rho_leg=e1["rho"], cos_leg=e1["cos"], nr_leg=e1["nr"])
    # costs: u first, then W1..W8, then E1..E6 (u appears twice)
    qnames = ["u"] + W_NAMES + E_NAMES
    qvecs = [u] + [combo["dirs"][x] for x in W_NAMES + E_NAMES]
    K0 = kernel(L, prob, st.th0)
    q0 = q_from_kernel(L, K0, qvecs)
    k0 = ksum(K0)
    del K0
    qe, ke = {}, {}
    for sigma in SIGNS:
        K = kernel(L, prob, ex[sigma][1]["th2"])
        qe[sigma] = q_from_kernel(L, K, qvecs)
        ke[sigma] = ksum(K)
        del K
    rec["q_names"] = qnames
    rec["q0"] = dict(zip(qnames, q0))
    for sigma in SIGNS:
        rec[f"q_end_sig{sigma:+d}"] = dict(zip(qnames, qe[sigma]))
        rec[f"dq_sig{sigma:+d}"] = {nm: qe[sigma][i] - q0[i] for i, nm in enumerate(qnames)}
    rec["dq"] = {nm: 0.5 * ((qe[1][i] - q0[i]) + (qe[-1][i] - q0[i])) for i, nm in enumerate(qnames)}
    rec["ksum0"] = k0
    rec["ksum_end_sig+1"], rec["ksum_end_sig-1"] = ke[1], ke[-1]
    rec["dksum"] = {k: 0.5 * ((ke[1][k] - k0[k]) + (ke[-1][k] - k0[k])) for k in k0}
    vecs = {"v+1": vP, "v-1": vM, "v1+1": ex[1][1]["v1"], "v1-1": ex[-1][1]["v1"]}
    return rec, vecs


def identity_unit(L, combo, u_name, evens):
    """Section 5 C2 IDENTITY. evens[size] = vNPe(u, size) for the sizes available (s1 required)."""
    prob, st = combo["prob"], combo["st"]
    u = combo["dirs"][u_name]
    ve1 = evens["s1"]
    h = L.vnorm(ve1)
    vh = ve1 / h
    qp = q(L, prob, st.th0 + h * vh, [u])[0]
    qm = q(L, prob, st.th0 - h * vh, [u])[0]
    Dq = (qp - qm) / (2.0 * h)
    rec = dict(part="A", kind="ID", combo=combo["name"], dir=u_name, h=h, q_plus=qp, q_minus=qm, Dq=Dq,
               pred={}, iota={}, status="OK")
    for size, ve in evens.items():
        s = combo["sizes"][size]
        rec["pred"][size] = -2.0 * L.vnorm(ve) / s ** 2
        rec["iota"][size] = Dq / rec["pred"][size]
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 6: Part B star of excursions
# ---------------------------------------------------------------------------------------------------------------
def part_b_unit(L, combo, Ld, n_m, evens, brk):
    """Section 6 UNIT. evens[name] = vNPe(name, s1) for the available names of (E1..E6, W1..W8)."""
    prob, st = combo["prob"], combo["st"]
    s = combo["sizes"]["s1"]
    names = list(REF14)
    qvecs = [combo["dirs"][x] for x in names]
    K0 = kernel(L, prob, st.th0)
    # spec note: q at theta0 is not listed in section 6 but HB2 / the dq_w(m) table need it: computed once per unit
    rec = dict(part="B", combo=combo["name"], learner=L.learner_name9(Ld), f=L.predict_f9(Ld), s1=s, n_m=n_m,
               q_names=names, q0=dict(zip(names, q_from_kernel(L, K0, qvecs))), ksum0=ksum(K0),
               ref_names=[x for x in names if x in evens], checkpoints=[], excursions=[])
    del K0
    th = st.th0
    status = "OK"
    vc = None
    for m in range(1, n_m + 1):
        E = combo["dirs"][f"E{m}"]
        for sigma in SIGNS:
            try:                                                        # NOTE (review): keep the checkpoints already made
                stc = L.Start(prob, th)                                 # from the learner's own current weights
                leg = two_legs(L, prob, stc, st.F0, Ld, E, s, sigma, brk)   # leg-1 target from the ORIGINAL st.F0
                del stc
            except Exception as e:
                rec["excursions"].append(dict(m=m, sign=sigma, status="EXC:" + type(e).__name__, error=repr(e)))
                status = "EXC:" + type(e).__name__
                break
            L.free_memory()
            leg["rec"]["m"] = m
            rec["excursions"].append(leg["rec"])
            # spec note: closure is checked after EVERY excursion (two_legs status), not only at the checkpoint
            if leg["rec"]["status"] != "OK":
                status = leg["rec"]["status"]
                break
            th = leg["th2"]
        if status != "OK":
            break
        vc = st.V0(th - st.th0)
        close = L.vnorm(prob.F(th) - st.F0) / s
        if brk != "noreturn" and not (close <= CLOSE_TOL):
            status = "NOT-CLOSED"
            break
        K = kernel(L, prob, th)
        cp = dict(m=m, nvc=L.vnorm(vc), nd=L.vnorm(th - st.th0), close=close,
                  dots={x: float(vc @ evens[x]) for x in names if x in evens},
                  q=dict(zip(names, q_from_kernel(L, K, qvecs))), ksum=ksum(K),
                  statuses=[e["status"] for e in rec["excursions"] if e["m"] == m])
        del K
        rec["checkpoints"].append(cp)
    rec["status"] = status
    rec["m_done"] = len(rec["checkpoints"])
    return rec, ({"vc": vc} if vc is not None else {})


# ---------------------------------------------------------------------------------------------------------------
# Section 7: Part C relabelling tasks, linear control
# ---------------------------------------------------------------------------------------------------------------
def task_target(L, combo, m):
    st = combo["st"]
    C = st.prob.C
    n = st.N // C
    lab = torch.randint(0, C, (n,), generator=L.gen(6000 + 100 * combo["sd"] + m))
    return torch.nn.functional.one_hot(lab, C).to(L.DT).reshape(-1).to(st.F0.device)


def part_c_unit(L, combo, Ld, n_tasks, final):
    prob, st = combo["prob"], combo["st"]
    wvecs = [combo["dirs"][x] for x in W_NAMES]
    K0 = kernel(L, prob, st.th0)
    # spec note: task 0 (the start) stored for the tables; not listed in section 7. A failing task is recorded
    #       (status, fit, seconds; no q/ksum) before the unit stops.
    rec = dict(part="C", combo=combo["name"], learner=L.learner_name9(Ld), f=L.predict_f9(Ld), n_tasks=n_tasks,
               start=dict(q=dict(zip(W_NAMES, q_from_kernel(L, K0, wvecs))), ksum=ksum(K0)), tasks=[])
    del K0
    th = st.th0
    status = "OK"
    for m in range(1, n_tasks + 1):
        t0 = time.time()
        Y = task_target(L, combo, m)
        try:                                                            # NOTE (review): keep the tasks already done
            stc = L.Start(prob, th)
            s_m = L.vnorm(Y - stc.F0)
            r = L.run_learner9(prob, stc, Y, s_m, Ld)
            del stc
        except Exception as e:
            status = "EXC:" + type(e).__name__
            rec["tasks"].append(dict(m=m, status=status, fit=float("nan"), error=repr(e), seconds=time.time() - t0))
            break
        L.free_memory()
        th = r["theta"]
        fit = L.vnorm(prob.F(th) - Y) / s_m if finite(th) else float("nan")
        cp = dict(m=m, status=r["status"], fit=fit, s_m=s_m, run=_leg_scalars(r))
        if r["status"] != "CONVERGED":
            status = r["status"]
        elif not (fit <= FIT_TOL):
            status = "NOT-FIT"
        if status == "OK":
            K = kernel(L, prob, th)
            cp.update(q=dict(zip(W_NAMES, q_from_kernel(L, K, wvecs))), ksum=ksum(K), dist=L.vnorm(th - st.th0))
            del K
        cp["seconds"] = time.time() - t0
        rec["tasks"].append(cp)
        if status != "OK":
            break
    rec["status"] = status
    rec["tasks_done"] = sum(1 for cp in rec["tasks"] if "q" in cp)
    if status == "OK" and final:
        t0 = time.time()
        stf = L.Start(prob, th)
        Yt = combo["Y"]
        sf = L.vnorm(Yt - stf.F0)
        rf = L.run_learner9(prob, stf, Yt, sf, make_learner(L, FINAL_LEARNER))
        del stf
        L.free_memory()
        fit = L.vnorm(prob.F(rf["theta"]) - Yt) / sf if finite(rf["theta"]) else float("nan")
        fin = dict(status=rf["status"], fit=fit, s=sf)
        if finite(rf["theta"]):
            td = combo["test"]()
            Ft = L.test_outputs(prob, td["X"], rf["theta"])
            fin["test_loss"], fin["test_acc"] = L.test_loss_acc(Ft, td)
        fin["seconds"] = time.time() - t0
        rec["final"] = fin
    return rec, {}


def linear_unit(L, combo, Ld, brk):
    """Section 7 LINEAR CONTROL: direction E1 (= ctx.dirs["label"]), size s1, sign +1."""
    r, _ = excursion(L, combo["prob"], combo["st"], Ld, combo["dirs"]["E1"], combo["sizes"]["s1"], 1, brk)
    rec = dict(part="V", kind="lin", combo=combo["name"], learner=L.learner_name9(Ld), excursion=r,
               nd=r.get("nd", float("nan")), n1=r["n1"], status=r["status"])
    rec["nd_over_n1"] = rec["nd"] / rec["n1"] if rec["n1"] > 0 else float("nan")
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 13 V1 / section 10 ref11.json: environment numbers
# ---------------------------------------------------------------------------------------------------------------
ENV_TOL = dict(S0=1e-8, F0_norm=1e-8, F0_sum=1e-8, lam1=1e-8, lamN=1e-6, trK=1e-8, loss_0=1e-8, acc_0=1e-8,
               label_rand_dot=1e-8)


def env_numbers(L, combo, with_test):
    st = combo["st"]
    d = dict(S0=combo["S0"], F0_norm=L.vnorm(st.F0), F0_sum=float(st.F0.sum()), lam1=st.lam1, lamN=st.lamN, trK=st.trK)
    if with_test:
        _, l0, a0 = combo["test0"]()
        d.update(loss_0=l0, acc_0=a0, label_rand_dot=float(combo["dirs"]["E1"] @ combo["dirs"]["E2"]))
    return d


def env_unit(L, combo, ref, with_test):
    now = env_numbers(L, combo, with_test)
    rec = dict(part="V", kind="env", combo=combo["name"], now=now, ref=ref)
    if ref is None:
        rec["status"] = "NO-REF11"
        return rec, {}
    rel = {}
    # spec note: relative difference read as |a - b| / |b|
    for k, v in now.items():
        b = ref[k]
        rel[k] = abs(v - b) / abs(b) if b != 0 else abs(v - b)
    rec["rel"] = rel
    rec["tol"] = {k: ENV_TOL[k] for k in now}
    rec["pass"] = all(rel[k] <= ENV_TOL[k] for k in now)
    rec["status"] = "OK"
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 8: names, plan, signature, atomic storage, resume
# ---------------------------------------------------------------------------------------------------------------
def ref_name(c, x, size, sigma):
    return f"R__{c}__{x}__{size}__{sigma:+d}"


def a_name(c, u, size, lname):
    return f"A__{c}__{u}__{size}__{lname}"


def plan(L, settings, combos, parts):
    """Every unit of the requested parts, in the order of work (section 8)."""
    items = []
    # spec note: QUICK says "combination tanh only"; the linear control and V1 still run when part V is requested
    if "V" in parts:
        items.append(dict(name=f"V__{LIN_NAME}__env", part="V", kind="env", combo=LIN_NAME))
        for spec in settings["LIN"]:
            nm = L.learner_name9(make_learner(L, spec))
            items.append(dict(name=f"V__{LIN_NAME}__lin__{nm}", part="V", kind="lin", combo=LIN_NAME, learner=spec,
                              lname=nm))
    for c in combos:
        if "V" in parts:
            items.append(dict(name=f"V__{c}__env", part="V", kind="env", combo=c))
        if "R" in parts:
            for size in ("s1", "s2"):
                for x in settings["REFS"][size]:
                    for sg in SIGNS:
                        items.append(dict(name=ref_name(c, x, size, sg), part="R", kind="ref", combo=c, dir=x,
                                          size=size, sign=sg))
            items.append(dict(name=f"G__{c}", part="R", kind="gram", combo=c))
        if "A" in parts:
            for u in settings["A_DIRS"]:
                for size in settings["A_SIZES"]:
                    for spec in settings["LA"]:
                        nm = L.learner_name9(make_learner(L, spec))
                        it = dict(name=a_name(c, u, size, nm), part="A", kind="A", combo=c, dir=u, size=size,
                                  learner=spec, lname=nm)
                        items.append(it)
                        if settings["BREAK"] == "repro" and (c, u, size, nm) == REPRO_UNIT:
                            items.append(dict(it, name=it["name"] + "__repro"))
            for u in settings["A_DIRS"]:
                items.append(dict(name=f"A__{c}__ID__{u}", part="A", kind="ID", combo=c, dir=u))
        if "B" in parts:
            for spec in settings["LB"]:
                nm = L.learner_name9(make_learner(L, spec))
                items.append(dict(name=f"B__{c}__{nm}", part="B", kind="B", combo=c, learner=spec, lname=nm))
        if "C" in parts:
            for spec in settings["LC"]:
                nm = L.learner_name9(make_learner(L, spec))
                items.append(dict(name=f"C__{c}__{nm}", part="C", kind="C", combo=c, learner=spec, lname=nm))
    return items


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(settings, lib_path, cfg_path):
    d = dict(settings=settings, part9lib_md5=file_md5(lib_path), cfg11_md5=file_md5(cfg_path))
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def unit_path(out, name):
    return os.path.join(out, "units", name + ".json")


def vec_path(out, name, what):
    return os.path.join(out, "vecs", f"{name}__{what}.pt")


def _unjson(o):
    """Inverse of the library's _jsonable for non-finite floats (stored as the strings 'nan', 'inf', '-inf')."""
    if isinstance(o, dict):
        return {k: _unjson(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unjson(v) for v in o]
    if isinstance(o, str) and o in ("nan", "inf", "-inf"):
        return float(o)
    return o


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return _unjson(json.load(f))


def read_unit(out, name):
    p = unit_path(out, name)
    if not os.path.exists(p):
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def unit_done(out, name, sig):
    """Resume rule: the json exists with the current signature.
    # spec note: section 8 says only 'json exists and signature equal'. Also required here: every vector file the unit
    #       recorded still exists (otherwise dependent units would silently miss their references). A unit stored with
    #       status ERROR counts as done (literal reading); delete its json to retry it."""
    rec = read_unit(out, name)
    if rec is None or rec.get("signature") != sig:
        return False
    if rec.get("status") == "ERROR":                 # NOTE (review): unexpected errors (out of memory, disk) are retried
        return False
    return all(os.path.exists(os.path.join(out, "vecs", fn)) for fn in rec.get("vec_files", []))


def load_vec(L, out, name, what):
    return torch.load(vec_path(out, name, what), map_location="cpu")[what].to(L.DEVICE)


def run_unit(L, out, name, sig, fn):
    """Run fn() -> (record, vectors); vectors first, then the json, each written atomically (temp file + rename), once,
    when the unit is complete. Exceptions inside fn are stored as status ERROR with the message."""
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    os.makedirs(os.path.join(out, "vecs"), exist_ok=True)
    t0 = time.time()
    try:
        rec, vecs = fn()
    except Exception as e:
        rec, vecs = dict(status="ERROR", error=repr(e), traceback=traceback.format_exc()), {}
    rec["name"] = name
    rec["signature"] = sig
    rec["seconds"] = time.time() - t0
    rec["vec_files"] = []
    for what, v in vecs.items():
        if v is None:
            continue
        p = vec_path(out, name, what)
        L.save_vecs_atomic(p, {what: v})
        rec["vec_files"].append(os.path.basename(p))
    L.write_json_atomic(unit_path(out, name), rec)
    L.free_memory()
    return rec


# ---------------------------------------------------------------------------------------------------------------
# Section 9 formulas shared by the report and the tests (stored scalars only)
# ---------------------------------------------------------------------------------------------------------------
def extrap(x_s1, x_s2):
    """x_x = (4 x(s2) - x(s1)) / 3."""
    return (4.0 * x_s2 - x_s1) / 3.0


def within(x, target, tol):
    """|x - target| <= tol * max(1, |target|) (False for non-finite x)."""
    return isinstance(x, (int, float)) and math.isfinite(x) and abs(x - target) <= tol * max(1.0, abs(target))


def sign(x):
    return (x > 0) - (x < 0)


def loop_law_ok(rho_x, cos2, nr2, g):
    """HP1 for one unit: |rho_x - g| <= 0.03 max(1,|g|); |g| >= 0.1: cos(s2) sign(g) >= 0.99; else nr(s2) <= 0.05."""
    if not within(rho_x, g, 0.03):
        return False
    if abs(g) >= 0.1:
        return math.isfinite(cos2) and cos2 * sign(g) >= 0.99
    return math.isfinite(nr2) and nr2 <= 0.05


def kappa(dq_u, nvNPe_sq, s):
    """HP3: kappa(s) = dq_u(s) / M_uu(s), M_uu(s) = 2 |vNPe(u, s)|^2 / s^2."""
    return dq_u / (2.0 * nvNPe_sq / s ** 2)


def gram_lookup(G, names, a, b):
    if a not in names or b not in names:
        return None
    return G[names.index(a)][names.index(b)]


def star_rho(dots, nvc, G, names, m):
    """Part B: Vpred(m) = 2 sum_{k<=m} vNPe(E_k, s1) from the stored dot products <vc, vNPe(a)> and the Gram matrix.
    Returns (rho_B, cos, |vc| / |Vpred|, |Vpred|) or None if a needed entry is missing."""
    Ek = [f"E{k}" for k in range(1, m + 1)]
    if any(e not in dots or e not in names for e in Ek):
        return None
    dot = 2.0 * sum(dots[e] for e in Ek)
    vp2 = 4.0 * sum(gram_lookup(G, names, a, b) for a in Ek for b in Ek)
    vpn = math.sqrt(vp2)
    return dot / vp2, (dot / (nvc * vpn) if nvc > 0 else float("nan")), nvc / vpn, vpn


def through_origin(xs, ys):
    """Least-squares line through the origin: slope = sum xy / sum x^2; R^2 = 1 - SSres / sum y^2."""
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    syy = sum(y * y for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan")
    slope = sxy / sxx
    ssres = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
    return slope, 1.0 - ssres / syy


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = 0.5 * (i + j) + 1.0          # average rank of ties
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, the = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - the) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - the) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


IGNORE_IN_REPRO = ("name", "seconds", "vec_files", "signature")


def compare_repro(a, b, path=""):
    """BREAK "repro": list of differences between two unit records (timings and names ignored; floats compared exactly)."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_IN_REPRO:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: present in one record only")
            else:
                diffs += compare_repro(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: lengths {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += compare_repro(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs
