"""Runs the SPEC4.md grids; writes results4.json incrementally (lock + merge, so several
worker processes may run at once; jobs are claimed in the SPEC4 section 6 run order).

Usage:  python run4.py [--scratch DIR] [--deadline EPOCH] [--skip-mom-widths 128] GRID [GRID ...]
Grids:  S6 MAIN EPS WIDE ADAM R2
  S6   : linear control (tcr.LinearModel), seeds 0-4, eps 0.1, GN and GD(1) (2000 steps/learn)
  MAIN : h in {32, 64, 128}, seeds 0-4, eps 0.025, 5 learners, 5 paths, measured + a-priori C
  EPS  : h in {32, 128}, seeds 0-4, eps in {0.2,...,0.0125}, 5 learners, oab and tri+
  WIDE : h = 512, seeds 0-2, eps 0.025, GN and GD(1), oab / tri+ / tri- / oneway
  ADAM : h in {32, 128}, seeds 0-2, eps 0.025, oab and tri+
  R2   : post-processing of MAIN (learning-rate dependence), from saved V_meas vectors
A job is one (grid, h, seed, eps, learner). Jobs already in results4.json are skipped.
Thread count is fixed by tcr4 (THREADS = 2) for every process.
"""
import tcr4  # first: fixes thread count
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import tcr3
from tcr4 import LEARNERS, CLOSED, Law, compare, setup, make_targets, walk_dirs, make_learn, \
    run_paths, apriori_C, c_parts
from tcr3 import nrm, cos, jk, proj_v
from tcr import Timeout

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results4.json")
LOCK = OUT + ".lock"

SEEDS = [0, 1, 2, 3, 4]
MAIN_WIDTHS = [32, 64, 128]
EPS_WIDTHS = [32, 128]
EPS_LIST = [0.2, 0.1, 0.05, 0.025, 0.0125]
MAIN_EPS = 0.025
MAIN_LEARNERS = ["GN", "GD0.25", "GD1", "GD1.8", "MOM"]
ALL_PATHS = ["oab", "tri+", "tri-", "walk8", "oneway"]
WIDE_SEEDS = [0, 1, 2]
ADAM_WIDTHS = [32, 128]
ADAM_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# results file: lock + merge
# ---------------------------------------------------------------------------
KEYF = ("grid", "h", "seed", "eps", "learner")


def key(rec):
    return "|".join(str(rec.get(k)) for k in KEYF)


def _lock():
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except (FileExistsError, PermissionError):
            if time.time() - t0 > 300:
                raise RuntimeError("could not acquire lock " + LOCK)
            time.sleep(0.05)


def _unlock():
    for _ in range(200):
        try:
            os.remove(LOCK)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.05)


def load():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh)
    return {"meta": {"runs": [], "threads": tcr4.THREADS}, "records": []}


def save_records(new_recs, meta_run=None):
    _lock()
    try:
        db = load()
        idx = {key(r): i for i, r in enumerate(db["records"])}
        for r in new_recs:
            k = key(r)
            if k in idx:
                db["records"][idx[k]] = r
            else:
                db["records"].append(r)
        if meta_run is not None:
            db["meta"]["runs"].append(meta_run)
        tmp = OUT + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(db, fh, indent=1)
        os.replace(tmp, OUT)
    finally:
        _unlock()


def done_keys():
    _lock()
    try:
        db = load()
    finally:
        _unlock()
    return {key(r) for r in db["records"] if r.get("status") not in ("ERROR", "TIMEOUT")}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# One MLP job: (grid, h, seed, eps, learner) over a list of paths
# ---------------------------------------------------------------------------
def path_record(S, L, base, name, res, apri_C, gn_oneway, eps):
    out = {"status": res["status"],
           "learns": [{k: v for k, v in i.items() if k != "theta"} for i in res["infos"]]}
    C = res["C"]
    Cs, Ca = c_parts(C)
    out.update({"C": C.tolist(), "C_F": float(np.linalg.norm(C)),
                "C_sym_F": float(np.linalg.norm(Cs)), "C_area_F": float(np.linalg.norm(Ca))})
    return out


def mlp_job(grid, h, seed, eps, learner, paths, deadline, vecdir):
    t0 = time.time()
    S = setup(h, seed)
    L = Law(S.model, S.theta_A, S.X)
    cu = tcr3.self_term(S.model, S.theta_A, S.X, S.u)
    buv = tcr3.bracket(S.model, S.theta_A, S.X, S.u, S.v)
    phiA = tcr3.stiffness(S.model, S.theta_A, S.X, S.u)
    KA = L.K.numpy()
    lamK = np.linalg.eigvalsh(KA)
    base = {"phi_u_A": phiA, "c_u_norm": nrm(cu), "b_uv_norm": nrm(buv),
            "lam_max_A": float(lamK[-1]), "lam_min_A": float(lamK[0]),
            "yA_norm": float(S.yA_norm), "P": S.model.P, "base_gn_iters": S.base_info["iters"],
            "law_time": time.time() - t0}
    wd = walk_dirs(S) if "walk8" in paths else None
    tg = make_targets(S, eps, wd)
    tg_np = {k: v.numpy() for k, v in tg.items()}
    learn = make_learn(S, learner, deadline=deadline)
    res = run_paths(S, learn, tg, paths)
    gn_one = None
    if "oneway" in paths and learner != "GN":
        gn_one = run_paths(S, make_learn(S, "GN"), tg, ["oneway"])["oneway"]
    has_apri = LEARNERS[learner][0] != "ADAM"
    gn_apri_one = (apriori_C(KA, S.F_A_np, [tg_np[t] for t in tcr4.PATHS["oneway"]], "GN")[0]
                   if gn_one is not None else None)
    rec_paths, vm_closed = {}, {}
    for p in paths:
        r = res[p]
        pr = {"status": r["status"],
              "learns": [dict(i) for i in r["infos"]]}
        C = r["C"]
        Cs, Ca = c_parts(C)
        pr.update({"C": C.tolist(), "C_F": float(np.linalg.norm(C)),
                   "C_sym_F": float(np.linalg.norm(Cs)), "C_area_F": float(np.linalg.norm(Ca))})
        if has_apri:
            Cap, ainf = apriori_C(KA, S.F_A_np, [tg_np[t] for t in tcr4.PATHS[p]], learner)
            Caps, Capa = c_parts(Cap)
            pr["apri"] = {"C": Cap.tolist(), "C_F": float(np.linalg.norm(Cap)),
                          "C_sym_F": float(np.linalg.norm(Caps)),
                          "C_area_F": float(np.linalg.norm(Capa)),
                          "status": ainf["status"], "steps": ainf["steps"]}
        if p == "oneway":
            if gn_one is None:
                rec_paths[p] = pr       # GN itself: D = 0 by definition
                continue
            pr["gn_status"] = gn_one["status"]
            Dm = L.Pv(r["theta"] - gn_one["theta"])
            Cd = C - gn_one["C"]
            Vp = L(Cd)
            pr["meas"] = compare(Dm, Vp)
            Cds, Cda = c_parts(Cd)
            pr["Vsym_norm"], pr["Varea_norm"] = nrm(L(Cds)), nrm(L(Cda))
            pr["Cdiff_sym_F"], pr["Cdiff_area_F"] = float(np.linalg.norm(Cds)), float(np.linalg.norm(Cda))
            if has_apri:
                Cad = np.array(pr["apri"]["C"]) - gn_apri_one
                pr["apri"]["cmp"] = compare(Dm, L(Cad))
            np.save(os.path.join(vecdir, f"{grid}_{h}_{seed}_{eps}_{learner}_{p}.npy"), Dm.numpy())
            rec_paths[p] = pr
            continue
        Vm = L.Pv(r["theta"] - S.theta_A)
        vm_closed[p] = Vm
        Vp = L(C)
        Vs, Va = L(Cs), L(Ca)
        pr["Vmeas_norm"] = nrm(Vm)
        pr["meas"] = compare(Vm, Vp)
        pr["Vsym_norm"], pr["Varea_norm"] = nrm(Vs), nrm(Va)
        pr["cos_Vmeas_Vsym"] = cos(Vm, Vs)
        pr["end_res_A"] = (S.model.f(r["theta"], S.X) - S.yA).abs().max().item()
        pr["horiz_frac"] = nrm(r["theta"] - S.theta_A - Vm) / max(nrm(r["theta"] - S.theta_A), 1e-300)
        phi_end = tcr3.stiffness(S.model, r["theta"], S.X, S.u)
        pr["dphi_meas"] = phi_end - phiA
        pr["dphi_pred_meas"] = -torch.dot(cu, Vp).item()
        if has_apri:
            Cap = np.array(pr["apri"]["C"])
            Vap = L(Cap)
            pr["apri"]["cmp"] = compare(Vm, Vap)
            pr["apri"]["Vsym_norm"] = nrm(L(c_parts(Cap)[0]))
            pr["apri"]["Varea_norm"] = nrm(L(c_parts(Cap)[1]))
            pr["dphi_pred_apri"] = -torch.dot(cu, Vap).item()
        np.save(os.path.join(vecdir, f"{grid}_{h}_{seed}_{eps}_{learner}_{p}.npy"), Vm.numpy())
        rec_paths[p] = pr
    cross = {}
    if "tri+" in vm_closed and "tri-" in vm_closed:
        a = (vm_closed["tri+"] - vm_closed["tri-"]) / 2
        cross["R4"] = {"cos_a_buv": cos(a, buv), "ratio": nrm(a) / (eps ** 2 / 2 * nrm(buv)),
                       "a_norm": nrm(a)}
    total_steps = sum(i.get("steps", 0) for r in res.values() for i in r["infos"])
    return {"grid": grid, "h": h, "seed": seed, "eps": eps, "learner": learner,
            "status": "OK", "base": base, "paths": rec_paths, "cross": cross,
            "wall": time.time() - t0}


# ---------------------------------------------------------------------------
# S6 (linear control)
# ---------------------------------------------------------------------------
def s6_job(seed, eps=0.1):
    t0 = time.time()
    out = {}
    for tag, uvt, gn_fixed in (("primary", False, False),
                               ("supp_uv_teacher_scale", True, False),
                               ("supp_gn_2000_iters", False, True)):
        s = tcr3.Setup(None, seed, linear=True, uv_from_teacher_for_linear=uvt)
        s.F_A = s.model.f(s.theta_A, s.X)
        s.F_A_np = s.F_A.numpy().copy()
        J, K = jk(s.model, s.theta_A, s.X)
        tg = make_targets(s, eps)
        vals = {}
        for name in ("GN", "GD1"):
            if name == "GN":
                learn = make_learn(s, "GN", fixed_steps=2000 if gn_fixed else None)
            else:
                learn = make_learn(s, "GD1", fixed_steps=2000)
            res = run_paths(s, learn, tg, ["oab", "tri+"])
            for p in ("oab", "tri+"):
                vals[f"{name}|{p}"] = {
                    "Pv_norm": nrm(proj_v(J, K, res[p]["theta"] - s.theta_A)),
                    "status": res[p]["status"],
                    "learns": [{k: v for k, v in i.items()} for i in res[p]["infos"]]}
        out[tag] = vals
    return {"grid": "S6", "h": None, "seed": seed, "eps": eps, "learner": "GN+GD1",
            "status": "OK", "S6": out, "wall": time.time() - t0}


# ---------------------------------------------------------------------------
# R2 post-processing
# ---------------------------------------------------------------------------
def r2_post(vecdir):
    db = load()
    recs = {key(r): r for r in db["records"]}
    out = []
    for h in MAIN_WIDTHS:
        for seed in SEEDS:
            rr = {}
            for p in ("oab", "tri+"):
                ref = recs.get(f"MAIN|{h}|{seed}|{MAIN_EPS}|GD1")
                fref = os.path.join(vecdir, f"MAIN_{h}_{seed}_{MAIN_EPS}_GD1_{p}.npy")
                if ref is None or not os.path.exists(fref):
                    continue
                V1 = np.load(fref)
                C1s = c_parts(np.array(ref["paths"][p]["C"]))[0]
                for c in ("0.25", "1.8"):
                    r = recs.get(f"MAIN|{h}|{seed}|{MAIN_EPS}|GD{c}")
                    f = os.path.join(vecdir, f"MAIN_{h}_{seed}_{MAIN_EPS}_GD{c}_{p}.npy")
                    if r is None or not os.path.exists(f):
                        continue
                    V = np.load(f)
                    Cs = c_parts(np.array(r["paths"][p]["C"]))[0]
                    rr[f"{p}|{c}"] = {
                        "V_rel": float(np.linalg.norm(V - V1) / np.linalg.norm(V1)),
                        "Csym_rel": float(np.linalg.norm(Cs - C1s) / np.linalg.norm(C1s)),
                        "status": r["paths"][p]["status"], "ref_status": ref["paths"][p]["status"]}
            out.append({"grid": "R2", "h": h, "seed": seed, "eps": MAIN_EPS, "learner": "GD",
                        "status": "OK", "R2": rr})
    save_records(out)
    log(f"R2: {len(out)} records")


# ---------------------------------------------------------------------------
# Job lists (SPEC4 section 6 run order) and the worker loop
# ---------------------------------------------------------------------------
def jobs(grid, skip_mom_widths=()):
    J = []
    if grid == "S6":
        J = [("S6", None, s, 0.1, "GN+GD1", None) for s in SEEDS]
    elif grid == "MAIN":
        for h in MAIN_WIDTHS:
            for s in SEEDS:
                for l in MAIN_LEARNERS:
                    if l == "MOM" and h in skip_mom_widths:
                        continue
                    J.append(("MAIN", h, s, MAIN_EPS, l, ALL_PATHS))
    elif grid == "EPS":
        for h in EPS_WIDTHS:
            for s in SEEDS:
                for e in EPS_LIST:
                    for l in MAIN_LEARNERS:
                        if l == "MOM" and h in skip_mom_widths:
                            continue
                        J.append(("EPS", h, s, e, l, ["oab", "tri+"]))
    elif grid == "WIDE":
        for s in WIDE_SEEDS:
            for l in ("GN", "GD1"):
                J.append(("WIDE", 512, s, MAIN_EPS, l, ["oab", "tri+", "tri-", "oneway"]))
    elif grid == "ADAM":
        for h in ADAM_WIDTHS:
            for s in ADAM_SEEDS:
                J.append(("ADAM", h, s, MAIN_EPS, "ADAM", ["oab", "tri+"]))
    return J


def claim(claimdir, k):
    fn = os.path.join(claimdir, k.replace("|", "_").replace("+", "p") + ".claim")
    try:
        fd = os.open(fn, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main(argv):
    scratch, deadline, skip_mom = None, None, ()
    grids = []
    i = 0
    while i < len(argv):
        if argv[i] == "--scratch":
            scratch = argv[i + 1]
            i += 2
        elif argv[i] == "--deadline":
            deadline = float(argv[i + 1])
            i += 2
        elif argv[i] == "--skip-mom-widths":
            skip_mom = tuple(int(x) for x in argv[i + 1].split(",") if x)
            i += 2
        else:
            grids.append(argv[i])
            i += 1
    assert scratch, "--scratch DIR required (vectors and claim files go there)"
    vecdir = os.path.join(scratch, "vecs4")
    claimdir = os.path.join(scratch, "claims4")
    os.makedirs(vecdir, exist_ok=True)
    os.makedirs(claimdir, exist_ok=True)
    log(f"pid {os.getpid()} threads torch={torch.get_num_threads()} grids={grids} "
        f"deadline={deadline} skip_mom={skip_mom}")
    for grid in grids:
        if grid == "R2":
            r2_post(vecdir)
            continue
        t_start = time.time()
        n_run = 0
        stopped = False
        done = done_keys()
        for (g, h, s, e, l, paths) in jobs(grid, skip_mom):
            kr = {"grid": g, "h": h, "seed": s, "eps": e, "learner": l}
            k = key(kr)
            if k in done or not claim(claimdir, k):
                continue
            if deadline is not None and time.time() > deadline:
                stopped = True
                rec = dict(kr, status="NOT RUN", error="budget deadline passed before start")
                save_records([rec])
                log(f"{k} NOT RUN (deadline)")
                continue
            t0 = time.time()
            try:
                if g == "S6":
                    rec = s6_job(s, e)
                else:
                    rec = mlp_job(g, h, s, e, l, paths, deadline, vecdir)
            except Timeout as ex:
                rec = dict(kr, status="TIMEOUT", error=str(ex))
                stopped = True
            except Exception as ex:  # noqa
                rec = dict(kr, status="ERROR", error=f"{type(ex).__name__}: {ex}",
                           tb=traceback.format_exc())
            rec["wall"] = time.time() - t0
            rec["pid"] = os.getpid()
            save_records([rec])
            n_run += 1
            st = ""
            if "paths" in rec:
                st = " ".join(f"{p}:{v['status']}" for p, v in rec["paths"].items())
            log(f"{k} {rec['status']} wall={rec['wall']:.1f}s {st}")
        save_records([], meta_run={"grid": grid, "pid": os.getpid(), "n_run": n_run,
                                   "wall": time.time() - t_start, "stopped": stopped,
                                   "threads": torch.get_num_threads(),
                                   "skip_mom_widths": list(skip_mom),
                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"done {grid}: {n_run} jobs in {time.time() - t_start:.1f}s stopped={stopped}")


if __name__ == "__main__":
    main(sys.argv[1:])
