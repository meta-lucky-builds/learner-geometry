"""Runs FG1-FG5 and RG1-RG5 of SPEC2.md; writes results2.json incrementally.

Usage:  python run2.py RG1 RG2 RG3 RG4 RG5      python run2.py FG1 FG2 FG3 FG4      python run2.py FG5
Conditions already present in results2.json are skipped (resume).
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

import tcr
import tcr2
from tcr import TaskFamily, NotConverged, Timeout, gn_learn, gd_learn, cosine

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results2.json")

WIDTHS = [32, 128, 512]
SEEDS = [0, 1, 2, 3, 4]
FG1_EPS = [0.1, 0.05, 0.025]
FG1_N = [3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
FG2_C = [0.25, 0.5, 1, 2]
FG3_EPS = [0.2, 0.1, 0.05, 0.025, 0.0125]
FG4_M = [1, 2, 4, 8, 16]
FG5_N = [4, 8, 16]
GRID_CAP_S = 60 * 60
FG5_CAP_S = 40 * 60

R_SEEDS = list(range(10))
RG1_KAPPA = [1.05, 1.25, 1.5, 2, 3, 4, 8, 16]
RG2_T = [10, 25, 50, 100]
RG3_N = [1, 4, 16]
PARTIAL_K = [1, 3, 10]
EIG_DMAX = 1600


def load():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh)
    return {"meta": {"runs": []}, "records": []}


def save(db):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(db, fh, indent=1)
    os.replace(tmp, OUT)


KEYF = ("grid", "learner", "h", "seed", "eps", "schedule", "k", "T", "n", "kappa", "E", "step_scale")


def key(rec):
    return "|".join(str(rec.get(k)) for k in KEYF)


def nrm(v):
    return torch.linalg.norm(v).item()


# ---------------------------------------------------------------------------
# Part F
# ---------------------------------------------------------------------------
_base = {}


def base_point(h, seed, learner="gn", deadline=None):
    k = (h, seed, learner)
    if k not in _base:
        fam = TaskFamily(seed)
        m = tcr2.FastMLP(h)
        th0 = m.init(seed)
        A = (fam.X_A, fam.teacher(fam.X_A))
        t0 = time.time()
        try:
            if learner == "gn":
                thA, info = gn_learn(m, th0, A)
            else:
                thA, info = gd_learn(m, th0, A, deadline=deadline)
            _base[k] = (fam, m, thA, dict(info, wall=time.time() - t0), None)
        except NotConverged as e:
            _base[k] = (fam, m, None, None, f"base point INVALID: {e}")
    return _base[k]


def info_summary(infos):
    out = {"n_learns": len(infos)}
    if infos and "iters" in infos[0]:
        out.update(max_iters=max(i["iters"] for i in infos),
                   ls_fail=sum(i["ls_fail"] for i in infos),
                   min_s=min(i["min_s"] for i in infos),
                   max_res_inf=max(i["res_inf"] for i in infos))
    if infos and "steps" in infos[0]:
        out.update(total_steps=sum(i["steps"] for i in infos),
                   max_res_inf=max(i["res_inf"] for i in infos),
                   n_not_converged=sum(not i["converged"] for i in infos),
                   converged=all(i["converged"] for i in infos))
    return out


def poly_scalars(res1, area_pts, name):
    return {"Na": nrm(res1["a"]), "Ns": nrm(res1["s"]), "Na_fn": nrm(res1["a_fn"]),
            "Ns_fn": nrm(res1["s_fn"]), "N_dplus": nrm(res1["d_plus"]),
            "N_dminus": nrm(res1["d_minus"]), "area": area_pts,
            "area_pred": tcr2.predicted_area(name)}


def f_condition(grid, h, seed, eps, poly_names):
    """GN learner. tri first; ratios/cosines of every polygon relative to tri."""
    rec = {"grid": grid, "learner": "gn", "h": h, "seed": seed, "eps": eps}
    fam, m, thA, binfo, err = base_point(h, seed)
    rec["base_info"] = binfo
    if err:
        return dict(rec, status="INVALID", error=err)
    pt = tcr2.PlaneTasks(fam, eps)
    learn = lambda th, task: gn_learn(m, th, task)
    polys, a_tri = {}, None
    for name in ["tri"] + [p for p in poly_names if p != "tri"]:
        pts = tcr2.get_poly(name)
        t0 = time.time()
        try:
            res, ip, im, _, _ = tcr2.polygon_run(m, learn, thA, pts, pt, fam.X_probe)
        except NotConverged as e:
            polys[name] = {"status": "INVALID", "error": str(e)}
            continue
        r1 = res[1]
        sc = poly_scalars(r1, tcr2.shoelace(pts), name)
        sc.update(status="OK", wall=time.time() - t0, learn=info_summary(ip + im))
        if name == "tri":
            a_tri = (r1["a"], r1["a_fn"])
        if a_tri is not None:
            sc["ratio"] = sc["Na"] / nrm(a_tri[0])
            sc["ratio_fn"] = sc["Na_fn"] / nrm(a_tri[1])
            sc["cos"] = cosine(r1["a"], a_tri[0])
            sc["cos_fn"] = cosine(r1["a_fn"], a_tri[1])
        polys[name] = sc
    bad = [k for k, v in polys.items() if v["status"] != "OK"]
    rec.update(status="OK" if not bad else ("INVALID" if "tri" in bad else "PARTIAL"),
               invalid_polys=bad, polys=polys)
    return rec


def fg4_condition(h, seed, eps=0.05, N=32):
    rec = {"grid": "FG4", "learner": "gn", "h": h, "seed": seed, "eps": eps}
    fam, m, thA, binfo, err = base_point(h, seed)
    rec["base_info"] = binfo
    if err:
        return dict(rec, status="INVALID", error=err)
    pt = tcr2.PlaneTasks(fam, eps)
    learn = lambda th, task: gn_learn(m, th, task)
    try:
        res, ip, im, thp, thm = tcr2.polygon_run(m, learn, thA, tcr2.poly_circ(N), pt,
                                                 fam.X_probe, m_list=FG4_M)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    XA, yA = pt.A
    na1, na1f = nrm(res[1]["a"]), nrm(res[1]["a_fn"])
    cyc = {}
    for mm in FG4_M:
        r = res[mm]
        rp, rm_ = m.f(thp[mm], XA) - yA, m.f(thm[mm], XA) - yA
        cyc[str(mm)] = {"Na": nrm(r["a"]), "Ns": nrm(r["s"]), "Na_fn": nrm(r["a_fn"]),
                        "Ns_fn": nrm(r["s_fn"]),
                        "ratio": nrm(r["a"]) / (mm * na1), "ratio_fn": nrm(r["a_fn"]) / (mm * na1f),
                        "cos_to_a1": cosine(r["a"], res[1]["a"]),
                        "LA_plus": (rp @ rp).item() / (2 * len(yA)),
                        "LA_minus": (rm_ @ rm_).item() / (2 * len(yA)),
                        "rinf_A_plus": rp.abs().max().item(), "rinf_A_minus": rm_.abs().max().item()}
    rec.update(status="OK", cycles=cyc, learn=info_summary(ip + im))
    return rec


def fg5_seed(seed, deadline, db_save):
    """GD learner, h=128, eps=0.05: tri, circ4, circ8, circ16; GN comparator for each."""
    h, eps = 128, 0.05
    rec = {"grid": "FG5", "learner": "gd", "h": h, "seed": seed, "eps": eps, "polys": {}}
    fam, m, thA_gn, binfo_gn, err = base_point(h, seed, "gn")
    if err:
        return dict(rec, status="INVALID", error="GN comparator " + err)
    pt = tcr2.PlaneTasks(fam, eps)
    try:
        _, _, thA_gd, binfo, _ = base_point(h, seed, "gd", deadline=deadline)
    except Timeout as e:
        _base.pop((h, seed, "gd"), None)
        return dict(rec, status="TIMEOUT", error=f"base point: {e}")
    rec["base_info"] = binfo
    rec["base_dist_gd_gn"] = nrm(thA_gd - thA_gn)
    learn_gd = lambda th, task: gd_learn(m, th, task, deadline=deadline)
    learn_gn = lambda th, task: gn_learn(m, th, task)
    a_tri = None
    status = "OK" if binfo["converged"] else "NOT CONVERGED"
    for name in ["tri"] + [f"circ{N}" for N in FG5_N]:
        pts = tcr2.get_poly(name)
        res_gn, *_ = tcr2.polygon_run(m, learn_gn, thA_gn, pts, pt, fam.X_probe)
        t0 = time.time()
        try:
            res, ip, im, _, _ = tcr2.polygon_run(m, learn_gd, thA_gd, pts, pt, fam.X_probe)
        except Timeout as e:
            rec["polys"][name] = {"status": "TIMEOUT", "error": str(e)}
            rec["status"] = "TIMEOUT"
            return rec
        r1 = res[1]
        li = info_summary(ip + im)
        sc = poly_scalars(r1, tcr2.shoelace(pts), name)
        sc.update(status="OK" if li["converged"] else "NOT CONVERGED", learn=li,
                  wall=time.time() - t0, Na_gn=nrm(res_gn[1]["a"]),
                  cos_gd_gn=cosine(r1["a"], res_gn[1]["a"]))
        if li["converged"] is False:
            status = "NOT CONVERGED"
        if name == "tri":
            a_tri = r1["a"]
            a_tri_gn = res_gn[1]["a"]
        sc["ratio"] = sc["Na"] / nrm(a_tri)
        sc["ratio_gn"] = sc["Na_gn"] / nrm(a_tri_gn)
        sc["cos"] = cosine(r1["a"], a_tri)
        rec["polys"][name] = sc
        rec["status"] = status
        db_save(rec)
    rec["status"] = status
    return rec


# ---------------------------------------------------------------------------
# Part R
# ---------------------------------------------------------------------------
def r_condition(grid, schedule, T, n, kappa, E, k=None, step_scale=None):
    d = int(round(kappa * T * n))
    rec = {"grid": grid, "learner": "exact" if k is None else "gd", "schedule": schedule,
           "k": k, "T": T, "n": n, "kappa": kappa, "d": d, "E": E, "step_scale": step_scale,
           "seeds": R_SEEDS}
    rho_all, js_all, preq_mean, eig, task0 = [], [], [], [], []
    for seed in R_SEEDS:
        rm = tcr2.RModel(seed, T, n, d)
        sched = "partial" if schedule.startswith("partial") else schedule
        Ltab, orders, _ = tcr2.run_schedule(rm, sched, E, seed, k=k,
                                            step_scale=1.0 if step_scale is None else step_scale)
        pq = tcr2.prequential(Ltab, orders, T, E)
        preq_mean.append(float(pq.mean()))
        if sched in ("cyclic", "partial"):
            rho, js = tcr2.window_measures(Ltab, T, E)
            rho_all.append(rho.tolist())
            js_all.append(js.tolist())
        if schedule == "cyclic" and d <= EIG_DMAX:
            eig.append(tcr2.eigen_summary(rm.one_cycle_operator()))
        # extra: loss of the task just learned (sanity), averaged
        just = [Ltab[i, orders[i // T][i % T]] for i in range(E * T)]
        task0.append({"mean_loss_just_learned": float(np.mean(just)),
                      "final_mean_loss": float(Ltab[-1].mean()),
                      "initial_mean_loss": float(rm.losses(torch.zeros(d)).mean())})
    rec.update(status="OK", rho=rho_all, jstar=js_all, preq_per_seed=preq_mean,
               preq=float(np.mean(preq_mean)), eig=eig, extra=task0)
    return rec


# ---------------------------------------------------------------------------
def main(grids):
    db = load()
    done = {key(r) for r in db["records"]}
    run_meta = {"grids": grids, "start": time.time(), "grid_walls": {}, "stopped_by_cap": {}}
    db["meta"]["runs"].append(run_meta)
    save(db)

    def emit(rec, t0, replace=False):
        rec["wall"] = time.time() - t0
        kk = key(rec)
        db["records"] = [r for r in db["records"] if key(r) != kk] + [rec]
        run_meta["end"] = time.time()
        save(db)
        if not replace:
            extra = ""
            if "polys" in rec:
                extra = " ".join(f"{p}:{v.get('ratio', float('nan')):.3f}" for p, v in rec["polys"].items()
                                 if isinstance(v, dict) and "ratio" in v)
            print(f"{rec['grid']} {rec.get('schedule') or ''} h={rec.get('h')} seed={rec.get('seed')} "
                  f"eps={rec.get('eps')} T={rec.get('T')} n={rec.get('n')} kappa={rec.get('kappa')} "
                  f"k={rec.get('k')} {rec['status']} wall={rec['wall']:.1f}s {extra}"
                  + (f" ERR={rec.get('error')}" if rec.get("error") else ""), flush=True)

    def stub_todo(stub):
        return key(stub) not in done

    for grid in grids:
        g0 = time.time()
        deadline = g0 + GRID_CAP_S
        stopped = False
        if grid in ("FG1", "FG2", "FG3", "FG4"):
            if grid == "FG1":
                eps_list, polys = FG1_EPS, ["tri"] + [f"circ{N}" for N in FG1_N]
            elif grid == "FG2":
                eps_list, polys = FG1_EPS, ["tri"] + [f"rect{c}" for c in FG2_C]
            elif grid == "FG3":
                eps_list, polys = FG3_EPS, ["tri", "fig8"]
            else:
                eps_list, polys = [0.05], None
            for h in WIDTHS:
                for seed in SEEDS:
                    for eps in eps_list:
                        stub = {"grid": grid, "learner": "gn", "h": h, "seed": seed, "eps": eps}
                        if not stub_todo(stub):
                            continue
                        if stopped or time.time() > deadline:
                            stopped = True
                            emit(dict(stub, status="NOT RUN", error="grid 60-min cap reached"), time.time())
                            continue
                        t0 = time.time()
                        if grid == "FG4":
                            rec = fg4_condition(h, seed, eps)
                        else:
                            rec = f_condition(grid, h, seed, eps, polys)
                        emit(rec, t0)
        elif grid == "FG5":
            deadline = g0 + FG5_CAP_S
            for seed in [0, 1, 2]:
                stub = {"grid": "FG5", "learner": "gd", "h": 128, "seed": seed, "eps": 0.05}
                if not stub_todo(stub):
                    continue
                if stopped or time.time() > deadline:
                    stopped = True
                    emit(dict(stub, status="TIMEOUT", error="not started: FG5 40-min cap reached"),
                         time.time())
                    continue
                t0 = time.time()
                rec = fg5_seed(seed, deadline, lambda r: emit(dict(r), t0, replace=True))
                if rec["status"] == "TIMEOUT":
                    stopped = True
                emit(rec, t0)
        else:
            conds = []
            if grid == "RG1":
                for kap in RG1_KAPPA:
                    for sch in ("cyclic", "shuffle"):
                        conds.append(dict(schedule=sch, T=25, n=4, kappa=kap, E=5))
            elif grid == "RG2":
                for T in RG2_T:
                    for sch in ("cyclic", "shuffle"):
                        conds.append(dict(schedule=sch, T=T, n=4, kappa=4, E=5))
            elif grid == "RG3":
                for n in RG3_N:
                    conds.append(dict(schedule="cyclic", T=25, n=n, kappa=4, E=5))
            elif grid == "RG4":
                conds.append(dict(schedule="cyclic", T=25, n=4, kappa=4, E=20))
            elif grid == "RG5":
                for k in PARTIAL_K:
                    conds.append(dict(schedule="partial", T=25, n=4, kappa=4, E=5, k=k, step_scale=1.0))
                conds.append(dict(schedule="cyclic", T=25, n=4, kappa=4, E=5))
                # SUPPLEMENTARY (not in SPEC2): half step size, see report
                for k in PARTIAL_K:
                    conds.append(dict(schedule="partialS", T=25, n=4, kappa=4, E=5, k=k, step_scale=0.5))
            else:
                raise ValueError(grid)
            for c in conds:
                stub = dict(c, grid=grid, learner="exact" if c.get("k") is None else "gd")
                if not stub_todo(stub):
                    continue
                if stopped or time.time() > deadline:
                    stopped = True
                    emit(dict(stub, status="NOT RUN", error="grid 60-min cap reached"), time.time())
                    continue
                t0 = time.time()
                emit(r_condition(grid, c["schedule"], c["T"], c["n"], c["kappa"], c["E"],
                                 k=c.get("k"), step_scale=c.get("step_scale")), t0)
        run_meta["grid_walls"][grid] = time.time() - g0
        run_meta["stopped_by_cap"][grid] = stopped
        save(db)
    run_meta["end"] = time.time()
    run_meta["wall"] = run_meta["end"] - run_meta["start"]
    save(db)
    print(f"done grids {grids} in {run_meta['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        traceback.print_exc()
        raise
