"""Runs grids G1-G4 of SPEC.md section 6; writes results.json incrementally.

Usage:  python run.py G1 G2 G3      then      python run.py G4
Conditions already present in results.json are skipped (resume).
"""
import json
import math
import os
import sys
import time
import traceback

import torch

import tcr
from tcr import (MLP, TaskFamily, NotConverged, Timeout, gn_learn, gd_learn, make_learn,
                 make_lin_learner, run_sequences, displacement_scalars, a_loop_vec, tang,
                 lin_dev, cosine, SEQUENCES)

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.json")

WIDTHS = [32, 128, 512]
SEEDS = [0, 1, 2, 3, 4]
EPS = [0.2, 0.1, 0.05, 0.025, 0.0125]
ALPHAS = [2.0, -1.0]
GAMMAS = {"pi/6": math.pi / 6, "pi/4": math.pi / 4, "pi/3": math.pi / 3,
          "pi/2": math.pi / 2, "2pi/3": 2 * math.pi / 3}
G4_CAP_S = 40 * 60


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


def key(rec):
    return "|".join(str(rec.get(k)) for k in ("grid", "learner", "h", "seed", "eps", "gamma", "alpha"))


def summarize_infos(infos):
    """infos: {seq: [per-learn info]} -> compact list of per-learn records (unique by prefix)."""
    out = {}
    for name, lst in infos.items():
        out[name] = [{k: v for k, v in i.items()} for i in lst]
    return out


_base_cache = {}


def base_point(h, seed, learner="gn", deadline=None):
    k = (h, seed, learner)
    if k not in _base_cache:
        fam = TaskFamily(seed)
        m = MLP(h)
        th0 = m.init(seed)
        A = (fam.X_A, fam.teacher(fam.X_A))
        t0 = time.time()
        try:
            if learner == "gn":
                thA, info = gn_learn(m, th0, A)
            else:
                thA, info = gd_learn(m, th0, A, deadline=deadline)
            info = dict(info, wall=time.time() - t0)
            _base_cache[k] = (fam, m, thA, info, None)
        except NotConverged as e:
            _base_cache[k] = (fam, m, None, None, f"base point INVALID: {e}")
    return _base_cache[k]


def g1_condition(h, seed, eps):
    rec = {"grid": "G1", "learner": "gn", "h": h, "seed": seed, "eps": eps,
           "gamma": "pi/2", "alpha": None}
    fam, m, thA, binfo, err = base_point(h, seed)
    rec["base_info"] = binfo
    if err:
        return dict(rec, status="INVALID", error=err)
    tasks = fam.tasks(eps, gamma=math.pi / 2)
    try:
        th, infos = run_sequences(make_learn(m, tasks, "gn"), thA, list(SEQUENCES))
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    sc, dl, ph = displacement_scalars(m, thA, th, fam.X_probe)
    a = a_loop_vec(thA, th)
    sc["tang"] = tang(m, thA, tasks["A"][0], a)
    sc["lin"] = lin_dev(m, thA, th["loop+"], tasks)
    thl, _ = run_sequences(make_lin_learner(m, thA, tasks), thA, ["loop+", "loop-"])
    a_lin = a_loop_vec(thA, thl)
    sc["N_a_lin"] = torch.linalg.norm(a_lin).item()
    sc["cos_lin"] = cosine(a, a_lin)
    sc["ratio_lin"] = sc["N_a"] / sc["N_a_lin"]
    rec.update(status="OK", scalars=sc, learn_info=summarize_infos(infos),
               max_gn_iters=max(i["iters"] for l in infos.values() for i in l),
               ls_fail_total=sum(i["ls_fail"] for l in infos.values() for i in l))
    return rec


def loop_condition(grid, h, seed, eps, gamma_name=None, alpha=None):
    rec = {"grid": grid, "learner": "gn", "h": h, "seed": seed, "eps": eps,
           "gamma": gamma_name, "alpha": alpha}
    fam, m, thA, binfo, err = base_point(h, seed)
    rec["base_info"] = binfo
    if err:
        return dict(rec, status="INVALID", error=err)
    if alpha is not None:
        tasks = fam.tasks(eps, alpha=alpha)
    else:
        tasks = fam.tasks(eps, gamma=GAMMAS[gamma_name])
    try:
        th, infos = run_sequences(make_learn(m, tasks, "gn"), thA, ["loop+", "loop-"])
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    sc, _, _ = displacement_scalars(m, thA, th, fam.X_probe)
    sc["tang"] = tang(m, thA, tasks["A"][0], a_loop_vec(thA, th))
    rec.update(status="OK", scalars=sc, learn_info=summarize_infos(infos),
               max_gn_iters=max(i["iters"] for l in infos.values() for i in l),
               ls_fail_total=sum(i["ls_fail"] for l in infos.values() for i in l))
    return rec


def g4_condition(seed, eps, deadline):
    h = 128
    seqs = ["loop+", "loop-", "retrace", "star+", "star-"]
    rec = {"grid": "G4", "learner": "gd", "h": h, "seed": seed, "eps": eps,
           "gamma": "pi/2", "alpha": None}
    # GN comparator (same condition)
    fam, m, thA_gn, binfo_gn, err = base_point(h, seed, "gn")
    tasks = fam.tasks(eps, gamma=math.pi / 2)
    if err:
        return dict(rec, status="INVALID", error="GN comparator " + err)
    try:
        th_gn, _ = run_sequences(make_learn(m, tasks, "gn"), thA_gn, seqs)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=f"GN comparator: {e}")
    sc_gn, _, _ = displacement_scalars(m, thA_gn, th_gn, fam.X_probe)
    a_gn = a_loop_vec(thA_gn, th_gn)
    # GD
    try:
        _, _, thA_gd, binfo_gd, _ = base_point(h, seed, "gd", deadline=deadline)
    except Timeout as e:
        _base_cache.pop((h, seed, "gd"), None)
        return dict(rec, status="TIMEOUT", error=f"base point: {e}", scalars_gn=sc_gn)
    rec["base_info"] = binfo_gd
    try:
        th_gd, infos = run_sequences(make_learn(m, tasks, "gd", deadline=deadline), thA_gd, seqs)
    except Timeout as e:
        return dict(rec, status="TIMEOUT", error=str(e), scalars_gn=sc_gn)
    sc_gd, _, _ = displacement_scalars(m, thA_gd, th_gd, fam.X_probe)
    a_gd = a_loop_vec(thA_gd, th_gd)
    all_infos = [binfo_gd] + [i for l in infos.values() for i in l]
    conv = all(i["converged"] for i in all_infos)
    rec.update(status="OK" if conv else "NOT CONVERGED", scalars=sc_gd, scalars_gn=sc_gn,
               cos_gd_gn=cosine(a_gd, a_gn),
               base_dist_gd_gn=torch.linalg.norm(thA_gd - thA_gn).item(),
               learn_info=summarize_infos(infos),
               max_res_inf=max(i["res_inf"] for i in all_infos),
               total_gd_steps=sum(i["steps"] for l in infos.values() for i in l),
               nonconverged_learns=[(s, j, i["task"], i["res_inf"]) for s, l in infos.items()
                                    for j, i in enumerate(l) if not i["converged"]]
               + ([("base", 0, "A", binfo_gd["res_inf"])] if not binfo_gd["converged"] else []))
    return rec


def main(grids):
    db = load()
    done = {key(r) for r in db["records"]}
    run_meta = {"grids": grids, "start": time.time()}
    db["meta"]["runs"].append(run_meta)
    save(db)

    def emit(rec, t0):
        rec["wall"] = time.time() - t0
        db["records"].append(rec)
        run_meta["end"] = time.time()
        save(db)
        sc = rec.get("scalars") or {}
        print(f"{rec['grid']} h={rec['h']} seed={rec['seed']} eps={rec['eps']} "
              f"g={rec['gamma']} a={rec['alpha']} {rec['status']} "
              f"N_a={sc.get('N_a', float('nan')):.4g} wall={rec['wall']:.1f}s"
              + (f" ERR={rec.get('error')}" if rec.get("error") else ""), flush=True)

    def todo(rec):
        return key(rec) not in done

    for grid in grids:
        if grid == "G1":
            for h in WIDTHS:
                for seed in SEEDS:
                    for eps in EPS:
                        stub = {"grid": "G1", "learner": "gn", "h": h, "seed": seed, "eps": eps,
                                "gamma": "pi/2", "alpha": None}
                        if todo(stub):
                            t0 = time.time()
                            emit(g1_condition(h, seed, eps), t0)
        elif grid == "G2":
            for h in WIDTHS:
                for seed in SEEDS:
                    for alpha in ALPHAS:
                        for eps in EPS:
                            stub = {"grid": "G2", "learner": "gn", "h": h, "seed": seed,
                                    "eps": eps, "gamma": None, "alpha": alpha}
                            if todo(stub):
                                t0 = time.time()
                                emit(loop_condition("G2", h, seed, eps, alpha=alpha), t0)
        elif grid == "G3":
            for h in WIDTHS:
                for seed in SEEDS:
                    for gname in GAMMAS:
                        stub = {"grid": "G3", "learner": "gn", "h": h, "seed": seed,
                                "eps": 0.05, "gamma": gname, "alpha": None}
                        if todo(stub):
                            t0 = time.time()
                            emit(loop_condition("G3", h, seed, 0.05, gamma_name=gname), t0)
        elif grid == "G4":
            g4_start = time.time()
            deadline = g4_start + G4_CAP_S
            run_meta["g4_start"] = g4_start
            stopped = False
            for seed in [0, 1, 2]:
                for eps in [0.2, 0.1]:
                    stub = {"grid": "G4", "learner": "gd", "h": 128, "seed": seed, "eps": eps,
                            "gamma": "pi/2", "alpha": None}
                    if not todo(stub):
                        continue
                    if stopped or time.time() > deadline:
                        rec = dict(stub, status="TIMEOUT", error="not started: G4 40-min cap reached")
                        emit(rec, time.time())
                        stopped = True
                        continue
                    t0 = time.time()
                    rec = g4_condition(seed, eps, deadline)
                    if rec["status"] == "TIMEOUT":
                        stopped = True
                    emit(rec, t0)
            run_meta["g4_wall"] = time.time() - g4_start
            run_meta["g4_stopped_by_cap"] = stopped
        run_meta["end"] = time.time()
        save(db)
    run_meta["end"] = time.time()
    run_meta["wall"] = run_meta["end"] - run_meta["start"]
    save(db)
    print(f"done grids {grids} in {run_meta['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1:] or ["G1", "G2", "G3"])
    except Exception:
        traceback.print_exc()
        raise
