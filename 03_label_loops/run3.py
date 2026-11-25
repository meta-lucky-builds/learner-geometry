"""Runs the SPEC3.md grids; writes results3.json incrementally (lock + merge, so several
processes may run different grids at once).

Usage:  python run3.py [--threads N] [--widths 32,64] GRID [GRID ...]
Grids:  LIN BASE LE L3 K1 K4 K2 L8 K3 M K5 G
  LIN  : L5 + K6 linear controls (tcr.LinearModel), seeds 0-4
  BASE : per (h, seed): theta_A, b_uv, c_u, phi, kernel stats, L6 curvature map
  LE   : eps-sweep of tri / fig8 / star (L1, L2, L3c, L3d, L4, L7)
  L3   : circ(N), rect(c) at eps in {0.05, 0.025} (L3a, L3b, L3e)
  K1   : out-and-back, all eps
  K4   : path refinement, L in {0.1, 0.05}, N in {1, 2, 4, 8, 16}
  K2   : repeated excursions, eps = 0.1, m up to 64
  K3   : random excursions, eps = 0.1, m up to 128
  L8   : circ(16), eps = 0.05, m up to 64 cycles
  M    : mixed input/label tri, widths {32, 128, 512}
  K5   : GD steps from S0 / S1, h in {32, 64}, seeds {0, 1, 2}
  G    : GD learner, h = 128, seeds {0, 1, 2}, eps in {0.1, 0.05}
Conditions already present in results3.json are skipped (resume).
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
import tcr3
from tcr import NotConverged, Timeout, gn_learn, gd_learn
from tcr3 import Setup, LabelPlane, MixedPlane, nrm, cos

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results3.json")
LOCK = OUT + ".lock"

SEEDS = [0, 1, 2, 3, 4]
WIDTHS = [32, 64, 128, 256, 512]
EPS = [0.2, 0.1, 0.05, 0.025, 0.0125]
L3_EPS = [0.05, 0.025]
L3_N = [3, 4, 8, 16, 32]
L3_C = [0.5, 1, 2]
K4_L = [0.1, 0.05]
K4_N = [1, 2, 4, 8, 16]
K2_M = [1, 2, 4, 8, 16, 32, 64]
K3_M = [1, 2, 4, 8, 16, 32, 64, 128]
L8_M = [1, 2, 4, 8, 16, 32, 64]
M_WIDTHS = [32, 128, 512]
K5_WIDTHS = [32, 64]
K5_SEEDS = [0, 1, 2]
G_SEEDS = [0, 1, 2]
G_EPS = [0.1, 0.05]
GRID_CAP_S = 60 * 60
G_CAP_S = 40 * 60

TRI = [(1.0, 0.0), (0.0, 1.0)]
FIG8 = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]
STAR = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]   # reverse = [(0,1),(0,0),(1,0)] = reversed list


# ---------------------------------------------------------------------------
# results file: lock + merge
# ---------------------------------------------------------------------------
KEYF = ("grid", "h", "seed", "eps", "L")


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
    return {"meta": {"runs": []}, "records": []}


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
    return {key(r) for r in db["records"]}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def tl(t):
    return t.tolist()


def info_summary(infos):
    out = {"n_learns": len(infos)}
    if infos and "iters" in infos[0]:
        out.update(max_iters=max(i["iters"] for i in infos),
                   total_iters=sum(i["iters"] for i in infos),
                   ls_fail=sum(i["ls_fail"] for i in infos),
                   min_s=min(i["min_s"] for i in infos),
                   max_res_inf=max(i["res_inf"] for i in infos))
    if infos and "steps" in infos[0]:
        out.update(total_steps=sum(i["steps"] for i in infos),
                   max_res_inf=max(i["res_inf"] for i in infos),
                   n_not_converged=sum(not i["converged"] for i in infos),
                   converged=all(i["converged"] for i in infos))
    return out


# ---------------------------------------------------------------------------
# cached base objects
# ---------------------------------------------------------------------------
_S, _BC = {}, {}


def get_setup(h, seed):
    k = (h, seed)
    if k not in _S:
        try:
            _S[k] = (Setup(h, seed), None)
        except NotConverged as e:
            _S[k] = (None, f"base point INVALID: {e}")
    return _S[k]


def base_fields(S):
    k = (S.h, S.seed)
    if k not in _BC:
        m, th, X = S.model, S.theta_A, S.X
        b = tcr3.bracket(m, th, X, S.u, S.v)
        c = tcr3.self_term(m, th, X, S.u)
        _BC[k] = (b, c)
    return _BC[k]


def gn(S):
    return lambda th, task: gn_learn(S.model, th, task)


def state_quants(S, th):
    q = tcr3.kernel_stats(S.model, th, S.X, S.u, S.v)
    fA = S.model.f(S.theta_A, S.X_probe)
    q.update(norm_theta=nrm(th), dist_thetaA=nrm(th - S.theta_A),
             probe_change=nrm(S.model.f(th, S.X_probe) - fA) / 16)
    return q


# ---------------------------------------------------------------------------
# condition functions (each returns one record)
# ---------------------------------------------------------------------------
def c_lin(seed):
    rec = {"grid": "LIN", "seed": seed}
    out = {}
    for tag, uvt in (("primary_uv_teacher_norm", True), ("suppl_uv_linear_norm", False)):
        S = Setup(None, seed, linear=True, uv_from_teacher_for_linear=uvt)
        learn = gn(S)
        r = {}
        res, _, _, _, _ = tcr3.poly_run(S, learn, S.theta_A, TRI, LabelPlane(S, 0.2))
        r["L5_Na_tri_eps0.2"] = nrm(res[1]["a"])
        r["K6_d_ret"] = {str(e): nrm(tcr3.out_and_back(S, learn, S.theta_A, S.u, e)[0] - S.theta_A)
                         for e in EPS}
        r["K6_d_sub4"] = {}
        for L in K4_L:
            t1, _ = tcr3.refine_path(S, learn, S.theta_A, S.u, L, 1)
            t4, _ = tcr3.refine_path(S, learn, S.theta_A, S.u, L, 4)
            r["K6_d_sub4"][str(L)] = nrm(t4 - t1)
        r["K6_d_K2_m8"] = nrm(tcr3.out_and_back(S, learn, S.theta_A, S.u, 0.1, m=8)[0] - S.theta_A)
        r["max_all"] = max([r["L5_Na_tri_eps0.2"], r["K6_d_K2_m8"]] + list(r["K6_d_ret"].values())
                           + list(r["K6_d_sub4"].values()))
        out[tag] = r
    rec.update(status="OK", **out)
    return rec


def c_base(h, seed):
    rec = {"grid": "BASE", "h": h, "seed": seed}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    b, c = base_fields(S)
    rec["base_info"] = S.base_info
    rec.update(norm_b_uv=nrm(b), norm_c_u=nrm(c), yA_norm=float(S.yA_norm),
               norm_theta_A=nrm(S.theta_A), dist_init=nrm(S.theta_A - S.theta_init),
               **tcr3.kernel_stats(S.model, S.theta_A, S.X, S.u, S.v))
    J = S.model.jac(S.theta_A, S.X)
    rec["vert_b"] = nrm(J @ b) / (torch.linalg.matrix_norm(J, ord=2).item() * nrm(b))
    rec["vert_c"] = nrm(J @ c) / (torch.linalg.matrix_norm(J, ord=2).item() * nrm(c))
    t0 = time.time()
    lam, B, kap, kap2 = tcr3.curvature_map(S.model, S.theta_A, S.X)
    rec["L6"] = {"lam": tl(lam), "B": tl(B), "kappa": tl(kap),
                 "kappa_alt_max_rel_diff": (torch.abs(kap - kap2).max() / kap.abs().max()).item(),
                 "wall": time.time() - t0}
    rec["status"] = "OK"
    return rec


def poly_scal(res1):
    return {"Na": nrm(res1["a"]), "Ns": nrm(res1["s"]), "Na_fn": nrm(res1["a_fn"]),
            "Ns_fn": nrm(res1["s_fn"])}


def c_le(h, seed, eps):
    rec = {"grid": "LE", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    b, c = base_fields(S)
    plane = LabelPlane(S, eps)
    learn = gn(S)
    polys, infos, ends = {}, [], []
    for name, pts in (("tri", TRI), ("fig8", FIG8), ("star", STAR)):
        try:
            res, inf, thp, thm, e_end = tcr3.poly_run(S, learn, S.theta_A, pts, plane)
        except NotConverged as e:
            polys[name] = {"status": "INVALID", "error": str(e)}
            continue
        infos += inf
        ends.append(e_end)
        sc = poly_scal(res[1])
        sc["status"] = "OK"
        if name == "tri":
            a_tri, a_tri_fn = res[1]["a"], res[1]["a_fn"]
            sc["cos_b"] = cos(a_tri, b)
            sc["ratio_b"] = nrm(a_tri) / (eps ** 2 / 2 * nrm(b))
            # L7
            KA = tcr3.jk(S.model, S.theta_A, S.X)[1]
            Kp = tcr3.jk(S.model, thp[1], S.X)[1]
            Km = tcr3.jk(S.model, thm[1], S.X)[1]
            nKA = torch.linalg.norm(KA).item()
            HA = tcr3.last_hidden(S.model, S.theta_A, S.X_probe)
            Hp = tcr3.last_hidden(S.model, thp[1], S.X_probe)
            Hm = tcr3.last_hidden(S.model, thm[1], S.X_probe)
            sc["L7"] = {"K_odd": torch.linalg.norm(Kp - Km).item() / nKA,
                        "K_even": torch.linalg.norm((Kp + Km) / 2 - KA).item() / nKA,
                        "probe_odd_per_input": nrm(a_tri_fn) / 16,
                        "one_minus_cka_plus_minus": 1 - tcr3.linear_cka(Hp, Hm),
                        "one_minus_cka_A_plus": 1 - tcr3.linear_cka(HA, Hp)}
        polys[name] = sc
    for name in ("fig8", "star"):
        if polys.get(name, {}).get("status") == "OK" and polys["tri"].get("status") == "OK":
            polys[name]["ratio_tri"] = polys[name]["Na"] / polys["tri"]["Na"]
    bad = [k for k, v in polys.items() if v["status"] != "OK"]
    rec.update(status="OK" if not bad else "PARTIAL", invalid=bad, polys=polys,
               learn=info_summary(infos), max_end_res=max(ends) if ends else None)
    return rec


def c_l3(h, seed, eps):
    rec = {"grid": "L3", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    plane = LabelPlane(S, eps)
    learn = gn(S)
    polys, infos, a_tri = {}, [], None
    names = ["tri"] + [f"circ{N}" for N in L3_N] + [f"rect{c}" for c in L3_C]
    for name in names:
        pts = tcr2.get_poly(name)
        try:
            res, inf, _, _, _ = tcr3.poly_run(S, learn, S.theta_A, pts, plane)
        except NotConverged as e:
            polys[name] = {"status": "INVALID", "error": str(e)}
            continue
        infos += inf
        sc = poly_scal(res[1])
        sc.update(status="OK", area=tcr2.shoelace(pts))
        if name == "tri":
            a_tri = res[1]["a"]
        if a_tri is not None:
            sc["ratio"] = sc["Na"] / nrm(a_tri)
            sc["cos"] = cos(res[1]["a"], a_tri)
        polys[name] = sc
    bad = [k for k, v in polys.items() if v["status"] != "OK"]
    rec.update(status="OK" if not bad else "PARTIAL", invalid=bad, polys=polys,
               learn=info_summary(infos))
    return rec


def c_k1(h, seed, eps):
    rec = {"grid": "K1", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    b, c = base_fields(S)
    try:
        th, _, inf = tcr3.out_and_back(S, gn(S), S.theta_A, S.u, eps)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    d = th - S.theta_A
    phA = tcr3.stiffness(S.model, S.theta_A, S.X, S.u)
    phR = tcr3.stiffness(S.model, th, S.X, S.u)
    rec.update(status="OK", norm_d=nrm(d), cos_c=cos(d, c), ratio_c=nrm(d) / (eps ** 2 * nrm(c)),
               phi_A=phA, phi_ret=phR, dphi=phR - phA,
               dphi_ratio=(phR - phA) / (eps ** 2 * nrm(c) ** 2),
               learn=info_summary(inf))
    return rec


def c_k4(h, seed, L):
    rec = {"grid": "K4", "h": h, "seed": seed, "L": L}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    b, c = base_fields(S)
    learn = gn(S)
    th, infos = {}, []
    try:
        for N in K4_N:
            th[N], inf = tcr3.refine_path(S, learn, S.theta_A, S.u, L, N)
            infos += inf
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    phi1 = tcr3.stiffness(S.model, th[1], S.X, S.u)
    out = {}
    for N in K4_N:
        ph = tcr3.stiffness(S.model, th[N], S.X, S.u)
        r = {"phi": ph, "dphi_vs_jump": ph - phi1}
        if N > 1:
            d = th[N] - th[1]
            r.update(norm_d=nrm(d), cos_c=cos(d, c),
                     ratio_c=nrm(d) / ((L ** 2 / 2) * (1 - 1 / N) * nrm(c)))
        out[str(N)] = r
    rec.update(status="OK", N=out, phi_A=tcr3.stiffness(S.model, S.theta_A, S.X, S.u),
               learn=info_summary(infos))
    return rec


def c_k2(h, seed, eps=0.1):
    rec = {"grid": "K2", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    try:
        _, ths, inf = tcr3.out_and_back(S, gn(S), S.theta_A, S.u, eps, m=max(K2_M), record=K2_M)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    q = {"0": state_quants(S, S.theta_A)}
    for mm in K2_M:
        q[str(mm)] = state_quants(S, ths[mm])
    rec.update(status="OK", m=q, learn=info_summary(inf))
    return rec


def c_k3(h, seed, eps=0.1):
    rec = {"grid": "K3", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    dirs = tcr3.random_dirs(99 + seed, max(K3_M), S.yA_norm)
    try:
        _, ths, inf = tcr3.random_excursions(S, gn(S), S.theta_A, dirs, eps, record=K3_M)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    q = {"0": state_quants(S, S.theta_A)}
    for mm in K3_M:
        q[str(mm)] = state_quants(S, ths[mm])
    rec.update(status="OK", m=q, learn=info_summary(inf))
    return rec


def c_l8(h, seed, eps=0.05):
    rec = {"grid": "L8", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    try:
        res, inf, thp, thm, e_end = tcr3.poly_run(S, gn(S), S.theta_A, tcr2.poly_circ(16),
                                                  LabelPlane(S, eps), m_list=L8_M)
    except NotConverged as e:
        return dict(rec, status="INVALID", error=str(e))
    a1 = res[1]["a"]
    q = {}
    for mm in L8_M:
        r = res[mm]
        ks = tcr3.kernel_stats(S.model, thp[mm], S.X, S.u, S.v)
        q[str(mm)] = {"Na": nrm(r["a"]), "ratio": nrm(r["a"]) / (mm * nrm(a1)),
                      "cos_a1": cos(r["a"], a1), "Ns": nrm(r["s"]), "Na_fn": nrm(r["a_fn"]),
                      "fwd_trKinv": ks["trKinv"], "fwd_lam_min": ks["lam_min"],
                      "fwd_phi_u": ks["phi_u"]}
    q["0"] = tcr3.kernel_stats(S.model, S.theta_A, S.X, S.u, S.v)
    rec.update(status="OK", m=q, learn=info_summary(inf), max_end_res=e_end)
    return rec


def c_m(h, seed, eps):
    rec = {"grid": "M", "h": h, "seed": seed, "eps": eps}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    learn = gn(S)
    out, infos = {}, []
    for name, plane in (("mixed", MixedPlane(S, eps)), ("input", tcr2.PlaneTasks(S.fam, eps)),
                        ("label", LabelPlane(S, eps))):
        try:
            res, inf, _, _, _ = tcr3.poly_run(S, learn, S.theta_A, TRI, plane)
        except NotConverged as e:
            out[name] = {"status": "INVALID", "error": str(e)}
            continue
        infos += inf
        out[name] = dict(poly_scal(res[1]), status="OK")
    if out["mixed"]["status"] == "OK":
        for name in ("input", "label"):
            if out[name]["status"] == "OK":
                out["mixed"][f"ratio_{name}"] = out["mixed"]["Na"] / out[name]["Na"]
                out["mixed"][f"ratio_{name}_fn"] = out["mixed"]["Na_fn"] / out[name]["Na_fn"]
    bad = [k for k, v in out.items() if v["status"] != "OK"]
    rec.update(status="OK" if not bad else "PARTIAL", invalid=bad, tri=out,
               learn=info_summary(infos))
    return rec


def c_k5(h, seed):
    rec = {"grid": "K5", "h": h, "seed": seed}
    S, err = get_setup(h, seed)
    if err:
        return dict(rec, status="INVALID", error=err)
    dirs = tcr3.random_dirs(99 + seed, 128, S.yA_norm)
    try:
        th1, _, _ = tcr3.random_excursions(S, gn(S), S.theta_A, dirs, 0.1, record=[128])
    except NotConverged as e:
        return dict(rec, status="INVALID", error="S1: " + str(e))
    targets = tcr3.random_dirs(4242 + seed, 3, S.yA_norm)
    states = {"S0": S.theta_A, "S1": th1}
    out = {}
    for sn, th in states.items():
        ks = tcr3.kernel_stats(S.model, th, S.X, S.u, S.v)
        runs = []
        for j, t in enumerate(targets, start=1):
            t0 = time.time()
            _, info = gd_learn(S.model, th, (S.X, S.yA + 0.2 * t))
            runs.append({"j": j, "steps": info["steps"], "converged": info["converged"],
                         "res_inf": info["res_inf"], "lr": info["lr"],
                         "lam_max_gd": info["lam_max"], "wall": time.time() - t0})
            log(f"  K5 h={h} seed={seed} {sn} j={j} steps={info['steps']} conv={info['converged']}")
        out[sn] = {"kernel": ks, "runs": runs}
    ratios = []
    for j in range(3):
        a, b = out["S1"]["runs"][j]["steps"], out["S0"]["runs"][j]["steps"]
        ratios.append(a / b)
    rec.update(status="OK", states=out, ratios=ratios,
               dist_S1_S0=nrm(th1 - S.theta_A))
    return rec


_GB = {}


def g_base(seed, deadline):
    if seed not in _GB:
        S = Setup(128, seed, learner="gd", deadline=deadline)
        b = tcr3.bracket(S.model, S.theta_A, S.X, S.u, S.v)
        c = tcr3.self_term(S.model, S.theta_A, S.X, S.u)
        Sgn, _ = get_setup(128, seed)
        _GB[seed] = (S, b, c, nrm(S.theta_A - Sgn.theta_A))
    return _GB[seed]


def c_g(seed, eps, deadline):
    rec = {"grid": "G", "h": 128, "seed": seed, "eps": eps}
    try:
        S, b, c, dgn = g_base(seed, deadline)
    except Timeout as e:
        return dict(rec, status="TIMEOUT", error=f"base point: {e}")
    rec["base_info"] = S.base_info
    rec["base_dist_gd_gn"] = dgn
    rec.update(norm_b_uv=nrm(b), norm_c_u=nrm(c))
    learn = lambda th, task: gd_learn(S.model, th, task, deadline=deadline)
    try:
        res, inf, _, _, _ = tcr3.poly_run(S, learn, S.theta_A, TRI, LabelPlane(S, eps), check=False)
        a = res[1]["a"]
        rec.update(Na=nrm(a), cos_b=cos(a, b), ratio_b=nrm(a) / (eps ** 2 / 2 * nrm(b)),
                   learn_tri=info_summary(inf))
        th, _, inf2 = tcr3.out_and_back(S, learn, S.theta_A, S.u, eps, check=False)
        d = th - S.theta_A
        rec.update(norm_d=nrm(d), cos_c=cos(d, c), ratio_c=nrm(d) / (eps ** 2 * nrm(c)),
                   learn_ret=info_summary(inf2))
    except Timeout as e:
        return dict(rec, status="TIMEOUT", error=str(e))
    conv = (S.base_info["converged"] and rec["learn_tri"]["converged"]
            and rec["learn_ret"]["converged"])
    rec["status"] = "OK" if conv else "NOT CONVERGED"
    return rec


# ---------------------------------------------------------------------------
# grid drivers
# ---------------------------------------------------------------------------
def conditions(grid, widths):
    if grid == "LIN":
        return [((s,), {"grid": "LIN", "seed": s}) for s in SEEDS]
    if grid in ("BASE", "K2", "K3", "L8"):
        eps = {"K2": 0.1, "K3": 0.1, "L8": 0.05}.get(grid)
        return [((h, s), {"grid": grid, "h": h, "seed": s, "eps": eps})
                for h in widths for s in SEEDS]
    if grid in ("LE", "K1"):
        return [((h, s, e), {"grid": grid, "h": h, "seed": s, "eps": e})
                for h in widths for s in SEEDS for e in EPS]
    if grid == "L3":
        return [((h, s, e), {"grid": grid, "h": h, "seed": s, "eps": e})
                for h in widths for s in SEEDS for e in L3_EPS]
    if grid == "K4":
        return [((h, s, L), {"grid": grid, "h": h, "seed": s, "L": L})
                for h in widths for s in SEEDS for L in K4_L]
    if grid == "M":
        return [((h, s, e), {"grid": grid, "h": h, "seed": s, "eps": e})
                for h in widths if h in M_WIDTHS for s in SEEDS for e in EPS]
    if grid == "K5":
        return [((h, s), {"grid": grid, "h": h, "seed": s})
                for h in K5_WIDTHS for s in K5_SEEDS]
    if grid == "G":
        return [((s, e), {"grid": grid, "h": 128, "seed": s, "eps": e})
                for s in G_SEEDS for e in G_EPS]
    raise ValueError(grid)


FUN = {"LIN": c_lin, "BASE": c_base, "LE": c_le, "L3": c_l3, "K1": c_k1, "K4": c_k4,
       "K2": c_k2, "K3": c_k3, "L8": c_l8, "M": c_m, "K5": c_k5}


def run_grid(grid, widths):
    t_start = time.time()
    cap = G_CAP_S if grid == "G" else GRID_CAP_S
    deadline = t_start + cap
    done = done_keys()
    conds = conditions(grid, widths)
    n_run, stopped = 0, False
    for args, kr in conds:
        if key(kr) in done:
            continue
        if time.time() > deadline:
            stopped = True
            log(f"{grid}: wall cap {cap}s reached; stopping")
            break
        t0 = time.time()
        try:
            if grid == "G":
                rec = c_g(*args, deadline=deadline)
            else:
                rec = FUN[grid](*args)
        except AssertionError as e:
            rec = dict(kr, status="INVALID", error="end-check: " + str(e))
        except Exception as e:  # noqa
            rec = dict(kr, status="ERROR", error=f"{type(e).__name__}: {e}",
                       tb=traceback.format_exc())
        rec["wall"] = time.time() - t0
        save_records([rec])
        n_run += 1
        log(f"{grid} {args} {rec['status']} wall={rec['wall']:.1f}s")
        if rec["status"] == "TIMEOUT":
            stopped = True
            break
    meta = {"grid": grid, "widths": widths, "n_run": n_run, "wall": time.time() - t_start,
            "stopped_at_cap": stopped, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_records([], meta_run=meta)
    log(f"done {grid}: {n_run} conditions in {time.time() - t_start:.1f}s stopped={stopped}")


def main(argv):
    widths = WIDTHS
    grids = []
    i = 0
    while i < len(argv):
        if argv[i] == "--threads":
            torch.set_num_threads(int(argv[i + 1]))
            i += 2
        elif argv[i] == "--widths":
            widths = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        else:
            grids.append(argv[i])
            i += 1
    for g in grids:
        run_grid(g, widths)


if __name__ == "__main__":
    main(sys.argv[1:])
