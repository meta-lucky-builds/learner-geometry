"""Runs SPEC7.md Tests 10-12 (Parts A, B, C) and the linear control; writes results7.json
incrementally (lock + merge, several worker processes may run at once; jobs are claimed through
claim files in the scratch dir). Path matrices C, Q of Part A / B and the per-step series of
Part C are stored in results7_mats/*.npz (referenced from results7.json); every scalar is in
results7.json.

Usage:  python run7.py --scratch DIR [--deadline EPOCH] [--widths 32,64] [--seeds 0,1] PART [PART ...]
Parts:  LIN  A  B  C  CSUPP8
Jobs already in results7.json (status not ERROR / NOT RUN) are skipped.
"""
import tcr7  # first: fixes thread count (via tcr4)
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import tcr4
import tcr6
from tcr7 import (Net, Pred, np_solve, gn_run, sl_run, gd_run, eta_c, identity_err, sharp_H,
                  nrm, safe)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results7.json")
LOCK = OUT + ".lock"
MATDIR = os.path.join(HERE, "results7_mats")

SEEDS = [0, 1, 2, 3, 4]
WIDTHS = [128, 64, 32]          # largest first (load balance)
N = 16

EIG_IDX = [1, 2, 4, 7, 10, 13, 16]
SIZES_A = [0.025, 0.05, 0.1]
CS_A = [0.25, 1.0, 1.5, 1.9]
KAPPAS = [0.5, 1.0, 2.0, 4.0]
RHOS = [0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4]
C_B = 0.1
TOL_B, CAP_B = 1e-6, 400_000
FS_C = [0.25, 0.5, 0.75]
CAP_C, TOL_C = 60_000, 1e-6
SH_EVERY = 50
WS = [16, 64]
MAXWIN = 150
SL_M = 64

KEYF = ("part", "h", "seed", "kappa", "sign", "f")


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
            if time.time() - t0 > 600:
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
            json.dump(db, fh, default=float)
        os.replace(tmp, OUT)
    finally:
        _unlock()


def find_record(**kw):
    _lock()
    try:
        db = load()
    finally:
        _unlock()
    k = key(kw)
    for r in db["records"]:
        if key(r) == k:
            return r
    return None


def done_keys():
    _lock()
    try:
        db = load()
    finally:
        _unlock()
    return {key(r) for r in db["records"] if r.get("status") not in ("ERROR", "NOT RUN")}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def lst(a):
    return [float(x) for x in np.asarray(a).ravel()]


# ---------------------------------------------------------------------------
# Part A (Test 10)
# ---------------------------------------------------------------------------
def partA_job(h, seed, linear=False, sizes=SIZES_A, tag="A"):
    S = tcr6.setup(h, seed, linear=linear)
    net = Net(S.model, S.X)
    L = tcr6.law(S)
    P = Pred(S.model, S.theta_A, S.X, L)
    thA = S.theta_A.numpy().copy()
    kA = net.kstats(thA)
    lam = S.lam.numpy()
    yA = S.yA.numpy()
    etas = {c: eta_c(S.model, S.theta_A, S.X, c) for c in CS_A}
    dirs = [(f"q{i}", S.Q[:, i - 1].numpy().copy(), i) for i in EIG_IDX]
    dirs += [(f"rand{j}", d, None) for j, d in enumerate(tcr6.unit_dirs(7000 + seed, 8), start=1)]
    dinfo = {}
    for name, e, i in dirs:
        xi = L.XI.numpy() @ e
        dinfo[name] = {"G": P(np.outer(e, e)), "Dlam_xi": P.dir(torch.from_numpy(xi)),
                       "Dtr_xi": P.dtr(torch.from_numpy(xi)), "eig_index": i,
                       "lam_i": float(lam[i - 1]) if i else None, "e": lst(e)}
    runs = []
    mats = {}
    for name, e, i in dirs:
        for s in sizes:
            for sg in (1, -1):
                t = yA + sg * s * e
                tt = torch.from_numpy(t)
                base = {"dir": name, "s": s, "sign": sg}
                # NP
                th, info = np_solve(net, thA, t)
                rec = dict(base, learner="NP", status=info["status"], steps=info["iters"],
                           res_inf=info["res_inf"], np_stat=info["stat"], np_dist=info["dist"],
                           time=info["time"])
                if th is not None:
                    rec.update(net.kstats(th))
                    rec["dY"] = lst(net.F(th) - S.F_A_np)
                    rec["dist"] = nrm(th - thA)
                runs.append(rec)
                # GN, SL, GD(c)
                learners = [("GN", lambda: gn_run(S.model, S.theta_A, S.X, tt)),
                            ("SL", lambda: sl_run(S.model, S.theta_A, S.X, tt, SL_M))]
                for c in CS_A:
                    learners.append((f"GD{c}", (lambda c=c: gd_run(net, thA, t, etas[c]))))
                for lname, fn in learners:
                    th, C, Q, dY, info = fn()
                    rec = dict(base, learner=lname, status=info["status"],
                               steps=info.get("steps"), res_inf=info.get("res_inf"),
                               time=info.get("time"))
                    if th is not None:
                        thn = th.numpy() if isinstance(th, torch.Tensor) else th
                        rec.update(net.kstats(thn))
                        rec["dY"] = lst(dY)
                        rec["dist"] = nrm(thn - thA)
                        rec["id_err"] = identity_err(C, Q, dY)
                        rec["pred_lit"] = P(C)
                        rec["pred_supp"] = P(C - np.outer(dY, dY))
                        rec["C_area_F"] = nrm((C - C.T) / 2)
                        rec["Q_tr"] = float(np.trace(Q))
                        mk = f"{name}|{s}|{sg}|{lname}"
                        mats[mk + "|C"] = C
                        mats[mk + "|Q"] = Q
                    runs.append(rec)
    os.makedirs(MATDIR, exist_ok=True)
    mf = os.path.join(MATDIR, f"{tag}_h{h}_s{seed}.npz")
    np.savez_compressed(mf, **mats)
    return {"lamA": lst(lam), "kA": kA, "etas": {str(c): v for c, v in etas.items()},
            "P": S.model.P, "dirs": dinfo, "runs": runs, "matfile": os.path.basename(mf)}


# ---------------------------------------------------------------------------
# Part B (Test 11)
# ---------------------------------------------------------------------------
def gd_flow(net, model, X, th0, y, c=C_B, rhos=RHOS):
    eta = eta_c(model, torch.from_numpy(th0), X, c)
    F0 = net.F(th0)
    r0n = nrm(F0 - y)
    ck = []
    pend = list(rhos)

    def after(k, th, F, r, C, Q):
        ratio = nrm(r) / r0n
        while pend and ratio <= pend[0]:
            rho = pend.pop(0)
            ck.append({"rho": rho, "step": k, "th": th.copy(), "C": C.copy(), "Q": Q.copy(),
                       "F": F.copy(), "ratio": ratio})
        return False
    th, C, Q, dY, info = gd_run(net, th0, y, eta, tol=TOL_B, cap=CAP_B, after_step=after)
    F = net.F(th)
    ck.append({"rho": "final", "step": info["steps"], "th": th.copy(), "C": C, "Q": Q, "F": F,
               "ratio": nrm(F - y) / r0n})
    info["rhos_not_reached"] = pend
    return ck, info


def partB_job(h, seed, kappa, sign, linear=False, tag="B"):
    S = tcr6.setup(h, seed, linear=linear)
    net = Net(S.model, S.X)
    th0 = S.theta_init.numpy().copy()
    F0 = net.F(th0)
    yA = S.yA.numpy()
    yp = F0 + kappa * (yA - F0)
    y = yp if sign == "+" else 2 * F0 - yp
    t0 = time.time()
    ck, ginfo = gd_flow(net, S.model, S.X, th0, y)
    t_gd = time.time() - t0
    L = tcr4.Law(S.model, S.theta_init, S.X)
    P = Pred(S.model, S.theta_init, S.X, L)
    k0 = net.kstats(th0)
    SH0 = sharp_H(net, th0, y)
    Kinv = np.linalg.inv(net.K(th0))
    rec = {"F0": lst(F0), "y": lst(y), "init": dict(k0, S_H=SH0), "gd": ginfo, "t_gd": t_gd,
           "difficulty_D": float((yp - F0) @ Kinv @ (yp - F0)), "move_sq": float((yp - F0) @ (yp - F0)),
           "r0_norm": nrm(F0 - y), "eta": ginfo["eta"]}
    cks = []
    mats = {}
    for c in ck:
        thg, C, Q, Ft = c["th"], c["C"], c["Q"], c["F"]
        dY = Ft - F0
        out = {"rho": c["rho"], "step": c["step"], "ratio": c["ratio"], "id_err": identity_err(C, Q, dY)}
        g = net.kstats(thg)
        g["S_H"] = sharp_H(net, thg, y)
        g["dist"] = nrm(thg - th0)
        out["GD"] = g
        thn, ni = np_solve(net, th0, Ft)
        out["NP_info"] = {k: v for k, v in ni.items() if k not in ("cont", "first_attempt")}
        if "first_attempt" in ni:
            out["NP_info"]["first_attempt"] = ni["first_attempt"]
            out["NP_info"]["cont_iters"] = [x["iters"] for x in ni["cont"]]
        if thn is not None:
            s_ = net.kstats(thn)
            s_["S_H"] = sharp_H(net, thn, y)
            s_["dist"] = nrm(thn - th0)
            out["NP"] = s_
        ths, Cs, Qs, dYs, si = sl_run(S.model, S.theta_init, S.X, torch.from_numpy(Ft), SL_M)
        out["SL_info"] = si
        if ths is not None:
            ths = ths.numpy()
            s_ = net.kstats(ths)
            s_["S_H"] = sharp_H(net, ths, y)
            s_["dist"] = nrm(ths - th0)
            out["SL"] = s_
            out["pred_SL_lit"] = P(Cs)
            out["pred_SL_supp"] = P(Cs - np.outer(dYs, dYs))
            out["pred_route_meas"] = P(C - Cs)
            mats[f"{c['rho']}|C_SL"] = Cs
        # split (three scalars: lam1, trK, S_H)
        for q, nm in (("lam1", "lam1"), ("trK", "trK"), ("S_H", "S_H")):
            base = rec["init"][q]
            tot = g[q] - base
            sp = {"total": tot}
            if "NP" in out:
                sp["position"] = out["NP"][q] - base
            if "SL" in out:
                sp["route"] = g[q] - out["SL"][q]
            if "NP" in out and "SL" in out:
                sp["travel"] = out["SL"][q] - out["NP"][q]
                err = abs(tot - (sp["position"] + sp["travel"] + sp["route"]))
                assert err <= 1e-12 * max(1.0, abs(base)), f"split identity {err}"
                sp["split_err"] = err
            out["split_" + nm] = sp
        DD = np.outer(dY, dY)
        tv = 0.5 * DD * (1 - 1 / SL_M)
        out["travel_pred"] = P(tv)
        out["route_pred"] = P(C - tv)
        out["travel_pred_supp"] = P(tv - DD)
        out["pathpred_lit"] = P(C)
        out["pathpred_supp"] = P(C - DD)
        mats[f"{c['rho']}|C"] = C
        mats[f"{c['rho']}|Q"] = Q
        out["dY"] = lst(dY)
        cks.append(out)
    rec["checkpoints"] = cks
    os.makedirs(MATDIR, exist_ok=True)
    mf = os.path.join(MATDIR, f"{tag}_h{h}_s{seed}_k{kappa}_{'p' if sign == '+' else 'm'}.npz")
    np.savez_compressed(mf, **mats)
    rec["matfile"] = os.path.basename(mf)
    return rec


def partBGN_job(h, seed, kappa, sign):
    """SUPPLEMENTARY (coordinator request): plain GN solve from theta_init to F_t at every Part-B
    checkpoint (default GN caps: tol 1e-12, 100 iterations, 30 halvings; nothing tuned)."""
    r = find_record(part="B", h=h, seed=seed, kappa=kappa, sign=sign)
    if r is None or r.get("status") != "OK":
        raise RuntimeError("Part-B record missing")
    S = tcr6.setup(h, seed)
    net = Net(S.model, S.X)
    th0 = S.theta_init.numpy().copy()
    F0 = net.F(th0)
    y = np.array(r["y"])
    cks = []
    for c in r["checkpoints"]:
        Ft = F0 + np.array(c["dY"])
        th, C, Q, dY, info = gn_run(S.model, S.theta_init, S.X, torch.from_numpy(Ft))
        o = {"rho": c["rho"], "status": info["status"], "steps": info.get("steps"),
             "error": info.get("error"), "time": info.get("time")}
        if th is not None:
            thn = th.numpy()
            o.update(net.kstats(thn))
            o["S_H"] = sharp_H(net, thn, y)
            o["dist"] = nrm(thn - th0)
        cks.append(o)
    return {"checkpoints": cks}


def partADIST_job(h, seed):
    """SUPPLEMENTARY: ||theta - theta_A|| for GN and SL for Part-A records produced before the
    distance field was added (NP distance is in np_dist; GD distances not recomputed)."""
    r = find_record(part="A", h=h, seed=seed)
    if r is None or r.get("status") != "OK":
        raise RuntimeError("Part-A record missing")
    if any("dist" in x for x in r["runs"]):
        return {"note": "record already has distances"}
    S = tcr6.setup(h, seed)
    thA = S.theta_A.numpy().copy()
    yA = S.yA.numpy()
    E = {k: np.array(v["e"]) for k, v in r["dirs"].items()}
    out = []
    for name, e in E.items():
        for s in SIZES_A:
            for sg in (1, -1):
                tt = torch.from_numpy(yA + sg * s * e)
                for lname, fn in (("GN", lambda: gn_run(S.model, S.theta_A, S.X, tt)),
                                  ("SL", lambda: sl_run(S.model, S.theta_A, S.X, tt, SL_M))):
                    th = fn()[0]
                    out.append({"dir": name, "s": s, "sign": sg, "learner": lname,
                                "dist": nrm(th.numpy() - thA) if th is not None else None})
    return {"dists": out}


# ---------------------------------------------------------------------------
# Part C (Test 12)
# ---------------------------------------------------------------------------
def lam_f_of(h, seed, kappa):
    r = find_record(part="B", h=h, seed=seed, kappa=kappa, sign="+")
    if r is not None and r.get("status") == "OK":
        fin = r["checkpoints"][-1]
        return fin["GD"]["lam1"], r["init"]["lam1"], "from Part-B record", r["gd"]["status"]
    S = tcr6.setup(h, seed)
    net = Net(S.model, S.X)
    th0 = S.theta_init.numpy().copy()
    F0 = net.F(th0)
    y = F0 + kappa * (S.yA.numpy() - F0)
    eta = eta_c(S.model, S.theta_init, S.X, C_B)
    th, C, Q, dY, info = gd_run(net, th0, y, eta, tol=TOL_B, cap=CAP_B)
    return net.lam1(th), net.lam1(th0), "recomputed (same GD(0.1) run)", info["status"]


def edge_run(net, th0, y, eta, n, cap=CAP_C, keep=None):
    """GD with fixed eta. Pass 1 (keep=None): per-step records. Pass 2 (keep = set of step
    indices): returns {k: theta_k} (no Jacobians)."""
    th = th0.copy()
    F, r, g = net.fg(th, y)
    r0 = float(np.abs(r).max())
    rec = {"lam1": [], "ceff": [], "x": [], "loss": [], "F": [], "sh_k": [], "sh": []}
    kept = {}
    k = 0
    status = None
    while True:
        if keep is None:
            K = net.K(th)
            ev, U = np.linalg.eigh(K)
            rec["lam1"].append(ev[-1])
            rec["ceff"].append(eta * ev[-1] / n)
            rec["x"].append(float(U[:, -1] @ r))
            rec["loss"].append(0.5 * float(r @ r) / n)
            rec["F"].append(F.copy())
            if k % SH_EVERY == 0:
                rec["sh_k"].append(k)
                rec["sh"].append(sharp_H(net, th, y))
        elif k in keep:
            kept[k] = (th.copy(), F.copy())
        res = float(np.abs(r).max())
        if not np.isfinite(res) or res > tcr4.DIV_FACTOR * r0:
            status = "DIVERGED"
            break
        if res <= TOL_C:
            status = "CONVERGED"
            break
        if k >= cap:
            status = "NOT CONVERGED (cap 60000)"
            break
        th = th + (-eta * g)
        F, r, g = net.fg(th, y)
        k += 1
    if keep is not None:
        return kept, k, status
    return rec, k, status


def select(idx, cap=MAXWIN):
    idx = list(idx)
    if len(idx) <= cap:
        return idx
    pos = np.unique(np.round(np.linspace(0, len(idx) - 1, cap)).astype(int))
    return [idx[p] for p in pos]


def partC_job(h, seed, f, kappa=4.0, tag="C"):
    lam_f, lam_i, src, bstat = lam_f_of(h, seed, kappa)
    out = {"lam_i": lam_i, "lam_f": lam_f, "lam_f_source": src, "partB_status": bstat,
           "kappa_C": kappa}
    if not lam_f >= 1.10 * lam_i:
        out["eligible"] = False
        out["note"] = "no threshold crossing possible (lam_f < 1.10 lam_i)"
        return out
    out["eligible"] = True
    S = tcr6.setup(h, seed)
    net = Net(S.model, S.X)
    n = N
    th0 = S.theta_init.numpy().copy()
    F0 = net.F(th0)
    y = F0 + kappa * (S.yA.numpy() - F0)
    lam_star = lam_i + f * (lam_f - lam_i)
    eta = 2 * n / lam_star
    t0 = time.time()
    rec, steps, status = edge_run(net, th0, y, eta, n)
    t_pass1 = time.time() - t0
    lam1 = np.array(rec["lam1"])
    ceff = np.array(rec["ceff"])
    Fs = np.array(rec["F"])
    sh_k = np.array(rec["sh_k"])
    sh = np.array(rec["sh"])
    cH = eta * sh
    # phases over steps taken: k = 0..steps-1
    pre = ceff[:steps] < 1.8
    edge = np.zeros(steps, bool)
    for k in range(steps):
        j = k // SH_EVERY
        if j + 1 < len(cH):
            edge[k] = cH[j] >= 1.9 and cH[j + 1] >= 1.9
    out.update({"lam_star": lam_star, "eta": eta, "steps": steps, "edge_status": status,
                "final_loss": float(rec["loss"][-1]), "final_lam1": float(lam1[-1]),
                "t_pass1": t_pass1, "frac_pre": float(pre.mean()) if steps else None,
                "frac_edge": float(edge.mean()) if steps else None,
                "cH_max": float(cH.max()), "ceff_max": float(ceff.max()),
                "edge_exists": bool(edge.any()), "n_sh": int(len(sh))})
    for nm, msk in (("PRE", pre), ("EDGE", edge)):
        if msk.any():
            out[f"ceff_{nm}_max"] = float(ceff[:steps][msk].max())
            out[f"ceff_{nm}_median"] = float(np.median(ceff[:steps][msk]))
            ksel = [i for i, k in enumerate(sh_k) if k < steps and msk[k]]
            if ksel:
                out[f"cH_{nm}_max"] = float(cH[ksel].max())
                out[f"cH_{nm}_median"] = float(np.median(cH[ksel]))
                out[f"cH_{nm}_samples"] = len(ksel)
    # windows
    wins = []
    need = set()
    plan = []
    for W in WS:
        nwin = steps // W
        for nm, msk in (("PRE", pre), ("EDGE", edge)):
            cand = [j for j in range(nwin) if msk[j * W:(j + 1) * W].all()]
            sel = select(cand)
            out[f"nwin_{nm}_W{W}_candidates"] = len(cand)
            for j in sel:
                plan.append((W, nm, j * W))
                need.add(j * W)
                need.add(j * W + W)
    t1 = time.time()
    kept, steps2, status2 = edge_run(net, th0, y, eta, n, keep=need)
    assert steps2 == steps and status2 == status
    for k, (_, Fk) in kept.items():
        assert np.array_equal(Fk, Fs[k]), "pass-2 trajectory differs from pass 1"
    preds = {}
    for (W, nm, a) in plan:
        b = a + W
        tha, thb = kept[a][0], kept[b][0]
        Fa = Fs[a]
        C = np.zeros((n, n))
        Q = np.zeros((n, n))
        for k in range(a, b):
            dF = Fs[k + 1] - Fs[k]
            C += np.outer(dF, Fs[k] - Fa)
            Q += np.outer(dF, dF)
        dY = Fs[b] - Fa
        if a not in preds:
            ta = torch.from_numpy(tha)
            preds[a] = Pred(S.model, ta, S.X)
        P = preds[a]
        thn, ni = np_solve(net, tha, Fs[b])
        la, lb = float(lam1[a]), float(lam1[b])
        w = {"W": W, "phase": nm, "a": a, "lam_a": la, "lam_b": lb, "np_status": ni["status"],
             "np_iters": ni["iters"], "id_err": identity_err(C, Q, dY), "dY_norm": nrm(dY),
             "Q_tr": float(np.trace(Q))}
        if thn is not None:
            ln = net.lam1(thn)
            w["position"] = ln - la
            w["path"] = lb - ln
        w["travel"] = P(0.5 * np.outer(dY, dY))
        w["jitter"] = P(-0.5 * Q)
        w["area"] = P(0.5 * (C - C.T))
        w["pathpred"] = w["travel"] + w["jitter"] + w["area"]
        w["travel_supp"] = P(-0.5 * np.outer(dY, dY))
        w["pathpred_supp"] = w["travel_supp"] + w["jitter"] + w["area"]
        K = net.K(tha)
        q1 = np.linalg.eigh(K)[1][:, -1]
        trQ = float(np.trace(Q))
        w["topshare"] = float(q1 @ Q @ q1 / trQ) if trQ > 0 else None
        wins.append(w)
    out["t_windows"] = time.time() - t1
    out["n_law_built"] = len(preds)
    out["windows"] = wins
    os.makedirs(MATDIR, exist_ok=True)
    mf = os.path.join(MATDIR, f"{tag}_h{h}_s{seed}_f{f}.npz")
    np.savez_compressed(mf, lam1=lam1, ceff=ceff, x=np.array(rec["x"]), loss=np.array(rec["loss"]),
                        sh_k=sh_k, sh=sh)
    out["matfile"] = os.path.basename(mf)
    return out


# ---------------------------------------------------------------------------
# job table and driver
# ---------------------------------------------------------------------------
def jobs_for(part, widths, seeds):
    J = []
    if part == "LIN":
        J.append(({"part": "LIN-A", "h": 32, "seed": 0},
                  lambda: partA_job(32, 0, linear=True, sizes=[0.05], tag="LINA")))
        J.append(({"part": "LIN-B", "h": 32, "seed": 0, "kappa": 1.0, "sign": "+"},
                  lambda: partB_job(32, 0, 1.0, "+", linear=True, tag="LINB")))
    elif part == "A":
        for h in widths:
            for s in seeds:
                J.append(({"part": "A", "h": h, "seed": s}, (lambda h=h, s=s: partA_job(h, s))))
    elif part == "B":
        for h in widths:
            for s in seeds:
                for kp in reversed(KAPPAS):
                    for sg in ("+", "-"):
                        J.append(({"part": "B", "h": h, "seed": s, "kappa": kp, "sign": sg},
                                  (lambda h=h, s=s, kp=kp, sg=sg: partB_job(h, s, kp, sg))))
    elif part == "C":
        for h in widths:
            for s in seeds:
                for f in FS_C:
                    J.append(({"part": "C", "h": h, "seed": s, "f": f},
                              (lambda h=h, s=s, f=f: partC_job(h, s, f))))
    elif part == "BGN":
        for h in widths:
            for s in seeds:
                for kp in reversed(KAPPAS):
                    for sg in ("+", "-"):
                        if find_record(part="B", h=h, seed=s, kappa=kp, sign=sg) is None:
                            continue
                        J.append(({"part": "BGN", "h": h, "seed": s, "kappa": kp, "sign": sg},
                                  (lambda h=h, s=s, kp=kp, sg=sg: partBGN_job(h, s, kp, sg))))
    elif part == "ADIST":
        for h in widths:
            for s in seeds:
                if find_record(part="A", h=h, seed=s) is None:
                    continue
                J.append(({"part": "ADIST", "h": h, "seed": s}, (lambda h=h, s=s: partADIST_job(h, s))))
    elif part == "CSUPP8":
        for h in widths:
            for s in seeds:
                for f in FS_C:
                    J.append(({"part": "C-SUPP-k8", "h": h, "seed": s, "f": f},
                              (lambda h=h, s=s, f=f: partC_job(h, s, f, kappa=8.0, tag="CS8"))))
    else:
        raise ValueError(part)
    return J


def claim(claimdir, k):
    fn = os.path.join(claimdir, k.replace("|", "_") + ".claim")
    try:
        fd = os.open(fn, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main(argv):
    scratch, deadline, parts = None, None, []
    widths, seeds = WIDTHS, SEEDS
    i = 0
    while i < len(argv):
        if argv[i] == "--scratch":
            scratch = argv[i + 1]
            i += 2
        elif argv[i] == "--deadline":
            deadline = float(argv[i + 1])
            i += 2
        elif argv[i] == "--widths":
            widths = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--seeds":
            seeds = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        else:
            parts.append(argv[i])
            i += 1
    assert scratch, "--scratch DIR required"
    claimdir = os.path.join(scratch, "claims7")
    os.makedirs(claimdir, exist_ok=True)
    log(f"pid {os.getpid()} threads torch={torch.get_num_threads()} parts={parts} deadline={deadline}")
    for part in parts:
        t_start = time.time()
        n_run = 0
        done = done_keys()
        for kr, fn in jobs_for(part, widths, seeds):
            k = key(kr)
            if k in done or not claim(claimdir, k):
                continue
            if deadline is not None and time.time() > deadline:
                save_records([dict(kr, status="NOT RUN", error="time budget passed before start")])
                log(f"{k} NOT RUN (deadline)")
                continue
            t0 = time.time()
            try:
                rec = dict(kr, status="OK", **fn())
            except Exception as ex:  # noqa: BLE001
                rec = dict(kr, status="ERROR", error=f"{type(ex).__name__}: {ex}",
                           tb=traceback.format_exc())
            rec["wall"] = time.time() - t0
            rec["pid"] = os.getpid()
            save_records([rec])
            n_run += 1
            log(f"{k} {rec['status']} wall={rec['wall']:.1f}s {rec.get('error', '')}")
        save_records([], meta_run={"part": part, "pid": os.getpid(), "n_run": n_run,
                                   "wall": time.time() - t_start,
                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"done {part}: {n_run} jobs in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
