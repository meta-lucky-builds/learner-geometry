"""Runs the SPEC6.md tests 5-9; writes results6.json incrementally (lock + merge, so several
worker processes may run at once; jobs are claimed through claim files in the scratch dir).

Usage:  python run6.py --scratch DIR [--deadline EPOCH] TEST [TEST ...]
Tests:  t5 t6 t7 t8 t9     (one job = one (test, h, seed); h in {32, 64, 128}, seeds 0-4)
Jobs already in results6.json (status not ERROR / TIMEOUT) are skipped.
Thread count is fixed by tcr4 (THREADS = 2) for every process.
"""
import tcr6  # first: fixes thread count (via tcr4)
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

import tcr
import tcr3
import tcr4
from tcr3 import nrm, cos
from tcr4 import c_parts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results6.json")
LOCK = OUT + ".lock"

SEEDS = [0, 1, 2, 3, 4]
WIDTHS = [32, 64, 128]
N = 16

C5 = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 1.95]
EPS5 = [0.1, 0.05, 0.025]
EPS7 = [0.1, 0.05, 0.025]
TTOT7 = [50, 200, 800, 3200]
PI9 = [16, 64, 256]
M9 = [1, 2, 4, 8, 16]
N8, R8 = 4000, 32


# ---------------------------------------------------------------------------
# results file: lock + merge
# ---------------------------------------------------------------------------
KEYF = ("test", "h", "seed")


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
            json.dump(db, fh, indent=1, default=float)
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


def fl(x):
    return float(x)


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------
def t5_job(h, seed):
    S = tcr6.setup(h, seed)
    L = tcr6.law(S)
    lamA = S.lam[0].item()
    f1 = lambda th: tcr6.lam1(S.model, th, S.X)  # noqa: E731
    gi = tcr6.grad_lam1_in_T_span(S)
    G = gi["G"]
    base = {k: v for k, v in gi.items() if k != "g"}
    base.update({"lam1_A": lamA, "lam16_A": S.lam[-1].item(), "P": S.model.P})
    q1 = S.Q[:, 0]
    A = {}
    for eps in EPS5:
        y = S.yA + eps * q1
        th_gn, C_gn, ign = tcr6.gn_C(S, y)
        l_gn = f1(th_gn)
        F_gn = S.model.f(th_gn, S.X)
        rows = {}
        for c in C5:
            th, C, info = tcr6.gd_C(S, y, c)
            dl = f1(th) - l_gn
            Cd = C - C_gn
            Cs, Ca = c_parts(Cd)
            V = L(Cd)
            p1 = tcr6.dderiv(f1, S.theta_A, V)
            p2 = eps ** 2 * tcr6.pred2_factor(c) * G / lamA ** 2
            p2s = eps ** 2 * tcr6.pred2_factor(c) * gi["sumsq"] / lamA ** 2
            rows[str(c)] = {"dlam1": dl, "pred1": p1, "pred2": p2, "pred2_sumsq": p2s,
                            "pred1_sym": tcr6.dderiv(f1, S.theta_A, L(Cs)),
                            "status": info["status"], "steps": info["steps"],
                            "res_inf": info["res_inf"], "eta": info["eta"],
                            "Cdiff_sym_F": fl(np.linalg.norm(Cs)), "Cdiff_area_F": fl(np.linalg.norm(Ca)),
                            "Cdiff_q1q1": fl(q1.numpy() @ Cd @ q1.numpy()),
                            "Cdiff_q1q1_pred": eps ** 2 * (1 - c) / (2 - c),
                            "out_diff_inf": (S.model.f(th, S.X) - F_gn).abs().max().item(),
                            "dtheta_norm": nrm(th - th_gn)}
        A[str(eps)] = {"gn_iters": ign["steps"], "lam1_gn": l_gn, "rows": rows}
    B = {}
    yB = S.yA + 0.05 * S.u
    for c in (2.05, 1.9):
        tr = tcr6.protocol_B(S, yB, c)
        lam = np.array(tr["lam1"])
        ce = np.array(tr["c_eff"])
        x = np.array(tr["x"])
        K = len(lam)
        dl = lam[1:] - lam[:-1]
        sel = np.where(ce[:-1] > 1.5)[0]
        slope, icpt, r2 = tcr6.ols(x[sel] ** 2, dl[sel]) if len(sel) else (np.nan, np.nan, np.nan)
        below = np.where(ce < 2.0)[0]
        k2 = int(below[0]) if len(below) else None
        pre = dl[:k2] if (k2 is not None and k2 > 0) else dl
        pred_slope = -(c ** 2 / 4) * G / lamA ** 2
        ce_med = float(np.median(ce[sel])) if len(sel) else float("nan")
        B[str(c)] = {"n_recorded": K, "stopped_at": tr["stopped_at"], "eta": tr["eta"],
                     "lam1_0": fl(lam[0]), "lam1_end": fl(lam[-1]), "lam1_min": fl(lam.min()),
                     "c_eff_0": fl(ce[0]), "c_eff_end": fl(ce[-1]), "c_eff_min": fl(ce.min()),
                     "first_k_ceff_below_2": k2,
                     "lam1_at_k2": fl(lam[k2]) if k2 is not None else None,
                     "frac_dlam_neg_before_k2": fl((pre < 0).mean()) if len(pre) else None,
                     "lam1_monotone_down_before_k2": bool((pre <= 0).all()) if len(pre) else None,
                     "loss_0": tr["loss"][0], "loss_end": tr["loss"][-1], "loss_max": max(tr["loss"]),
                     "rnorm_end": tr["rnorm"][-1],
                     "reg_n": int(len(sel)), "reg_slope": slope, "reg_intercept": icpt, "reg_R2": r2,
                     "pred_slope_ceff_c": pred_slope,
                     "pred_slope_ceff_median": -(ce_med ** 2 / 4) * G / lamA ** 2, "ceff_median_reg": ce_med,
                     "trace_lam1_every10": lam[::10].tolist(), "trace_ceff_every10": ce[::10].tolist(),
                     "trace_x_every10": x[::10].tolist(), "trace_loss_every10": tr["loss"][::10]}
    return {"base": base, "A": A, "B": B}


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------
def t6_job(h, seed):
    S = tcr6.setup(h, seed)
    states = {"S0": S.theta_A}
    infos = {}
    for tag, e, ln in (("u", S.u, "GN"), ("v", S.v, "GN"), ("GD", S.u, "GD")):
        out, inf = tcr6.practice_states(S, e, ln)
        for m, th in out.items():
            states[f"{tag}{m}"] = th
        infos[tag] = {"n_learns": len(inf),
                      "steps": [i.get("steps", i.get("iters")) for i in inf],
                      "all_converged": all(i.get("converged", True) for i in inf),
                      "max_res_inf": max(i.get("res_inf", 0.0) for i in inf)}
    bd = tcr6.unit_dirs(6000 + seed, 16)
    delta = 0.05 * torch.linalg.norm(S.yA).item()
    eta0 = 0.5 * N / S.lam[0].item()
    rec = {}
    for name, th in states.items():
        pr = tcr6.interference_probe(S, th, bd, eta0, delta)
        pr.update(tcr3.kernel_stats(S.model, th, S.X, S.u, S.v))
        pr["dtheta_norm"] = nrm(th - S.theta_A)
        rec[name] = pr
    return {"states": rec, "practice": infos, "delta": delta, "eta0": eta0}


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------
def t7_job(h, seed):
    S = tcr6.setup(h, seed)
    L = tcr6.law(S)
    s, w, s2, w2 = tcr6.strong_weak_dirs(S)
    pairs = {"ss'": (s, s2), "ww'": (w, w2), "sw": (s, w)}
    fpA = S.model.f(S.theta_A, S.X_probe)
    part_a = {}
    for pn, (p, q) in pairs.items():
        b = tcr3.bracket(S.model, S.theta_A, S.X, p, q)
        Vlaw = L(np.outer(q.numpy(), p.numpy()) - np.outer(p.numpy(), q.numpy()))
        part_a[pn] = {"b_norm": nrm(b), "cos_b_lawpred": cos(b, Vlaw), "ratio_lawpred_b": nrm(Vlaw) / nrm(b)}
        for eps in EPS7:
            tp, tq = tcr6.order_pair(S, p, q, eps)
            O = tp - tq
            o = S.model.f(tp, S.X_probe) - S.model.f(tq, S.X_probe)
            part_a[pn][str(eps)] = {"O_norm": nrm(O), "cos_O_b": cos(O, b),
                                    "ratio": nrm(O) / (eps ** 2 * nrm(b)),
                                    "cos_O_law": cos(O, Vlaw),
                                    "ratio_law": nrm(O) / (eps ** 2 * nrm(Vlaw)),
                                    "probe_rms": nrm(o) / math.sqrt(o.numel()),
                                    "end_out_diff_inf": (S.model.f(tp, S.X) - S.model.f(tq, S.X)).abs().max().item()}
    # part (b)
    eps = 0.1
    ystar = S.yA + eps * (s + w)
    firsts = {"JOINT": None, "EASY": S.yA + eps * s, "HARD": S.yA + eps * w}
    KA, FA = S.K.numpy(), S.F_A_np
    Qn = S.Q.numpy()
    sh, wh = (s / torch.linalg.norm(s)).numpy(), (w / torch.linalg.norm(w)).numpy()
    part_b = {}
    for T in TTOT7:
        F, Fl, FP, st = {}, {}, {}, {}
        for cn, y1 in firsts.items():
            th, inf = tcr6.curriculum(S, y1, ystar, T)
            F[cn] = S.model.f(th, S.X).numpy()
            FP[cn] = S.model.f(th, S.X_probe)
            Fl[cn] = tcr6.curriculum_lin(KA, FA, None if y1 is None else y1.numpy(), ystar.numpy(), T)
            st[cn] = [i["status"] for i in inf]
        r = {}
        for a, b in (("EASY", "HARD"), ("JOINT", "EASY"), ("JOINT", "HARD")):
            d = F[a] - F[b]
            dl = Fl[a] - Fl[b]
            proj = Qn.T @ d
            r[f"{a}-{b}"] = {"d_norm": fl(np.linalg.norm(d)), "dlin_norm": fl(np.linalg.norm(dl)),
                             "weak_share": fl((proj[8:] ** 2).sum() / (d @ d)),
                             "weak_share_lin": fl(((Qn.T @ dl)[8:] ** 2).sum() / (dl @ dl)),
                             "cos_d_dlin": fl(d @ dl / (np.linalg.norm(d) * np.linalg.norm(dl))),
                             "ratio_d_dlin": fl(np.linalg.norm(d) / np.linalg.norm(dl))}
        oEH = FP["EASY"] - FP["HARD"]
        rs = ystar.numpy()
        losses = {cn: {"loss_s": fl((sh @ (F[cn] - rs)) ** 2 / (2 * N)),
                       "loss_w": fl((wh @ (F[cn] - rs)) ** 2 / (2 * N)),
                       "loss_total": fl(((F[cn] - rs) ** 2).sum() / (2 * N))} for cn in F}
        part_b[str(T)] = {"pairs": r, "probe_rms_EH": nrm(oEH) / math.sqrt(oEH.numel()),
                          "losses": losses, "status": st}
    return {"part_a": part_a, "part_b": part_b,
            "lam": S.lam.tolist(), "probe_rms_A_scale": nrm(fpA) / math.sqrt(fpA.numel())}


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------
def t8_job(h, seed):
    S = tcr6.setup(h, seed)
    L = tcr6.law(S)
    KA = S.K.numpy()
    Pm, ndraw, minre = tcr6.draw_P(seed, KA)
    npm = tcr6.NpMLP6(S.model, S.X)
    zetas = [tcr6.td_noise(seed, st, N8) for st in range(R8)]
    al0, tr0 = tcr6.kernel_alignment(S.model, S.theta_A, S.X, Pm)
    fal = lambda th: tcr6.kernel_alignment(S.model, th, S.X, Pm)[0]  # noqa: E731
    ftr = lambda th: tcr6.kernel_alignment(S.model, th, S.X, Pm)[1]  # noqa: E731
    G = {}
    vecs = {}
    for gm in tcr6.GAMMAS8:
        q = tcr6.lyap_quantities(KA, Pm, gm)
        res, asym, emin, emax = tcr6.lyap_residual(q)
        pr = tcr6.drift_predictions(S, q, Pm, gm)
        AS = q["A"] @ q["Sig"]
        symAS = (AS + AS.T) / 2
        Ctr = tcr6.transient_expected_C(q, N8)
        d_tr = L(Ctr)
        # simulation
        Vsum = torch.zeros(S.model.P)
        Vsq = 0.0
        Csum = np.zeros((N, N))
        al_sum, tr_sum = 0.0, 0.0
        n_ok, gn_iters, pre_res, blew = 0, [], [], 0
        Vs = []
        for st in range(R8):
            th_end, C, info = tcr6.td_simulate(S, npm, q["M"], q["etap"], zetas[st])
            V = th_end - S.theta_A
            Vs.append(V)
            Vsum += V
            Csum += C
            a_, t_ = tcr6.kernel_alignment(S.model, th_end, S.X, Pm)
            al_sum += a_
            tr_sum += t_
            gn_iters.append(info["steps"])
            pre_res.append(info["pre_return_res_inf"])
            blew += int(info["blew_up"])
            n_ok += 1
        Vm = Vsum / R8
        Vstack = torch.stack(Vs)
        se = nrm(Vstack.std(0, unbiased=True)) / math.sqrt(R8)
        Cmean = Csum / R8
        d_meanC = L(Cmean)
        vecs[gm] = {"V": Vm, "d_law": pr["d_law"], "d_tr": d_tr, "d_meanC": d_meanC, "Vs": Vstack}
        nd = N8 * pr["d_law"]
        G[str(gm)] = {
            "etap": q["etap"], "spec_rad_IA": q["spec_rad_IA"], "lyap_resid": res, "lyap_asym": asym,
            "Sig_eig_min": emin, "Sig_eig_max": emax,
            "d_law_norm": nrm(pr["d_law"]), "d_dir_norm": nrm(pr["d_dir"]),
            "H8a_rel": nrm(pr["d_law"] - pr["d_dir"]) / nrm(pr["d_law"]),
            "cos_law_dir": cos(pr["d_law"], pr["d_dir"]),
            "rel_pot1": nrm(pr["d_pot1"] - pr["d_law"]) / nrm(pr["d_law"]),
            "rel_pot2": nrm(pr["d_pot2"] - pr["d_law"]) / nrm(pr["d_law"]),
            "cos_pot1": cos(pr["d_pot1"], pr["d_law"]), "cos_pot2": cos(pr["d_pot2"], pr["d_law"]),
            "kappa_star": pr["kappa_star"],
            "H8b_rel_symAS_halfBB": fl(np.linalg.norm(symAS - 0.5 * q["BB"]) / np.linalg.norm(0.5 * q["BB"])),
            "supp_rel_symAS_half_BB_plus_ASAt": fl(np.linalg.norm(symAS - 0.5 * (q["BB"] + q["A"] @ q["Sig"] @ q["A"].T))
                                                  / np.linalg.norm(symAS)),
            "law_area_over_sym": nrm(pr["law_E_area"]) / nrm(pr["law_E_sym"]),
            "law_E_sym_norm": nrm(pr["law_E_sym"]),
            # simulation vs predictions
            "V_norm": nrm(Vm), "V_se_norm": se,
            "cos_V_Ndlaw": cos(Vm, nd), "ratio_V_Ndlaw": nrm(Vm) / nrm(nd),
            "cos_V_transient": cos(Vm, d_tr), "ratio_V_transient": nrm(Vm) / nrm(d_tr),
            "cos_V_meanC": cos(Vm, d_meanC), "ratio_V_meanC": nrm(Vm) / nrm(d_meanC),
            "cos_Ndlaw_transient": cos(nd, d_tr), "ratio_transient_Ndlaw": nrm(d_tr) / nrm(nd),
            "Cmean_sym_F": fl(np.linalg.norm(c_parts(Cmean)[0])), "Cmean_area_F": fl(np.linalg.norm(c_parts(Cmean)[1])),
            "NE_sym_F": fl(N8 * np.linalg.norm(c_parts(q["E"])[0])), "NE_area_F": fl(N8 * np.linalg.norm(c_parts(q["E"])[1])),
            "Ctr_sym_F": fl(np.linalg.norm(c_parts(Ctr)[0])), "Ctr_area_F": fl(np.linalg.norm(c_parts(Ctr)[1])),
            "gn_iters_max": max(gn_iters), "pre_return_res_inf_median": fl(np.median(pre_res)),
            "blew_up": blew, "streams": n_ok,
            # kernel alignment / trace: measured per step vs predicted per step
            "align_meas_per_step": (al_sum / R8 - al0) / N8, "align_pred_per_step": tcr6.dderiv(fal, S.theta_A, pr["d_law"]),
            "trK_meas_per_step": (tr_sum / R8 - tr0) / N8, "trK_pred_per_step": tcr6.dderiv(ftr, S.theta_A, pr["d_law"]),
        }
    # H8b across gamma; H8e
    e0 = None
    for gm in tcr6.GAMMAS8:
        q = tcr6.lyap_quantities(KA, Pm, gm)
        Es = c_parts(q["E"])[0]
        les = L(Es)
        if e0 is None:
            e0 = les
        G[str(gm)]["H8b_law_Esym_rel_vs_gamma0"] = nrm(les - e0) / nrm(e0)
        Vd = vecs[gm]["V"] - vecs[0.0]["V"]
        pd = N8 * (vecs[gm]["d_law"] - vecs[0.0]["d_law"])
        G[str(gm)]["cos_dV_dpred"] = cos(Vd, pd) if gm != 0.0 else None
        G[str(gm)]["ratio_dV_dpred"] = nrm(Vd) / nrm(pd) if gm != 0.0 else None
        if gm != 0.0:
            tdd = vecs[gm]["d_tr"] - vecs[0.0]["d_tr"]
            G[str(gm)]["cos_dV_dtransient"] = cos(Vd, tdd)
            G[str(gm)]["ratio_dV_dtransient"] = nrm(Vd) / nrm(tdd)
            Dst = vecs[gm]["Vs"] - vecs[0.0]["Vs"]          # paired (common random numbers)
            G[str(gm)]["dV_se_norm"] = nrm(Dst.std(0, unbiased=True)) / math.sqrt(R8)
            G[str(gm)]["dV_norm"] = nrm(Vd)
    return {"P_draws": ndraw, "minre_KM": {str(k): v for k, v in minre.items()},
            "lam_max": S.lam[0].item(), "align_A": al0, "trK_A": tr0, "gammas": G}


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------
def t9_job(h, seed):
    S = tcr6.setup(h, seed)
    L = tcr6.law(S)
    npm = tcr4.np_model(S.model, S.X)
    eta = 0.5 * N / S.lam[0].item()
    KA, FA, yA = S.K.numpy(), S.F_A_np, S.yA.numpy()
    x1 = S.X_probe[0]
    g1 = S.model.jac(S.theta_A, x1[None, :])[0]
    ps, qs, sinfo = tcr6.steered_plane(S, x1)
    planes = {"uv": (S.u, S.v), "steered": (ps, qs)}
    fpA = S.model.f(S.theta_A, S.X_probe)
    out = {"steer_info": sinfo}
    for pl, (p, q) in planes.items():
        out[pl] = {}
        for Pi in PI9:
            om = 2 * math.pi / Pi
            drives = {"ROT+": ("ROT", om), "ROT-": ("ROT", -om), "LINE+": ("LINE", om), "LINE-": ("LINE", -om)}
            runs, apri = {}, {}
            for dn, (kind, w) in drives.items():
                runs[dn] = tcr6.run_drive(S, npm, eta, p, q, w, Pi, kind, M9)
                apri[dn] = tcr6.apriori_drive_C(KA, FA, yA, eta / N, p, q, w, Pi, kind, M9)
            res = {}
            for m in M9:
                Vp = runs["ROT+"][m][0] - S.theta_A
                Vm = runs["ROT-"][m][0] - S.theta_A
                odd, even = (Vp - Vm) / 2, (Vp + Vm) / 2
                Cp, Cm = apri["ROT+"][m], apri["ROT-"][m]
                Cps, Cpa = c_parts(Cp)
                p_odd, p_even = L(Cpa), L(Cps)
                p_odd_c, p_even_c = L((Cp - Cm) / 2), L((Cp + Cm) / 2)
                Cmp, Cmm = runs["ROT+"][m][1], runs["ROT-"][m][1]
                p_odd_meas = L((Cmp - Cmm) / 2)
                VLp = runs["LINE+"][m][0] - S.theta_A
                VLm = runs["LINE-"][m][0] - S.theta_A
                CL = apri["LINE+"][m]
                fp = S.model.f(runs["ROT+"][m][0], S.X_probe)
                fm = S.model.f(runs["ROT-"][m][0], S.X_probe)
                of = ((fp - fm) / 2).numpy()
                pred_f1 = torch.dot(g1, p_odd).item()
                pred_f1_c = torch.dot(g1, p_odd_c).item()
                res[str(m)] = {
                    "odd_norm": nrm(odd), "even_norm": nrm(even),
                    "cos_odd_lit": cos(odd, p_odd), "ratio_odd_lit": nrm(odd) / nrm(p_odd),
                    "cos_even_lit": cos(even, p_even), "ratio_even_lit": nrm(even) / nrm(p_even),
                    "cos_odd_combo": cos(odd, p_odd_c), "ratio_odd_combo": nrm(odd) / nrm(p_odd_c),
                    "cos_even_combo": cos(even, p_even_c), "ratio_even_combo": nrm(even) / nrm(p_even_c),
                    "cos_odd_meas": cos(odd, p_odd_meas), "ratio_odd_meas": nrm(odd) / nrm(p_odd_meas),
                    "pred_odd_lit_norm": nrm(p_odd), "pred_even_lit_norm": nrm(p_even),
                    "Capri_sym_F": fl(np.linalg.norm(Cps)), "Capri_area_F": fl(np.linalg.norm(Cpa)),
                    "Capri_diff_half_sym_F": fl(np.linalg.norm(c_parts((Cp - Cm) / 2)[0])),
                    "line_odd_lit_norm": nrm((VLp - VLm) / 2),
                    "line_V_norm": nrm(VLp),
                    "line_law_area_apri_norm": nrm(L(c_parts(CL)[1])),
                    "line_law_area_meas_norm": nrm(L(c_parts(runs["LINE+"][m][1])[1])),
                    "f1_odd": fl(of[0]), "f1_pred_lit": pred_f1, "f1_pred_combo": pred_f1_c,
                    "f1_ratio_lit": fl(of[0]) / pred_f1 if pred_f1 != 0 else float("nan"),
                    "f1_ratio_combo": fl(of[0]) / pred_f1_c if pred_f1_c != 0 else float("nan"),
                    "others_median_abs_odd": fl(np.median(np.abs(of[1:]))),
                    "f1_over_others_median": fl(abs(of[0]) / np.median(np.abs(of[1:]))),
                    "probe_rms_odd": fl(np.linalg.norm(of) / math.sqrt(len(of))),
                    "gn_iters": [runs[d][m][2]["steps"] for d in drives],
                }
            out[pl][str(Pi)] = res
    out["probe_rms_A"] = nrm(fpA) / math.sqrt(fpA.numel())
    return out


JOBS = {"t5": t5_job, "t6": t6_job, "t7": t7_job, "t8": t8_job, "t9": t9_job}


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
    scratch, deadline, tests = None, None, []
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
            tests.append(argv[i])
            i += 1
    assert scratch, "--scratch DIR required (claim files go there)"
    claimdir = os.path.join(scratch, "claims6")
    os.makedirs(claimdir, exist_ok=True)
    log(f"pid {os.getpid()} threads torch={torch.get_num_threads()} tests={tests} deadline={deadline}")
    for t in tests:
        t_start = time.time()
        n_run, stopped = 0, False
        done = done_keys()
        for h in widths:
            for s in seeds:
                kr = {"test": t, "h": h, "seed": s}
                k = key(kr)
                if k in done or not claim(claimdir, k):
                    continue
                if deadline is not None and time.time() > deadline:
                    stopped = True
                    save_records([dict(kr, status="NOT RUN", error="time budget passed before start")])
                    log(f"{k} NOT RUN (deadline)")
                    continue
                t0 = time.time()
                try:
                    rec = dict(kr, status="OK", **JOBS[t](h, s))
                except Exception as ex:  # noqa
                    rec = dict(kr, status="ERROR", error=f"{type(ex).__name__}: {ex}",
                               tb=traceback.format_exc())
                rec["wall"] = time.time() - t0
                rec["pid"] = os.getpid()
                rec["threads"] = torch.get_num_threads()
                save_records([rec])
                n_run += 1
                log(f"{k} {rec['status']} wall={rec['wall']:.1f}s {rec.get('error', '')}")
        save_records([], meta_run={"test": t, "pid": os.getpid(), "n_run": n_run,
                                   "wall": time.time() - t_start, "stopped": stopped,
                                   "threads": torch.get_num_threads(),
                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"done {t}: {n_run} jobs in {time.time() - t_start:.1f}s stopped={stopped}")


if __name__ == "__main__":
    main(sys.argv[1:])
