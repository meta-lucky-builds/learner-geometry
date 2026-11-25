"""SPEC12 shared code: learners, legs, closing, polygons, readings, units, predictions, plan, unit-level rules.

Imported by spec12_run.py, spec12_report.py and test_spec12.py. The library part9lib is passed in explicitly as the
argument L (no module-level state here). The reviewed SPEC11 helpers are imported unchanged from ../spec11 as S.
All tensors are float64 (the library sets the default dtype; every tensor here is made from library tensors).
"""
import os
import sys
import math

import torch

sys.dont_write_bytecode = True                   # never write __pycache__ into the read-only spec11 directory
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC11_DIR = os.path.normpath(os.path.join(HERE, "..", "spec11"))
if SPEC11_DIR not in sys.path:
    sys.path.insert(0, SPEC11_DIR)
import spec11_common as S                        # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Section 3: constants
# ---------------------------------------------------------------------------------------------------------------
EPS = 1e-8
B1 = 0.9
B2 = 0.999
GSTAR = 9.5                                       # = (1 + B1) / (2 (1 - B1))
CLOSE_TOL = 1e-6
TAU_MODE = 1e-5
TAU_GEN = 1e-3
CAP_MODE = 4000
CAP_GEN = 6000
CAP_ADAM = 3000
DIVERGE = 1e4
SIGNS = (1, -1)

W_NAMES = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
QNAMES = ["u"] + W_NAMES                          # probe list QV = [u, W1, ..., W8]
BREAK_MODES = ("", "stalelook", "noclose", "repro")
REPRO_COMBO = "tanh-init-s0"
CHK_STEPS = 60                                    # Part V chk (section 7)
OP_PROBE_STEP = 50                                # r50 (section 6.3)

# For `spec12_run.py --list` ONLY (the network is not loaded there): the tanh lam_ratios quoted in section 3.
# The run itself always uses float(ctx.dinfo[nm]["lam_ratio"]) of the loaded combination.
LIST_NOTE_LAM_RATIOS = {"eigA": 1.0, "eigB": 0.2314, "eigC": 0.1007}


def check_break(brk):
    if brk not in BREAK_MODES:
        raise ValueError(f"--break {brk!r} is not one of {BREAK_MODES}")
    return brk


# ---------------------------------------------------------------------------------------------------------------
# Section 1.4: reference implementation of the one-mode alignment fraction (VERBATIM from the spec)
# ---------------------------------------------------------------------------------------------------------------
def f_scalar(c, mu=0.0, gamma=0.0, mid=False, tol=1e-14, cap=2_000_000):
    """Alignment fraction of a sum-of-gradients learner on ONE kernel mode (linearised model).
    x = position of the outputs along the journey (0 start, 1 target); u = last step (same units).
    Step: look y = x + gamma*u  (mid=True: y = x + (c/2)*(1-x); needs mu == 0 and gamma == 0);
          pay = c*(1-y); u = mu*u + pay; x = x + u.
    The gradient taken at y pays for the share w = pay/(1-mu) of the journey. f = sum of w*y.
    Returns (f, steps); f = nan if the recursion diverges or does not converge within cap steps."""
    assert not (mid and (mu != 0.0 or gamma != 0.0))
    x = 0.0; u = 0.0; f = 0.0; k = 0
    while k < cap:
        y = x + (0.5 * c * (1.0 - x) if mid else gamma * u)
        pay = c * (1.0 - y)
        f += (pay / (1.0 - mu)) * y
        u = mu * u + pay
        x = x + u
        k += 1
        if abs(1.0 - x) < tol and abs(u) < tol:
            return f, k
        if abs(x) > 1e6:
            break
    return float("nan"), k


# closed forms of section 1.4 (used by test T1 and printed by the report next to the measured values)
def f_gd(c):
    return (1.0 - c) / (2.0 - c)


def f_gdmid(c):
    return 0.5 + c * c / (2.0 * (c * c - 2.0 * c + 4.0))


def f_hb(c, mu):
    return 1.0 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))


def a_gd(c):
    return c / (2.0 - c)


def a_hb(c, mu):
    return c * (1 + mu) ** 2 / ((1 - mu) ** 2 * (2 * (1 + mu) - c))


# ---------------------------------------------------------------------------------------------------------------
# Section 4: learners
# ---------------------------------------------------------------------------------------------------------------
def SL1():
    return dict(kind="X", name="SL1")


def EM():
    return dict(kind="X", name="EM")


def GD(c):
    return dict(kind="E", c=float(c), mu=0.0, gamma=0.0, mid=False)


def GDmid(c):
    return dict(kind="E", c=float(c), mu=0.0, gamma=0.0, mid=True)


def HB(c, mu):
    return dict(kind="E", c=float(c), mu=float(mu), gamma=0.0, mid=False)


def HBLA(c, mu, gamma):
    assert gamma != 0
    return dict(kind="E", c=float(c), mu=float(mu), gamma=float(gamma), mid=False)


def ADAM(kappa, b1=B1, b2=B2, gamma=0.0, keep="none"):
    assert keep in ("none", "m", "v", "mv")
    return dict(kind="A", kappa=float(kappa), b1=float(b1), b2=float(b2), gamma=float(gamma), keep=keep)


def PRE(kappa):
    return dict(kind="F", kappa=float(kappa), b1=0.0)


def PREM(kappa):
    return dict(kind="F", kappa=float(kappa), b1=B1)


def learner_name(Ld):
    k = Ld["kind"]
    if k == "X":
        return Ld["name"]
    if k == "E":
        if Ld["mid"]:
            assert Ld["mu"] == 0 and Ld["gamma"] == 0
            return "GDmid(%g)" % Ld["c"]
        if Ld["gamma"] != 0:
            return "HBLA(%g,%g,%g)" % (Ld["c"], Ld["mu"], Ld["gamma"])
        if Ld["mu"] == 0:
            return "GD(%g)" % Ld["c"]
        return "HB(%g,%g)" % (Ld["c"], Ld["mu"])
    if k == "A":
        return "ADAM(%g,%g,%g,%g,%s)" % (Ld["kappa"], Ld["b1"], Ld["b2"], Ld["gamma"], Ld["keep"])
    if k == "F":
        assert Ld["b1"] in (0.0, B1)
        return ("PRE(%g)" if Ld["b1"] == 0 else "PREM(%g)") % Ld["kappa"]
    raise ValueError(k)


def learner_lr(Ld, st, N, P, dJ):
    """Kind E: eta = c * N / lambda_1(K0) of the unit's start. Kinds A, F: kappa * dJ / sqrt(P), dJ = exact-jump distance
    of the unit's FIRST leg with sign +1 (reference unit). Kind X: no learning rate (None)."""
    k = Ld["kind"]
    if k == "E":
        return Ld["c"] * N / st.lam1
    if k in ("A", "F"):
        return Ld["kappa"] * dJ / math.sqrt(P)
    return None


def keep_state(Ld, state):
    """Section 4 keep policy, applied to the Adam-family state returned by the previous leg (or polygon).
    None means the zero state. Kinds other than "A" never carry state across legs."""
    if Ld["kind"] != "A" or state is None:
        return None
    keep = Ld["keep"]
    if keep == "none":
        return None
    if keep == "m":
        return dict(m=state["m"], km=state["km"], v=torch.zeros_like(state["v"]), kv=0)
    if keep == "v":
        return dict(m=torch.zeros_like(state["m"]), km=0, v=state["v"], kv=state["kv"])
    if keep == "mv":
        return dict(m=state["m"], km=state["km"], v=state["v"], kv=state["kv"])
    raise ValueError(keep)


def direction(combo, name):
    """Section 3: a direction name is looked up in combo["dirs"] first, then in combo["eig"]."""
    if name in combo["dirs"]:
        return combo["dirs"][name]
    return combo["eig"][name][0]


def probes(combo, u):
    """QV = [u, W1..W8] (names QNAMES)."""
    return [u] + [combo["dirs"][w] for w in W_NAMES]


def finite_t(x):
    return x is not None and S.finite(x)


def div(a, b):
    """a / b, nan when b == 0 (e.g. M_uu = 0 on the linear test model, where vNP vanishes)."""
    return a / b if b != 0 else float("nan")


def d_problem(L, prob, D):
    """Section 6.5: the re-parametrised problem F_D(phi) = F(D^(1/2) phi). Returns (probD, sq = D^(1/2))."""
    N, P = prob.N, prob.P
    sq = torch.sqrt(D)
    probD = L.Problem(F=lambda ph: prob.F(sq * ph), J=lambda ph: prob.J(sq * ph) * sq, N=N, P=P, C=prob.C,
                      fwd=lambda ph: prob.fwd(sq * ph), bwd=lambda cache, r: sq * prob.vjp(cache, r), name="D")
    return probD, sq


# ---------------------------------------------------------------------------------------------------------------
# Section 5.1: learner phase of one first-order leg
# ---------------------------------------------------------------------------------------------------------------
def fo_leg(L, prob, th_start, T, Ld, lr, state, tau, cap, D=None, look=None, hover=0, brk="", hover_snaps=(),
           probe_step=None):
    """Runs the step rule of section 4 until the stopping test of section 5.1 says stop.
    state: Adam-family state entering this leg (None = zero state; ignored by kinds E and F).
    look:  None, or dict(C=N x N accumulator, F0=st.F0, K0=st.K0) of the polygon (kind "E" only; section 5.4).
    hover_snaps: hover step counts at which a clone of th is kept (section 5.1).
    probe_step:  if set, rn at the loop iteration with kleg == probe_step is returned as rn_probe (section 6.3 r50)."""
    kind = Ld["kind"]
    assert kind in ("E", "A", "F"), kind
    assert look is None or kind == "E", "the look matrix is defined for kind E (and X) only"
    N = prob.N
    gamma = Ld.get("gamma", 0.0) if kind in ("E", "A") else 0.0
    mid = kind == "E" and Ld["mid"]
    mu = Ld["mu"] if kind == "E" else 0.0
    if kind == "F":
        assert D is not None
    th = th_start.clone()
    u = torch.zeros_like(th)
    if kind == "A":
        if state is None:
            m, v, km, kv = torch.zeros_like(th), torch.zeros_like(th), 0, 0
        else:
            m, v, km, kv = state["m"], state["v"], state["km"], state["kv"]
        b1, b2 = Ld["b1"], Ld["b2"]
    elif kind == "F":
        m, km = torch.zeros_like(th), 0                # reset at the start of every leg
        b1 = Ld["b1"]
    if look is not None:
        A1 = torch.zeros(N, N, dtype=th.dtype, device=th.device)
        A2 = torch.zeros(N, N, dtype=th.dtype, device=th.device)
    kleg = 0
    hovering = False
    left = 0
    status = None
    th_stop, steps_stop, res_stop = None, None, None
    Qh, res_sum, res_max = 0.0, 0.0, float("nan")
    snaps, snap_records = {}, {}
    rn_probe = None
    Fprev = None
    Fk, cache_k = prob.fwd(th)                         # first forward pass, reused for s_leg
    s_leg = L.vnorm(T - Fk)
    first = True
    while True:
        if not first:
            Fk, cache_k = prob.fwd(th)
        first = False
        rn = L.vnorm(Fk - T)
        if probe_step is not None and kleg == probe_step:
            rn_probe = rn
        if not math.isfinite(rn) or rn > DIVERGE * s_leg:
            status = "DIVERGED"
            break
        if not hovering:
            if rn <= tau * s_leg and (kleg == 0 or L.vnorm(u) <= tau * L.vnorm(th - th_start)):
                status = "CONVERGED"
                if hover == 0:
                    break
                hovering = True
                left = hover
                th_stop = th.clone()
                steps_stop = kleg
                res_stop = rn / s_leg
            elif kleg >= cap:
                status = "UNCONVERGED"
                break
        if hovering:
            n_h = hover - left                         # hover steps already taken at the current th
            if n_h > 0:
                # spec note: the hover statistics run over the positions reached by the hover steps (n_h = 1..H);
                #       th_stop itself (n_h = 0) is not counted in res_max / res_mean.
                Qh += L.vnorm(Fk - Fprev) ** 2 / s_leg ** 2
                res_sum += rn / s_leg
                if n_h == 1 or rn / s_leg > res_max:
                    res_max = rn / s_leg
            if n_h in hover_snaps:
                snaps[n_h] = th.clone()
                snap_records[n_h] = dict(Qh=Qh, res_max=res_max,
                                         res_mean=res_sum / n_h if n_h > 0 else float("nan"))
            if left == 0:
                break
            left -= 1
        Fprev = Fk
        # look point (BREAK "stalelook": th - gamma*u)
        if gamma != 0:
            y = th - gamma * u if brk == "stalelook" else th + gamma * u
            Fy, cache_y = prob.fwd(y)
        else:
            Fy, cache_y = Fk, cache_k                  # reuse the forward pass when y is th
        g = prob.vjp(cache_y, Fy - T) / N
        if kind == "E":
            if mid:
                p = -lr * g
                Fh, cache_h = prob.fwd(th + 0.5 * p)
                u_new = -lr * (prob.vjp(cache_h, Fh - T) / N)
                if look is not None:
                    o = torch.outer(Fh - T, Fh - look["F0"])
                    A1 += o                            # A2 stays 0
            else:
                u_new = mu * u - lr * g
                if look is not None:
                    o = torch.outer(Fy - T, Fy - look["F0"])
                    A1 += o
                    A2 = mu * (A2 + o)
        elif kind == "A":
            km += 1
            kv += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            u_new = -lr * (m / (1 - b1 ** km)) / (torch.sqrt(v / (1 - b2 ** kv)) + EPS)
        else:                                          # kind "F": frozen metric D
            km += 1
            m = b1 * m + (1 - b1) * g
            u_new = -lr * (m / (1 - b1 ** km)) * D
        th = th + u_new
        u = u_new
        kleg += 1
    if look is not None:
        look["C"] += -(lr / N) / (1 - mu) * (look["K0"] @ (A1 - A2))
    out = dict(theta=th, status=status, steps=kleg, res=rn / s_leg, s_leg=s_leg, rn_probe=rn_probe)
    if kind == "A":
        out["state"] = dict(m=m, v=v, km=km, kv=kv)
    elif kind == "F":
        out["state"] = dict(m=m, km=km)
    else:
        out["state"] = None
    if hover > 0:
        out.update(theta_stop=th_stop, steps_stop=steps_stop, res_stop=res_stop, Qh=Qh, res_max=res_max,
                   res_mean=(res_sum / (hover - left) if hovering and hover - left > 0 else float("nan")),
                   hover_snaps=snaps, hover_records=snap_records)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Section 5.2: closing (explicit midpoint; also the EM learner)
# ---------------------------------------------------------------------------------------------------------------
def em_close(L, prob, th, T):
    F = prob.F(th)
    J = prob.J(th)
    h = L.jump(prob, th, F + 0.5 * (T - F), Jb=J)                 # half-way point, Jacobian at th
    del J                                                          # (memory only: at most three Jacobians alive)
    Jh = prob.J(h["theta"])
    r = L.jump_a(prob, th, T, Jb=Jh)                               # the whole move from th with the half-way Jacobian
    del Jh
    status = "CONVERGED" if h["status"] == r["status"] == "CONVERGED" else "CLOSE-FAILED"
    return dict(theta=r["theta"], status=status, pay=T - F, look=F + 0.5 * (T - F))


STOP_STATUSES = ("DIVERGED", "CLOSE-FAILED", "NOT-CLOSED", "NONFINITE")


def _closed_status(L, prob, th_c, T, s_leg, close_status, check_close):
    """Status of a leg from its closed weights. Returns (status or None if fine, close)."""
    if close_status == "CLOSE-FAILED":
        close = L.vnorm(prob.F(th_c) - T) / s_leg if finite_t(th_c) else float("nan")
        return "CLOSE-FAILED", close
    if not finite_t(th_c):
        return "NONFINITE", float("nan")
    close = L.vnorm(prob.F(th_c) - T) / s_leg
    if not math.isfinite(close):
        return "NONFINITE", close
    if check_close and not (close <= CLOSE_TOL):
        return "NOT-CLOSED", close
    return None, close


def run_leg(L, prob, Ld, lr, th, T, tau, cap, state, D, look, brk):
    """One leg of a polygon from the closed weights th to the target T. Returns (theta_closed or None, leg record,
    state returned by the learner phase)."""
    kind = Ld["kind"]
    if kind == "X":
        F = prob.F(th)
        s_leg = L.vnorm(T - F)
        if Ld["name"] == "SL1":
            j = L.jump(prob, th, T)                                 # Jacobian at the start th of the leg
            th_c = j["theta"]
            # spec note: a failed exact jump is recorded as "CLOSE-FAILED" (the spec names no status for it)
            cs = "CONVERGED" if j["status"] == "CONVERGED" else "CLOSE-FAILED"
            if look is not None:
                look["C"] += torch.outer(T - F, F - look["F0"])
        else:
            c = em_close(L, prob, th, T)
            th_c = c["theta"]
            cs = c["status"]
            if look is not None:
                look["C"] += torch.outer(c["pay"], c["look"] - look["F0"])
        bad, close = _closed_status(L, prob, th_c, T, s_leg, cs, True)
        rec = dict(status=bad or "OK", phase_status="EXACT", steps=0, res=0.0, s_leg=s_leg, close_status=cs,
                   close_w=0.0, close=close)
        return (th_c if bad is None else None), rec, None
    r = fo_leg(L, prob, th, T, Ld, lr, state, tau, cap, D=D, look=look, brk=brk)
    rec = dict(phase_status=r["status"], steps=r["steps"], res=r["res"], s_leg=r["s_leg"])
    if r["state"] is not None and "kv" in r["state"]:
        rec.update(km=r["state"]["km"], kv=r["state"]["kv"])
    if r["status"] == "DIVERGED":
        rec.update(status="DIVERGED", close_status="NOT-RUN", close_w=float("nan"), close=float("nan"))
        return None, rec, r["state"]
    th_raw = r["theta"]
    if brk == "noclose":
        th_c = th_raw                                               # BREAK: closing skipped
        cs = "SKIPPED-BREAK-noclose"
        close_w = 0.0
    else:
        c = em_close(L, prob, th_raw, T)
        th_c = c["theta"]
        cs = c["status"]
        if look is not None:
            look["C"] += torch.outer(c["pay"], c["look"] - look["F0"])
        close_w = L.vnorm(th_c - th_raw) / L.vnorm(th_c - th) if finite_t(th_c) else float("nan")
    bad, close = _closed_status(L, prob, th_c, T, r["s_leg"], cs, brk != "noclose")
    # spec note: a leg whose learner phase is UNCONVERGED and whose closing ALSO fails gets the closing reason (the
    #       polygon stops there); the spec's "first failing reason" order would label it UNCONVERGED.
    if bad is not None:
        status = bad
    elif r["status"] != "CONVERGED":
        status = r["status"]                                        # UNCONVERGED: closed, the polygon continues
    else:
        status = "OK"
    rec.update(status=status, close_status=cs, close_w=close_w, close=close)
    return (th_c if bad is None else None), rec, r["state"]


# ---------------------------------------------------------------------------------------------------------------
# Section 5.3: polygon
# ---------------------------------------------------------------------------------------------------------------
def polygon(L, prob, st, Ld, lr, th_start, targets, tau, cap, state=None, D=None, look=None, brk=""):
    """Closed path of legs. targets are built by the caller from the ORIGINAL st.F0, the last one being st.F0 itself.
    state: Adam-family state entering leg 1 (None = zero); before each later leg the keep policy is applied to the
    state returned by the previous leg. look: None or dict(C, F0=st.F0, K0=st.K0) (section 5.4).
    Returns dict(legs, th1, thE, state, status, complete)."""
    legs = []
    th = th_start
    th1, thE = None, None
    st_in = state
    st_out = None
    complete = True
    for i, T in enumerate(targets):
        if i > 0:
            st_in = keep_state(Ld, st_out)
        th_c, rec, st_out = run_leg(L, prob, Ld, lr, th, T, tau, cap, st_in, D, look, brk)
        rec["leg"] = i + 1
        legs.append(rec)
        if th_c is None:                                            # DIVERGED, CLOSE-FAILED, NOT-CLOSED, NONFINITE
            complete = False
            break
        th = th_c
        if i == 0:
            th1 = th_c
        if i == len(targets) - 1:
            thE = th_c
    bad = [g["status"] for g in legs if g["status"] != "OK"]
    return dict(legs=legs, th1=th1, thE=thE, state=st_out, status=bad[0] if bad else "OK", complete=complete)


# ---------------------------------------------------------------------------------------------------------------
# Section 6.1: reference unit
# ---------------------------------------------------------------------------------------------------------------
def ref_name(c, x, size):
    # spec note: one reference unit holds both signs (section 6.1 defines R(combination, direction, size)); name R__c__x__size
    return f"R__{c}__{x}__{size}"


def reference_unit(L, combo, xname, size):
    prob, st = combo["prob"], combo["st"]
    x = direction(combo, xname)
    s = combo["sizes"][size]
    QV = probes(combo, x)
    rec = dict(part="R", kind="ref", combo=combo["name"], dir=xname, size=size, s=s)
    vecs = {}
    for sigma in SIGNS:
        t = st.F0 + sigma * s * x
        r, vNP = S.reference(L, prob, st, x, s, sigma)
        usable = S.ref_usable(r)
        jr = L.jump(prob, st.th0, t, Jb=st.J0)
        thJ = jr["theta"]
        dJ = L.vnorm(thJ - st.th0)
        qJ = S.q(L, prob, thJ, QV)
        rec[f"sig{sigma:+d}"] = dict(ref=r, usable=bool(usable), thJ_status=jr["status"], dJ=dJ,
                                     qJ=dict(zip(QNAMES, qJ)))
        vecs["vNPp" if sigma == 1 else "vNPm"] = vNP
    rec["q0"] = dict(zip(QNAMES, S.q_from_kernel(L, st.K0, QV)))
    rec["ksum0"] = S.ksum(st.K0)
    ve = S.even(vecs["vNPp"], vecs["vNPm"])
    rec["nvNPe"] = L.vnorm(ve)
    rec["Muu"] = 2.0 * rec["nvNPe"] ** 2 / s ** 2
    rec["agree"] = L.cosine(vecs["vNPp"], vecs["vNPm"])
    rec["status"] = "OK" if rec["sig+1"]["usable"] and rec["sig-1"]["usable"] else "UNUSABLE"
    return rec, vecs


def load_ref(L, out, name):
    """dict(rec, vNP={1:, -1:}) of a stored reference unit, or None if it is missing or has no vectors."""
    rec = S.read_unit(out, name)
    if rec is None or rec.get("status") == "ERROR":
        return None
    files = rec.get("vec_files", [])
    if not (name + "__vNPp.pt" in files and name + "__vNPm.pt" in files):
        return None
    return dict(rec=rec, vNP={1: S.load_vec(L, out, name, "vNPp"), -1: S.load_vec(L, out, name, "vNPm")})


def ref_in_memory(rec, vecs):
    """The same structure as load_ref, from the return value of reference_unit (tests)."""
    return dict(rec=rec, vNP={1: vecs["vNPp"], -1: vecs["vNPm"]})


# ---------------------------------------------------------------------------------------------------------------
# Kind "F": the frozen metric D from the Adam twin (section 4)
# ---------------------------------------------------------------------------------------------------------------
def frozen_D(L, prob, st, kappa, lr, T, tau, cap):
    """Learner phase (no closing) of ADAM(kappa, B1, B2, 0, none) from theta0 toward T; D from its final v, kv.
    Returns (D or None, info)."""
    r = fo_leg(L, prob, st.th0, T, ADAM(kappa), lr, None, tau, cap)
    info = dict(status=r["status"], steps=r["steps"], res=r["res"])
    if r["status"] != "CONVERGED":
        return None, info
    v, kv = r["state"]["v"], r["state"]["kv"]
    D = 1.0 / (torch.sqrt(v / (1 - B2 ** kv)) + EPS)
    return D, info


def d_summary(D):
    sub = D[::7]
    qs = [float(torch.quantile(sub, q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)]
    return dict(min=qs[0], q01=qs[1], median=qs[2], q99=qs[3], max=qs[4],
                share_cap=float((D > 0.5 / EPS).to(D.dtype).mean()))


# ---------------------------------------------------------------------------------------------------------------
# Section 6.2: pair unit
# ---------------------------------------------------------------------------------------------------------------
def _even(a, b):
    return 0.5 * (a + b)


def _odd(a, b):
    return 0.5 * (a - b)


def pair_unit(L, combo, Ld, dirs, size, tau, cap, ref, brk="", want_law=False, want_uCu=False, meta=None):
    """One learner, one closed polygon (excursion: dirs = [u]; triangle: dirs = [u, w]), both signs.
    ref = load_ref(...) of (dirs[0], size). Returns (record, {}) - no vectors are stored."""
    prob, st = combo["prob"], combo["st"]
    N, P = prob.N, prob.P
    s = combo["sizes"][size]
    X = [direction(combo, nm) for nm in dirs]
    u = X[0]
    QV = probes(combo, u)
    rr = ref["rec"]
    dJ = rr["sig+1"]["dJ"]
    lr = learner_lr(Ld, st, N, P, dJ)
    exc = len(dirs) == 1
    rec = dict(meta or {})
    # spec note: a pair unit also runs when its reference is not usable; the reference status is stored as ref_status
    #       (V6 reports unusable references) and the unit status is not changed by it
    rec.update(kind="pair", combo=combo["name"], learner=learner_name(Ld), Ld=Ld, dirs=list(dirs),
               dirs_name="+".join(dirs), size=size, s=s, tau=tau, cap=cap, lr=lr, dJ=dJ, nvNPe=rr["nvNPe"],
               Muu=rr["Muu"], ref_status=rr["status"], excursion=exc, brk=brk)
    D = None
    if Ld["kind"] == "F":
        D, twin = frozen_D(L, prob, st, Ld["kappa"], lr, st.F0 + 1 * s * u, tau, cap)
        rec["twin"] = twin
        if D is None:
            rec["status"] = "NO-TWIN"
            return rec, {}
        rec["D_summary"] = d_summary(D)
    acc = want_law or want_uCu
    vec = {}
    Cs = {}
    for sigma in SIGNS:
        targets = [st.F0 + sigma * s * x for x in X] + [st.F0]       # F0 is st.F0 itself
        look = None
        if acc:
            look = dict(C=torch.zeros(N, N, dtype=st.F0.dtype, device=st.F0.device), F0=st.F0, K0=st.K0)
        pg = polygon(L, prob, st, Ld, lr, st.th0, targets, tau, cap, state=None, D=D, look=look, brk=brk)
        sr = dict(status=pg["status"], complete=pg["complete"], legs=pg["legs"])
        vv = {}
        th1, thE = pg["th1"], pg["thE"]
        if finite_t(th1):
            sr["n1"] = L.vnorm(th1 - st.th0)
            vv["v1"] = st.V0(th1 - st.th0)
            sr["q1"] = dict(zip(QNAMES, S.q(L, prob, th1, QV)))
        if finite_t(thE):
            d = thE - st.th0
            v = st.V0(d)
            nd = L.vnorm(d)
            vv["v"] = v
            sr.update(nd=nd, nv=L.vnorm(v), horiz=L.vnorm(d - v) / max(nd, 1e-300),
                      close=L.vnorm(prob.F(thE) - st.F0) / s)
            KE = S.kernel(L, prob, thE)
            sr["qE"] = dict(zip(QNAMES, S.q_from_kernel(L, KE, QV)))
            sr["ksumE"] = S.ksum(KE)
            del KE
        vec[sigma] = vv
        if acc and pg["complete"]:
            Cs[sigma] = look["C"]
        rec[f"sig{sigma:+d}"] = sr
        L.free_memory()
    bad = [rec[f"sig{sg:+d}"]["status"] for sg in SIGNS if rec[f"sig{sg:+d}"]["status"] != "OK"]
    rec["status"] = bad[0] if bad else "OK"
    # per-unit maxima over legs (report tables)
    legs = rec["sig+1"]["legs"] + rec["sig-1"]["legs"]
    rec["steps_max"] = max(g["steps"] for g in legs)
    rec["res_max"] = max(g["res"] for g in legs)
    cw = [g["close_w"] for g in legs if g["phase_status"] != "EXACT"]
    rec["close_w_max"] = max(cw) if cw else 0.0
    rec["closings_ok"] = bool(all(rec[f"sig{sg:+d}"]["complete"] for sg in SIGNS)
                              and all(g["status"] in ("OK", "UNCONVERGED") for g in legs))
    if not all("v" in vec[sg] and "v1" in vec[sg] for sg in SIGNS):
        return rec, {}
    # combined over the two signs
    sp, sm = rec["sig+1"], rec["sig-1"]
    q0 = rr["q0"]
    ksum0 = rr["ksum0"]
    ve, vo = _even(vec[1]["v"], vec[-1]["v"]), _odd(vec[1]["v"], vec[-1]["v"])
    v1e, v1o = _even(vec[1]["v1"], vec[-1]["v1"]), _odd(vec[1]["v1"], vec[-1]["v1"])
    rec.update(nve=L.vnorm(ve), nvo=L.vnorm(vo), nv1e=L.vnorm(v1e), nv1o=L.vnorm(v1o))
    rec["close_max"] = max(sp["close"], sm["close"])
    rec["horiz_max"] = max(sp["horiz"], sm["horiz"])
    dq = {sg: {x: rec[f"sig{sg:+d}"]["qE"][x] - q0[x] for x in QNAMES} for sg in SIGNS}
    rec["dqe"] = {x: _even(dq[1][x], dq[-1][x]) for x in QNAMES}
    rec["dqo"] = {x: _odd(dq[1][x], dq[-1][x]) for x in QNAMES}
    G = {sg: sum(q0[x] / rec[f"sig{sg:+d}"]["qE"][x] for x in W_NAMES) / len(W_NAMES) for sg in SIGNS}
    rec["G"] = {"+1": G[1], "-1": G[-1]}
    rec["dGe"] = _even(G[1] - 1.0, G[-1] - 1.0)
    rec["dGo"] = _odd(G[1] - 1.0, G[-1] - 1.0)
    h = {sg: {x: rec[f"sig{sg:+d}"]["q1"][x] - rr[f"sig{sg:+d}"]["qJ"][x] for x in QNAMES} for sg in SIGNS}
    rec["he"] = {x: _even(h[1][x], h[-1][x]) for x in QNAMES}
    rec["ho"] = {x: _odd(h[1][x], h[-1][x]) for x in QNAMES}
    rec["Ho"] = sum(abs(rec["ho"][x]) / q0[x] for x in W_NAMES) / len(W_NAMES)
    for key, nm in (("lamN", "lamN_e"), ("trK", "trK_e"), ("inv_mean", "inv_e")):
        rec[nm] = _even(sp["ksumE"][key] / ksum0[key] - 1.0, sm["ksumE"][key] / ksum0[key] - 1.0)
    if exc:
        vNP = ref["vNP"]
        es = L.even_stats(vec[1]["v"], vNP[1], vec[-1]["v"], vNP[-1])
        es1 = L.even_stats(vec[1]["v1"], vNP[1], vec[-1]["v1"], vNP[-1])
        rec.update(rho=es["rho"], cos=es["cos"], nr=es["nr"], rho1=es1["rho"], cos1=es1["cos"], nr1=es1["nr"])
        rec["a_u"] = div(rec["dqe"]["u"], rr["Muu"])
        rec["phi_u"] = div(-rec["he"]["u"], rr["Muu"])
    if acc and len(Cs) == 2:
        Ce = 0.5 * (Cs[1] + Cs[-1])
        rec["uCu"] = float(u @ Ce @ u) / s ** 2
        if want_law:
            Vlaw, info = L.law(prob, st, Ce)
            nl = L.vnorm(Vlaw)
            rec.update(law_ratio=float(ve @ Vlaw) / float(Vlaw @ Vlaw) if nl > 0 else float("nan"),
                       law_cos=L.cosine(ve, Vlaw), n_law=nl, kept=info["kept"], kept_frac=info["kept_frac"],
                       n_sym=L.vnorm(0.5 * (Ce + Ce.T)), n_asym=L.vnorm(0.5 * (Ce - Ce.T)))
            del Vlaw
            # NOTE (review, A1.6): spectral area. Predicted antisymmetric part of the loop look matrix of a
            # gradient-flow-like learner, in the eigenbasis of K0: -s^2 a_i a_j (lam_i - lam_j) / (lam_i + lam_j).
            a_ = st.Q.T @ u
            lam = st.lam
            Tm = (lam[:, None] - lam[None, :]) / (lam[:, None] + lam[None, :])
            Apred = st.Q @ (-(s ** 2) * a_[:, None] * a_[None, :] * Tm) @ st.Q.T
            Casym = 0.5 * (Ce - Ce.T)
            num_ = float((Casym * Apred).sum())
            nA, nC = L.vnorm(Apred), L.vnorm(Casym)
            rec.update(n_asym_pred=nA, area_ratio=div(num_, nA * nA), area_cos=div(num_, nA * nC))
    return rec, {}


def mode_prediction(Ld, lam_ratio):
    """Part M prediction: c_i = c * lam_ratio; f_pred = f_scalar(c_i, mu, gamma, mid); g_pred = 2 f_pred - 1."""
    c_i = Ld["c"] * lam_ratio
    f, steps = f_scalar(c_i, Ld["mu"], Ld["gamma"], Ld["mid"])
    return dict(lam_ratio=lam_ratio, c_i=c_i, f_pred=f, g_pred=2.0 * f - 1.0, f_steps=steps)


def mode_unit(L, combo, Ld, dname, ref, brk="", meta=None):
    """Part M unit: prediction first; a pair whose f_pred is nan is not run (status SKIPPED-UNSTABLE)."""
    pred = mode_prediction(Ld, float(combo["eig"][dname][1]))
    if not math.isfinite(pred["f_pred"]):
        rec = dict(meta or {})
        rec.update(kind="pair", combo=combo["name"], learner=learner_name(Ld), Ld=Ld, dirs=[dname], dirs_name=dname,
                   size="s1", status="SKIPPED-UNSTABLE", **pred)
        return rec, {}
    rec, v = pair_unit(L, combo, Ld, [dname], "s1", TAU_MODE, CAP_MODE, ref, brk=brk, want_uCu=True, meta=meta)
    rec.update(pred)
    return rec, v


# ---------------------------------------------------------------------------------------------------------------
# Section 6.3: star unit and operational reading
# ---------------------------------------------------------------------------------------------------------------
def op_reading(L, prob, th, lr, s1, xs):
    """Learner phase (no closing) of ADAM(0.1, B1, B2, 0, none), zero state, from th toward F(th) + s1 x."""
    F = prob.F(th)
    out = {}
    for name, x in xs:
        T = F + s1 * x
        r = fo_leg(L, prob, th, T, ADAM(0.1), lr, None, TAU_GEN, CAP_ADAM, probe_step=OP_PROBE_STEP)
        out[name] = dict(status=r["status"], steps=r["steps"], res=r["res"],
                         r50=(r["rn_probe"] / s1) if r["rn_probe"] is not None else float("nan"))
    return out


def star_unit(L, combo, Ld, ref_E1, n_mm, brk="", meta=None):
    prob, st = combo["prob"], combo["st"]
    N, P = prob.N, prob.P
    s1 = combo["sizes"]["s1"]
    dJ = ref_E1["rec"]["sig+1"]["dJ"]
    lr = learner_lr(Ld, st, N, P, dJ)
    lr_op = learner_lr(ADAM(0.1), st, N, P, dJ)
    tau, cap = TAU_GEN, CAP_ADAM
    wv = [combo["dirs"][w] for w in W_NAMES]
    q0 = S.q_from_kernel(L, st.K0, wv)
    rec = dict(meta or {})
    rec.update(kind="star", combo=combo["name"], learner=learner_name(Ld), Ld=Ld, s1=s1, lr=lr, lr_op=lr_op, dJ=dJ,
               n_mm=n_mm, tau=tau, cap=cap, q0=dict(zip(W_NAMES, q0)), ksum0=S.ksum(st.K0), checkpoints=[])
    th = st.th0
    state = None
    first = True
    status = "OK"
    for mm in range(1, n_mm + 1):
        E = combo["dirs"][f"E{mm}"]
        legs_mm = []
        for sigma in SIGNS:
            st_in = None if first else keep_state(Ld, state)
            first = False
            # spec note: only the learner runs (polygons, operational reading) are guarded; an exception in a kernel / q
            #       computation propagates to S.run_unit (status ERROR, retried on the next start)
            try:
                pg = polygon(L, prob, st, Ld, lr, th, [st.F0 + sigma * s1 * E, st.F0], tau, cap, state=st_in,
                             brk=brk)
            except Exception as e:
                status = "EXC:" + type(e).__name__
                rec["error"] = repr(e)
                break
            L.free_memory()
            legs_mm.append(dict(sign=sigma, status=pg["status"], legs=pg["legs"]))
            if pg["status"] != "OK":
                status = pg["status"]
                break
            th = pg["thE"]
            state = pg["state"]
        if status != "OK":
            rec["failed_mm"] = dict(mm=mm, polygons=legs_mm)
            break
        K = S.kernel(L, prob, th)
        q = S.q_from_kernel(L, K, wv)
        rec["checkpoints"].append(dict(
            mm=mm, ksum=S.ksum(K), q=dict(zip(W_NAMES, q)),
            G=sum(q0[i] / q[i] for i in range(len(W_NAMES))) / len(W_NAMES),
            nvc=L.vnorm(st.V0(th - st.th0)), nd=L.vnorm(th - st.th0), close=L.vnorm(prob.F(th) - st.F0) / s1,
            polygons=legs_mm))
        del K
    rec["status"] = status
    rec["mm_done"] = len(rec["checkpoints"])
    # spec note: "a star with all 6 checkpoints" read as "all n_mm planned checkpoints" (QUICK plans 2)
    if status == "OK" and rec["mm_done"] == n_mm:
        try:
            rec["op"] = op_reading(L, prob, th, lr_op, s1, [("W1", combo["dirs"]["W1"]), ("W5", combo["dirs"]["W5"])])
        except Exception as e:
            rec["status"] = "EXC:" + type(e).__name__
            rec["error"] = repr(e)
    return rec, {}


def op0_unit(L, combo, ref_E1, meta=None):
    prob, st = combo["prob"], combo["st"]
    s1 = combo["sizes"]["s1"]
    dJ = ref_E1["rec"]["sig+1"]["dJ"]
    lr_op = learner_lr(ADAM(0.1), st, prob.N, prob.P, dJ)
    rec = dict(meta or {})
    rec.update(kind="op0", combo=combo["name"], s1=s1, lr_op=lr_op, dJ=dJ)
    try:
        rec["op"] = op_reading(L, prob, st.th0, lr_op, s1, [("W1", combo["dirs"]["W1"]), ("W5", combo["dirs"]["W5"])])
        rec["status"] = "OK"
    except Exception as e:
        rec["status"] = "EXC:" + type(e).__name__
        rec["error"] = repr(e)
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 6.4: hover unit
# ---------------------------------------------------------------------------------------------------------------
def hover_unit(L, combo, Ld, ref, hover, snaps, brk="", meta=None):
    prob, st = combo["prob"], combo["st"]
    N, P = prob.N, prob.P
    s1 = combo["sizes"]["s1"]
    u = combo["dirs"]["E1"]
    QV = probes(combo, u)
    rr = ref["rec"]
    q0 = rr["q0"]
    Muu = rr["Muu"]
    lr = learner_lr(Ld, st, N, P, rr["sig+1"]["dJ"])
    tau = TAU_GEN
    cap = CAP_GEN if Ld["kind"] == "E" else CAP_ADAM
    snaps = list(snaps)
    rec = dict(meta or {})
    rec.update(kind="hover", combo=combo["name"], learner=learner_name(Ld), Ld=Ld, s1=s1, lr=lr, tau=tau, cap=cap,
               hover=hover, snaps=snaps, Muu=Muu, nvNPe=rr["nvNPe"], brk=brk)
    statuses = []
    # spec note: a sign whose learner phase is not CONVERGED does not stop the other sign (it is still run and stored);
    #       per-snapshot records are stored as lists in the order of `snaps`
    for sigma in SIGNS:
        T = st.F0 + sigma * s1 * u
        sr = {}
        rec[f"sig{sigma:+d}"] = sr
        try:
            r = fo_leg(L, prob, st.th0, T, Ld, lr, None, tau, cap, hover=hover, brk=brk, hover_snaps=snaps)
        except Exception as e:
            sr["status"] = "EXC:" + type(e).__name__
            sr["error"] = repr(e)
            statuses.append(sr["status"])
            break
        sr.update(phase_status=r["status"], steps=r["steps"], res=r["res"])
        if r["status"] != "CONVERGED":
            sr["status"] = r["status"]
            statuses.append(sr["status"])
            continue
        sr.update(steps_stop=r["steps_stop"], res_stop=r["res_stop"], Qh=r["Qh"], res_max=r["res_max"],
                  res_mean=r["res_mean"], records=[r["hover_records"].get(n) for n in snaps])
        s_leg = r["s_leg"]
        # spec note: BREAK "noclose" also skips these closings (they close a learner phase like every leg closing)
        pts = [("A", r["theta_stop"])] + [(n, r["hover_snaps"][n]) for n in snaps]
        closed = {}
        cstat = []
        try:
            for key, thr in pts:
                if brk == "noclose":
                    th_c, cs = thr, "SKIPPED-BREAK-noclose"
                else:
                    c = em_close(L, prob, thr, T)
                    th_c, cs = c["theta"], c["status"]
                bad, close = _closed_status(L, prob, th_c, T, s_leg, cs, brk != "noclose")
                cstat.append(dict(point=str(key), close_status=cs, close=close, status=bad or "OK"))
                closed[key] = th_c
        except Exception as e:
            sr["status"] = "EXC:" + type(e).__name__
            sr["error"] = repr(e)
            statuses.append(sr["status"])
            break
        sr["closings"] = cstat
        badc = [c["status"] for c in cstat if c["status"] != "OK"]
        if badc:
            sr["status"] = badc[0]
            statuses.append(sr["status"])
            continue
        qA = dict(zip(QNAMES, S.q(L, prob, closed["A"], QV)))
        GA = sum(q0[x] / qA[x] for x in W_NAMES) / len(W_NAMES)
        sr.update(qA=qA, GA=GA, qB=[], GB=[], w=[], wu=[], nvh=[])
        for n in snaps:
            qB = dict(zip(QNAMES, S.q(L, prob, closed[n], QV)))
            GB = sum(q0[x] / qB[x] for x in W_NAMES) / len(W_NAMES)
            sr["qB"].append(qB)
            sr["GB"].append(GB)
            sr["w"].append(GB / GA - 1.0)
            sr["wu"].append(div(qB["u"] - qA["u"], Muu))
            sr["nvh"].append(L.vnorm(st.V0(closed[n] - closed["A"])))
        sr["status"] = "OK"
        statuses.append("OK")
        L.free_memory()
    bad = [x for x in statuses if x != "OK"]
    rec["status"] = bad[0] if bad else ("OK" if len(statuses) == 2 else "INCOMPLETE")
    if rec["status"] == "OK":
        sp, sm = rec["sig+1"], rec["sig-1"]
        rec["we"] = [_even(a, b) for a, b in zip(sp["w"], sm["w"])]
        rec["wue"] = [_even(a, b) for a, b in zip(sp["wu"], sm["wu"])]
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Section 6.5: own-metric unit
# ---------------------------------------------------------------------------------------------------------------
def own_unit(L, combo, kappa, ref, brk="", meta=None):
    prob, st = combo["prob"], combo["st"]
    N, P = prob.N, prob.P
    s1 = combo["sizes"]["s1"]
    u = combo["dirs"]["E1"]
    Ld = ADAM(kappa)
    lr = learner_lr(Ld, st, N, P, ref["rec"]["sig+1"]["dJ"])
    tau, cap = TAU_GEN, CAP_ADAM
    rec = dict(meta or {})
    rec.update(kind="own", combo=combo["name"], kappa=kappa, learner=learner_name(Ld), Ld=Ld, s1=s1, lr=lr)
    try:
        D, twin = frozen_D(L, prob, st, kappa, lr, st.F0 + 1 * s1 * u, tau, cap)
    except Exception as e:
        rec.update(status="EXC:" + type(e).__name__, error=repr(e))
        return rec, {}
    rec["twin"] = twin
    if D is None:
        rec["status"] = "NO-TWIN"
        return rec, {}
    rec["D_summary"] = d_summary(D)
    probD, sq = d_problem(L, prob, D)
    try:
        stD = L.Start(probD, st.th0 / sq)
    except Exception as e:
        rec.update(status="D-FAILED:Start-" + type(e).__name__, error=repr(e))
        return rec, {}
    rec.update(lamD1=stD.lam1, lamDN=stD.lamN)
    if stD.ill:
        rec["status"] = "D-FAILED:ill-conditioned"
        return rec, {}
    vD_ref = {}
    for sigma in SIGNS:
        try:
            recD, vD = S.reference(L, probD, stD, u, s1, sigma)
        except Exception as e:
            rec.update(status="D-FAILED:reference-" + type(e).__name__, error=repr(e))
            return rec, {}
        rec[f"refD_sig{sigma:+d}"] = recD
        if not S.ref_usable(recD):
            rec["status"] = "D-FAILED:reference-unusable"
            return rec, {}
        vD_ref[sigma] = vD
    # control (covariance): exact-jump loop on probD
    vc = {}
    for sigma in SIGNS:
        j1 = L.jump(probD, stD.th0, stD.F0 + sigma * s1 * u, Jb=stD.J0)
        j2 = L.jump(probD, j1["theta"], stD.F0)
        rec[f"ctrl_jumps_sig{sigma:+d}"] = [j1["status"], j2["status"]]
        vc[sigma] = stD.V0(j2["theta"] - stD.th0)
    es = L.even_stats(vc[1], vD_ref[1], vc[-1], vD_ref[-1])
    rec["ctrl"] = dict(rho=es["rho"], cos=es["cos"], nr=es["nr"])
    # the ordinary ADAM(kappa) excursion (theta-space polygon, usual closing)
    vDs, vs = {}, {}
    statuses = []
    for sigma in SIGNS:
        try:
            pg = polygon(L, prob, st, Ld, lr, st.th0, [st.F0 + sigma * s1 * u, st.F0], tau, cap, brk=brk)
        except Exception as e:
            statuses.append("EXC:" + type(e).__name__)
            rec["error"] = repr(e)
            break
        rec[f"sig{sigma:+d}"] = dict(status=pg["status"], legs=pg["legs"])
        statuses.append(pg["status"])
        if finite_t(pg["thE"]):
            vDs[sigma] = stD.V0((pg["thE"] - st.th0) / sq)
            vs[sigma] = st.V0(pg["thE"] - st.th0)
        L.free_memory()
    bad = [x for x in statuses if x != "OK"]
    rec["status"] = bad[0] if bad else "OK"
    if len(vDs) == 2:
        own = L.even_stats(vDs[1], vD_ref[1], vDs[-1], vD_ref[-1])
        eu = L.even_stats(vs[1], ref["vNP"][1], vs[-1], ref["vNP"][-1])
        rec["own"] = dict(rho=own["rho"], cos=own["cos"], nr=own["nr"])
        rec["eucl"] = dict(rho=eu["rho"], cos=eu["cos"], nr=eu["nr"])
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Part V chk: equivalence with the library learners (section 7)
# ---------------------------------------------------------------------------------------------------------------
def chk_unit(L, combo, ref_E1, steps=CHK_STEPS, meta=None):
    prob, st = combo["prob"], combo["st"]
    N, P = prob.N, prob.P
    s1 = combo["sizes"]["s1"]
    T = st.F0 + 1 * s1 * combo["dirs"]["E1"]
    dJ = ref_E1["rec"]["sig+1"]["dJ"]
    rec = dict(meta or {})
    rec.update(kind="chk", combo=combo["name"], steps=steps, rel={}, status="OK")
    cases = [("GD(1)", GD(1)), ("HB(0.02,0.9)", HB(0.02, 0.9)), ("ADAM(0.1,0.9,0.999,0,none)", ADAM(0.1))]
    for nm, Ld in cases:
        lr = learner_lr(Ld, st, N, P, dJ)
        if Ld["kind"] == "E" and Ld["mu"] == 0:
            Lib = L.L_GD(Ld["c"], 0.0, steps)
        elif Ld["kind"] == "E":
            Lib = L.L_HB(Ld["c"], Ld["mu"], 0.0, steps)
        else:
            Lib = L.L_ADAM(lr, 0.0, steps)
        mine = fo_leg(L, prob, st.th0, T, Ld, lr, None, 0.0, steps)
        lib = L.run_first_order(prob, st, T, s1, Lib)
        rec["rel"][nm] = L.vnorm(mine["theta"] - lib["theta"]) / L.vnorm(lib["theta"] - st.th0)
        rec[f"steps_{nm}"] = [mine["steps"], lib["steps"]]
    rec["max_rel"] = max(rec["rel"].values())
    return rec, {}


# ---------------------------------------------------------------------------------------------------------------
# Sections 7, 8: settings (enter the signature) and the plan
# ---------------------------------------------------------------------------------------------------------------
LM = [GD(0.5), GD(1), GDmid(1), HB(0.02, 0.9), HB(0.1, 0.9), HB(0.1, 0.5), HBLA(0.02, 0.9, 0.9),
      HBLA(0.02, 0.9, 4.75), HBLA(0.02, 0.9, 9.5), HBLA(0.02, 0.9, 14.25), HBLA(0.1, 0.9, 9.5), HBLA(0.2, 0, 0.5)]


def settings12(quick, brk):
    """Every constant of section 3, every grid of section 7, the quick flag and the break value."""
    check_break(brk)
    consts = dict(EPS=EPS, B1=B1, B2=B2, GSTAR=GSTAR, CLOSE_TOL=CLOSE_TOL, TAU_MODE=TAU_MODE, TAU_GEN=TAU_GEN,
                  CAP_MODE=CAP_MODE, CAP_GEN=CAP_GEN, CAP_ADAM=CAP_ADAM, DIVERGE=DIVERGE, SIGNS=list(SIGNS),
                  CHK_STEPS=CHK_STEPS, OP_PROBE_STEP=OP_PROBE_STEP)
    A = []

    def a(group, dirn, size, Ld, tau=None, cap=None, suffix=""):
        A.append(dict(group=group, dir=dirn, size=size, Ld=Ld, tau=tau, cap=cap, suffix=suffix))

    if not quick:
        R = [["eigA", "s1"], ["eigB", "s1"], ["eigC", "s1"], ["E1", "s1"], ["E2", "s1"], ["E1", "s2"], ["E2", "s2"]]
        M = [["eigA", Ld] for Ld in LM]
        M += [["eigB", Ld] for Ld in (GD(1), GDmid(1), HB(0.1, 0.9), HBLA(0.1, 0.9, 9.5))]
        M += [["eigC", Ld] for Ld in (HB(0.1, 0.9), HBLA(0.1, 0.9, 9.5))]
        G = [["E1", Ld] for Ld in (SL1(), EM(), GD(1), HB(0.1, 0.9), HBLA(0.1, 0.9, 9.5), HB(0.02, 0.9))]
        G += [["E2", Ld] for Ld in (HB(0.1, 0.9), HBLA(0.1, 0.9, 9.5))]
        for d in ("E1", "E2"):
            for size in ("s1", "s2"):
                for Ld in (SL1(), EM()):
                    a("A0", d, size, Ld)
        for d in ("E1", "E2"):
            for size in ("s1", "s2"):
                for k in (0.03, 0.1, 0.3, 1.0):
                    a("A1", d, size, ADAM(k))
        for size in ("s1", "s2"):                      # NOTE (review, A1.1): k = 0.01 on E1
            a("A1", "E1", size, ADAM(0.01))
        for size in ("s1", "s2"):
            for b1 in (0.0, 0.5, 0.8, 0.95):
                a("A2", "E1", size, ADAM(0.1, b1=b1))
        for b1 in (0.5, 0.8, 0.95):                    # NOTE (review, A1.1): first-moment series also at k = 1.0 (s1)
            a("A2", "E1", "s1", ADAM(1.0, b1=b1))
        for k in (0.1, 1.0):                           # NOTE (review, A1.1): was (0.03, 0.1)
            for b2 in (0.5, 0.9, 0.99):
                a("A3", "E1", "s1", ADAM(k, b2=b2))
        for g in (0.5, 1.0, 2.0, 4.0, 9.5):
            a("A4", "E1", "s1", ADAM(0.1, gamma=g))
        for g in (1.0, 4.0, 9.5):
            a("A4", "E1", "s1", ADAM(0.03, gamma=g))
        for size in ("s1", "s2"):
            for k in (0.03, 0.1):
                a("A5", "E1", size, PRE(k))
                a("A5", "E1", size, PREM(k))
        for d in ("E1", "E2"):
            for size in ("s1", "s2"):
                for keep in ("m", "v", "mv"):
                    a("A6", d, size, ADAM(0.1, keep=keep))
        for size in ("s1", "s2"):
            a("A6", "E1", size, ADAM(0.3, keep="mv"))
        a("A7", "E1", "s1", ADAM(0.1), tau=1e-4, cap=6000, suffix="__tau1e-4")
        T = [[size, Ld] for size in ("s1", "s2") for Ld in (SL1(), EM(), GD(1), ADAM(0.03), ADAM(0.1), ADAM(0.3))]
        H = [GD(1), ADAM(0.03), ADAM(0.1), ADAM(0.3), ADAM(0.1, b2=0.9), ADAM(0.1, b1=0.95),
             ADAM(1.0)]                                # NOTE (review, A1.2): ADAM(1.0) added
        HOVER, HS = 2000, [500, 2000]
        SL = [SL1(), EM(), ADAM(0.03), ADAM(0.1), ADAM(0.3), ADAM(0.1, keep="mv"), ADAM(0.1, b1=0.95),
              ADAM(0.1, b2=0.9), ADAM(1.0)]            # NOTE (review, A1.2): ADAM(1.0) added
        N_MM = 6
        DK = [0.03, 0.1, 0.3]
    else:
        R = [["eigA", "s1"], ["E1", "s1"], ["E2", "s1"]]
        M = [["eigA", Ld] for Ld in (GD(1), HB(0.1, 0.5), HBLA(0.2, 0, 0.5))]
        G = [["E1", Ld] for Ld in (EM(), GD(1))]
        a("A0", "E1", "s1", SL1())
        a("A0", "E1", "s1", EM())
        a("A1", "E1", "s1", ADAM(0.3))
        a("A5", "E1", "s1", PREM(0.3))
        a("A6", "E1", "s1", ADAM(0.3, keep="mv"))
        T = [["s1", Ld] for Ld in (SL1(), ADAM(0.3))]
        H = [ADAM(0.3)]
        HOVER, HS = 300, [100, 300]
        SL = [SL1(), ADAM(0.3)]
        N_MM = 2
        DK = [0.3]
    return dict(CONSTS=consts, R=R, M=M, G=G, A=A, T=T, H=H, HOVER=HOVER, HS=HS, S=SL, N_MM=N_MM, D=DK,
                QUICK=bool(quick), BREAK=brk)


def default_tau_cap(part, Ld):
    if Ld["kind"] == "X":
        return None, None
    if part == "M":
        return TAU_MODE, CAP_MODE
    if Ld["kind"] == "E":
        return TAU_GEN, CAP_GEN
    return TAU_GEN, CAP_ADAM


def pair_name(part, c, dirs, size, Ld, safe_name, suffix=""):
    return f"{part}__{c}__{'+'.join(dirs)}__{size}__{safe_name(learner_name(Ld))}{suffix}"


def plan12(settings, combos, parts, safe_name, repro_combo=REPRO_COMBO):
    """Every unit of the requested parts, in the order of work (section 8): per combination V, R, M, G, A, T, H, S, D.
    safe_name = L.safe_name (passed in so that the plan can be printed without loading the network)."""
    items = []
    brk = settings["BREAK"]

    def pair(part, c, dirs, size, Ld, tau=None, cap=None, group=None, suffix="", want_law=False, want_uCu=False,
             kind="pair"):
        t0, c0 = default_tau_cap(part, Ld)
        return dict(name=pair_name(part, c, dirs, size, Ld, safe_name, suffix), part=part, kind=kind, combo=c,
                    dirs=list(dirs), size=size, Ld=Ld, lname=learner_name(Ld), tau=t0 if tau is None else tau,
                    cap=c0 if cap is None else cap, group=group or part, suffix=suffix, want_law=want_law,
                    want_uCu=want_uCu, needs=[[dirs[0], size]])

    for c in combos:
        # spec note: the QUICK list of section 8 does not mention Part V: V (env, chk) is planned whenever "V" is in --parts
        if "V" in parts:
            items.append(dict(name=f"V__{c}__env", part="V", kind="env", combo=c, needs=[]))
            items.append(dict(name=f"V__{c}__chk", part="V", kind="chk", combo=c, needs=[["E1", "s1"]]))
        if "R" in parts:
            for x, size in settings["R"]:
                items.append(dict(name=ref_name(c, x, size), part="R", kind="ref", combo=c, dir=x, size=size,
                                  needs=[]))
        if "M" in parts:
            for d, Ld in settings["M"]:
                it = pair("M", c, [d], "s1", Ld, want_uCu=True, kind="mode")
                items.append(it)
                if brk == "repro" and c == repro_combo and d == "eigA" and learner_name(Ld) == "GD(1)":
                    items.append(dict(it, name=it["name"] + "__repro"))
        if "G" in parts:
            for d, Ld in settings["G"]:
                items.append(pair("G", c, [d], "s1", Ld, want_law=True))
        if "A" in parts:
            for e in settings["A"]:
                items.append(pair("A", c, [e["dir"]], e["size"], e["Ld"], tau=e["tau"], cap=e["cap"],
                                  group=e["group"], suffix=e["suffix"]))
        if "T" in parts:
            for size, Ld in settings["T"]:
                items.append(pair("T", c, ["E1", "E2"], size, Ld))
        if "H" in parts:
            # spec note: hover and own-metric unit names use <dirs> = E1 and <size> = s1 (their fixed direction and size);
            #       the own-metric unit is named by its learner ADAM(k, B1, B2, 0, none)
            for Ld in settings["H"]:
                items.append(dict(name=f"H__{c}__E1__s1__{safe_name(learner_name(Ld))}", part="H", kind="hover",
                                  combo=c, Ld=Ld, lname=learner_name(Ld), needs=[["E1", "s1"]]))
        if "S" in parts:
            # spec note: the spec gives no <dirs>/<size> for star units: "star" and "s1" are used
            for Ld in settings["S"]:
                items.append(dict(name=f"S__{c}__star__s1__{safe_name(learner_name(Ld))}", part="S", kind="star",
                                  combo=c, Ld=Ld, lname=learner_name(Ld), needs=[["E1", "s1"]]))
            items.append(dict(name=f"S__{c}__OP0", part="S", kind="op0", combo=c, needs=[["E1", "s1"]]))
        if "D" in parts:
            for k in settings["D"]:
                Ld = ADAM(k)
                items.append(dict(name=f"D__{c}__E1__s1__{safe_name(learner_name(Ld))}", part="D", kind="own",
                                  combo=c, kappa=k, Ld=Ld, lname=learner_name(Ld), needs=[["E1", "s1"]]))
    return items


def item_meta(it):
    """Plan fields copied into the unit record (so that the report can read units without the plan)."""
    keep = ("part", "group", "suffix", "lname", "want_law", "want_uCu")
    return {k: it[k] for k in keep if k in it}


# ---------------------------------------------------------------------------------------------------------------
# Section 9: unit-level rules (shared by the report and the tests)
# ---------------------------------------------------------------------------------------------------------------
def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def hm1_ok(rec):
    g = rec.get("g_pred")
    rho, cos, nr = rec.get("rho"), rec.get("cos"), rec.get("nr")
    if not (num(g) and num(rho)):
        return False
    if not S.within(rho, g, 0.03):
        return False
    if abs(g) >= 0.1:
        return num(cos) and cos * S.sign(g) >= 0.99
    return num(nr) and nr <= abs(g) + 0.05


def hm2_ok(rec):
    g, a_u = rec.get("g_pred"), rec.get("a_u")
    return num(g) and num(a_u) and abs(a_u + g) <= 0.05 * max(1.0, abs(g))


def hm3_ok(rec):
    f, rho1, phi = rec.get("f_pred"), rec.get("rho1"), rec.get("phi_u")
    return num(f) and num(rho1) and num(phi) and S.within(rho1, f, 0.03) and S.within(phi, f, 0.05)


def hm4_ok(rec):
    g, uCu = rec.get("g_pred"), rec.get("uCu")
    return num(g) and num(uCu) and S.within(uCu, g, 0.02)


def hg1_ok(rec):
    nvNPe = rec.get("nvNPe")
    if not (num(nvNPe) and nvNPe > 0 and num(rec.get("n_law"))):
        return False
    if rec["n_law"] / nvNPe >= 0.05:
        return (num(rec.get("law_ratio")) and abs(rec["law_ratio"] - 1.0) <= 0.05
                and num(rec.get("law_cos")) and rec["law_cos"] >= 0.98)
    return num(rec.get("nve")) and rec["nve"] / nvNPe <= 0.1


def v4_ok(rec):
    a_u, rho = rec.get("a_u"), rec.get("rho")
    return num(a_u) and num(rho) and abs(a_u + rho) <= 0.03 * max(1.0, abs(rho))


def usable_M(rec):
    """Part M: OK, or UNCONVERGED with every leg res <= 1e-4 (and every closing done). Returns (usable, star)."""
    # spec note: an UNCONVERGED Part M unit is usable only if, in addition, both polygons are complete and every closing
    #       converged and closed (closings_ok; "the closing finished the job")
    if rec.get("status") == "OK":
        return True, False
    if rec.get("status") == "UNCONVERGED" and rec.get("closings_ok"):
        legs = rec["sig+1"]["legs"] + rec["sig-1"]["legs"]
        if all(num(g["res"]) and g["res"] <= 1e-4 for g in legs):
            return True, True
    return False, False


def hg3_ok(rec):
    """NOTE (review, A1.6): spectral area rule for one unit."""
    return (num(rec.get("area_cos")) and rec["area_cos"] >= 0.98
            and num(rec.get("area_ratio")) and abs(rec["area_ratio"] - 1.0) <= 0.05)


def usable_G(rec):
    """Part G: OK, or UNCONVERGED with every closing converged and closed."""
    return rec.get("status") == "OK" or (rec.get("status") == "UNCONVERGED" and bool(rec.get("closings_ok")))
