"""Task-cycle rotation, part 4 (SPEC4.md): the path law.

Path matrix C (measured and a-priori), the LAW operator C -> V, the GD / momentum / Adam
learners of SPEC4 section 2 with C bookkeeping, and the path runner.
Reuses tcr.py / tcr2.py / tcr3.py unchanged (imported). No experiment logic here.
All arithmetic float64, CPU, deterministic. BLAS / torch thread count is fixed to THREADS
(set the environment variables before numpy is imported: import this module first).
"""
import os

THREADS = 2
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = str(THREADS)

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tcr  # noqa: E402
import tcr3  # noqa: E402
from tcr import NotConverged, Timeout  # noqa: E402,F401
from tcr3 import xi, jk, proj_v, nrm  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(THREADS)

FD_STEP = 1e-4             # s = 1e-4 / ||w||  (SPEC3 section 1, SPEC4 section 0)
FO_TOL = 1e-8              # GD / MOM stop rule
FO_CAP = 400_000           # GD / MOM cap
DIV_FACTOR = 1e3           # DIVERGED if ||r||_inf > 1e3 x initial
MOM_MU = 0.9
MOM_C = 0.25
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS = 1e-3, 0.9, 0.999, 1e-8
ADAM_TOL, ADAM_CAP = 1e-6, 200_000

# learner name -> (kind, c)
LEARNERS = {"GN": ("GN", None), "GD0.25": ("GD", 0.25), "GD1": ("GD", 1.0),
            "GD1.8": ("GD", 1.8), "MOM": ("MOM", MOM_C), "ADAM": ("ADAM", None)}

# sequences of targets after theta_A (names resolved by make_targets)
PATHS = {"oab": ("u", "A"),
         "tri+": ("u", "v", "A"),
         "tri-": ("v", "u", "A"),
         "walk8": ("w1", "w2", "w3", "w4", "w5", "w6", "w7", "A"),
         "oneway": ("u",)}
CLOSED = ("oab", "tri+", "tri-", "walk8")


# ---------------------------------------------------------------------------
# C bookkeeping
# ---------------------------------------------------------------------------
def c_increment(C, F_next, F_k, F_A):
    """GN / GD rule: C += (F_{k+1} - F_k) (x) (F_k - F_A)   (increment = column vector)."""
    C += np.outer(F_next - F_k, F_k - F_A)
    return C


def c_parts(C):
    S = (C + C.T) / 2
    A = (C - C.T) / 2
    return S, A


# ---------------------------------------------------------------------------
# numpy forward / gradient / JVP (fast per-step routines; checked in test_part4.py
# against J^T r / n and K r / n from the exact Jacobian tcr2.FastMLP.jac)
# ---------------------------------------------------------------------------
class NpMLP:
    def __init__(self, model, X):
        self.model = model
        self.Xn = X.numpy().copy()
        self.n = self.Xn.shape[0]
        self.P = model.P
        self.sl = dict(model.slices)
        self.sd, self.sh = model.sd, model.sh
        self.b3i = self.sl["b3"][0]

    def _v(self, t, k):
        a, b, shp = self.sl[k]
        return t[a:b].reshape(shp)

    def fg(self, t, y):
        """Returns F = f(X;t), r = F - y, g = J^T r / n, activations."""
        sd, sh, n = self.sd, self.sh, self.n
        W1, b1, W2, b2, w3 = (self._v(t, k) for k in ("W1", "b1", "W2", "b2", "w3"))
        a1 = np.tanh(self.Xn @ W1.T / sd + b1)
        a2 = np.tanh(a1 @ W2.T / sh + b2)
        F = a2 @ w3 / sh + t[self.b3i]
        r = F - y
        e = r / n
        g = np.empty(self.P)
        self._v(g, "w3")[:] = a2.T @ e / sh
        g[self.b3i] = e.sum()
        dz2 = (e[:, None] * w3[None, :] / sh) * (1.0 - a2 * a2)
        self._v(g, "W2")[:] = dz2.T @ a1 / sh
        self._v(g, "b2")[:] = dz2.sum(0)
        dz1 = (dz2 @ W2 / sh) * (1.0 - a1 * a1)
        self._v(g, "W1")[:] = dz1.T @ self.Xn / sd
        self._v(g, "b1")[:] = dz1.sum(0)
        return F, r, g, (a1, a2)

    def jvp(self, t, d, acts):
        """J(t) d  (forward-mode, exact)."""
        a1, a2 = acts
        sd, sh = self.sd, self.sh
        W2, w3 = self._v(t, "W2"), self._v(t, "w3")
        dW1, db1, dW2, db2, dw3 = (self._v(d, k) for k in ("W1", "b1", "W2", "b2", "w3"))
        dz1 = self.Xn @ dW1.T / sd + db1
        da1 = (1.0 - a1 * a1) * dz1
        dz2 = (da1 @ W2.T + a1 @ dW2.T) / sh + db2
        da2 = (1.0 - a2 * a2) * dz2
        return (da2 @ w3 + a2 @ dw3) / sh + d[self.b3i]


class NpLinear:
    """Same interface for tcr.LinearModel (f = Phi theta)."""

    def __init__(self, model, X):
        self.model = model
        self.Phi = model.phi(X).numpy().copy()
        self.n = self.Phi.shape[0]
        self.P = model.P

    def fg(self, t, y):
        F = self.Phi @ t
        r = F - y
        return F, r, self.Phi.T @ r / self.n, None

    def jvp(self, t, d, acts):
        return self.Phi @ d


def np_model(model, X):
    return NpLinear(model, X) if isinstance(model, tcr.LinearModel) else NpMLP(model, X)


def lam_max_K(model, theta, X):
    J = model.jac(theta, X)
    return torch.linalg.eigvalsh(J @ J.T)[-1].item()


# ---------------------------------------------------------------------------
# Learners (SPEC4 section 2) with C bookkeeping (SPEC4 section 3)
# ---------------------------------------------------------------------------
def gn_learn_C(model, theta, X, y, F_A, C, tol=tcr.GN_TOL, max_iter=tcr.GN_MAX_ITER):
    """tcr.gn_learn, step by step (same tcr.gn_step), accumulating C over every GN iteration."""
    t0 = time.time()
    F_An = F_A
    F = model.f(theta, X)
    r = F - y
    it, n_lsf, min_s = 0, 0, 1.0
    while r.abs().max().item() > tol:
        if it >= max_iter:
            raise NotConverged(f"GN not converged after {max_iter} iterations, "
                               f"||r||_inf={r.abs().max().item():.3e}")
        theta, r, s, lsf = tcr.gn_step(model, theta, X, y)
        F_new = model.f(theta, X)
        c_increment(C, F_new.numpy(), F.numpy(), F_An)
        F = F_new
        n_lsf += int(lsf)
        min_s = min(min_s, s)
        it += 1
    return theta, C, {"steps": it, "res_inf": r.abs().max().item(), "status": "CONVERGED",
                      "ls_fail": n_lsf, "min_s": min_s, "time": time.time() - t0}


def fo_learn_C(npm, model, theta, X, y, F_A, C, kind, c=None, mu=MOM_MU, tol=None, cap=None,
               fixed_steps=None, hook=None, deadline=None):
    """First-order learners. kind in {'GD', 'MOM', 'ADAM'}. theta: torch tensor (P,).
    y, F_A: numpy (n,). C: numpy (n,n), updated in place.
    GD : theta <- theta - eta g,                 C += (F_{k+1}-F_k) (x) (F_k - F_A)
    MOM: v <- mu v - eta g; theta <- theta + v,  C += e~_k (x) (F_k - F_A),
         e~_k = -(eta/((1-mu) n)) K(theta_k) r_k = -(eta/(1-mu)) J(theta_k) g_k
    ADAM: torch-style Adam (lerp first moment), plain C rule as GD.
    eta = c n / lambda_max(K(theta_in)) for GD / MOM.
    fixed_steps: run exactly that many steps, ignoring the tolerance (S6).
    hook(k, theta_k_np, dtheta_np): called for each step (test 2).
    Status: CONVERGED / NOT CONVERGED (cap) / DIVERGED (||r||_inf > 1e3 x initial or non-finite)."""
    t0 = time.time()
    n = npm.n
    if kind == "ADAM":
        tol = ADAM_TOL if tol is None else tol
        cap = ADAM_CAP if cap is None else cap
        lam, eta = None, None
    else:
        tol = FO_TOL if tol is None else tol
        cap = FO_CAP if cap is None else cap
        lam = lam_max_K(model, theta, X)
        eta = c * n / lam
    if fixed_steps is not None:
        cap = fixed_steps
    th = theta.numpy().copy()
    F, r, g, acts = npm.fg(th, y)
    r0 = float(np.abs(r).max())
    vel = np.zeros_like(th)
    if kind == "ADAM":
        m1 = np.zeros_like(th)
        m2 = np.zeros_like(th)
        bc2s = None
    k = 0
    status = None
    while True:
        res = float(np.abs(r).max())
        if not np.isfinite(res) or res > DIV_FACTOR * r0:
            status = "DIVERGED"
            break
        if fixed_steps is None and res <= tol:
            status = "CONVERGED"
            break
        if k >= cap:
            status = "FIXED" if fixed_steps is not None else "NOT CONVERGED"
            break
        if deadline is not None and (k & 4095) == 0 and time.time() > deadline:
            raise Timeout(f"deadline passed after {k} {kind} steps (res_inf={res:.3e})")
        if kind == "GD":
            dth = -eta * g
        elif kind == "MOM":
            et = -(eta / (1.0 - mu)) * npm.jvp(th, g, acts)
            C += np.outer(et, F - F_A)
            vel = mu * vel - eta * g
            dth = vel
        elif kind == "ADAM":
            t = k + 1
            m1 += (g - m1) * (1.0 - ADAM_B1)          # exp_avg.lerp_(grad, 1 - beta1)
            m2 *= ADAM_B2
            m2 += (1.0 - ADAM_B2) * g * g              # exp_avg_sq.mul_(b2).addcmul_(g, g, 1 - b2)
            bc1 = 1.0 - ADAM_B1 ** t
            bc2 = 1.0 - ADAM_B2 ** t
            step_size = ADAM_LR / bc1
            denom = np.sqrt(m2) / np.sqrt(bc2) + ADAM_EPS
            dth = -step_size * (m1 / denom)
        else:
            raise ValueError(kind)
        if hook is not None:
            hook(k, th.copy(), dth.copy())
        th = th + dth
        F_old = F
        F, r, g, acts = npm.fg(th, y)
        if kind != "MOM":
            C += np.outer(F - F_old, F_old - F_A)
        k += 1
    info = {"steps": k, "res_inf": float(np.abs(r).max()), "res0_inf": r0, "status": status,
            "eta": eta, "lam_max_in": lam, "time": time.time() - t0}
    return torch.from_numpy(th.copy()), C, info


# ---------------------------------------------------------------------------
# Targets and path runner (prefix sharing; deterministic so identical to separate runs)
# ---------------------------------------------------------------------------
def make_targets(S, eps, walk_dirs=None):
    t = {"A": S.yA.clone(), "u": S.yA + eps * S.u, "v": S.yA + eps * S.v}
    if walk_dirs is not None:
        for i, w in enumerate(walk_dirs, start=1):
            t[f"w{i}"] = S.yA + eps * w
    return t


def walk_dirs(S):
    """Seven vectors from Generator(777 + seed), 16 entries ~ N(0,1) each, scaled to ||y_A||."""
    return tcr3.random_dirs(777 + S.seed, 7, S.yA_norm)


def make_learn(S, name, npm=None, deadline=None, fixed_steps=None):
    """Returns learn(theta, y_torch, C_np) -> (theta, C, info)."""
    kind, c = LEARNERS[name]
    F_A = S.F_A_np
    if kind == "GN":
        def learn(theta, y, C):
            if fixed_steps is not None:
                return gn_fixed_C(S.model, theta, S.X, y, F_A, C, fixed_steps)
            return gn_learn_C(S.model, theta, S.X, y, F_A, C)
        return learn
    npm = npm or np_model(S.model, S.X)

    def learn(theta, y, C):
        return fo_learn_C(npm, S.model, theta, S.X, y.numpy(), F_A, C, kind, c,
                          deadline=deadline, fixed_steps=fixed_steps)
    return learn


def gn_fixed_C(model, theta, X, y, F_A, C, n_steps):
    """Supplementary (S6): exactly n_steps tcr.gn_step iterations, no stop rule."""
    t0 = time.time()
    F = model.f(theta, X)
    n_lsf = 0
    r = F - y
    for _ in range(n_steps):
        theta, r, s, lsf = tcr.gn_step(model, theta, X, y)
        F_new = model.f(theta, X)
        c_increment(C, F_new.numpy(), F.numpy(), F_A)
        F = F_new
        n_lsf += int(lsf)
    return theta, C, {"steps": n_steps, "res_inf": r.abs().max().item(), "status": "FIXED",
                      "ls_fail": n_lsf, "time": time.time() - t0}


def run_paths(S, learn, targets, path_names, paths=PATHS):
    """Run each path from theta_A with prefix sharing. Returns {path: dict(theta, C, infos,
    status)}; status 'OK', or the first non-converged learn's status (DIVERGED stops the
    path; NOT CONVERGED continues), or 'INVALID' (GN NotConverged)."""
    n = S.yA.shape[0]
    cache = {(): (S.theta_A, np.zeros((n, n)), [], "OK")}
    out = {}
    for pname in path_names:
        seq = tuple(paths[pname])
        k = len(seq)
        while seq[:k] not in cache:
            k -= 1
        theta, C, infos, status = cache[seq[:k]]
        infos = list(infos)
        for j in range(k, len(seq)):
            if status in ("DIVERGED", "INVALID"):
                cache[seq[:j + 1]] = (theta, C, infos, status)
                continue
            Cn = C.copy()
            try:
                theta, Cn, info = learn(theta, targets[seq[j]], Cn)
            except NotConverged as e:
                info = {"status": "INVALID", "error": str(e)}
                infos.append(dict(info, target=seq[j]))
                status = "INVALID"
                cache[seq[:j + 1]] = (theta, C, infos, status)
                continue
            C = Cn
            infos.append(dict(info, target=seq[j]))
            st = info["status"]
            if st == "DIVERGED":
                status = "DIVERGED"
            elif st == "NOT CONVERGED" and status == "OK":
                status = "NOT CONVERGED"
            cache[seq[:j + 1]] = (theta, C, list(infos), status)
        out[pname] = {"theta": theta, "C": C, "infos": infos, "status": status}
    return out


# ---------------------------------------------------------------------------
# A-priori C (SPEC4 section 3): label-space simulation with K_A only
# ---------------------------------------------------------------------------
def apriori_C(KA, F_A, target_seq, name, tol=FO_TOL, cap=FO_CAP, mu=MOM_MU):
    """KA, F_A numpy; target_seq: list of numpy targets. Returns (C, info).
    cap: safety cap per learn (not in the spec; status NOT CONVERGED if hit)."""
    kind, c = LEARNERS[name]
    n = F_A.shape[0]
    C = np.zeros((n, n))
    F = F_A.copy()
    lam = float(np.linalg.eigvalsh(KA)[-1])
    steps, status = [], "OK"
    for y in target_seq:
        if kind == "GN":
            r = F - y
            e = -r
            C += np.outer(e, F - F_A)
            F = F + e
            steps.append(1)
            continue
        if kind == "GD":
            etap = c / lam
            k = 0
            r = F - y
            while np.abs(r).max() > tol:
                if k >= cap:
                    status = "NOT CONVERGED"
                    break
                e = -etap * (KA @ r)
                C += np.outer(e, F - F_A)
                F = F + e
                r = F - y
                k += 1
            steps.append(k)
        elif kind == "MOM":
            etap = c / lam
            w = np.zeros(n)
            k = 0
            r = F - y
            while np.abs(r).max() > tol:
                if k >= cap:
                    status = "NOT CONVERGED"
                    break
                Kr = KA @ r
                w = mu * w - etap * Kr
                et = -(etap / (1.0 - mu)) * Kr
                C += np.outer(et, F - F_A)
                F = F + w
                r = F - y
                k += 1
            steps.append(k)
        else:
            raise ValueError(f"no a-priori C for {name}")
    return C, {"steps": steps, "status": status}


# ---------------------------------------------------------------------------
# The LAW operator  V = sum_j P_v [ D xi_{C e_j} . xi_{e_j} ] (theta_A)
# ---------------------------------------------------------------------------
class Law:
    """Precomputes T_j = D xi_I . xi_{e_j} (P x n, column i = D xi_{e_i} . xi_{e_j}) with the
    finite-difference step s = 1e-4 / ||xi_{e_j}||, so that by linearity of xi_E in E
    D xi_{C e_j} . xi_{e_j} = T_j C e_j. LAW(C) = P_v sum_j T_j C[:, j].
    Checked against the literal column-by-column evaluation (law_direct) in test_part4.py."""

    def __init__(self, model, theta, X, step=FD_STEP):
        self.model, self.theta, self.X = model, theta, X
        self.J, self.K = jk(model, theta, X)
        n = self.K.shape[0]
        I = torch.eye(n)
        self.XI = self.J.T @ torch.linalg.solve(self.K, I)          # column j = xi_{e_j}
        T = torch.empty(n, model.P, n)
        for j in range(n):
            w = self.XI[:, j]
            s = step / torch.linalg.norm(w).item()
            T[j] = (xi(model, theta + s * w, X, I) - xi(model, theta - s * w, X, I)) / (2 * s)
        self.T = T

    def raw(self, C):
        Ct = torch.as_tensor(np.asarray(C), dtype=torch.float64)
        return torch.einsum("jpi,ij->p", self.T, Ct)

    def __call__(self, C):
        return proj_v(self.J, self.K, self.raw(C))

    def Pv(self, w):
        return proj_v(self.J, self.K, w)


def law_direct(model, theta, X, C, step=FD_STEP):
    """Literal SPEC4 (LAW): sum_j P_v [ D xi_{C e_j} . xi_{e_j} ], one FD pair per column."""
    J, K = jk(model, theta, X)
    n = K.shape[0]
    Ct = torch.as_tensor(np.asarray(C), dtype=torch.float64)
    tot = torch.zeros(model.P)
    for j in range(n):
        ej = torch.zeros(n)
        ej[j] = 1.0
        w = xi(model, theta, X, ej, (J, K))
        tot += proj_v(J, K, tcr3.dxi(model, theta, X, Ct[:, j].clone(), w, step))
    return tot


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def compare(meas, pred):
    nm, npd = nrm(meas), nrm(pred)
    c = tcr.cosine(meas, pred)
    return {"cos": c, "ratio": nm / npd if npd > 0 else float("nan"),
            "rel_err": nrm(meas - pred) / nm if nm > 0 else float("nan"),
            "norm_meas": nm, "norm_pred": npd}


def setup(h, seed, linear=False):
    """tcr3.Setup (GN base point) plus F_A (numpy)."""
    S = tcr3.Setup(h, seed, linear=linear, uv_from_teacher_for_linear=not linear)
    S.F_A = S.model.f(S.theta_A, S.X)
    S.F_A_np = S.F_A.numpy().copy()
    return S
