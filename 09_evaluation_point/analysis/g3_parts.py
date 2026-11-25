"""Second read-only readout for the SPEC9 run. Uses the saved Part A vectors and records; runs no learner; CPU only.
Safe to run while the main run is still working. Needs the vector files, so run it before the rig is released.

G = even part of G_NP at size s2 (the one held-out function along which all learners differ), F0 = F_test(theta0),
Y = one-hot test labels, r0 = F0 - Y, N = number of entries, b = <r0, G>/N (test-loss change per unit f), as in g3_split.py.

1. Is any part of b related to WHICH test image has WHICH label?
     b = free + lab,   free = (<F0, G> - <Yc, Gc>)/N   (unchanged if the test labels are shuffled over the test images)
                       lab  = -<Y, Gv>/N               (Gv = G minus its per-class mean over the test inputs)
   lab = lab_scale + pattern, lab_scale = -e <Y, F0v>/N (growth of the start function times its agreement with the labels),
   pattern = -<Y, Gp>/N (Gp = Gv - e F0v; the 'pattern' column of g3_split.py).
   Null: the test labels are shuffled over the test images K_PERM times; z = value / (standard deviation of the shuffled
   values) (the shuffled mean is 0 by construction). |z| < 2: no evidence of a relation to the labels.
   The functions G of different directions of one combination are correlated, so their z values are not independent.

2. Trained starts only. The start function on the test inputs is F0 = Nf + S, Nf = F_test(theta_init) (before fitting),
   S = F0 - Nf (what fitting added). Least squares on the centred vectors: Gv = eN Nv + eS Sv + rest.
   eN, eS = relative growth of each part per unit f. b = const + bN + bS + brest with bN = eN <r0, Nv>/N etc.
   optS = <Y, Sv>/|Sv|^2: > 1 means the fitted part is too SMALL on the test inputs (growing it lowers the test loss).

3. Directions with a defined sign only (label, rand, true). S1 = first-order change of the held-out outputs when the
   training outputs move the FULL length of the direction (tau = 1), by central differences along h = J0^T K0^-1 d.
   cS1 = cos(Gv, S1v): > 0 the shift toward the nearest point extends the first-order change on unseen inputs, < 0 it
   takes part of it back. Fit Gv = eF F0v + e1 S1v + rest (R2b). To first order in tau the residual at the exact-jump end
   point is r0 + tau S1, so the slope b changes sign at tau* = -<r0, G>/<S1, G> (printed when positive).
   chk = |first-order change at size s2| / |F_test(theta_TH) - F_test(theta0)| from the stored record (must be near 1).

4. Writes g3_bundle.pt next to the notebook (about 25 MB in full mode): G of every combination and direction, F0, Nf and
   the test labels, so later questions need no rig.

usage on the rig:   python g3_parts.py                       (expects /workspace/evaluation_point_tests.ipynb)
local test:         python g3_parts.py <notebook> <mode>     with PART9LIB_DIR and PART9_RUN_DIR set
"""
import os, sys, json, re

nbp = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/workspace/evaluation_point_tests.ipynb")
mode = sys.argv[2] if len(sys.argv) > 2 else None
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

import torch
torch.set_num_threads(4)
import part9lib as L
L.configure(CFG)
L.set_device("cpu")
L._load_mnist_raw()
rd = os.environ.get("PART9_RUN_DIR", CFG["RUN_DIR"])
C = CFG["N_CLASSES"]
K_PERM = 1000
print("g3_parts: MODE %s data_source %s run_dir %s" % (CFG["MODE"], L.DATA_SOURCE, os.path.abspath(rd)))


def nm(cn, Ls, d, size, sg):
    fr = CFG["S1_FRAC"] if size == "s1" else CFG["S2_FRAC"]
    return L._case9(cn, "A", "A", Ls, d, size, fr, sg)["name"]


def vec(name):
    p = os.path.join(rd, "vecs", L.safe_name(name) + ".pt")
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def centre(x):
    """(centred per class, per-class mean expanded), both as (n_test, C) matrices."""
    X = x.view(-1, C)
    m = X.mean(0, keepdim=True)
    return X - m, m.expand_as(X)


def dot(a, b):
    return float((a * b).sum())


def cos(a, b):
    return dot(a, b) / max((dot(a, a) * dot(b, b)) ** 0.5, 1e-300)


def pick(M, y):
    """sum_i M[i, y_i] = <Y, M> for one-hot Y."""
    return float(M.gather(1, y.view(-1, 1)).sum())


NP_L = L.L_AJ(1.0)
bundle = dict(mode=CFG["MODE"], data_source=L.DATA_SOURCE, combos={})
print("\nPart 1 (label split, all starts) and part 2 (trained starts: growth of the part before fitting, eN, and of the fitted "
      "part, eS). Columns: b = free + lab; z_lab; lab = lab_scale + pattern; z_pat; then for trained starts eN, eS, "
      "b = const + bN + bS + brest, R2 of the two-part fit, cos(Nv, Sv), optS; then part 3 (label, rand, true): cS1, eF, e1, "
      "R2b, tau*, chk.")
print("%-17s %-5s %11s %11s %11s %7s | %11s %11s %7s | %10s %10s | %10s %10s %10s %10s | %6s %6s %7s | %7s %10s %10s %6s %7s %6s" % (
    "combination", "dir", "b", "free", "lab", "z_lab", "lab_scale", "pattern", "z_pat", "eN", "eS", "const", "bN", "bS",
    "brest", "R2", "cosNS", "optS", "cS1", "eF", "e1", "R2b", "tau*", "chk"))
for cb in L.combos("A"):
    cn = L.cname9(cb)
    try:
        _, th_init, _, _ = L.combo_problem9(cb)
        ctx = L.Ctx9(cb, rd)
        prob, st, th0 = ctx.prob, ctx.st, ctx.st.th0
    except Exception as e:
        print("%-17s start not available (%s)" % (cn, type(e).__name__))
        continue
    td = L.get_test_data(cb["seed"], CFG["N_TRAIN"])
    y = td["labels"].long().cpu()
    n_test = y.numel()
    F0 = L.test_outputs(prob, td["X"], th0).reshape(-1)
    Yt = td["Y"].reshape(-1)
    r0 = (F0 - Yt).view(-1, C)
    N = F0.numel()
    F0v, F0c = centre(F0)
    entry = dict(labels=y.to(torch.int16), F0=F0.view(-1, C).float(), G={}, b={})
    trained = cb["start"] == "trained"
    if trained:
        Nf = L.test_outputs(prob, td["X"], th_init.to(L.DT)).reshape(-1)
        Nv, _ = centre(Nf)
        Sv = F0v - Nv
        entry["Nf"] = Nf.view(-1, C).float()
        a11, a12, a22 = dot(Nv, Nv), dot(Nv, Sv), dot(Sv, Sv)
        det = a11 * a22 - a12 * a12
        optS = pick(Sv, y) / a22
        ip = L.start_paths(rd, L.cname9(dict(cb, start="init")))[0]
        if os.path.exists(ip):
            dmax = float((torch.load(ip, map_location="cpu").to(L.DT) - th_init.cpu().to(L.DT)).abs().max())
            print("%-17s check: theta_init used here vs the stored start of the init combination, max |difference| = %.1e" % (cn, dmax))
    gp = L.gen(987654)
    perms = [torch.randperm(n_test, generator=gp) for _ in range(K_PERM)]
    dirs = list(CFG["A_DIRS"]) + (["true"] if cb["start"] == "init" else [])
    for d in dirs:
        vp, vm = vec(nm(cn, NP_L, d, "s2", 1)), vec(nm(cn, NP_L, d, "s2", -1))
        if vp is None or vm is None or "GNP" not in vp or "GNP" not in vm:
            print("%-17s %-5s missing" % (cn, d))
            continue
        G = 0.5 * (vp["GNP"].reshape(-1).to(L.DT) + vm["GNP"].reshape(-1).to(L.DT))
        Gm = G.view(-1, C)
        Gv, Gc = centre(G)
        e = dot(Gv, F0v) / dot(F0v, F0v)
        Gp = Gv - e * F0v
        b = dot(r0, Gm) / N
        lab = -pick(Gv, y) / N
        free = b - lab
        free_direct = (dot(F0.view(-1, C), Gm) - pick(Gc, y)) / N
        assert abs(free - free_direct) <= 1e-9 * max(abs(b), abs(free), 1e-300), (free, free_direct)
        lab_scale = -e * pick(F0v, y) / N
        pattern = -pick(Gp, y) / N
        nl = torch.tensor([-pick(Gv, y[p]) / N for p in perms], dtype=L.DT)
        npat = torch.tensor([-pick(Gp, y[p]) / N for p in perms], dtype=L.DT)
        z_lab = lab / max(float(nl.std()), 1e-300)
        z_pat = pattern / max(float(npat.std()), 1e-300)
        tail = "%10s %10s | %10s %10s %10s %10s | %6s %6s %7s" % ("-", "-", "-", "-", "-", "-", "-", "-", "-")
        if trained:
            g1, g2 = dot(Nv, Gv), dot(Sv, Gv)
            eN = (g1 * a22 - g2 * a12) / det
            eS = (g2 * a11 - g1 * a12) / det
            rest = Gv - eN * Nv - eS * Sv
            const, bN, bS, br = dot(r0, Gc) / N, eN * dot(r0, Nv) / N, eS * dot(r0, Sv) / N, dot(r0, rest) / N
            assert abs(const + bN + bS + br - b) <= 1e-9 * max(abs(b), abs(bN), abs(bS), 1e-300)
            tail = "%+10.2e %+10.2e | %+10.2e %+10.2e %+10.2e %+10.2e | %6.3f %+6.3f %7.3f" % (
                eN, eS, const, bN, bS, br, 1.0 - dot(rest, rest) / dot(Gv, Gv), cos(Nv, Sv), optS)
        t3 = "%7s %10s %10s %6s %7s %6s" % ("-", "-", "-", "-", "-", "-")
        if d in ("label", "rand", "true"):
            h = st.J0.T @ L.csolve(st.L0, ctx.dirs[d])
            sz = CFG["S2_FRAC"] * ctx.S[d]
            T2 = 0.5 * (L.test_outputs(prob, td["X"], th0 + sz * h) - L.test_outputs(prob, td["X"], th0 - sz * h))
            S1 = (T2 / CFG["S2_FRAC"]).reshape(-1, C)
            S1v, _ = centre(S1)
            c11, c12, c22 = dot(F0v, F0v), dot(F0v, S1v), dot(S1v, S1v)
            q1, q2 = dot(F0v, Gv), dot(S1v, Gv)
            dt2 = c11 * c22 - c12 * c12
            eF, e1 = (q1 * c22 - q2 * c12) / dt2, (q2 * c11 - q1 * c12) / dt2
            rb = Gv - eF * F0v - e1 * S1v
            sg = dot(S1, Gm)
            tstar = -dot(r0, Gm) / sg if sg != 0 else float("nan")
            rp = L.read_json(os.path.join(rd, "done", L.safe_name(nm(cn, NP_L, d, "s2", 1)) + ".json"))
            chk = L.vnorm(T2) * rp["GNP_over_dTH"] / rp["nGNP"]
            t3 = "%+7.3f %+10.2e %+10.2e %6.3f %7s %6.3f" % (
                cos(Gv, S1v), eF, e1, 1.0 - dot(rb, rb) / dot(Gv, Gv), ("%7.3f" % tstar) if tstar > 0 else "none", chk)
            entry.setdefault("S1", {})[d] = S1.float()
        print("%-17s %-5s %+11.3e %+11.3e %+11.3e %+7.1f | %+11.3e %+11.3e %+7.1f | %s | %s" % (
            cn, d, b, free, lab, z_lab, lab_scale, pattern, z_pat, tail, t3))
        entry["G"][d] = Gm.float()
        entry["b"][d] = b
    bundle["combos"][cn] = entry
    del ctx, st, prob
    L.free_memory()
out = os.path.join(os.getcwd(), "g3_bundle.pt")
torch.save(bundle, out)
print("g3_parts: wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
print("g3_parts: done")
