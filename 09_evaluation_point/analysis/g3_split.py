"""Read-only extra readout for the SPEC9 run. Uses the saved Part A vectors and records; runs no learner; CPU only.

G3 found b > 0 everywhere: at equal training outputs, a larger alignment fraction f gives a HIGHER test loss.
To leading order  b = <r0, G> / N_test  with r0 = F_test(theta0) - Y_test and G = even part of G_NP (the one held-out
function along which all learners differ). This script splits G into three orthogonal pieces and b with it:
  const   : the per-class mean of G over the test inputs (a shift of every prediction by the same 10 numbers)
  scale   : the piece along the centred start outputs F0v (e > 0: the function gets LARGER on unseen inputs)
  pattern : the remainder (b_pattern < 0: the pattern moves TOWARD the true labels, > 0: away from them)
It also prints test-accuracy changes (accuracy ignores the size of the outputs) and the cosines between the functions G of
different target directions.

usage on the rig:   python g3_split.py                       (expects /workspace/evaluation_point_tests.ipynb)
local test:         python g3_split.py <notebook> <mode>     with PART9LIB_DIR and PART9_RUN_DIR set
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
print("g3_split: MODE %s data_source %s run_dir %s" % (CFG["MODE"], L.DATA_SOURCE, os.path.abspath(rd)))


def nm(cn, Ls, d, size, sg):
    fr = CFG["S1_FRAC"] if size == "s1" else CFG["S2_FRAC"]
    return L._case9(cn, "A", "A", Ls, d, size, fr, sg)["name"]


def vec(name):
    p = os.path.join(rd, "vecs", L.safe_name(name) + ".pt")
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def rec(name):
    p = os.path.join(rd, "done", L.safe_name(name) + ".json")
    return L.read_json(p) if os.path.exists(p) else None


def cos(a, b):
    return float(a @ b) / max(float(a.norm() * b.norm()), 1e-300)


def centre(x):
    X = x.view(-1, C)
    m = X.mean(0, keepdim=True)
    return (X - m).reshape(-1), m.expand_as(X).reshape(-1)


NP_L = L.L_AJ(1.0)
ACC_L = [("AJ(2)", L.L_AJ(2.0), 2.0), ("AJ(1)", L.L_AJ(1.0), 1.0), ("DJ(1.5)", L.L_DJ(1.5), -1.0)]
print("\nb = test-loss change per unit f (even part, size s2). b_vec = <r0, G>/N from the vectors; b_rec = from the stored losses "
      "(must agree). const + scale + pattern = b_vec. e = relative growth of the centred start outputs per unit f. "
      "cos columns: G against the centred start outputs F0v, against the centred labels Yv, against r0.")
print("%-17s %-5s %11s %11s | %10s %10s %10s | %10s | %7s %7s %7s | %s" % (
    "combination", "dir", "b_vec", "b_rec", "const", "scale", "pattern", "e", "cF0v", "cYv", "cr0",
    "test images gained/lost vs f=0 at s1, + sign: AJ(2) AJ(1) DJ(1.5)  [acc at theta0 -> at f=0]"))
for cb in L.combos("A"):
    cn = L.cname9(cb)
    try:
        prob, _, Ytr, _ = L.combo_problem9(cb)
        th0 = torch.load(L.start_paths(rd, cn)[0], map_location="cpu").to(L.DT)
    except Exception as e:
        print("%-17s start not available (%s)" % (cn, type(e).__name__))
        continue
    td = L.get_test_data(cb["seed"], CFG["N_TRAIN"])
    F0 = L.test_outputs(prob, td["X"], th0).reshape(-1)
    Yt = td["Y"].reshape(-1)
    r0 = F0 - Yt
    N = r0.numel()
    F0v, F0c = centre(F0)
    Yv, _ = centre(Yt)
    dirs = list(CFG["A_DIRS"]) + (["true"] if cb["start"] == "init" else [])
    Gs = {}
    for d in dirs:
        vp, vm = vec(nm(cn, NP_L, d, "s2", 1)), vec(nm(cn, NP_L, d, "s2", -1))
        rp, rm = rec(nm(cn, NP_L, d, "s2", 1)), rec(nm(cn, NP_L, d, "s2", -1))
        if vp is None or vm is None or "GNP" not in vp or "GNP" not in vm or rp is None or rm is None:
            print("%-17s %-5s missing" % (cn, d))
            continue
        G = 0.5 * (vp["GNP"].reshape(-1).to(L.DT) + vm["GNP"].reshape(-1).to(L.DT))
        Gs[d] = G
        Gv, Gc = centre(G)
        e = float(Gv @ F0v) / float(F0v @ F0v)
        Gs_ = e * F0v
        Gp = Gv - Gs_
        b_vec = float(r0 @ G) / N
        b_rec = 0.5 * ((rp["loss_NP"] - rp["loss_TH"]) + (rm["loss_NP"] - rm["loss_TH"]))
        acc = []
        for ln, Ls, f in ACC_L:
            r = rec(nm(cn, Ls, d, "s1", 1))
            acc.append("%+5d" % round((r["acc_L"] - r["acc_TH"]) * r["n_test"]) if r and "acc_L" in r else "    ?")
        r1 = rec(nm(cn, NP_L, d, "s1", 1))
        a0 = "[%.4f -> %.4f]" % (r1["acc_0"], r1["acc_TH"]) if r1 and "acc_0" in r1 else ""
        print("%-17s %-5s %+11.3e %+11.3e | %+10.2e %+10.2e %+10.2e | %+10.2e | %+7.3f %+7.3f %+7.3f | %s %s" % (
            cn, d, b_vec, b_rec, float(r0 @ Gc) / N, float(r0 @ Gs_) / N, float(r0 @ Gp) / N, e,
            cos(G, F0v), cos(G, Yv), cos(G, r0), " ".join(acc), a0))
    if len(Gs) > 1:
        ks = list(Gs)
        print("   cosines between the functions G of the directions (%s):" % " ".join(ks))
        for a in ks:
            print("      %-5s " % a + " ".join("%+6.3f" % cos(Gs[a], Gs[b_]) for b_ in ks))
print("g3_split: done")
