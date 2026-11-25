"""Layer structure of the nearest-point shift v_NP on the real MNIST nets (vectors from the rig). Per combination and direction
(even part over signs, size s2): share of |v_NP|^2 per parameter block, relative growth of each block's squared norm per unit
|v_NP| (2 <W, v>/|W|^2 normalised), and cosines between v_NP of different directions."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, glob, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc
L, CFG = loc.load(threads=2)
VD = DATA + "/vecs_np/part9_run/vecs/"
def blocks(shapes):
    o, out = 0, []
    for l, (wo, wi) in enumerate(shapes):
        out.append(("W%d" % (l + 1), o, o + wo * wi)); o += wo * wi
        out.append(("b%d" % (l + 1), o, o + wo)); o += wo
    return out
print("share of |v_NP|^2 per block (W1 b1 W2 b2 W3 b3) | relative growth of |W_l|^2 along v_NP, sign and size: g_l = 2<W_l, v_l>/(|W_l|^2 |v|) x |theta|")
for cb in L.combos("A"):
    cn = L.cname9(cb)
    prob, _, _, _ = L.combo_problem9(cb)
    th0 = torch.load(L.start_paths(loc.RD, cn)[0], map_location="cpu").to(L.DT)
    bl = blocks(prob.shapes); vs = {}
    dirs = list(CFG["A_DIRS"]) + (["true"] if cb["start"] == "init" else [])
    for d in dirs:
        fs = [VD + L.safe_name(L._case9(cn, "A", "A", L.L_AJ(1.0), d, "s2", CFG["S2_FRAC"], sg)["name"]) + ".pt" for sg in (1, -1)]
        if not all(os.path.exists(f) for f in fs): print(cn, d, "missing"); continue
        v = 0.5 * sum(torch.load(f, map_location="cpu")["vNP"].to(L.DT) for f in fs)
        vs[d] = v; n2 = float(v @ v)
        sh = [float(v[a:b] @ v[a:b]) / n2 for _, a, b in bl]
        gr = [2 * float(th0[a:b] @ v[a:b]) / float(th0[a:b] @ th0[a:b]) / n2 ** 0.5 * float(th0.norm()) if float(th0[a:b] @ th0[a:b]) > 0 else float("nan") for _, a, b in bl]
        print("%-17s %-5s shares %s | growth W1 %+7.2f W2 %+7.2f W3 %+7.2f | total |theta|^2 growth %+7.3f" % (
            cn, d, " ".join("%.3f" % x for x in sh), gr[0], gr[2], gr[4], 2 * float(th0 @ v) / float(th0 @ th0) / n2 ** 0.5 * float(th0.norm())))
    ks = list(vs)
    print("   cos between v_NP of directions (%s): " % " ".join(ks) + " | ".join(" ".join("%+.2f" % (float(vs[a] @ vs[b]) / float(vs[a].norm() * vs[b].norm())) for b in ks) for a in ks))
