import torch, sys
NM = ["DJ(0.5)", "DJ(1.5)", "SL(2)", "SL(4)", "SL(8)"]
def coef(nm):
    if nm.startswith("DJ"):
        c = float(nm[3:-1]); q = 1 - c; p2 = c / (2 - c); p3 = c * c / (3 - 3 * c + c * c)
        return [(1 - c) / (2 - c), (1 - 3 * p2 + 2 * p3) / 6, 0.5 * (1 - 2 / (2 - c) + c / (1 - q ** 3))]
    m = int(nm[3:-1]); return [0.5 * (1 - 1 / m), (1 - 1 / m) * (1 - 2 / m) / 6, (m - 1) * (2 * m - 1) / (12 * m * m)]
for fn in sys.argv[1:]:
    b = torch.load(fn, map_location="cpu", weights_only=False); o = b["outs"]
    D = torch.stack([(o[n] - o["DJ(1)"]).reshape(-1) for n in NM])            # 5 x 100000
    nr = D.norm(dim=1); print("\n%s tau %g   |D| rms: %s" % (b["combo"], b["tau"], " ".join("%s %.3e" % (n, float(x) / D.shape[1] ** 0.5) for n, x in zip(NM, nr))))
    G = D @ D.T; cs = G / (nr[:, None] * nr[None, :]); print(" cosines between difference vectors:"); [print("   %-8s " % NM[i] + " ".join("%+.4f" % float(cs[i, j]) for j in range(5))) for i in range(5)]
    sv = torch.linalg.svdvals(D); print(" singular values / largest: " + " ".join("%.4f" % float(x / sv[0]) for x in sv))
    Dn = D / nr[:, None]; svn = torch.linalg.svdvals(Dn); print(" same after normalising each vector: " + " ".join("%.4f" % float(x / svn[0]) for x in svn))
    Cf = torch.tensor([coef(n) for n in NM], dtype=D.dtype)                   # columns e2, e3, b
    tot = float((D * D).sum())
    for name, cols in [("one function, weight e2", Cf[:, :1]), ("one function, weight e2+e3 (diagonal-model law)", (Cf[:, 0] + Cf[:, 1])[:, None]), ("one function, weight e3", Cf[:, 1:2]), ("one function, weight b", Cf[:, 2:3]),
                       ("two functions, weights e2 and e3", Cf[:, :2]), ("two functions, weights e2 and b", Cf[:, [0, 2]]), ("three functions e2, e3, b", Cf)]:
        A = torch.linalg.lstsq(cols, D).solution; R = D - cols @ A
        print("   %-48s residual %.4f of sum|D|^2 ; per learner %s" % (name, float((R * R).sum()) / tot, " ".join("%.3f" % float((R[i] * R[i]).sum() / (D[i] * D[i]).sum()) for i in range(5))))
    print(" norm ratios SL(2):SL(4):SL(8) = %.3f : %.3f : 1   DJ(1.5)/SL(8) = %.3f (cos %+.4f)   DJ(0.5)/SL(8) = %.3f | e2: .571:.857:1, -2.286, .762 | e2+e3: .457:.800:1, -2.438, .697" % (
        float(nr[2] / nr[4]), float(nr[3] / nr[4]), float(nr[1] / nr[4]), float(cs[1, 4]), float(nr[0] / nr[4])))
