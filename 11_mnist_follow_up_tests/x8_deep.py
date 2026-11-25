import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import torch, glob, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("xr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "x8_read.py"))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "x8_read.py")).read().split("EXPL = ")[0]; ns = {}; exec(src, ns); weights = ns["weights"]
OUT = OUT + "/x8_out"
NAMES = ["SL(2)", "SL(3)", "SL(4)", "SL(8)", "DJ(0.5)", "DJ(0.75)", "DJ(1.25)", "DJ(1.5)", "AJ(0.5)", "AJ(1)", "SLa(2,1)", "SLa(4,2)", "TJ(1.5,1)"]
key = lambda nm: nm.replace("(", "_").replace(")", "").replace(",", "_")
# MNIST test labels (same order for every start)
import loc
L, CFG = loc.load(threads=2)
td = L.get_test_data(0, 32); Y = td["Y"].reshape(-1); lab = td["labels"]; C = 10
FN = {}
for blk in ["tanh-init-s0_0.03", "tanh-init-s0_0.1", "tanh-init-s1_0.03", "tanh-init-s1_0.1", "gelu-init-s0_0.03", "gelu-init-s0_0.1"]:
    tau = float(blk.split("_")[1]); ld = lambda nm: torch.load("%s/%s_%s.pt" % (OUT, blk, key(nm)), map_location="cpu", weights_only=False)["out"].reshape(-1)
    ref = ld("DJ(1)"); D = torch.stack([ld(n) - ref for n in NAMES]); W = torch.tensor([weights(n) for n in NAMES], dtype=D.dtype)
    F = torch.linalg.lstsq(W, D).solution                      # rows: functions for A2, A3, B3
    FN[blk] = F; n = F.norm(dim=1); G = (F @ F.T) / (n[:, None] * n[None, :])
    r = ref - tau * Y                                           # residual of the reference learner on test (doubled net: start function 0)
    N = Y.numel(); slope = (F @ r) / N                          # d(test loss)/d(weight)
    # split of each function: rescaling of the learned function S = ref (centred per class column) and the rest
    cen = lambda o: (o.reshape(-1, C) - o.reshape(-1, C).mean(0, keepdim=True)).reshape(-1)
    Sv = cen(ref); eS = [float((cen(f) @ Sv) / (Sv @ Sv)) for f in F]
    drest = [float(((cen(f) - e * Sv) @ r) / N) for f, e in zip(F, eS)]
    # change with the weights along flow-centred coordinates
    print("\n%s  |function| rms: A2 %.3e  A3 %.3e  B3 %.3e   ratios A3/A2 %.3f  B3/A2 %.3f" % (blk, *(float(x) / math.sqrt(N) for x in n), float(n[1] / n[0]), float(n[2] / n[0])))
    print("   cosines: A2-A3 %+.4f  A2-B3 %+.4f  A3-B3 %+.4f" % (float(G[0, 1]), float(G[0, 2]), float(G[1, 2])))
    print("   test-loss slope per unit weight: A2 %+.3e  A3 %+.3e  B3 %+.3e | shrink coefficient eS: %+.3e %+.3e %+.3e | pattern part: %+.3e %+.3e %+.3e" % (*[float(x) for x in slope], *eS, *drest))
    a, b, c = [float(x) for x in slope]
    # best single alpha-point learner at this order: loss(alpha) = a alpha + b alpha^2 + c alpha^2 / 2
    q = b + c / 2
    print("   alpha-point family: dloss = %.3e a + %.3e a^2 -> %s; flow (1/2,1/6,1/6): %+.3e ; AJ(1): %+.3e ; SL(8): %+.3e ; DJ(1.5): %+.3e" % (
        a, q, ("minimum at alpha* = %.2f" % (-a / (2 * q)) if q > 0 else "no minimum (keeps falling)" if a < 0 else "maximum"), a / 2 + b / 6 + c / 6, a + b + c / 2,
        *[sum(x * y for x, y in zip(weights(nm), (a, b, c))) for nm in ("SL(8)", "DJ(1.5)")]))
    # accuracy change predicted to first order is not available; report measured accuracies
    acc = lambda o: float((o.reshape(-1, C).argmax(1) == lab).double().mean())
    print("   accuracy: DJ(1) %.4f | " % acc(ref) + " ".join("%s %+.4f" % (nm, acc(ld(nm)) - acc(ref)) for nm in ("DJ(1.5)", "SL(8)", "AJ(0.5)", "AJ(1)", "SLa(4,2)", "TJ(1.5,1)")))
print()
for a_, b_ in (("tanh-init-s0_0.03", "tanh-init-s0_0.1"), ("tanh-init-s1_0.03", "tanh-init-s1_0.1"), ("gelu-init-s0_0.03", "gelu-init-s0_0.1")):
    Fa, Fb = FN[a_], FN[b_]
    print("%s vs tau .1: same function shape? cos %s | norm ratio (expect (0.1/0.03)^3 = 37.0): %s" % (a_, " ".join("%+.4f" % float((Fa[i] @ Fb[i]) / (Fa[i].norm() * Fb[i].norm())) for i in range(3)),
          " ".join("%.1f" % float(Fb[i].norm() / Fa[i].norm()) for i in range(3))))
