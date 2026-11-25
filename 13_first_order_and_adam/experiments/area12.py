import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", 2, OUT + "/pre12_out")
ctx = L.Ctx9(S.COMBOS["tanh-init-s0"], B + "/run"); st = ctx.st; s = CFG["S1_FRAC"] * ctx.S0
for nm in ("label", "rand"):
    u = ctx.dirs[nm]; a = st.Q.T @ u; lam = st.lam
    T = (lam[:, None] - lam[None, :]) / (lam[:, None] + lam[None, :])
    A = -s * s * a[:, None] * a[None, :] * T          # predicted antisymmetric part of the loop look matrix (eigenbasis)
    print(nm, "predicted |antisym| for a gradient-flow out-and-back: %.4e  (s^2 = %.4e ; ratio %.4f)" % (float(torch.linalg.matrix_norm(A)), s * s, float(torch.linalg.matrix_norm(A)) / (s * s)))
