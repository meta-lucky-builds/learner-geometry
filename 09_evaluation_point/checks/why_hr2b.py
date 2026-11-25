from common9 import load
L, CFG = load("tiny")
import torch
vn = L.vnorm
for act in ("relu", "tanh"):
    sm = L._small_mlp(act); prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
    s = 1e-2 * S0; d = -sm["dirs"]["eigA"]; t = st.F0 + s * d
    ref = L.RefPath(prob, st, d, s, 64); vPP, om, omp = L.R_profile(ref)
    log = []; L.run_DJ_logged(prob, st, t, 0.5, log, keep_points=True)
    b = log[1]["b"]; tau = float(d @ (log[1]["Fe"] - st.F0)) / s; beta = float(d @ b) / s
    b_on = beta * s * d; b_off = b - b_on
    g = torch.Generator().manual_seed(1); R = torch.randn(prob.N, 6, generator=g, dtype=L.DT); R = R / R.norm(dim=0)
    X, _ = ref.W(tau, torch.cat([torch.stack([b_on, b_off], 1), R], 1))
    print(act, "step 1 at tau %.3f: |W b_off|/|W b_on| = %.3f ; cos(W b_on, vPP) = %.4f ; cos(W b_off, vPP) = %.4f" % (
        tau, vn(X[:, 1]) / vn(X[:, 0]), L.cosine(X[:, 0], vPP), L.cosine(X[:, 1], vPP)))
    print("    6 random unit output vectors: |cos(W r, vPP)| =", " ".join("%.3f" % abs(L.cosine(X[:, 2 + j], vPP)) for j in range(6)),
          "| |W r| / |W d| =", " ".join("%.1f" % (vn(X[:, 2 + j]) / (vn(X[:, 0]) / abs(beta * s))) for j in range(6)))
    sv = torch.linalg.svdvals(ref.W(tau, torch.eye(prob.N, dtype=L.DT))[0])
    print("    singular values of the field W(tau) (top 5, normalised):", " ".join("%.3f" % float(x / sv[0]) for x in sv[:5]))
