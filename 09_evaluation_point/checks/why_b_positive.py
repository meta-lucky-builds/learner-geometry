"""Why is b > 0 for every direction? Hypothesis: the nearest-point shift v_NP always contains a growth of the LAST layer
(compensated on the training inputs), which makes the function larger on unseen inputs. Small tanh MLP (n = 8), local."""
import sys
from common9 import load
L, CFG = load("tiny")
import torch
vn = L.vnorm
sm = L._small_mlp("tanh")
prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
th0 = st.th0
td = L.get_test_data(0, 8)
Ft = lambda th: L.test_outputs(prob, td["X"], th).reshape(-1)
F0 = Ft(th0)
C = CFG["N_CLASSES"]
cen = lambda x: (x.view(-1, C) - x.view(-1, C).mean(0, keepdim=True)).reshape(-1)
F0v = cen(F0)
r0 = F0 - td["Y"].reshape(-1)
cos = lambda a, b: float(a @ b) / max(vn(a) * vn(b), 1e-300)
sl, o = [], 0
for wo, wi in prob.shapes:
    sl.append((o, o + wo * wi, o + wo * wi + wo))
    o += wo * wi + wo
U = []
for a, b, c in sl:                       # scaling direction of each layer's weight matrix (biases not scaled)
    u = torch.zeros_like(th0)
    u[a:b] = th0[a:b]
    U.append(u)
VU = [st.V0(u) for u in U]
eps = 1e-6
GU = [(Ft(th0 + eps * w) - Ft(th0 - eps * w)) / (2 * eps) for w in VU]     # held-out change per unit projected layer scaling
print("projected scaling of layer l (training outputs fixed to first order): share of the scaling direction that survives the projection, "
      "and cosine of its held-out change with the centred start outputs")
for l in range(3):
    print("   layer %d: |V0 u|/|u| %.3f   cos(J_test V0 u, F0v) %+.3f" % (l + 1, vn(VU[l]) / vn(U[l]), cos(GU[l], F0v)))
print("\ndir    | share of |v_NP|^2 in layer 1/2/3 | weight-norm growth per unit f, layer 1/2/3 (relative) | cos(v_NP, V0 u_l) l=1/2/3 | "
      "held-out: e, cos(G,F0v) | b total | b from the part of v_NP along V0 u_3 | along span(V0 u_1..3)")
for dn in ("eigA", "eigB", "eigC", "eigD", "label", "rand"):
    d = sm["dirs"][dn]
    s = 0.01 * S0
    vs = []
    for sg in (1, -1):
        res = L.run_learner9(prob, st, st.F0 + sg * s * d, s, L.L_AJ(1.0))
        assert res["status"] == "CONVERGED", res["status"]
        vs.append(st.V0(res["theta"] - th0))
    v = 0.5 * (vs[0] + vs[1])
    G = (Ft(th0 + v) - Ft(th0 - v)) / 2
    N = G.numel()
    share = [float((v[a:c] ** 2).sum() / (v @ v)) for a, b, c in sl]
    grow = [float(v @ U[l]) / float(U[l] @ U[l]) for l in range(3)]
    cs = [cos(v, VU[l]) for l in range(3)]
    c3 = float(v @ VU[2]) / float(VU[2] @ VU[2])
    b3 = c3 * float(r0 @ GU[2]) / N
    A = torch.stack(VU, 1)
    co = torch.linalg.lstsq(A, v.unsqueeze(1)).solution.squeeze(1)
    b123 = sum(float(co[l]) * float(r0 @ GU[l]) for l in range(3)) / N
    print("%-6s | %.3f %.3f %.3f | %+.2e %+.2e %+.2e | %+.3f %+.3f %+.3f | %+.2e %+.3f | %+.3e | %+.3e | %+.3e" % (
        dn, *share, *grow, *cs, float(cen(G) @ F0v) / float(F0v @ F0v), cos(G, F0v), float(r0 @ G) / N, b3, b123))
