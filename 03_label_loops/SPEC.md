# Part 3 — label-space loops: the curvature of the network itself

Conventions as in SPEC.md / SPEC2.md: float64, CPU, deterministic, seeded `torch.Generator`.
Reuse `tcr.py` and `tcr2.py` by import. Do NOT modify any existing file. New files only (§9).

## 0. What is being tested, in one paragraph

Inputs are FIXED at X = X_A (n = 16 points). Only the label vector y ∈ R^n changes from task
to task. For a model linear in its weights, learning a sequence of label vectors and returning
to the first leaves the weights exactly where they started. For a real MLP it does not. Two
claims, each with an exact formula:

- (L) a closed loop of label vectors rotates the weights by an amount equal to the loop's
  signed area times a Lie bracket of two vector fields built from the network's Jacobian;
- (K) an out-and-back excursion of the labels moves the weights, at fixed outputs on X, in the
  direction that makes the network's tangent kernel WEAKER along the excursion direction;
  refining a label path into smaller steps moves them the opposite way.

Both effects are exactly zero for a model linear in θ. Everything is at fixed function values
on X at the start and end, so every displacement measured here lies along the set of weights
that fit y_A exactly.

## 1. Objects

Student MLP f(x;θ), widths h, teacher g, per-seed X_A, probe set: exactly SPEC.md §2
(`tcr.TaskFamily(seed)`, `tcr2.FastMLP(h)`, `model.init(seed)`).

    X      = X_A                       (fixed for every task in this spec)
    y_A    = g(X_A) ∈ R^16
    θ_A    = Learn(θ_init, (X, y_A))                      (GN learner, SPEC.md §3.1)
    J(θ)   = ∂f(X;θ)/∂θ ∈ R^{16×P},   K(θ) = J Jᵀ ∈ R^{16×16}
    P_v(θ) w = w − Jᵀ K⁻¹ J w          ("vertical" projector: onto directions that do not change f on X)

Label directions, per seed: from `torch.Generator().manual_seed(555 + seed)` draw u_raw then
v_raw, each 16 entries ~ N(0,1);

    v' = v_raw − (v_raw·u_raw / u_raw·u_raw) u_raw
    u  = u_raw · ‖y_A‖ / ‖u_raw‖ ,   v = v' · ‖y_A‖ / ‖v'‖         (so ‖u‖ = ‖v‖ = ‖y_A‖, u·v = 0)

A "label task at plane point (x,y) with size ε" is (X, y_A + ε (x u + y v)). The point (0,0)
is task A. "Learn" always means the GN learner unless a section says GD.

Vector fields. For a fixed label direction e ∈ R^16:

    ξ_e(θ) = J(θ)ᵀ K(θ)⁻¹ e                                  (min-norm weight step that moves f(X) by e)

Finite-difference directional derivative (use exactly this):

    Dξ_e·w (θ) = [ ξ_e(θ + s w) − ξ_e(θ − s w) ] / (2s),   s = 1e-4 / ‖w‖

    bracket     b_{uv}(θ) = Dξ_v·ξ_u (θ) − Dξ_u·ξ_v (θ)       (Lie bracket [ξ_u, ξ_v])
    self-term   c_u(θ)    = P_v(θ) [ Dξ_u·ξ_u (θ) ]
    stiffness   φ_e(θ)    = ½ eᵀ K(θ)⁻¹ e                     (identity: P_v ∇φ_u = − c_u)

## 2. Part L — loops (order-odd displacement)

For a vertex list `pts` in the label plane: forward = learn pts in order, then A; reverse =
learn pts in reversed order, then A. δ₊, δ₋ = final θ minus θ_A.

    a = (δ₊ − δ₋)/2 ,  s = (δ₊ + δ₋)/2          (and the same from probe outputs: a_fn, s_fn)

Polygons (same definitions as SPEC2.md §F.2 but in the LABEL plane): tri = [(1,0),(0,1)];
circ(N); rect(c) = [(1,0),(1,c),(0,c)]; fig8 = [(1,0),(0,1),(0,0),(−1,0),(0,1)];
star: forward = [(1,0),(0,0),(0,1)], reverse = [(0,1),(0,0),(1,0)] (each followed by the final A).

### Predictions

| id | statement | pass rule (median over seeds, per width) | pre-check |
|----|-----------|------|------|
| L1 | ‖a_tri‖ ∝ ε² | slope of log‖a_tri‖ on log ε over all 5 ε in [1.8, 2.2] | 3 seeds: ×4.0 per halving at h=512 |
| L2 | a_tri = +(ε²/2) · b_{uv}(θ_A) as VECTORS | at ε = 0.0125: cos(a_tri, b_uv) ≥ 0.99 AND ‖a_tri‖ / ((ε²/2)‖b_uv‖) ∈ [0.97, 1.03] | ε=0.025: cos 0.994–0.9995, ratio 0.989–1.002 (h=32,128; 2 seeds) |
| L3a | ‖a_circ(N)‖/‖a_tri‖ = N sin(2π/N) | at ε = 0.025, every N ∈ {3,4,8,16,32}: within 5% | h=32: 3.974, 5.628, 6.091 for N=4,8,16 |
| L3b | ‖a_rect(c)‖/‖a_tri‖ = 2c, c ∈ {0.5,1,2} | at ε = 0.025: within 5% | rect(2): 4.090 (h=32), 4.052 (h=128) |
| L3c | fig8: ‖a_fig8‖/‖a_tri‖ → 0 | slope of log ratio on log ε > 0.5 | none |
| L3d | star: ‖a_star‖ ∝ ε⁴ | slope in [3.4, 4.4] | none |
| L3e | cos(a_poly, a_tri) → 1 for circ and rect | at ε = 0.025: ≥ 0.98 | 0.990 (h=32), 0.999 (h=128) |
| L4 | ‖b_uv(θ_A)‖ and ‖a_tri‖(ε=0.05) fall with width | REPORT the fitted exponent of each on h (log-log, all widths). No pass rule. | rough: halves per 4× width |
| L5 | linear control: with `tcr.LinearModel(seed)` in place of the MLP (labels y_A = model.labels(X), same u, v), ‖a_tri‖ < 1e-10 at ε = 0.2 | must hold | 2e-14 |

### L6 — where the curvature lives (no pass rule; report)

At θ_A let K = Σ_i λ_i q_i q_iᵀ (λ_1 ≥ … ≥ λ_16, unit q_i). For all pairs i < j compute

    B_ij = ‖ b_{q_i q_j}(θ_A) ‖ ,   κ_ij = B_ij / ( ‖ξ_{q_i}‖ ‖ξ_{q_j}‖ ) = B_ij · sqrt(λ_i λ_j)

Report, per width (median over seeds): the mean of κ_ij and of B_ij over three blocks —
top×top (i,j ≤ 4), bottom×bottom (i,j ≥ 13), top×bottom (i ≤ 4, j ≥ 13) — and the Spearman
correlation of κ_ij with (i + j) and of B_ij with (i + j). Save the full 16×16 median matrices
in results3.json.

### L7 — does the loop direction change the kernel and the features? (report)

For tri at each ε, with θ₊, θ₋ the forward and reverse endpoints:

    kernel odd change   ‖K(θ₊) − K(θ₋)‖_F / ‖K(θ_A)‖_F
    kernel even change  ‖(K(θ₊) + K(θ₋))/2 − K(θ_A)‖_F / ‖K(θ_A)‖_F
    probe output odd change per input   ‖a_fn‖ / 16
    feature similarity  linear CKA between last-hidden-layer activations tanh(z2) on X_probe
                        (256×h matrices, column-centred) for the pairs (θ₊, θ₋) and (θ_A, θ₊)

Report slopes in ε of the first three, and 1 − CKA for both pairs.

### L8 — many cycles (report)

circ(16) at ε = 0.05, m ∈ {1,2,4,8,16,32,64} forward cycles vs m reverse cycles:
‖a_m‖/(m‖a_1‖), cos(a_m, a_1), ‖s_m‖, and tr K⁻¹, λ_min(K), φ_u at the forward endpoint.

## 3. Part K — the kernel ratchet (order-even displacement)

### K1 — out-and-back law
Sequence: learn (X, y_A + εu), then learn A. d_ret = θ − θ_A.

| id | statement | pass rule |
|----|-----------|-----------|
| K1a | d_ret = − ε² · c_u(θ_A) as vectors | at ε = 0.0125: cos(d_ret, c_u) ≤ −0.99 AND ‖d_ret‖/(ε²‖c_u‖) ∈ [0.95, 1.05] |
| K1b | φ_u increases: φ_u(θ_ret) − φ_u(θ_A) > 0 | every seed, every width, every ε |
| K1c | φ_u(θ_ret) − φ_u(θ_A) = ε² ‖c_u‖² | at ε = 0.0125: ratio in [0.9, 1.1] |

Pre-check (h = 32, 128; 2 seeds; ε = 0.025): cos −0.9993 to −0.9998, ratio 0.993–1.019,
φ_u rose in every case.

### K2 — repeated excursions (report + one rule)
From θ_A, repeat m times: learn (X, y_A + εu), learn A. ε = 0.1. Record at
m ∈ {1,2,4,8,16,32,64}: φ_u, φ_v, tr K⁻¹, λ_min(K), λ_max(K), ‖θ‖, ‖θ − θ_A‖, and probe
output change per input ‖f(X_probe;θ) − f(X_probe;θ_A)‖/16.
Rule K2: φ_u is strictly increasing in m in at least 90% of (seed, width) pairs.
Pre-check (seed 0): h=32 φ_u ×2.49, tr K⁻¹ ×1.58, λ_min ×0.50 at m=32; h=128 ×1.18, ×1.06, ×0.92.

### K3 — random excursions (report + one rule)
As K2 but each excursion uses a fresh random direction e_k: from
`Generator(99 + seed)` draw 16 entries ~ N(0,1), scale to ‖y_A‖. ε = 0.1, m up to 128, record
the K2 quantities (φ_u, φ_v for the fixed u, v) at m ∈ {1,2,4,8,16,32,64,128}.
Rule K3: tr K⁻¹ at m = 128 exceeds its starting value in every (seed, width).
Pre-check (seed 0, m = 64): h=32 tr K⁻¹ ×5.3, λ_min ×0.098; h=128 ×1.27, ×0.73.

### K4 — path refinement
Total label move L·u, done in N equal steps: learn (X, y_A + (kL/N) u) for k = 1..N.
θ^{(N)} is the endpoint; θ^{(1)} is the single jump. d_sub(N) = θ^{(N)} − θ^{(1)}.

| id | statement | pass rule |
|----|-----------|-----------|
| K4a | d_sub(N) = + (L²/2)(1 − 1/N) · c_u(θ_A) | at L = 0.05, N ∈ {2,4,16}: cos ≥ 0.99, ratio in [0.95, 1.05] |
| K4b | φ_u(θ^{(N)}) < φ_u(θ^{(1)}) | every seed, width, N at L ∈ {0.1, 0.05} |

Pre-check (seed 0, L = 0.05): cos 0.998–0.9995, ratio 0.986–1.011.

### K5 — is the stiffening real for a learner? (GD, report + one rule)
Widths h ∈ {32, 64}, seeds {0,1,2}. Start states: S0 = θ_A; S1 = the K3 state after 128 random
excursions (recompute it with the GN learner, same directions). From each state, with the GD
learner of SPEC.md §3.2 (tolerance 1e-8, cap 400,000 steps), fit three fresh targets
(X, y_A + 0.2·t_j), t_j from `Generator(4242 + seed)`, 16 entries ~ N(0,1) scaled to ‖y_A‖,
j = 1,2,3. Record the number of GD steps to converge. NOTE: the GD step size is recomputed
from λ_max at each start state, as §3.2 says; also record λ_max and λ_min at both states.
Rule K5: median over (seed, j) of steps(S1)/steps(S0) > 1 at both widths.
If a run hits the cap, record it as "> cap" and count the ratio using the cap.

### K6 — linear control
With `tcr.LinearModel(seed)`: d_ret, d_sub(4) and the K2 displacement at m = 8 all have norm
< 1e-10. Must hold.

## 4. Part G — the same two laws under plain gradient descent

h = 128, seeds {0,1,2}, ε ∈ {0.1, 0.05}, GD learner for every learn (including θ_A).
Compute b_uv and c_u at the GD base point θ_A^GD.
Report cos(a_tri^GD, b_uv), ‖a_tri^GD‖/((ε²/2)‖b_uv‖), cos(d_ret^GD, c_u), ‖d_ret^GD‖/(ε²‖c_u‖).
No pass rule (GD is not an exact projection; the question is how close it is). 40-minute cap.

## 5. Part M — one exploratory mixed loop (report only)

tri where vertex B moves the INPUTS and vertex C moves the LABELS:
B = (X_A + ε U, g(X_A + ε U)) with U from `tcr.TaskFamily`; C = (X_A, y_A + ε v).
Widths {32, 128, 512}, all 5 ε, 5 seeds. Report ‖a‖ and its slope in ε, and ‖a‖ relative to
the pure-input tri (SPEC.md generic γ = π/2 loop) and the pure-label tri at the same ε.

## 6. Grid

Seeds {0,1,2,3,4}. Widths h ∈ {32, 64, 128, 256, 512}. ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}
for every ε-sweep (L1, L2, L3c, L3d, L7, K1, M). L3a/L3b/L3e at ε ∈ {0.05, 0.025}.
K4 at L ∈ {0.1, 0.05}, N ∈ {2, 4, 8, 16}.
If any single grid would exceed 60 minutes, stop it, report what finished, and say so.

## 7. Tests that must pass before any grid is run (`test_part3.py`)

1. `ξ_e` check: ‖J ξ_e − e‖_∞ < 1e-10 at θ_A (h = 32, seed 0, e = u).
2. Bracket is vertical: ‖J b_uv‖₂ / (‖J‖₂ ‖b_uv‖) < 1e-6 at θ_A (h = 32 and 128, seed 0).
3. Identity P_v∇φ_u = −c_u: directional derivative of φ_u along c_u, by central finite
   difference with step 1e-4/‖c_u‖, equals −‖c_u‖² to relative error < 1e-3 (h = 32, seed 0).
4. Finite-difference step sanity: b_uv computed with s = 1e-4/‖w‖ and with s = 1e-5/‖w‖ agree
   to relative error < 1e-4 (h = 32, seed 0).
5. Linear controls L5 and K6.
6. Reproduce the pre-check, h = 32, seed 0, ε = 0.025: cos(a_tri, b_uv) > 0.99 and
   ratio in [0.97, 1.03]; cos(d_ret, c_u) < −0.99 and ratio in [0.95, 1.05].
7. Every sequence ends with ‖f(X;θ) − y_A‖_∞ ≤ 1e-12 (check inside the runner; assert).

Tests 2, 3 and 6 encode results verified in a separate implementation. If one fails, the bug
is in the code until proven otherwise. Do not loosen a test; report the evidence instead.

## 8. Reporting rules

- Verdict per prediction per width: PASS if the rule holds for the median over seeds AND for
  at least 4 of 5 seeds; INCONCLUSIVE if it holds for the median only; FAIL otherwise. Give the
  per-seed values wherever a verdict is not PASS.
- A FAIL is a valid result. Never tune seeds, ε, tolerances or step sizes to make a
  prediction pass. Never drop a condition without listing it.
- Report-only items: give the numbers; add no interpretation.
- Flag anything in the data that looks anomalous or unexpected (non-monotone trends, a width
  or seed that behaves differently, sign flips), stating only what was measured.

## 9. Deliverables (in `task_cycle_rotation/`)

`tcr3.py` (fields ξ, bracket, self-term, stiffness, label-plane runner, kernel and CKA
measures), `test_part3.py`, `run3.py` (writes `results3.json` incrementally), `report3.py`
(writes `REPORT3.md`: one table per prediction with the predicted value next to the measured
one, per width, median and min–max over seeds; the report-only tables; verdicts by the §8 rule).

Final message back: `test_part3.py` output verbatim; every table; verdicts; every INVALID or
NOT CONVERGED condition; anomalies; wall time; and every ambiguity or decision you made.
