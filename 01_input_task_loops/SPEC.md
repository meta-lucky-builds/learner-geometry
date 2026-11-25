# Task-cycle rotation (geometric phase of learning) — experiment specification

All arithmetic is float64 (`torch.set_default_dtype(torch.float64)`), CPU only. Every random
object is drawn from a `torch.Generator` seeded as stated. No stochastic training anywhere.

## 1. Question

A model learns task A to convergence, then other tasks, then A again. It is back at zero loss
on A, but at a different A-solution. Claim under test: the order-dependent part of that
displacement behaves like a geometric phase (holonomy):

- it needs a loop that encloses area in task space (A→B→C→A), and is absent or much smaller
  for a retraced path that visits the same tasks (A→B→A→C→A);
- it scales with the enclosed area (∝ ε² for a loop of linear size ε, ∝ sin γ in the angle
  between the two displacement directions);
- it adds when the loop is repeated and cancels when the loop is then run backwards.

These were verified in a model linear in its parameters. The experiment asks whether they hold
in a nonlinear MLP, and how they change from wide (near-linear) to narrow networks.

## 2. Objects

### 2.1 Student network f(x; θ)
MLP, input dimension d = 8, two hidden layers of width h, scalar output, tanh, NTK
parametrisation:

    z1 = W1 x / sqrt(d)  + b1          W1 ∈ R^{h×d}
    z2 = W2 tanh(z1) / sqrt(h) + b2    W2 ∈ R^{h×h}
    f  = w3 · tanh(z2) / sqrt(h) + b3  w3 ∈ R^{h}

Initial values: every entry of W1, W2, w3 ~ N(0,1); b1, b2, b3 = 0. θ ∈ R^P is the flat
concatenation (W1, b1, W2, b2, w3, b3), P = h·d + h + h² + h + h + 1. All weights and biases
are trained. Widths: h ∈ {32, 128, 512}.

### 2.2 Teacher
Fixed function g(x) = v · tanh(M x / sqrt(d) + c) / sqrt(16), with M ∈ R^{16×8}, c ∈ R^{16},
v ∈ R^{16}, all entries ~ N(0,1). Drawn once per seed.

### 2.3 Tasks
n = 16 inputs per task. With ‖·‖_F the Frobenius norm and ⟨M,N⟩ = Σ_ij M_ij N_ij:

    X_A ∈ R^{n×d},  entries ~ N(0,1)
    U_raw, V_raw ∈ R^{n×d}, entries ~ N(0,1)
    U  = U_raw · ‖X_A‖_F / ‖U_raw‖_F
    V' = V_raw − (⟨V_raw,U⟩/⟨U,U⟩) U ;   U⊥ = V' · ‖X_A‖_F / ‖V'‖_F
    (so ‖U‖_F = ‖U⊥‖_F = ‖X_A‖_F and ⟨U,U⊥⟩ = 0)

    X_B(ε)        = X_A + ε U
    X_C(ε, γ)     = X_A + ε (cos γ · U + sin γ · U⊥)        "generic" family
    X_C^col(ε, α) = X_A + α ε U                              "collinear" family

Labels for every task T: y_T = g(X_T) (teacher applied row-wise). Task loss
L_T(θ) = (1/2n) ‖f(X_T; θ) − y_T‖². Residual r_T(θ) = f(X_T; θ) − y_T ∈ R^n.

### 2.4 Probe set
X_probe ∈ R^{256×8}, entries ~ N(0,1), drawn once per seed. Never trained on.

## 3. "Learn task T to convergence": two learners

Both map (θ_in, T) → θ_out with ‖r_T(θ_out)‖_∞ ≤ tol.

### 3.1 GN learner (primary; exact, fast)
Iterated minimum-norm Gauss–Newton projection. With J = ∂f(X_T; θ)/∂θ ∈ R^{n×P} at the
current θ and r = r_T(θ):

    Δ = Jᵀ (J Jᵀ)⁻¹ r         (solve the n×n system with torch.linalg.solve; no regulariser)
    θ ← θ − s Δ,  s = 1; if ‖r_T(θ − sΔ)‖₂ ≥ ‖r‖₂ halve s (up to 30 halvings)

Stop when ‖r_T(θ)‖_∞ ≤ 1e-12. At most 100 iterations; if not converged, raise and mark the
condition INVALID (never silently continue). For a model linear in θ this is exactly the
Euclidean orthogonal projection onto {θ : f(X_T;θ) = y_T}, in one step.
Compute J with `torch.func.jacrev` + `functional_call` (or an equivalent exact method), and
unit-test it against finite differences once (relative error < 1e-6).

### 3.2 GD learner (validation that the effect is not an artefact of GN)
Plain full-batch gradient descent on L_T, constant step size
lr = 1.0 · n / λ_max(J Jᵀ) with J evaluated at θ_in (λ_max = largest eigenvalue of the n×n
Gram matrix). Stop when ‖r_T(θ)‖_∞ ≤ 1e-8. Cap 400,000 steps. If the cap is hit, record the
achieved ‖r‖_∞ and mark that condition NOT CONVERGED; do not discard it silently and do not
use it in fitted slopes. No momentum, no weight decay, no minibatches.

## 4. Sequences and measured quantities

Base point: θ_A = Learn(θ_init, A). For a sequence s = (T₁,…,T_k):
θ_s = Learn(…Learn(Learn(θ_A, T₁), T₂)…, T_k). Every sequence ends in A.

    weight displacement    δ_s = θ_s − θ_A                       ∈ R^P
    function displacement  φ_s = f(X_probe; θ_s) − f(X_probe; θ_A) ∈ R^256

Sequences (written as the tasks learned after the base point):

    retrace  = (B, A)
    loop+    = (B, C, A)          loop−  = (C, B, A)
    star+    = (B, A, C, A)       star−  = (C, A, B, A)
    ff       = (B, C, A, B, C, A)             (loop forward twice)
    fr       = (B, C, A, C, B, A)             (loop forward, then backward)

Derived vectors, each computed both in weight space (from δ) and in function space (from φ):

    a_loop = (δ_loop+ − δ_loop−) / 2      order-antisymmetric part of the loop (the "phase")
    s_loop = (δ_loop+ + δ_loop−) / 2      order-symmetric part (the "shrinkage")
    a_star = (δ_star+ − δ_star−) / 2      order-antisymmetric part of the retraced star

Scalars reported (‖·‖ = Euclidean norm; use weight space as primary, function space alongside):

    N_a = ‖a_loop‖,  N_s = ‖s_loop‖,  N_star = ‖a_star‖,  N_ret = ‖δ_retrace‖
    E_add    = ‖(δ_ff − δ_fr) − 2 a_loop‖ / ‖2 a_loop‖        (phases add / cancel)
    E_shrink = ‖δ_fr − 2 s_loop‖ / ‖2 s_loop‖
    tang     = ‖J_A a_loop‖₂ / (‖J_A‖₂ · ‖a_loop‖₂)   with J_A at θ_A, ‖J_A‖₂ the spectral norm
    lin      = max over tasks T∈{A,B,C} of ‖J_T(θ_loop+) − J_T(θ_A)‖_F / ‖J_T(θ_A)‖_F
               (how far the network is from the linear regime during the cycle)

### 4.1 Linearised prediction (comparator)
Freeze the Jacobians at θ_A: J_T⁰ = J_T(θ_A), r_T⁰ = r_T(θ_A) for T ∈ {A,B,C}. Linear learner:

    Learn_lin(θ, T) = θ − J_T⁰ᵀ (J_T⁰ J_T⁰ᵀ)⁻¹ ( r_T⁰ + J_T⁰ (θ − θ_A) )

Run loop+ and loop− with Learn_lin from θ_A to get a_loop^lin. Report

    cos_lin = ⟨a_loop, a_loop^lin⟩ / (‖a_loop‖ ‖a_loop^lin‖),   ratio_lin = ‖a_loop‖ / ‖a_loop^lin‖

## 5. Predictions (state each as PASS / FAIL / INCONCLUSIVE with the numbers)

Fit slopes by ordinary least squares of log(quantity) on log(ε), per seed, then report the
median and the min–max range over seeds. Use only valid (converged) points.

| id | prediction | linear-model value |
|----|------------|--------------------|
| H1 | N_a ∝ ε^p with p ≈ 2 (generic, γ = π/2) | 1.8–2.0 |
| H2 | N_star ∝ ε^q with q ≈ 4, so N_star / N_a → 0 as ε → 0 | 3.8–4.0 |
| H3 | collinear loops (α = 2 and α = −1): N_a ∝ ε^p with p ≈ 3, and N_a^col / N_a^generic → 0 | 2.7–3.0 |
| H4 | at fixed ε = 0.05, N_a(γ) / N_a(π/2) ≈ sin γ for γ ∈ {π/6, π/4, π/3, 2π/3} | within 0.02 |
| H5 | E_add → 0 and E_shrink → 0 as ε → 0 (slope ≈ 2) | 0.25 → 0.003 |
| H6 | tang ≪ 1 (the phase is tangent to A's solution set) | — |
| H7 | N_ret ∝ ε² and N_s ∝ ε² (shrinkage is second order too; reference only) | 2.0 |

Reference numbers from the linear-in-θ pre-check (random-feature model, n=16, d=8, 400
features, ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}): N_a = [0.253, 0.094, 0.027, 0.007, 0.002];
N_star = [4.8e-2, 4.9e-3, 3.6e-4, 2.3e-5, 1.5e-6]; angle sweep ratios
[0.517, 0.724, 0.879, 1.0, 0.864] against sin γ = [0.5, 0.707, 0.866, 1.0, 0.866].

A FAIL is a real result. Do not tune anything to make a prediction pass. If a prediction
fails, report the measured exponent and stop there; do not explain it away.

## 6. Experiment grid

Seeds: 5 (seed ∈ {0,1,2,3,4}); a seed fixes teacher, X_A, U, U⊥, X_probe and θ_init
(θ_init is drawn per (seed, h)).

- G1 (main, GN learner): h ∈ {32, 128, 512} × ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125} × generic
  γ = π/2 × all seven sequences × 5 seeds. All scalars of §4, plus §4.1.
- G2 (collinear, GN): h ∈ {32, 128, 512} × same ε × α ∈ {2, −1} × loop± × 5 seeds.
- G3 (angle, GN): h ∈ {32, 128, 512}, ε = 0.05, γ ∈ {π/6, π/4, π/3, π/2, 2π/3} × loop± × 5 seeds.
- G4 (GD validation): h = 128, ε ∈ {0.2, 0.1}, γ = π/2, sequences loop±, star±, retrace,
  seeds {0,1,2}. Report N_a, N_star, N_ret and, for the same conditions, the GN values and
  cos(a_loop^GD, a_loop^GN). If G4 exceeds 40 minutes of wall time, stop it, report what
  finished, and say so.

## 7. Required tests (must pass before any experiment is run)

`test_linear.py`, using the same sequence/measurement code with a model linear in θ
(f = φ(x)·θ, φ(x) = tanh(Wx/√d + b)/√400, 400 fixed random features, labels from a random
θ_true):

1. Jacobian of the MLP vs central finite differences: relative error < 1e-6.
2. One GN step on the linear model reaches ‖r‖_∞ < 1e-10 and equals the closed-form projection.
3. δ for the empty continuation (sequence (A) from θ_A) has norm < 1e-10.
4. Loop reversal: restricted to range of P_A, the operator of loop− is the transpose of the
   operator of loop+ (build explicit projectors P_T = I − J_T⁺ J_T; check
   ‖P_A P_B P_C P_A − (P_A P_C P_B P_A)ᵀ‖ < 1e-10).
5. Slopes on ε ∈ {0.2,…,0.0125}: N_a in [1.6, 2.2], N_star in [3.4, 4.3], collinear α=2 in
   [2.5, 3.3].
6. With C = B (γ = 0): N_a < 1e-10 and N_star < 1e-10.
7. Learn_lin (§4.1) applied to the linear model reproduces the GN learner to < 1e-9.

## 8. Deliverables (all inside this folder, `task_cycle_rotation/`)

- `tcr.py` — model, tasks, learners, sequence runner, measurements. No experiment logic.
- `test_linear.py` — §7, runnable as `python test_linear.py`, prints PASS/FAIL per test.
- `run.py` — runs G1–G4, writes `results.json` (one record per condition, every scalar of §4,
  convergence flags, iteration counts, wall time) incrementally so a crash loses nothing.
- `report.py` — reads `results.json`, prints one table per prediction H1–H7 (per width:
  median and min–max over seeds), the G4 comparison, and writes `REPORT.md` with the same
  tables and a PASS/FAIL/INCONCLUSIVE line per prediction per width. Plots only if
  matplotlib is already installed; do not install packages.

Final message back: the test output, the H1–H7 tables, every INVALID / NOT CONVERGED
condition, total wall time, and anything in this spec you found ambiguous or had to decide.
