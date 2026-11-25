# Laws

Every law found in this project, with its formula and how well it held. ✅ confirmed, 🟡 partial, ❌ failed or refuted.
The laws are not independent: 1-9 are one principle (where a learner evaluates its gradients decides where it lands)
in different forms, 10-12 extend it to third order, 13-24 are its consequences, and 25-30 are the task-loop side the
project started from. Details, numbers and caveats for each are in [FINDINGS.md](FINDINGS.md).

Notation: J Jacobian of the training outputs, K = JJᵀ tangent kernel, c = ηλ/n step size relative to a kernel
eigenvalue, GN one exact Gauss-Newton jump, NP nearest point of the solution set (fibre) to the start,
v_NP = NP − GN shift, f alignment fraction.

## Core laws

| # | Law | Formula | Status | Stage |
|---|---|---|---|---|
| 1 | Alignment-fraction law | learner lands at GN + f·(NP − GN); GN 0, SL(m) ½(1 − 1/m), gradient flow ½, NP 1, damped GN / GD per kernel mode (1−c)/(2−c) | ✅ 960/960 on MNIST | 06, 08 |
| 2 | Full GD matrix law | f_ij = ½ − ½·c_ic_j/(c_i + c_j − c_ic_j); first order = implicit gradient regularisation, pole = edge of stability | ✅ | 06, 08 |
| 3 | Heavy-ball law | closed form; zero at c* = (1−μ)²(1+μ)/(1+μ²), pole at c = 2(1+μ) | ✅ 16/16 | 08 |
| 4 | Theta-method laws | TJ(c, α): f = (1+k)/(2+k), k = c(2α − 1); proximal point (1+c)/(2+c); implicit midpoint ½ at any step size | ✅ | 09 |
| 5 | α-point rule | a jump with its Jacobian evaluated at fraction α of the move has f = α, for any α | ✅ 3-4 decimals | 08, 09 |
| 6 | Sub-step rule | SLa(m, α): f = ½ + (α − ½)/m | ✅ | 09 |
| 7 | Evaluation-point principle | ρ_L = ∫ Φ_L dω: pre-point (Itô) 0, midpoint (Stratonovich) ½, post-point 1 | ✅ | 08, 09 |
| 8 | Ground formula | f = ½ + Σ(α_k − ½)δ_k² / (Σδ_k)² = z² coefficient of the stability function R(z) | ✅ reproduces every closed form | 11 |
| 9 | Look-position rule | f = Σ w_j e_j (share of the journey paid by gradient j × where it looks) | ✅ 7/7 | 13 |

## Third-order laws

| # | Law | Formula | Status | Stage |
|---|---|---|---|---|
| 10 | Three-function law | learner differences = A2·F₁ + A3·F₂ + B3·F₃, A2 = Σbc, A3 = Σbac, B3 = ½Σbc² (Runge-Kutta order-condition sums) | ✅ rank 3, residual 0.00000 | 11 |
| 11 | Jensen bound | B3 ≥ A2²/2: an interior optimum of alignment is forced | ✅ derived; predicted optimum not yet run | 11 |
| 12 | Solvable-model series | β/α² = 8s + 32f(1+f)s³; conserved quantity of gradient flow broken ∝ (f − ½) | ✅ 3 digits | 11 |

## Prediction and generalisation laws

| # | Law | Formula | Status | Stage |
|---|---|---|---|---|
| 13 | Midpoint-kernel identity | gradient flow = kernel regression with the tangent kernel at the path midpoint; learner = that + (f − ½)G(x) | ✅ | 11 |
| 14 | One-function family | all learners differ on unseen inputs along one function G(x), coefficient f | ✅ | 08, 09 |
| 15 | Test loss linear in f | start function removed: larger f, lower test loss at equal training outputs | ✅ all pre-registered rules | 10 |
| 16 | Readout-flip symmetry | θ′(τ) = Sθ(−τ): second-order effects odd, third-order even, doubling cancels the odd part | ✅ cos ±0.99 | 11 |
| 17 | Weight-distance law | ‖θ_L − θ₀‖² − ‖θ_NP − θ₀‖² ∝ (1 − ρ)² | ✅ ~4% | 09 |

## Trainability laws

| # | Law | Formula | Status | Stage |
|---|---|---|---|---|
| 18 | Loop law | an out-and-back excursion moves the weights by (2f − 1)·v_NP | ✅ 36/36 | 12 |
| 19 | Mixed-learner loop law | displacement = (f_out + f_back − 1)·v_NP; exact jump + implicit jump cancel exactly | ✅ 5/5 | 12 |
| 20 | Cost-change law | cost of direction w changes by (1 − 2f)·2⟨v_NP,w, v_NP,u⟩/s² | ✅ | 12 |
| 21 | One-clock depletion law | λ_N/λ_N0 = 1 − t/36.5 with t = (1 − 2f) × excursions | ✅ R² 0.994-0.999; linear, then floors near 4% | 12 |
| 22 | State / heat / area decomposition | heat = (f − ½) × route term; area learner-independent | ✅ 4/4 | 12 |
| 23 | a + b/N law | trainability loss with N sub-steps, 1 − 2f = 1/N | ✅ | 07, 12 |
| 24 | Hover-wear law | explicit jitter around a target adds wear ∝ Σ squared step fractions; Adam with β₁ = 0: ∝ lr² | ✅ roughly (factor 10-12 vs 9 predicted) | 13 |

## Task-loop laws (where the project started)

| # | Law | Formula | Status | Stage |
|---|---|---|---|---|
| 25 | Path law | V = Σ C_ij T_ij for any learner whose steps lie in the row space of J | ✅ cos 0.999 | 04 |
| 26 | Three-force identity | C = ½ΔYΔYᵀ − ½Q + Area | ✅ exact to 1e-9 | 05 |
| 27 | Input-loop area law | rotation = signed polygon area × bracket; polygon ratios N·sin(2π/N) | ✅ within 2% | 01, 02 |
| 28 | Label-loop bracket law | shift = (ε²/2)[ξ_u, ξ_v]; 50-60× larger in weak kernel directions | ✅ | 03 |
| 29 | Spectral-area law | antisymmetric part = −s²a_ia_j(λ_i − λ_j)/(λ_i + λ_j) | ✅ 8.788e-3 vs 8.793e-3 | 13 |
| 30 | Flatness closed form | Δλ₁ = ε²(1−c)/(2(2−c))·‖P_v∇λ₁‖²/λ₁² | ✅ ratio 0.95-1.03 | 05 |

## Failed or refuted

| # | Claim | Status | Stage |
|---|---|---|---|
| 31 | Dose law: loss of plasticity follows one cumulative kernel-weighted dose across optimizers | ❌ | 07 |
| 32 | Moving beyond the nearest point always helps (n = 32) | ❌ interior optimum instead | 11 |
| 33 | Larger f gives better features for a fresh linear readout | ❌ | 11 |
| 34 | Finite-time collapse of trainability | ❌ depletion floors instead | 12 |
| 35 | Scalar ω form of the ReLU law | 🟡 the vector form holds, the scalar form fails | 09 |
