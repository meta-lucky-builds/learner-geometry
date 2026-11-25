# Part 6 — five CPU tests of the three-force picture (Tests 5–9)

Conventions as in SPEC.md–SPEC4.md: float64, CPU, deterministic, seeded generators,
`torch.set_num_threads(2)` (and the matching environment variables) for every run. Reuse
`tcr.py`, `tcr2.py`, `tcr3.py`, `tcr4.py` by import where their functions match the
definitions below (read them first); otherwise implement the definitions here directly.
Do NOT modify any existing file. New files only (§9).

## 0. Shared objects (all at fixed inputs X = X_A, n = 16)

Network f, teacher, X_A, probe set X_probe (256 inputs), y_A = teacher(X_A),
θ_A = GN-learn(θ_init, (X, y_A)), label directions u, v: exactly SPEC3.md §1.
F(θ) = f(X;θ) ∈ R^16, J = ∂F/∂θ, K = J Jᵀ = Σ_i λ_i q_i q_iᵀ (λ_1 ≥ … ≥ λ_16, unit q_i),
P_v w = w − Jᵀ K⁻¹ J w, ξ_e = Jᵀ K⁻¹ e, Dξ_e·w by the finite difference of SPEC3.md §1.

    T_ij = P_v [ Dξ_{e_i} · ξ_{e_j} ] (θ_A)         (e_i standard basis of R^16)
    LAW(C) = Σ_ij C_ij T_ij                          (SPEC4.md: predicted output-preserving weight shift)

Path matrix of a training run (SPEC4.md §3): C += (F_{k+1} − F_k) ⊗ (F_k − F_start) at every
optimizer step ("measured C"); or the same recursion on the linearised label-space dynamics
with K frozen at K(θ_A) ("a-priori C"). Split C = C_sym + C_area (symmetric / antisymmetric).

Established facts these tests build on (verified earlier; do not re-derive, but test 0 below
re-checks them): LAW(C) predicts the measured shift for GN, GD and momentum (cosine ≈ 0.999);
LAW(C_sym) = −P_v ∇_θ [ ½ tr(C_sym K(θ)⁻¹) ] at θ_A; LAW(C_area) is not a gradient.

Learners: GN = `tcr.gn_learn`. GD(c): θ ← θ − η Jᵀr/n with η = c·n/λ_max(K(θ_in)) fixed for
the learn, stop at ‖r‖_∞ ≤ 1e-8 unless a step count is specified (as in SPEC4.md §2).
Widths h ∈ {32, 64, 128}; seeds {0,1,2,3,4} unless a test says otherwise.
"Dλ[V]" means the directional derivative of a scalar function of θ along the vector V at θ_A,
by central finite difference with step 1e-4/‖V‖, multiplied by ‖V‖.

Verdict rule for every H: PASS if the rule holds for the median over seeds AND for at least 4
of 5 seeds; INCONCLUSIVE if median only; FAIL otherwise. A FAIL is a valid result; tune nothing.
Predictions marked [derived] come from algebra that has NOT been checked numerically — report
what is measured even if the derivation turns out wrong, and say so.

---

## Test 5 — large steps along the strongest direction lower the top eigenvalue

Claim: when gradient descent overshoots (step factor c > 1) the outputs zigzag along the
strongest kernel direction q_1; that jitter pushes the weights so that λ_1 falls. For c < 1 the
opposite happens. This would be the mechanism of "self-stabilisation" at the edge of stability.

Protocol A (single direction): target y_A + ε q_1(θ_A). From θ_A run GD(c) to convergence,
giving θ_c, and GN, giving θ_GN (same final outputs). Let Δλ_1(c) = λ_1(K(θ_c)) − λ_1(K(θ_GN)).
c ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 1.95}; ε ∈ {0.1, 0.05, 0.025}.
Predictions:
- pred1 = Dλ_1[ LAW(C_c − C_GN) ] with measured path matrices.
- pred2 [derived] = ε² · (1−c) / (2(2−c)) · ‖P_v ∇λ_1‖² / λ_1² , where ‖P_v ∇λ_1‖² is obtained
  as Dλ_1[g] with g = P_v∇λ_1 computed by finite differences over an orthonormal basis of the
  span of {T_ij} (at most 256 directions) — state clearly how it was computed.

| id | rule (at ε = 0.025) |
|----|------|
| H5a | sign(Δλ_1) = sign(1 − c) for every c ≠ 1 (positive for c < 1, negative for c > 1) |
| H5b | Δλ_1 / pred1 ∈ [0.8, 1.25] for every c with |c − 1| ≥ 0.25 |
| H5c | Δλ_1 / pred2 ∈ [0.67, 1.5] for the same c  [derived] |
| H5d | |Δλ_1| at c = 1.95 is at least 5× |Δλ_1| at c = 1.5 (the predicted growth as c → 2) |

Protocol B (report only — the unstable regime): fixed target y_A + 0.05·u. Run GD with
c = 2.05 for 3000 steps (do not stop on divergence unless ‖r‖ > 1e6). Every step record
λ_1(K(θ_k)), the residual component x_k = q_1(θ_k)·r_k, loss, and c_eff,k = η λ_1(θ_k)/n.
Report: whether λ_1 falls until c_eff < 2; the regression of (λ_1,k+1 − λ_1,k) on x_k² over the
steps where c_eff > 1.5 (slope, R²); and the slope predicted [derived] as
−(c_eff²/4)·‖P_v∇λ_1‖²/λ_1² evaluated at θ_A. Do the same at c = 1.9 as a stable comparison.

---

## Test 6 — stiffening protects what was practised (consolidation)

Claim: a direction that has become hard to learn is equally hard to overwrite. Repeated exact
out-and-back practice along u stiffens the kernel along u, so later brief training on OTHER
targets moves the outputs along u less.

States (all have outputs y_A on X): S_0 = θ_A; S_m^u = θ_A after m exact (GN) out-and-back
excursions along u with ε = 0.1 (targets y_A + εu then y_A), m ∈ {4, 8, 16};
S_m^v = the same along v (control for specificity); S_m^{GD} = the same along u but each learn
done by GD(1) (small steps: predicted little effect).
Interference probe from a state S: for each of B = 16 unit directions b_j ∈ R^16
(`Generator(6000+seed)`, Gaussian, normalised), run exactly k = 50 steps of GD from S toward
y_A + δ b_j with δ = 0.05·‖y_A‖ and the SAME absolute step size for every state,
η_0 = 0.5·n/λ_max(K(θ_A)). With û = u/‖u‖:

    vulnerability  ν_u(S) = mean_j ( û · (F_end − y_A) )² / δ²          (and ν_v likewise)
    plasticity     π_u(S) = 1 − ‖r_50‖/‖r_0‖ when the target is y_A + δ û (k = 50, same η_0)
    linear predictions: with A = (I − (η_0/n) K(S))^{50}:  ν_u^lin = ‖(I − A) û‖² / 16 ,  π_u^lin = 1 − ‖A û‖

| id | rule |
|----|------|
| H6a | ν_u(S_m^u)/ν_u(S_0) is decreasing in m and < 0.8 at m = 16 |
| H6b | specificity: ν_u(S_16^u)/ν_u(S_0) < ν_v(S_16^u)/ν_v(S_0), and ν_v(S_16^v)/ν_v(S_0) < ν_u(S_16^v)/ν_u(S_0) |
| H6c | pooled over all states, widths and seeds: Spearman(ν_u, π_u) ≥ 0.9 (hard to learn ⇔ hard to overwrite) |
| H6d | ν_u/ν_u^lin ∈ [0.8, 1.25] and π_u/π_u^lin ∈ [0.8, 1.25] for every state (median) |
| H6e | GD practice does little: |log(ν_u(S_16^{GD})/ν_u(S_0))| < 0.25·|log(ν_u(S_16^u)/ν_u(S_0))| |

Report only: φ_u, φ_v, λ_min, tr K⁻¹ for every state; generic plasticity (mean π over the 16
b_j as targets); the price of protection = ratio of generic plasticity S_16^u / S_0.

---

## Test 7 — order matters for the weak-kernel parts, not the strong ones

Let s = a unit vector in span{q_1..q_4} and w = a unit vector in span{q_13..q_16} (random
coefficients from `Generator(7000+seed)`, normalised), both scaled to ‖y_A‖; also s′, w′
(second independent draws from the same spans, orthogonalised against s resp. w).

Part (a) — at convergence, order leaves a trace only through the area term. For a pair (p,q),
"p-first" = GN-learn y_A + εp then y_A + ε(p+q); "q-first" = y_A + εq then y_A + ε(p+q).
O_pq = θ(p-first) − θ(q-first); probe difference o_pq = f(X_probe;·) difference, per-input RMS.
Pairs: strong×strong (s,s′), weak×weak (w,w′), strong×weak (s,w). ε ∈ {0.1, 0.05, 0.025}.
Prediction: O_pq = ε²·κ·b_pq(θ_A) with κ = ±1 (bracket of SPEC3.md; determine κ and report).

| id | rule (ε = 0.025) |
|----|------|
| H7a | |cos(O_pq, b_pq)| ≥ 0.99 and ‖O_pq‖/(ε²‖b_pq‖) ∈ [0.95, 1.05] for all three pairs |
| H7b | ‖O_ww′‖ ≥ 10·‖O_ss′‖ and RMS o_ww′ ≥ 5·RMS o_ss′ |

Part (b) — with a fixed step budget, order changes the training outputs only in weak
directions. Final target y* = y_A + ε(s + w), ε = 0.1. GD(1) with total budget T_tot steps,
T_tot ∈ {50, 200, 800, 3200}: JOINT = all steps on y*; EASY-FIRST = T_tot/2 steps on
y_A + εs then T_tot/2 on y*; HARD-FIRST = T_tot/2 on y_A + εw then T_tot/2 on y*.
For each pair of curricula let d = difference of final training outputs.
weak share = Σ_{i≥9} (q_i·d)² / ‖d‖². Linear prediction of d from K(θ_A) (frozen-kernel GD).

| id | rule |
|----|------|
| H7c | weak share ≥ 0.9 for EASY-FIRST vs HARD-FIRST at every T_tot |
| H7d | cos(d, d_lin) ≥ 0.95 at every T_tot |

Report only: per-input RMS difference of PROBE outputs between EASY-FIRST and HARD-FIRST as a
function of T_tot, next to the converged-order probe difference of part (a) for (s,w) at the
same ε (shows where budget effects end and geometric effects remain); final loss on the s and
w components separately for each curriculum.

---

## Test 8 — bootstrapped targets: the extra drift, three ways

Setting: 16 states = the 16 inputs. Transition matrix P: draw `torch.rand(16,16)**4` rows
normalised to sum 1 from `Generator(8000+seed)`; redraw until every eigenvalue of K(θ_A)·M has
real part > 1e-3·λ_max, where M = I − γP. Rewards are set so the fixed point is y_A.
Noisy semi-gradient update, step k: θ ← θ − η′ Jᵀ( M (F − y_A) − σ ζ_k ), ζ_k ~ N(0, I_16),
η′ = 0.2/λ_max(K(θ_A)), σ = 0.05. γ ∈ {0, 0.5, 0.9, 0.99}.

Stationary quantities at θ_A: A = η′ K M, BBᵀ = η′² σ² K², Σ_y = solution of the DISCRETE
Lyapunov equation Σ = (I − A) Σ (I − A)ᵀ + BBᵀ, per-step expected path matrix E = −A Σ_y,
Σ_θ = Jᵀ K⁻¹ Σ_y K⁻¹ J. Per-step predicted drift, three forms:

    d_law   = LAW(E)
    d_dir   = −η′ P_v Σ_a H_a Σ_θ ( g_a − γ Σ_b P_ab g_b )        g_a = ∇_θ F_a = row a of J,
              H_a w computed as the finite-difference derivative of row a of J along w
    d_pot(κ) = −(η′/2) P_v ∇_θ [ Σ_a g_aᵀ Σ_θ g_a − κ γ Σ_a g_aᵀ Σ_θ sg(Σ_b P_ab g_b) ]
              with Σ_θ and the sg(·) factor held fixed while differentiating (finite differences
              along the needed directions); evaluate for κ = 1 and κ = 2 and also report the
              least-squares κ* that best matches d_law.

Simulation with common random numbers: for each γ run R = 32 independent noise streams
(`Generator(8100+seed)+stream`) for N = 4000 steps from θ_A, then a clean GN return to y_A;
V_γ = mean over streams of (θ_end − θ_A). The SAME noise streams are used for every γ.

| id | rule |
|----|------|
| H8a | d_law = d_dir: relative difference < 1e-5 for every γ  [derived] |
| H8b | Sym(A Σ_y) = ½ BBᵀ to relative 1e-8 for every γ, hence LAW(E_sym) is identical across γ to 1e-6 |
| H8c | κ* ∈ [1.9, 2.1] for γ ∈ {0.5, 0.9, 0.99}  [derived; a result near 1 instead is equally informative — report it] |
| H8d | simulation, total drift: cos(V_γ, N·d_law) ≥ 0.95 and ratio ∈ [0.8, 1.25] for every γ |
| H8e | simulation, the part specific to bootstrapping: cos(V_γ − V_0, N·(d_law(γ) − d_law(0))) ≥ 0.8 for γ ∈ {0.9, 0.99} |

Report only: ‖LAW(E_area)‖/‖LAW(E_sym)‖ per γ; the measured and predicted change, per step, of
the mean normalised kernel entry between each state and its expected successor,
(1/16) Σ_a Σ_b P_ab K_ab/√(K_aa K_bb), and of tr K — i.e. whether successive states' gradients
become more aligned.

---

## Test 9 — rotating targets give steady drift whose direction follows the rotation

Targets at step k: y_k = y_A + a( cos(ωk) p + sin(ωk) q ), a = 0.05, tracked by GD with
η = 0.5·n/λ_max(K(θ_A)), one step per k. Three drives with the same per-step target power:
ROT+ (as written), ROT− (ω → −ω), LINE (y_k = y_A + a·cos(ωk)·(p + q)).
Run m full periods (period Π = 2π/ω steps, Π ∈ {16, 64, 256}), then a clean GN return to y_A;
V = θ_end − θ_A. m ∈ {1, 2, 4, 8, 16}. A-priori path matrix from the linearised recursion.

Planes: (p,q) = (u,v) of SPEC3.md; and a STEERED plane chosen for the first probe input x_1:
with S_ij = J_x1 · T_ij and Ā = antisymmetric part of S, take (p,q) as the real and imaginary
parts (scaled by √2, then each normalised to ‖y_A‖) of the eigenvector of Ā with the largest
imaginary eigenvalue.

| id | rule (per Π, at m = 8 unless stated) |
|----|------|
| H9a | odd part (V₊ − V₋)/2 vs LAW(C_area a-priori): cos ≥ 0.98, ratio ∈ [0.9, 1.1] |
| H9b | even part (V₊ + V₋)/2 vs LAW(C_sym a-priori): cos ≥ 0.98, ratio ∈ [0.9, 1.1] |
| H9c | steady drift: ‖odd part‖ at m = 16 is within [1.6, 2.2]× its value at m = 8 |
| H9d | LINE has no odd part: ‖(V_LINE,ω − V_LINE,−ω)/2‖ < 0.02·‖(V₊ − V₋)/2‖ |
| H9e | steering: for the STEERED plane, the odd part of the change in f(x_1) has the predicted sign and magnitude within [0.8, 1.25]× of J_x1·LAW(C_area), and is at least 3× the median |odd change| over the other 255 probe inputs |

Report only: odd-part norm per period as a function of Π next to the prediction (frequency
response); odd-part change of f(x_1) as a function of m for the STEERED plane.

---

## 7. Test 0 and required unit tests (`test_part6.py`, must pass before any grid)

0. Re-check on h = 32, seed 0: (i) GN out-and-back along u at ε = 0.025: cos(V, LAW(C)) > 0.999;
   (ii) for a random symmetric C: directional derivative of ½tr(C K⁻¹) along 5 random
   output-preserving directions w equals −w·LAW(C) to relative 1e-6.
1. T is vertical: ‖J T_ij‖/(‖J‖‖T_ij‖) < 1e-6 for 10 random (i,j).
2. A-priori vs measured C for GD(1) one-way along u at ε = 0.025: relative difference < 0.05.
3. Test-5 bookkeeping: at c = 1 the GD run converges in one step in the q_1 direction and
   |Δλ_1| is at least 20× smaller than at c = 1.5 (same ε).
4. Test-8 Lyapunov: the returned Σ_y satisfies its equation to 1e-10 and is symmetric PSD.
5. Test-9 drives: ROT+ and ROT− target sequences are mirror images; LINE has zero a-priori
   area matrix (< 1e-12).
6. Linear control (`tcr.LinearModel`): Test-5 Δλ_1, Test-7 O_pq and Test-9 V are all < 1e-10.

## 8. Reporting

`report6.py` writes REPORT6.md: for each test, the claim in one sentence, the H-table with
predicted and measured numbers per width (median and min–max over seeds), verdicts by the rule
of §0, the report-only tables, and a short "anomalies" list stated as measurements only
(non-convergence, sign surprises, a width or seed that behaves differently). For every
[derived] prediction say explicitly whether the derivation was confirmed or contradicted.

## 9. Deliverables (in `task_cycle_rotation/`)

`tcr6.py`, `test_part6.py`, `run6.py` (writes `results6.json` incrementally; one sub-command
per test), `report6.py`, `REPORT6.md`. Logs and debug scripts in the scratchpad.
Budget 3 hours wall in total; if a test would exceed 45 minutes, stop it, report what finished
and list what did not run.

Final message back: `test_part6.py` output verbatim; every H-table with verdicts and per-seed
values where not PASS; the report-only tables; κ and κ* found; which [derived] predictions were
confirmed or contradicted; anything not run; anomalies; wall time; every ambiguity or decision.
