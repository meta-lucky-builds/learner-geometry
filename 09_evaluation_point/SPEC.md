# SPEC9 — Three follow-up tests of the evaluation-point law

Deliverable: one Jupyter notebook `evaluation_point_tests.ipynb`, saved to `the working folder` and copied
into this project folder. It runs on a rented Linux GPU box with `jupyter nbconvert --execute` (PyTorch present; the
notebook must NOT install packages) and on a CPU-only Windows machine in MODE="tiny".

Read ALL of this spec, then SPEC8.md (sections 2, 3, 4, 10, 11), before writing code. Section 12 lists the mistakes
made in earlier rounds of this project; each one must be designed out. A reference implementation of every new
learner and of the Part R prediction, written by the spec author and already checked numerically, is in
`gpu_run_part8/pre9.py` (read it; reproduce its numbers; do not copy its shortcuts where this spec is stricter).

====================================================================================================
## 1. What is being tested

Setting (unchanged from SPEC8): start weights θ₀, training outputs F(θ) ∈ ℝᴺ, J = ∂F/∂θ, K = JJᵀ,
vertical projector at the start V₀w = w − J₀ᵀK₀⁻¹J₀w, nearest point θ_NP of the learner's own final outputs F_L,
exact jump θ_TH = JUMP(θ₀, F_L), v_L = V₀(θ_L − θ₀), v_NP = V₀(θ_NP − θ₀),
ρ_L = ⟨v_L, v_NP⟩/‖v_NP‖², cos, nr, even part over the two signs (SPEC8 3.5).

THE EVALUATION-POINT LAW. Every learner in this spec changes the weights in steps of the form
        Δθ_k = J(θ_e,k)ᵀ a_k          (a_k ∈ ℝᴺ; θ_e,k = the point where step k evaluates the Jacobian).
Because V₀J₀ᵀ = 0, the vertical displacement is  v_L = V₀ Σ_k [J(θ_e,k) − J₀]ᵀ a_k  (an exact identity).
The LAW is the claim that, to leading order in the size of the move, this depends only on WHERE ALONG THE OUTPUT PATH
each step evaluated its Jacobian. For a smooth network J changes linearly along the path, and the alignment fraction is
        f_L = Σ_k (output increment of step k, as a fraction of the total move) × (position of its evaluation point, as a fraction of the total move).
Evaluating at the start of a jump gives 0 (exact jump TH), at the end gives 1 (nearest point NP), midway ½ (gradient flow).

Three parts:
- **Part A (new closed forms from the law).** Four families of learners whose evaluation point is a free parameter.
  Each has a closed-form f. None has been run on a real network.
- **Part G (unseen inputs).** All learners fit the same training outputs; the law implies that their predictions on
  held-out inputs differ along ONE function, with coefficient f. Then the open question with no prediction:
  does a larger f lower or raise the test loss?
- **Part R (ReLU, no smoothness).** For ReLU the Jacobian changes in jumps (activation switches), so the closed-form f fails
  (seen in SPEC8). The general form of the law replaces "position × increment" by the Jacobian actually found at that
  position on a REFERENCE path. This is tested with no fitted parameter and without the nearest-point solver.

====================================================================================================
## 2. Reused, already reviewed code (copy VERBATIM)

From `alignment_fraction_test.ipynb` (final reviewed version, md5 787d12332b4bf0eb4f862ecbfc06d571) copy these
functions and classes without changing a character of their bodies:
`Problem, _act, _act_deriv, mlp_forward, mlp_backward, layered_problem, toy_problem, gram, chol, csolve, eig_desc, vnorm,
Start, jump, run_SL, run_DJ, run_first_order, run_PROX, _np_eval, nearest_point, measure, even_stats, usable,
get_data, _load_mnist_raw, combo_problem, fit_dj, make_directions, matches, cosine` and the unit tests U1–U14.
They will be diffed against the original. Allowed additions (new functions/wrappers only; list each in your report):
- `jump_a`: same algorithm as `jump` but also returns the coefficient vector `a` (needed for logging in Part R).
  Implement by a wrapper or a copy; `jump` itself stays untouched.
- `measure_ext`: calls the same steps as `measure` and additionally returns θ_NP and θ_TH (needed for Part G).
Setup constants of SPEC8 section 2 (float64, data, model 196→W→W→10, loss L = ‖F − t‖²/(2N), η = c·N/λ₁(K₀)),
section 3.1–3.5 (V₀, JUMP, NP with the step-length merit of A1.5, measurement, even part), section 4 (directions eigA–eigD,
label, rand; S₀; sizes s₁ = 0.02·S₀, s₂ = 0.01·S₀) and the direction-converged stop rule A2.1 apply unchanged.

====================================================================================================
## 3. New learners (exact definitions)

### 3.1 α-point jump  AJ(θ_b, t; α)
The point θ with F(θ) = t and θ − θ_b = J(θ_e)ᵀa for some a, where θ_e = θ_b + α(θ − θ_b).   (α = 0 is JUMP.)
Algorithm (outer fixed point on θ, inner exact solve for a):
  θ⁽⁰⁾ = θ_b. For i = 0, 1, …:
    θ_e = θ_b + α(θ⁽ⁱ⁾ − θ_b);  J_e = J(θ_e);  factorise K_e = J_eJ_eᵀ.
    inner: a = 0; repeat a ← a + K_e⁻¹(t − F(θ_b + J_eᵀa)) until ‖F − t‖∞ ≤ 1e-11 (cap 200; if not reached, Newton on a
           with M = J(θ_b + J_eᵀa)·J_eᵀ, cap 50; inner failure → AJ-FAILED).
    θ_new = θ_b + J_eᵀa.
    if ‖θ_new − θ⁽ⁱ⁾‖ ≤ 1e-11·‖θ_new − θ_b‖ + 1e-15: converged, θ = θ_new.
    else θ⁽ⁱ⁺¹⁾ = θ_new; if the outer step length ‖θ_new − θ⁽ⁱ⁾‖ did not decrease compared with the previous outer
         iteration, use θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + ½(θ_new − θ⁽ⁱ⁾) instead (count these dampings).
  Outer cap 100 → AJ-FAILED (recorded, excluded from rules, listed).
After convergence verify and store: res = ‖F(θ) − t‖∞ and
  hz = ‖V_e(θ − θ_b)‖/‖θ − θ_b‖ with V_e the vertical projector built at θ_e = θ_b + α(θ − θ_b) for the FINAL θ
  (must be ≤ 1e-7; larger → AJ-INCONSISTENT, excluded and listed).
As a learner: AJ(α) := AJ(θ₀, t; α).            Prediction: f = α.

### 3.2 Sub-stepped α-point transport  SLα(m, α)
θ ← θ₀; for k = 1..m: θ ← AJ(θ, F₀ + (k/m)(t − F₀); α).     Prediction: f = ½ + (α − ½)/m.
(α = 0 is SL(m). α = ½ gives ½ for every m.)

### 3.3 θ-method in output space  TJ(c, α)
q := (1 − c(1 − α))/(1 + cα).  θ ← θ₀; repeat: r = F(θ) − t; stop when ‖r‖₂ ≤ 1e-7·‖t − F₀‖₂ (cap 400 steps);
θ ← AJ(θ, t + q·r; α).        (Each step multiplies the residual by q and evaluates the Jacobian at the α-point of the step.)
Prediction: f = (1 + κ)/(2 + κ) with κ = c(2α − 1).   (α = 0: (1−c)/(2−c), the DJ curve. α = ½: ½ for every c. α = 1: (1+c)/(2+c).)

### 3.4 θ-method in weight space  TM(c, α)   (α = 1: proximal-point / implicit Euler; α = ½: implicit midpoint; α = 0: gradient descent)
η = c·N/λ₁(K₀). Step: θ⁺ = θ − η∇L(θ + α(θ⁺ − θ); t).  Solve for δ = θ⁺ − θ by the quasi-Newton iteration
  δ ← δ − (I + a J_eᵀJ_e)⁻¹ g(δ),  g(δ) = δ + (η/N) J_eᵀ(F(θ_e) − t),  θ_e = θ + αδ,  J_e = J(θ_e),  a = ηα/N,
  with (I + a J_eᵀJ_e)⁻¹g = g − J_eᵀ(J_eJ_eᵀ + I/a)⁻¹J_e g; start δ = 0; stop when ‖g‖ ≤ 1e-12·‖δ‖ + 1e-16
  (cap 60 inner iterations → TM-INNER-FAILED, recorded and listed). For α = 0 the step is explicit (no inner iteration).
Stop rule of the run: direction-converged (A2.1): |d·(F − t)| ≤ 1e-6·s for 100 consecutive steps; cap 20 000 steps.
Used only with targets along kernel eigen-directions eigA and eigB.
Prediction for eigen-direction i: f = (1 + κ_i)/(2 + κ_i), κ_i = c_i(2α − 1), c_i = c·λ_i/λ₁.

All four predictions were verified by the spec author on the solvable model (values in U-tests below) and on a small
tanh MLP (`gpu_run_part8/pre9.py`).

====================================================================================================
## 4. Part A — the evaluation-point families

Combinations: act ∈ {tanh, gelu} × start ∈ {init, trained} × seed ∈ {0, 1}  (n = 32, W = 256).
Targets: the six directions, sizes s₁ and s₂, both signs (TM: eigA and eigB only).
Learners and predictions:
| family | parameters | f |
|---|---|---|
| AJ(α) | α ∈ {−1, 0.25, 0.5, 0.75, 1, 1.5, 2} | α |
| SLα(m, α) | (2, 0.5), (2, 1), (2, 2), (8, 1), (8, 2), (4, −1) | 0.5, 0.75, 1.25, 0.5625, 0.6875, 0.125 |
| TJ(c, α) | (0.5, 0.5), (1.5, 0.5), (0.5, 1), (1.5, 1), (8, 1), (3, 0.25) | 0.5, 0.5, 0.6, 0.714286, 0.9, −1 |
| TM(c, α) | (0.5, 0.5), (1.5, 0.5), (0.5, 1), (1.5, 1), (6, 1) | (1+κ_i)/(2+κ_i) |
| anchors (as in SPEC8) | SL(8), DJ(1.5) | 0.4375, −1 |
Every case goes through `measure_ext` (same measurement as SPEC8). Even part; extrapolated value
ρ_x = (4ρ(s₂) − ρ(s₁))/3 with cos and nr taken at s₂ (A1.2); the function `matches` with the stated tolerance.

Pre-registered rules, evaluated per (act, start), pooling seeds and directions. A FAIL is a valid result.
- HA1: AJ: ≥ 90% of (learner, direction) units match, tol 0.02.
- HA2: SLα: ≥ 90%, tol 0.02.
- HA3: TJ: ≥ 90%, tol 0.02.
- HA4: TM: ≥ 90% of units whose four runs are CONVERGED match, tol 0.03; and ≥ 75% of planned TM units converged.
- HA5 (midpoint has no step-size bias; sharper): for AJ(0.5), SLα(2, 0.5), TJ(0.5, 0.5), TJ(1.5, 0.5), TM(0.5, 0.5), TM(1.5, 0.5):
  |ρ_x − 0.5| ≤ 0.01 and cos(s₂) ≥ 0.99 in ≥ 90% of units.
- HA6 (two solvers of the post-point condition agree): for AJ(1) runs (every size and sign) do ONE extra nearest-point
  solve STARTED FROM θ₀ (not from the learner's endpoint): θ_NP0 := nearest_point(θ₀, F_L, start = θ₀). Rule:
  ‖θ_AJ(1) − θ_NP0‖ ≤ 1e-5·‖θ_NP0 − θ₀‖ in ≥ 90% of runs.
  (Note: `measure_ext` starts its nearest-point solve at the learner's endpoint, which for AJ(1) already satisfies the
  post-point condition; so ρ = 1 for AJ(1) is true by construction and HA6 is the non-trivial check. Say so in the output.)
- HA7 (anchors reproduce SPEC8): SL(8) and DJ(1.5) match with tol 0.02 in ≥ 90% of units.
Report (no verdict): literal single-size matches; order check (ratio |ρ(s₁)−f|/|ρ(s₂)−f|, expected ≈ 4); every failed /
excluded unit with its reason; outer-iteration counts and dampings of AJ per α.

====================================================================================================
## 5. Part G — predictions on unseen inputs

Held-out set: the MNIST TEST split, all 10 000 images in dataset order, preprocessed like the training inputs (2×2 average
pooling to 14×14, flatten) and standardised with **the mean and std of that seed's TRAINING selection** (the same two
scalars used for the training inputs; store them when the training data are built). Y_test = one-hot test labels.
N_test = 10 000·C. F_test(θ) = network outputs on the held-out inputs (forward pass only, float64, in chunks).
Test loss ℓ(θ) = ‖F_test(θ) − Y_test‖²/(2N_test). Test accuracy = fraction with argmax equal to the label.
(Synthetic fallback when MNIST is unavailable: 2 000 Gaussian test inputs from `gen(5000+seed)` with random labels,
standardised with the training scalars; label every result data_source="synthetic".)

### G1 (law-implied): one function, coefficient f
For every Part A case (all learners incl. anchors), with θ_L, θ_TH, θ_NP of the SAME fibre from `measure_ext`:
   G_L = F_test(θ_L) − F_test(θ_TH),   G_NP = F_test(θ_NP) − F_test(θ_TH).
Even parts over the two signs; γ = ⟨G_Lᵉ, G_NPᵉ⟩/‖G_NPᵉ‖², cos_test, nr_test; extrapolated γ_x = (4γ(s₂) − γ(s₁))/3.
- HG1: `matches(γ_x, cos_test(s₂), nr_test(s₂), f, tol 0.02)` in ≥ 90% of (learner, direction) units, per (act, start).
Store per run also: ℓ(θ_L), ℓ(θ_TH), ℓ(θ_NP), ℓ(θ₀), test accuracy at the same four points, ‖G_NP‖/‖F_test(θ_TH) − F_test(θ₀)‖.

### G2 (law-implied): test loss is linear in f
For each (combination, direction, size s₂): over the learners with a closed-form f (even part of
Δℓ_L := ℓ(θ_L) − ℓ(θ_TH)), the Pearson r² between Δℓ_Lᵉ and f_L.
- HG2: r² ≥ 0.99 in ≥ 90% of (combination, direction) pairs, per (act, start).
Report the fitted slope b (Δℓ per unit f; its SIGN is the object of G3).

### G3 (OPEN QUESTION — no prediction, two-sided): does alignment help?
Extra direction for start = init only: **true** := (Y_true − F₀)/‖Y_true − F₀‖ with Y_true the one-hot TRUE training labels
(the realistic direction of training). Run the Part A learner set on "true" as on the other directions (sizes s₁, s₂ with
S₀ replaced by S_true := ‖Y_true − F₀‖; both signs; the − sign is needed only for the even part).
Report, for direction "true" and separately for "label" and "rand", for every combination:
 (i) the sign of the slope b of G2 (b < 0: moving toward the nearest solution LOWERS the test loss), with b itself;
 (ii) from the + sign runs only (the realistic direction), at each size: E₊ := [ℓ(θ_NP) − ℓ(θ_TH)] / [ℓ(θ₀) − ℓ(θ_TH)], i.e. the
      share of the test-loss change caused by the move that is added (E₊ > 0 when the move lowered the test loss and the
      nearest point lowers it further) by going from fraction 0 to fraction 1; and E₊/(s/S_true) (effect per unit relative move).
      If |ℓ(θ₀) − ℓ(θ_TH)| < 1e-12 report E₊ as undefined.
 (iii) the count of combinations with b < 0.
State in the output: "no prediction was registered for the sign of b".

### G4 (report only): large moves toward the true labels
Combinations: act ∈ {tanh, gelu}, start = init, seed ∈ {0, 1}, and training-set size n ∈ {32, 128} (n = 128 only here).
Targets t_τ = F₀ + τ(Y_true − F₀), τ ∈ {0.1, 0.25, 0.5, 1.0}, sign + only.
Learners (nominal f): DJ(1) [0], SL(2) [0.25], SL(4) [0.375], DJ(0.25) [0.4286], SL(32) [0.4844],
GD(0.25) [≈ ½; stop ‖F − t‖₂ ≤ 1e-3‖t − F₀‖₂ or 50 000 steps], TJ(1.5, 1) [0.7143], SLα(4, 2) [0.875], AJ(1) [1], DJ(1.5) [−1].
For each run store: status, training residual ‖F − t‖₂/‖t − F₀‖₂ (runs with residual > 1e-3 are listed and not used in the
rank statistics), ℓ(θ_L), test accuracy, ‖θ_L − θ₀‖, λ₁(K(θ_L)), tr K(θ_L). For n = 32 and τ ∈ {0.1, 0.25} additionally the
SPEC8 measurement with the nearest-point solve capped at 120 iterations (ρ, cos; failures recorded). For τ ≥ 0.5 and for
n = 128: no NP, no TH (cost). For n = 128 set the Jacobian chunk to 256 rows.
Report per (combination, τ): the table f → (test loss, accuracy, distance, λ₁), and the Spearman correlation between nominal
f and test loss over the usable learners. Stated expectation (not a rule): at τ = 1 the sign of that correlation equals
the sign of b found in G3 (direction "true") for the same act/seed.

====================================================================================================
## 6. Part R — ReLU and the general (reference-path) form of the law

NO nearest-point solve anywhere in Part R.
Combinations: relu × {init, trained} × seeds {0, 1}; smooth controls: tanh-trained-s0 and gelu-init-s0 (n = 32, W = 256).
Directions eigA, label, rand. Sizes s/S₀ ∈ {1e-5 (relu only), 1e-4, 1e-3, 1e-2, 0.03, 0.1}. Both signs; **every sign is a
separate case (no even part, no extrapolation)**, because a switch is a one-sided, first-order event.
For one case let d be the SIGNED unit direction, t = F₀ + s·d, and τ the position along the line: F = F₀ + τ·s·d.

### 6.1 Reference path
Grid τ_j = 2j/256, j = 0..256 (so τ ∈ [0, 2]; 128 cells per unit of τ). θ_ref[0] = θ₀;
θ_ref[j] = JUMP(θ_ref[j−1], F₀ + τ_j·s·d). If any JUMP fails: case REF-FAILED (recorded, listed, excluded).
(General: G_REF cells on [0, 2], τ_j = 2j/G_REF; G_REF = 256 except in MODE "tiny", where it is 64.)
Exact position: for any τ ∈ [0, 2]: θ_ref(τ) := JUMP(θ_ref[j], F₀ + τ·s·d) with j = floor(τ·G_REF/2) (j = G_REF → use G_REF − 1).
For τ outside [0, 2]: clamp τ to the range and flag the learner step OUT-OF-RANGE (count them).
Field: for b ∈ ℝᴺ, W(τ)b := V₀ J(θ_ref(τ))ᵀ K(θ_ref(τ))⁻¹ b   (K factorised at that point).
Post-point reference: v_PP := W(1)(s·d).      (For a smooth network v_PP = v_NP to leading order.)
Profile: ω(τ) := ⟨W(τ)(s·d), v_PP⟩/‖v_PP‖², and its off-axis part ω⊥(τ) := ‖W(τ)(s·d) − ω(τ)v_PP‖/‖v_PP‖,
stored at τ = 0, 1/16, …, 2 (33 values; these are grid points: θ_ref[k·G_REF/32]).
Switches (all activations; for tanh/gelu "switch" = sign change of a hidden pre-activation, stored for completeness):
for each grid point the number of hidden (unit, input) pairs whose pre-activation sign differs from the sign at θ₀
(n_sw(τ_j)), and for each cell whether any pair changed sign between its two ends. Store n_sw at τ = 1/16 steps and the
total number of cells with a change on [0, 1] and on [0, 2].

### 6.2 Logged learners
Each learner step must record (F_e, b): F_e = outputs at the step's evaluation point θ_e, and b ∈ ℝᴺ such that the step is
EXACTLY Δθ = J(θ_e)ᵀK(θ_e)⁻¹b.
- SL(m) (m = 2, 8, 32): step = JUMP from θ_k; θ_e = θ_k; b = K(θ_k)a with a from `jump_a`.
- DJ(c) (c = 0.5, 1): θ_e = θ_k; b = −c(F(θ_k) − t).
- AJ(α) (α = 0.5, 1) and SLα(8, 1): θ_e = the final evaluation point of the (sub-)jump, θ_e = θ_b + α(θ − θ_b);
  b = J(θ_e)(θ − θ_b)   (because θ − θ_b = J(θ_e)ᵀa, this equals K(θ_e)a).
- DJ needs a logging variant `run_DJ_logged` (new function, same arithmetic as `run_DJ` plus the log); V5 must also check
  that it returns exactly the same weights as `run_DJ`.
- Overshooting learners, reported separately (not in the primary rules): DJ(1.5), DJ(1.9).
Primary learner set P := {SL(2), SL(8), SL(32), DJ(0.5), DJ(1), AJ(0.5), AJ(1), SLα(8, 1)}.
Smooth closed forms for comparison: ½(1−1/m); (1−c)/(2−c); α; ½ + (α−½)/m.

### 6.3 Measured and predicted quantities (per case and learner)
measured:   v_L = V₀(θ_L − θ₀);   ρᴾᴾ_L = ⟨v_L, v_PP⟩/‖v_PP‖².
positions:  τ_k = d·(F_e,k − F₀)/s;   β_k = d·b_k/s.
P-field (the general law, no fitted parameter):   v_pred = Σ_k W(τ_k)b_k;   compare vectors: cosF = cos(v_L, v_pred), nrF = ‖v_L‖/‖v_pred‖;  ρ_field = ⟨v_pred, v_PP⟩/‖v_PP‖².
P-omega (scalar form):   ρ_ω = Σ_k β_k·ω(τ_k)  with ω(τ_k) computed from the same W(τ_k).
P-smooth (closed form):  f_L.
Also store: number of steps, OUT-OF-RANGE count, max |τ_k|, the off-line part of the path max_k ‖(F_e,k − F₀) − τ_k s d‖/s,
n_sw at the learner's own endpoint, cost (Jacobian count), and whether the reference path had any switch on [0, 1] and on [0, max τ_k].

### 6.4 Pre-registered rules (per activation; single-sign cases; learners in P)
A (case, learner) pair is USABLE when the reference path is complete, the learner run did not diverge or fail, no step is
OUT-OF-RANGE, and ‖v_PP‖ ≥ 1e-9·‖θ_ref(1) − θ₀‖. HR1 and HR2 do NOT require the learner to have converged (the general law
holds for the path actually taken; record the status). HR3, HR4, HR5 compare with a closed form f_L and therefore use only
CONVERGED learner runs.
- HR1 (general law, vectors): cosF ≥ 0.95 and nrF ∈ [0.9, 1.1] in ≥ 90% of usable (case, learner) pairs. Evaluated for
  relu (the test) and for each smooth control (must hold there too). Report the pass fraction per size as well.
- HR2 (general law, scalar): |ρᴾᴾ_L − ρ_ω| ≤ 0.05·max(1, |ρᴾᴾ_L|) in ≥ 90%.
- HR3 (smooth controls): for sizes ≤ 1e-2: max over τ ∈ {1/16..1} of |ω(τ) − τ| ≤ 0.03 in ≥ 90% of cases, and
  |ρᴾᴾ_L − f_L| ≤ 0.03·max(1, |f_L|) in ≥ 90% of (case, learner) pairs.
- HR4 (ReLU inside one linear region): among relu (case, learner) pairs whose reference path has NO switch on [0, max(1, max τ_k)]:
  |ρᴾᴾ_L − f_L| ≤ 0.03·max(1, |f_L|) in ≥ 90%. (If there are fewer than 10 such pairs: NO DATA, say so.)
- HR5 (ReLU with switches is not described by the smooth fractions, while the general law still holds): among relu
  (case, learner) pairs whose reference path has ≥ 1 switch on [0, 1]: the smooth closed form misses, |ρᴾᴾ_L − f_L| > 0.05,
  in ≥ 50% of pairs, AND HR1's condition holds in ≥ 90% of the SAME pairs.
Report (no verdict):
- R-a: ω profiles: for every case the 17 values on [0, 1]; summary per (act, size): median over cases of max|ω(τ) − τ|,
  median ω⊥, median n_sw(1), and the fraction of the total variation of ω on [0, 1] that falls in cells containing a switch.
- R-b: homogenisation: per (act, size): median over cases of ρᴾᴾ for SL(32) and DJ(0.5) next to 0.4844 and 0.3333, and of
  |ρᴾᴾ_L − f_L| over P. Stated expectation (not a rule): for relu the deviation from the smooth fractions first rises with
  size (first switches) and then falls again when switches become many.
- R-c: overshooting learners DJ(1.5), DJ(1.9): cosF, nrF, ρᴾᴾ, ρ_field, f per size.
- R-d: accuracy of the general law versus number of switches: cosF and nrF binned by n_sw(1) ∈ {0, 1–2, 3–10, 11–100, > 100}.
- R-e: v_PP against the SPEC8 nearest point, smooth controls only, sizes ≤ 1e-2 (one NP solve per case is allowed here,
  and only here): cos(v_PP, v_NP), ‖v_PP‖/‖v_NP‖.

====================================================================================================
## 7. Linear control
Model 196 → 10 linear, start init, seed 0, directions label and rand, size s₁, both signs, learners AJ(0.5), AJ(2),
SLα(2, 1), TJ(1.5, 1): ‖v_L‖/‖θ_L − θ₀‖ ≤ 1e-9 and ‖v_NP‖/‖θ_NP − θ₀‖ ≤ 1e-9 (rule H0: all), and for Part G:
‖G_L‖/‖F_test(θ_L) − F_test(θ₀)‖ ≤ 1e-9 (all learners predict the same on unseen inputs in a linear model).

====================================================================================================
## 8. Unit tests (ALL must pass before any grid; print PASS/FAIL with numbers; ALSO write the full test output to
`part9_run/tests_output.txt` immediately after the tests finish)

U1–U14: the reviewed tests, unchanged.
New tests (ground truth verified by the spec author):
- V1 solvable model F = u² − v², θ₀ = (1, 1), s = 0.02, even part, tolerance 0.01·max(1, |value|):
  AJ(−1) → −1, AJ(0.25) → 0.25, AJ(0.5) → 0.5, AJ(1) → 1, AJ(2) → 2;
  SLα(2, 1) → 0.75, SLα(8, 2) → 0.6875, SLα(4, −1) → 0.125, SLα(2, 0.5) → 0.5;
  TJ(0.5, 0.5) → 0.5, TJ(1.5, 0.5) → 0.5, TJ(0.5, 1) → 0.6, TJ(1.5, 1) → 0.7143, TJ(8, 1) → 0.9, TJ(3, 0.25) → −1;
  TM(0.5, 0.5) → 0.5, TM(1.5, 0.5) → 0.5, TM(0.5, 1) → 0.6, TM(1.5, 1) → 0.7143, TM(6, 1) → 0.875, TM(1.5, 0) → −1.
- V2 the U6 network (8-32-32-1 tanh), direction q₁ and the random direction, s as in U6, even part:
  AJ(0.5) → 0.5 (±0.01), AJ(2) → 2 (±0.03), SLα(8, 1) → 0.5625 (±0.01), TJ(1.5, 1) → 0.7143 (±0.01);
  TM(1.5, 1) on q₁ → 0.7143 (±0.02), TM(1.5, 0.5) on q₁ → 0.5 (±0.01).
- V3 AJ consistency on the U6 network: AJ(0) equals JUMP (‖Δθ‖ ≤ 1e-10); for AJ(0.5), AJ(1), AJ(2): res ≤ 1e-11 and hz ≤ 1e-7;
  AJ(1) equals the nearest point of its own outputs computed by `nearest_point` STARTED FROM θ₀ (‖θ_AJ(1) − θ_NP0‖ ≤ 1e-6·‖θ_NP0 − θ₀‖).
- V4 residual recursions on the linear control (the mapping from (c, α) to the per-step factor; relative 1e-9):
  TM(c, α), c = 0.7, α ∈ {0, 0.5, 1}: after ONE step the residual component along q_i equals q_i·(initial component), with
  q_i = (1 − c_i(1 − α))/(1 + c_iα), c_i = cλ_i/λ₁, for i ∈ {1, 2, 5};  TJ(0.7, α), α ∈ {0.5, 1}: after one step r = q·r₀ (relative 1e-8).
- V5 step-log exactness (Part R logging): on the U6 network and on a small ReLU MLP (196-32-32-10, n = 8, trained start,
  direction eigA, s = 1e-2·S₀, + sign), for SL(8), DJ(0.5), AJ(0.5), SLα(8, 1): recompute V₀Σ_k J(θ_e,k)ᵀK(θ_e,k)⁻¹b_k from
  the log with the learner's OWN evaluation points (stored for this test) and compare with V₀(θ_L − θ₀): relative error ≤ 1e-8.
- V6 reference-path prediction, smooth: U6 network, direction q₁, s as in U6, + sign: for SL(8) and DJ(0.5):
  cosF ≥ 0.9999 and |nrF − 1| ≤ 0.01; max_τ|ω(τ) − τ| ≤ 0.02 on [0, 1]; cos(v_PP, v_NP) ≥ 0.999.
- V7 ReLU inside a linear region: the small ReLU MLP of V5, direction eigA: starting at s = 1e-3·S₀ and dividing by 10
  until the reference path has no switch on [0, 2] (at most 4 divisions; report the size used): ρᴾᴾ of SL(8) → 0.4375 and of
  DJ(1.5) → −1, tolerance 0.01·max(1, |value|), single sign (+).
- V8 ReLU with switches: the same network, direction eigA, s = 1e-2·S₀, both signs: for SL(8), SL(32), DJ(0.5), DJ(1):
  cosF ≥ 0.98 and nrF ∈ [0.95, 1.05] (the spec author measured 1.0000 / 1.000 here), and in the sign whose reference path has a
  switch on [0, 1] (report which) |ρᴾᴾ(SL(8)) − 0.4375| > 0.05 (the smooth form must visibly fail where the general one holds).
- V9 held-out family: small tanh MLP (196-32-32-10, n = 8, trained start), direction eigA, s = 0.01·S₀, even part, 200 Gaussian
  held-out inputs: γ of SL(8) → 0.4375, DJ(1.5) → −1, AJ(2) → 2, tolerance 0.01·max(1, |value|), cos_test ≥ 0.999.
- V10 test-set standardisation: the mean and std applied to the held-out inputs are bit-identical to those of the training inputs.
- V11 determinism (one Part A case twice, one Part R case twice: identical to 1e-10) and UW worker identity as before.

====================================================================================================
## 9. Modes
| MODE | n | W | Part A / G1–G3 combinations | G4 | Part R | purpose |
|---|---|---|---|---|---|---|
| "tiny" | 8 | 32 | tanh-trained-s0 and tanh-init-s0; directions eigA, label (+ true for init); AJ α ∈ {0.5, 1, 2}; one SLα, one TJ, one TM | tanh-init-s0, n = 8, τ ∈ {0.25, 1} | relu-trained-s0 and tanh-trained-s0; direction eigA; sizes {1e-3, 1e-2}; grid 64 cells on [0, 2] | CPU validation, < 30 min |
| "quick" | 32 | 256 | tanh-trained-s0, tanh-init-s0 (full learner set) | tanh-init-s0, n = 32 | relu-trained-s0 + tanh-trained-s0 (full) | first GPU run |
| "full" | 32 | 256 | as section 4/5 | as G4 | as section 6 | the test |
In "tiny" the reduced grids are part of the mode definition (state them in the output); all rules are evaluated on what the mode runs.

====================================================================================================
## 10. Engineering requirements (each one answers a past failure)

1. **Results must not depend on the run finishing.** The analysis is a library function that works on whatever cases are on
   disk. The notebook has a CONFIG flag `ANALYSIS_ONLY` (default False): when True, the grid cell is skipped and only the
   analysis and save cells run. The analysis is also executed automatically after EACH Part finishes (A, then G, then R) and
   its text is appended to `part9_run/analysis_output.txt`, and `results_part9.json` + `part9_outputs.zip` are rewritten
   each time. So stopping the run at any point leaves complete tables for the finished Parts on disk.
2. **Order of work:** linear control → Part A (with G1–G3 measured in the same cases) → G4 → Part R smooth controls → Part R relu.
   Within Part R, smaller sizes first.
3. **Every iteration has a cap and a status**; no solver may loop on a failing case (the SPEC8 ReLU nearest-point solves cost
   ~400 s each). Print in the timing cell the measured cost of one case of each kind and an estimate per Part.
4. One "case" = one JSON file in `part9_run/done/` written atomically, vectors in `part9_run/vecs/` (v_L, v_NP, G_L, G_NP for
   A/G; v_L, v_pred, v_PP for R); resumable by configuration signature; failures in `part9_run/failed/` with traceback, listed
   loudly; nothing dropped silently. The signature must NOT include the ANALYSIS_ONLY flag.
5. Several GPUs: one worker process per GPU, static split of the combinations of the current Part; identical numbers to the
   in-kernel path (UW test). Progress line at least every 60 s, appended also to `part9_run/progress.log`.
6. Memory: never more than three Jacobians alive; the reference path stores only weight vectors (257 × P float64 ≈ 0.25 GB);
   held-out forward passes in chunks of 2 000 inputs; for n = 128 use Jacobian chunks of 256 rows.
7. The notebook file itself is written by nbconvert only at the end; therefore everything needed (tests output,
   analysis text, results JSON, zip) must already be on disk as plain files (items 1 and section 8).
8. float64 everywhere; no package installs; console text ASCII-safe; delivered state MODE = "quick", outputs cleared.

====================================================================================================
## 11. What to hand back
1. TESTS output verbatim (real execution of the delivered notebook, MODE = "tiny", this machine; synthetic data allowed and labelled).
2. The H-tables of Parts A, G, R from that tiny run with the numbers behind each verdict; the G3 and G4 report tables; the
   R-a … R-e report tables.
3. Evidence for: resume; ANALYSIS_ONLY on an interrupted run (kill the grid in the middle of Part R, rerun with the flag,
   show that Parts A and G tables are complete and identical to those of an uninterrupted run); failure listing; worker identity.
4. A diff statement for section 2: the list of copied functions with "identical" confirmed by a programmatic comparison
   against the original notebook, and the list of every new function.
5. Every deviation and suspected spec error with evidence (run the literal version, add a labelled SUPPLEMENTARY variant).
6. Time estimates for "quick" and "full" on one and on four RTX 3090s, per Part.
7. The section-12 checklist, each item with the line or test that guarantees it.
8. Delivered state: MODE = "quick", ANALYSIS_ONLY = False, outputs cleared, no installs, both copies identical (md5).

====================================================================================================
## 12. Mistakes of earlier rounds that must not recur (check each one)

1. Testing a simplification instead of the claim (the "dose" episode). Here: Part R's primary prediction is the vector
   v_pred built from the learner's logged evaluation positions and the reference-path field — not a scalar summary.
2. Comparing a finite-size value with a limit value. Parts A and G1 use two sizes and the extrapolated value; Part R makes
   no limit statement (single size, single sign).
3. A stop rule that can never be met (weak kernel modes): TM uses the direction-converged rule; TJ, SLα, DJ stop on the
   full residual, which is fine for function-space learners; GD in G4 has an explicit loose rule and cap.
4. A reference solver that fails on the system under test: no nearest-point solve in Part R (relu).
5. Nearest point computed for the nominal target instead of the learner's own final outputs; projector built anywhere but
   at θ₀; float32; transposed index order — all inherited from the reviewed `measure` and must stay so.
6. For Part R: τ_k must be computed from the outputs AT THE EVALUATION POINT of the step (not at the end of the step), with
   the SIGNED direction d of the case; b_k must reproduce the step exactly (V5); the reference path must be built for the
   same signed direction and the same size as the case.
7. For Part G: the held-out inputs must be standardised with the training scalars; θ_TH and θ_NP must be those of the SAME
   fibre as θ_L (from `measure_ext` of that very run); never compare learners at different training outputs.
8. In G4 never rank learners whose training residual differs (list and exclude runs above the residual bound).
9. Pairing the wrong signs or sizes in even parts / extrapolation; using cos from s₁ where the rule says s₂.
10. Losing results because the grid did not finish, or because a printout existed only inside the notebook (section 10.1, 10.7).
11. Tuning any threshold, grid, seed or stopping rule after seeing results; dropping or silently retrying cases.
12. Reporting a ratio without the cosine, or a verdict without the numbers behind it.

====================================================================================================
## 13. Amendment A3 (written after the spec author's line-by-line review of the delivered notebook; BEFORE any GPU run)

Reviewed file: `evaluation_point_tests.ipynb`, md5 3dcfe9187066d56a9690244488057e6b. Result of the review:
- every function of section 2 is character-identical to the reviewed notebook (spec author's own AST comparison);
- all new code (2 508 lines) was read line by line; no error was found that changes a tested quantity;
- the spec author ran the unit tests himself from the delivered file (MODE "quick", CPU): U1–U14 and V1–V10 pass
  (V11 still running when this amendment was written; UW needs a GPU or CPU workers);
- an independent re-implementation (own forward pass, torch.autograd Jacobian, numpy solvers; `review9/check9.py`) agrees
  with the library on Part R ReLU cases (every learner: ρᴾᴾ, ρ_field, cosF, nrF, ρ_ω to 5 decimals; ω profile to 2e-13;
  τ ranges identical for the sign −1 case, so the SIGNED direction is used) and on Part A / Part G vectors;
- single cases at the real size (n = 32, width 256) run through the library's own case runner converge with no damping.
The three "SPEC ERROR?" reports of the implementation are CONFIRMED as errors of the spec, not of the code. They are corrected
here. A3 changes NO learner definition, NO prediction, NO threshold of any rule of Parts A, G, R, and NO grid.
Everything not mentioned below stays exactly as delivered. Do not refactor or "improve" anything else.

### A3.1 α-point jump: the consistency check gets a rounding-aware tolerance (replaces the last three lines of 3.1)
Why the old rule was wrong: hz is computed from the STORED vectors, θ − θ_b. The stored θ = fl(θ_b + J_eᵀa) carries a rounding
error of up to ½·eps·|θ_i| per component, so the difference θ − θ_b has a relative error of order
   B := eps·‖θ‖₂ / ‖θ − θ_b‖₂        (eps = 2.220446049250313e-16; θ = the final point of the jump),
which has nothing to do with the solver. Late TJ steps move by about 1e-9·‖θ‖, so B is 1e-7…1e-6 and correct runs were marked
AJ-INCONSISTENT (HA3 had NO DATA in the tiny run). Measured by the spec author (`review9/probe_A.log`; solvable model, U6
network, small tanh MLP; AJ, SLα, TJ; about 1 000 jumps): whenever hz > 1e-9, hz/B lies between 0.17 and 0.28; the maximum
over all jumps of (hz − B) is 3e-15.
New rule (the ONLY status rule): after convergence compute hz exactly as now (from the stored vectors) and B as above.
   AJ-INCONSISTENT  ⇔  hz > AJ_HZ_TOL + AJ_HZ_FP·B,     AJ_HZ_TOL = 1e-7 (unchanged),   AJ_HZ_FP = 1.0 (new CONFIG constant;
   add it to the configuration signature).   If ‖θ − θ_b‖ = 0: hz := 0 and B := 0.
Implementation: put the check into ONE new pure function
   `aj_consistency(prob, thb, th, alpha, Jf=None) -> dict(hz, hz_bound, inconsistent)`        (hz_bound = B)
called by `alpha_jump` (Jf = the Jacobian at the final evaluation point θ_b + α(θ − θ_b) if the caller already has it;
otherwise the function computes it; the arithmetic of hz stays as delivered). Keep `hz_step` (step-vector variant) as a
reported number, computed as now.
Per jump store: hz, hz_bound, hz_step. Per run store in addition to the existing fields:
   `aj_hz_excess_max` = max over the run's jumps of (hz − AJ_HZ_FP·hz_bound),
   `aj_lit_flags`     = number of jumps with hz > 1e-7 (the ORIGINAL literal rule; report only).
`status_supp` and the "SUPPLEMENTARY" analysis blocks stay in the code with unchanged logic (they print only when a
supplementary status differs from the status; after A3.1 that should not happen — if it does, it is printed). Change their
label text from "see SPEC ERROR? hz floor" to "A3.1: status from hz on the step vector".
The unit-test helper `_ok_runs` must now use the run's `status` (the A3.1 status), not `status_supp`.
Part A report table "alpha-point jumps": add the columns max `aj_hz_excess_max` and total `aj_lit_flags`.

### A3.2 TM inner iteration: stop rule with a rounding floor (replaces "stop when ‖g‖ ≤ 1e-12·‖δ‖ + 1e-16" in 3.4)
Why the old rule was wrong: g = δ + (η/N)J_eᵀ(F(θ_e) − t) is evaluated in floating point; rounding of θ_e = θ + αδ and of F
leaves ‖g‖ of order c·eps·‖θ‖ even at the exact solution, far above 1e-12·‖δ‖ + 1e-16 when δ is small. TM(6, 1) therefore ran
60 inner iterations (60 Jacobians) on most steps without changing δ.
New inner stop: accept δ when
   (i)  ‖g‖₂ ≤ TM_INNER_TOL_REL·‖δ‖₂ + TM_INNER_TOL_ABS            (unchanged constants 1e-12 and 1e-16),   OR
   (ii) ‖g‖₂ ≤ TM_INNER_FP·eps·(1 + c)·‖θ‖₂                         (θ = the weights at the START of the step; c = the run's c),
with TM_INNER_FP = 8.0 (new CONFIG constant; add it to the configuration signature). Inside the loop: evaluate g; test (i);
then test (ii); then the cap; then update δ (the order as delivered, with (ii) inserted after (i)). Cap 60 → TM-INNER-FAILED
as before.
Per run store in addition: `tm_inner_floor_accepts` = number of steps accepted by (ii) and not by (i);
   `tm_floor_g_max` = max over those steps of ‖g‖/(eps·‖θ‖)   (dimensionless; ≤ TM_INNER_FP·(1 + c) by construction; 0 if none).
Measured by the spec author with exactly this rule (`review9/probe_B.log`, solvable model): TM(6, 1): inner iterations
625 → 27, capped steps 10 → 0, vertical displacement changed by 7e-11 (relative), even ρ identical to 6 decimals (0.875017);
the other TM settings likewise (ρ unchanged to 6 decimals).
The report "TM inner solves that hit the cap" and the SUPPLEMENTARY "HA4 restricted to units with no TM-INNER-FAILED step"
stay; add to that report the group's total `tm_inner_floor_accepts` and its max `tm_floor_g_max`.

### A3.3 V9 (held-out family): the cosine condition is sign-aware
The rule "cos_test ≥ 0.999" cannot hold for a learner with f < 0. Correct rule: cos_test·sign(value) ≥ 0.999 (as in `matches`).
This is what the delivered notebook already does; keep it and change the comment/label from "SPEC ERROR?" to "A3.3".

### A3.4 Part R: two more reported numbers per (case, learner), and a note printed with HR2 (NO rule changes)
Background (found by the spec author's independent check; small ReLU MLP, eigA, s = 1e-2·S₀, sign −1, two switches in the
first 1/16 of the path): DJ(0.5): ρᴾᴾ = 0.9236, ρ_field = 0.9234 (cosF = 1.000000), but ρ_ω = 0.5024. The step vectors b_k are
not exactly parallel to d (after a switch the linearised increment has an off-line part), and K⁻¹ weights soft directions
λ₁/λ_i times more than the stiff direction d = eigA, so the off-line part of b_k contributes as much as the on-line part. The
scalar form P-omega drops it. HR2 may therefore fail for relu while HR1 (the general law) holds.
Add to each learner record of Part R:
   `bperp_max` = max_k ‖b_k − β_k·s·d‖₂/‖b_k‖₂     (0 when ‖b_k‖ = 0; β_k = d·b_k/s as now; d the SIGNED direction),
   `rho_perp`  = ρ_field − ρ_ω.
Per-case learner table: two more columns (bperp_max, rho_perp). Directly under each HR2 verdict line print:
   "HR2 note (A3.4, written before the GPU run): P-omega keeps only the component of each b_k along d. Usable pairs:
    median / max bperp_max = … / …; HR2-passing pairs: median |rho_perp| = …; HR2-failing pairs: median |rho_perp| = …,
    median bperp_max = …. A FAIL of HR2 together with a PASS of HR1 means that the scalar reduction is invalid there,
    not the general law."
HR2 itself (rule, threshold, verdict) is unchanged and stays in the verdict table.

A3.4b (added after the amendment was first sent; report only): status table R-f.
Background: in the spec author's cross-check (small ReLU MLP, eigA, s = 1e-3·S₀, sign −1, ONE switch at τ ≈ 0.27) the library's
AJ(0.5) did not converge: the outer iteration oscillated with step ≈ 1e-4 (outer iteration 74 and counting) while AJ(1), SL, DJ
converged. Reason: with a kink on the way, neither activation pattern is self-consistent for the evaluation point (the
α-point equation has no solution on either side of the kink) — the same structure that made the nearest-point solve fail on
ReLU. This is a result, not a bug: AJ-FAILED must stay recorded, excluded from the rules and listed, exactly as now.
Add report R-f: for each activation and each size, one row per learner of R_LEARNERS with the counts of learner statuses over
the cases (CONVERGED / UNCONVERGED / AJ-FAILED / AJ-INCONSISTENT / JUMP-FAILED / DIVERGED / PRED-FAILED / other), and for the
AJ-type learners (AJ, SLα) the median and max number of outer iterations and of dampings. Store it in the report tables as "R_f".

### A4 (2026-11-22, after the first GPU start): unit test V5 must not require α-point jumps to converge across ReLU kinks
What happened on the GPU (real MNIST): all tests passed except V5. Its ReLU half (small ReLU MLP 196-32-32-10, n = 8, trained,
eigA, s = 1e-2·S₀, + sign) reported SL(8) 2.4e-12 and DJ(0.5) 4.4e-12 (exact), but AJ(0.5): AJ-FAILED (0 steps logged) and
SLα(8, 1): AJ-FAILED at sub-jump 3. With MNIST inputs this path has switches (V8 on the same machine: 4 switching cells on
[0, 1] for the + sign), and an α-point jump can have no self-consistent solution at a kink (A3.4b). So V5 as written by the
spec author demanded something that is not true for ReLU; the logging itself is exact. This is a SPEC error in a unit test;
no measured quantity, learner, rule or grid is affected.
New V5 (only the ReLU half changes; the U6 half and the two "bitwise equal weights" checks stay exactly as they are):
 (i) small ReLU MLP, eigA, s = 1e-2·S₀, + sign:
     - SL(8) and DJ(0.5): status CONVERGED and relative error ≤ 1e-8 (unchanged).
     - AJ(0.5) and SLα(8, 1): run them. CONVERGED → relative error ≤ 1e-8 required. AJ-FAILED → NOT a test failure: print
       "AJ-FAILED (allowed on ReLU: no self-consistent alpha-point at a kink, A4)" together with the run's aj_outer_max,
       aj_dampings and aj_why. Any other status (AJ-INCONSISTENT, …) → test FAILS.
 (ii) same network and direction, + sign, at the size found exactly as in V7 (start at s = 1e-3·S₀ and divide by 10 until the
     reference path has no switch on [0, 2]; at most 4 divisions; report the size): AJ(0.5) and SLα(8, 1) must be CONVERGED
     with relative error ≤ 1e-8. (If no such size is found within 4 divisions: test FAILS, as in V7.)
Hand back: the changed function list (expected: t_V5 only, plus a shared helper if you factor the V7 size search — then V7's
printed numbers must be unchanged), TESTS output (MODE "tiny"), md5, delivered state as always. Do NOT rerun the tiny grid.

### A5 (2026-11-22, follows the implementation's own observation on A4): V5's bound must be rounding-aware (same principle as A3.1)
Testing measured in A4 part (ii) relative errors 2.2e-9 (AJ(0.5), 1 jump) and 5.5e-9 (SLα(8, 1), 8 jumps) at s = 1e-3·S₀, against
2e-11 at 1e-2·S₀: a factor 100 for a factor 10 in size, i.e. the error is an ABSOLUTE rounding error of the stored weights
(about 0.22·eps·‖θ‖ per jump, as in A3.1) divided by ‖v_L‖ ∝ s². On MNIST the no-switch size may be 1e-4·S₀ or smaller, where the
fixed bound 1e-8 would fail although the log is exact. Fixed-size tolerance against a scale-dependent quantity: the same
mistake as before, corrected here BEFORE it costs another GPU start.
New bound for EVERY relative-error check of V5 (U6 half, ReLU (i), ReLU (ii)):
   rel err ≤ 1e-8 + V5_FP·m·eps·‖θ_L‖₂/‖v_L‖₂,     V5_FP = 2.0 (a constant local to t_V5, not a CONFIG/signature constant),
   m = number of logged steps of that run, θ_L = the run's final weights, v_L = V₀(θ_L − θ₀), eps = FP_EPS.
Print, next to each relative error, the bound used. Everything else in V5 stays as in A4.
Hand back as for A4 (changed function: t_V5 only; TESTS in MODE "tiny"; md5; both copies; no grid rerun), and state the
largest ratio (rel err)/(bound) over all V5 checks.

### A3.5 Text only
- Timing cell caveat: replace the sentence about TM(6, 1) hitting the inner cap by "TM cases cost about (steps) × 1–3 Jacobians".
- Markdown note in the TESTS section of the notebook ("Note on TM … SPEC ERROR?"): replace it by two lines describing A3.1 and A3.2.
- Docstrings/comments/printed labels that say "SPEC ERROR?" for these three items: say "A3.1" / "A3.2" / "A3.3" instead.
- First markdown cell: add one line "Amendment A3 (rounding-aware consistency check, TM inner stop with rounding floor, V9
  sign, two Part R report columns) is implemented."

### A3.6 New unit test V12 (A3.1 and A3.2 are tested, including a negative control)
V12, on the U6 network, direction q₁, size s as in U6, + sign:
 (a) run TJ(0.5, 0.5): run status CONVERGED; `aj_lit_flags` ≥ 1 (the literal rule would have flagged it; print the number);
     `aj_hz_excess_max` ≤ 1e-9.
 (b) negative control: θ := AJ(θ₀, t; 0.5) (converged). Let θ_e = θ₀ + 0.5(θ − θ₀), V_e the vertical projector at θ_e
     (V_e w = w − J(θ_e)ᵀK(θ_e)⁻¹J(θ_e)w), w a standard Gaussian vector from gen(9200), u = V_e w/‖V_e w‖, and
     θ_bad = θ + 1e-5·‖θ − θ₀‖·u.  `aj_consistency(prob, θ₀, θ_bad, 0.5)` must report inconsistent = True with hz between
     0.9e-5 and 1.1e-5;  `aj_consistency(prob, θ₀, θ, 0.5)` must report inconsistent = False.
 (c) run TM(6, 1) on the solvable model (as in V1, + sign): status CONVERGED, `tm_inner_capped` = 0, `tm_inner_total` ≤ 3·steps.
V1 and V2 stay as they are; their printed values must not change beyond the 4th decimal.

### A3.7 What to hand back for A3
1. A programmatic statement of what changed: the list of functions whose source differs from the delivered notebook
   (md5 3dcfe918…) and "all other functions identical" confirmed by comparison. Expected to change: alpha_jump, the AJ
   accumulators, run_TM, R_learner, the analysis/report functions that print the new columns and the HR2 note, _ok_runs,
   label text in t_V9, TESTS_V, SCIENCE_KEYS9, the timing text; new: aj_consistency, t_V12. Anything else: say why.
2. TESTS output verbatim (MODE "tiny"): U1–U14, V1–V12 (+ UW status).
3. The tiny grid rerun from scratch (new signature): the verdict table, and explicitly: on how many units HA3, HG1, HG2 are now
   evaluated (HA3 was NO DATA); the number of AJ-INCONSISTENT runs (expected 0) and the total `aj_lit_flags`; TM inner
   iterations in total before/after; the HR2 note lines.
4. Any deviation or newly suspected spec error, with evidence, as before.
5. Delivered state: MODE = "quick", ANALYSIS_ONLY = False, outputs cleared, no installs; md5 of the new notebook; both copies
   identical (project folder and Downloads).
