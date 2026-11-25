# SPEC8 — Test of the alignment-fraction law on a real network

Deliverable: one Jupyter notebook, `alignment_fraction_test.ipynb`, saved to
`the working folder` and copied into this project folder. It will be executed on a rented
Linux GPU machine with `jupyter nbconvert --execute` (PyTorch already installed; the notebook must
NOT install packages). It must also run on a CPU-only Windows machine in MODE="tiny" for validation.

This spec is self-contained. Read all of it before writing code. Where the spec gives a formula,
an index order or a normalisation, implement exactly that; section 9 lists the mistakes that would
make the test meaningless.

---------------------------------------------------------------------------------------------------
## 1. What is being tested (plain statement)

Fix a network with weights θ₀ (the START) and a fixed set of training inputs. Let F(θ) be the vector
of all outputs on those inputs (length N = n·C). A learner starts at θ₀ and ends at θ_L with outputs
F_L = F(θ_L). Many other weight vectors have exactly the same outputs F_L; that set is the FIBRE of F_L.

Split any weight displacement w at the start into
  horizontal part  H₀w = J₀ᵀK₀⁻¹J₀ w      (moves the outputs, to first order)
  vertical part    V₀w = w − H₀w          (leaves the outputs unchanged, to first order)
with J₀ = ∂F/∂θ at θ₀ (N×P) and K₀ = J₀J₀ᵀ (N×N). **The projector is always built at the START θ₀.**

One special point on the fibre: the NEAREST POINT θ_NP = the point of the fibre of F_L closest to θ₀.

CLAIM (valid to leading order in the size of the move): the vertical part of the learner's displacement
is a fixed multiple of the vertical part of the nearest point's displacement,

    V₀(θ_L − θ₀)  =  f_L · V₀(θ_NP − θ₀)  (+ a path-shape term that vanishes for straight output paths),

and the multiple f_L ("alignment fraction") is known in closed form:

| learner | f |
|---|---|
| one exact jump (JUMP, sec. 3.2) | 0 |
| straight-line transport in m exact jumps, SL(m) | ½(1 − 1/m) |
| damped jumps with step fraction c, DJ(c) | (1 − c)/(2 − c) |
| gradient descent with step factor c, target along kernel eigenvector q_i | (1 − c_i)/(2 − c_i), c_i = c·λ_i/λ₁ |
| heavy-ball momentum μ, same target | 1 − [(1−μ)²(1+μ) + 2μc_i] / [(1−μ)²(2(1+μ) − c_i)] |
| minimiser of loss + (γ/2)‖θ−θ₀‖² (PROX) | 1 (exactly, at any move size) |

For learners whose output path is not a straight line (gradient descent or minibatch SGD on a generic
target) the claim is the full path law of section 5 (H5), which needs the path matrix C.

The measured quantity is always

    ρ_L  = ⟨v_L, v_NP⟩ / ⟨v_NP, v_NP⟩ ,      cos_L = ⟨v_L, v_NP⟩ / (‖v_L‖‖v_NP‖),
    v_L  = V₀(θ_L − θ₀),   v_NP = V₀(θ_NP(F_L) − θ₀),

where θ_NP(F_L) is the nearest point **of the learner's own final outputs F_L** (never of the nominal
target). The test asks: is ρ_L equal to the predicted f_L, with cos_L = ±1?

---------------------------------------------------------------------------------------------------
## 2. Fixed setup

- torch, float64 EVERYWHERE (`torch.set_default_dtype(torch.float64)`; data, weights, Jacobians, solves).
- Data: MNIST training images, average-pooled 28×28 → 14×14 (2×2 mean), flattened to 196, standardised
  with the mean and std of the selected images (one scalar mean, one scalar std). For seed s select n
  images with `torch.Generator().manual_seed(1000+s)` `randperm`. If MNIST cannot be loaded (no
  torchvision / no network): fall back to N(0,1) inputs with random labels, print a loud warning, and
  store `data_source="synthetic"` in every result. (The law does not depend on the data; synthetic
  results are valid but must be labelled.)
- Model: MLP 196 → W → W → 10 with biases, activation `act` ∈ {tanh, gelu, relu}; default PyTorch
  `nn.Linear` initialisation with `torch.manual_seed(s)`.
- Outputs flattened row-major: index a = (input index)·C + (class index), N = n·C.
- Loss  L(θ; t) = (1/(2N)) ‖F(θ) − t‖².   Gradient ∇L = Jᵀ(F − t)/N.
- J(θ): N×P float64 Jacobian (torch.func `jacrev`+`vmap`, or per-output backward). K(θ) = J Jᵀ.
- λ₁ ≥ λ₂ ≥ … eigenvalues of K₀ = K(θ₀), eigenvectors q₁, q₂, … (unit, `torch.linalg.eigh`, re-sorted descending).
- Step factor: gradient descent with learning rate η moves kernel mode i by the factor (1 − ηλ_i/N).
  **"GD with step factor c" means η = c·N/λ₁(K₀), fixed for the run.** (Unit test U8 checks this mapping.)

MODES (one variable `MODE` at the top):
| MODE | n | W | acts | starts | seeds | purpose |
|---|---|---|---|---|---|---|
| "tiny"  | 8  | 32  | tanh | trained | 0 | local CPU validation, must finish in < 20 min on CPU |
| "quick" | 32 | 256 | tanh | trained | 0 | first run on the GPU box |
| "full"  | 32 | 256 | tanh, gelu, relu | init, trained | 0, 1, 2 | the test |

STARTS:
- "init": θ₀ = the freshly initialised weights.
- "trained": θ₀ = weights after fitting the true one-hot MNIST labels by DJ(0.5) iterations (sec. 3.3)
  from the initialisation until mse = ‖F − Y‖²/N ≤ 1e-6 (cap 200 iterations; record iterations and final mse;
  if the cap is hit keep the weights and record it — the start point only has to be a fixed point in
  weight space, its provenance does not matter).
- Record for every start: λ₁, λ_N, tr K₀, cond(K₀). If λ_N/λ₁ < 1e-11 mark the (act, start, seed) combination
  ILL-CONDITIONED, skip it and list it (do not regularise K to rescue it).

---------------------------------------------------------------------------------------------------
## 3. Building blocks (implement once, in a library cell; every test uses these same functions)

### 3.1 Vertical projector at the start
V₀w = w − J₀ᵀ (K₀⁻¹ (J₀ w)), using one Cholesky (or eigh) factorisation of K₀ computed once per start.

### 3.2 Exact jump  JUMP(θ_b, t)
The point θ = θ_b + J_bᵀa (a ∈ ℝᴺ, J_b = J(θ_b)) with F(θ) = t: displacement exactly horizontal at θ_b.
Solve for a: start a = 0; repeat a ← a + K_b⁻¹(t − F(θ_b + J_bᵀa)) (K_b = J_bJ_bᵀ factorised once);
stop when ‖F − t‖∞ ≤ 1e-11. If not reached in 200 iterations, switch to Newton on a with matrix
M = J(θ)J_bᵀ (recomputed each iteration, cap 50). Status CONVERGED / FAILED. Returns θ and iteration counts.
TH(t) := JUMP(θ₀, t) (the "one exact jump" reference; by construction V₀(θ_TH − θ₀) = 0).

### 3.3 Learners (all start from θ₀; nominal target t)
- SL(m): θ ← θ₀; for k = 1..m: θ ← JUMP(θ, F₀ + (k/m)(t − F₀)).
- DJ(c): repeat θ ← θ − c·J(θ)ᵀK(θ)⁻¹(F(θ) − t) until ‖F − t‖₂ ≤ 1e-7·‖t − F₀‖₂ (cap 400 steps).
- GD(c): θ ← θ − η∇L(θ; t), η = c·N/λ₁(K₀).
- HB(c, μ): u ← μu − η∇L(θ; t); θ ← θ + u; u₀ = 0; η = c·N/λ₁(K₀); μ = 0.9. (c is the RAW step factor.)
- SGD(c, B): minibatches of B inputs (all C outputs of those inputs), reshuffled every epoch with
  `torch.Generator().manual_seed(7000+seed)`; step θ ← θ − η∇L_B with L_B = (1/(2BC))‖F_B − t_B‖², η = c·N/λ₁(K₀).
- ADAM: `torch.optim.Adam(lr=1e-3)` defaults, full batch, loss L.
- PROX(γ̃): the stationary point of L(θ; t) + (γ/2)‖θ − θ₀‖² with γ = γ̃/N. Iterate from θ₀:
  θ ← θ₀ + JᵀK̃⁻¹[(t − F) + J(θ − θ₀)], K̃ = K(θ) + γ̃I, until ‖Jᵀ(F − t)/N + γ(θ − θ₀)‖₂ ≤ 1e-10·‖J₀ᵀ(F₀ − t)/N‖₂ (cap 200).
Stopping rules for GD / HB / SGD / ADAM are given per test. Every learner returns θ_L, its step count and status.
A learner that hits its cap is UNCONVERGED: it is still measured, but excluded from rules that assume
convergence (H3, H4), and listed.

### 3.4 Nearest point  NP(F_L)
Minimise ‖θ − θ₀‖² subject to F(θ) = F_L. Iterate from θ = θ_L (the learner's endpoint is already on the fibre;
for TH-type references start from θ₀):
   θ ← θ₀ + J(θ)ᵀK(θ)⁻¹[(F_L − F(θ)) + J(θ)(θ − θ₀)]
until BOTH ‖F(θ) − F_L‖∞ ≤ 1e-10 and stat := ‖V_θ(θ − θ₀)‖₂/‖θ − θ₀‖₂ ≤ 1e-8, where V_θ is the vertical
projector **at the current θ** (this is the stationarity condition of the nearest point; it is NOT V₀).
If the residual increases in an iteration, halve the step (θ_new = θ + s·(θ_iter − θ), s = 1, ½, ¼, … ≥ 2⁻¹⁰).
Cap 300 iterations → NP-FAILED (record, exclude the case from rules, list it).
Runtime assertion for every case: ‖θ_NP − θ₀‖ ≤ ‖θ_L − θ₀‖·(1 + 1e-9). If violated, the case is INVALID (list it).

### 3.5 The measurement (the same function for every learner)
Input: θ₀, J₀, K₀ factor, θ_L. Steps:
1. F_L = F(θ_L); ΔY = F_L − F₀.
2. θ_NP = NP(F_L); θ_TH = JUMP(θ₀, F_L).
3. v_L = V₀(θ_L − θ₀); v_NP = V₀(θ_NP − θ₀).
4. Store: ρ, cos, ‖v_L‖, ‖v_NP‖, ‖θ_L−θ₀‖, ‖θ_NP−θ₀‖, ‖θ_TH−θ₀‖, ‖ΔY‖, NP iterations/stat/residual, and
   for g ∈ {λ₁(K), tr K}: g(θ_L), g(θ_NP), g(θ_TH), g(θ₀), and R_g = (g(θ_L) − g(θ_TH))/(g(θ_NP) − g(θ_TH)).
5. Resolution floor: floor = ‖V₀(θ_TH − θ₀)‖ (should be ~1e-12). If ‖v_NP‖ < 1e4·max(floor, 1e-13·‖θ_NP−θ₀‖)
   mark the case BELOW-RESOLUTION, exclude from rules, list it.

EVEN PART (used by all rules of the small-move tests): each case is run with the two nominal targets
t± = F₀ ± s·d. With v_L(±), v_NP(±) from the two runs (each with its own F_L and its own NP):
   v_L^e = ½[v_L(+) + v_L(−)],  v_NP^e = ½[v_NP(+) + v_NP(−)],
   ρ = ⟨v_L^e, v_NP^e⟩/‖v_NP^e‖²,  cos = ⟨v_L^e, v_NP^e⟩/(‖v_L^e‖‖v_NP^e‖),  nr = ‖v_L^e‖/‖v_NP^e‖.
(The leading term is even in s; averaging cancels the next order.) Also store the single-sign values.

### 3.6 Path matrix and the full law (used only by H5)
While a learner runs, with F evaluated on the FULL training set after every step:
   C += outer(F_{k+1} − F_k , F_k − F₀)        # C[a,b] = (increment)_a · (position before the step)_b
   Q += outer(F_{k+1} − F_k , F_{k+1} − F_k)
Assert at the end (relative 1e-9): C + Cᵀ = ΔYΔYᵀ − Q.
LAW(C) at θ₀: with ξ_e(θ) := J(θ)ᵀK(θ)⁻¹e and the directional derivative
   Dξ_a[w] := (ξ_a(θ₀ + δŵ) − ξ_a(θ₀ − δŵ))·‖w‖/(2δ),  ŵ = w/‖w‖,  δ = 1e-4 (a length in weight space),
LAW(C) = V₀ Σ_k S_k · Dξ_{a_k}[ ξ_{b_k}(θ₀) ], where C = Σ_k S_k a_k b_kᵀ is the SVD of C (a_k = left singular
vector = INCREMENT side, b_k = right singular vector = POSITION side). Keep singular values S_k ≥ 1e-5·S₁,
at most 96 terms; store the kept count and Σ kept S_k² / Σ S_k². (Each term costs two Jacobians.)
The order (derivative of ξ_{increment side}, taken along ξ_{position side}) is essential; unit test U11 checks it.

---------------------------------------------------------------------------------------------------
## 4. Targets

For a start θ₀ (outputs F₀): let Y_new = one-hot of labels drawn with `torch.Generator().manual_seed(2000+seed)`,
S₀ = ‖Y_new − F₀‖₂ (the size of a full random relabelling).
Directions d (unit vectors in ℝᴺ):
- eigA, eigB, eigC, eigD: the eigenvector q_i of K₀ whose λ_i/λ₁ is closest to 1, 0.3, 0.1, 0.01 respectively
  (store i and λ_i/λ₁; if two requests give the same i, take the next unused index).
- label: (Y_new − F₀)/S₀.
- rand: Gaussian vector from `manual_seed(3000+seed)`, normalised.
Sizes: s₁ = 0.02·S₀ and s₂ = 0.01·S₀. Nominal targets t± = F₀ ± s·d.
Small-move grid: all six directions at s₁ (both signs); additionally eigA and label at s₂ (both signs).

---------------------------------------------------------------------------------------------------
## 5. Tests and pre-registered rules

A FAIL is a valid result. Change no threshold, grid, seed or stopping rule to alter a verdict. If you believe a rule
or formula here is wrong: run it literally, report it as "SPEC ERROR?" with evidence, and add a clearly labelled
SUPPLEMENTARY variant next to it. Rules are evaluated separately for each (act, start), pooling seeds, directions
and sizes; print the count passed/total and every failing case. tanh and gelu are the primary arms; relu is run with
the same rules and reported as "robustness (non-smooth activation)".

Predictions f: SL(m): ½(1−1/m).  DJ(c): (1−c)/(2−c).  GD(c) on eig-direction i: (1−c_i)/(2−c_i), c_i = cλ_i/λ₁.
HB(c, μ): 1 − [(1−μ)²(1+μ) + 2μc_i]/[(1−μ)²(2(1+μ) − c_i)].  PROX: 1.
A case "matches" when |ρ − f| ≤ tol·max(1, |f|) AND (if |f| ≥ 0.05) cos·sign(f) ≥ 0.99; if |f| < 0.05 the cosine
condition is replaced by nr ≤ 0.05 + tol.

- **H1 (halfway and the sub-step ladder)** — SL(m), m ∈ {2, 8, 32}, all six directions, all sizes. tol = 0.02.
  Rule: ≥ 90% of cases match. (Predictions 0.25, 0.4375, 0.484.)
- **H2 (step-fraction curve)** — DJ(c), c ∈ {0.25, 0.5, 1.0, 1.5, 1.9}, all six directions, all sizes. tol = 0.02.
  Rule: ≥ 90% match. (Predictions 0.4286, 0.3333, 0, −1, −9.)
- **H3 (gradient descent, one kernel direction)** — GD(c), c ∈ {0.25, 1.0, 1.5, 1.9}, directions eigA–eigD, all sizes.
  Stop when ‖F − t‖₂ ≤ 1e-6·s (cap 300 000 steps). tol = 0.03. Rule: ≥ 90% of CONVERGED cases match, and at least
  75% of the planned cases converged.
- **H4 (momentum)** — HB(c, 0.9), c ∈ {0.005, 0.0105, 0.1, 1.0}, directions eigA–eigD, size s₁.
  Stop when ‖F − t‖₂ ≤ 1e-6·s and ‖u‖₂ ≤ 1e-9·‖θ_L − θ₀‖ (cap 300 000). tol = 0.05. Same rule form as H3.
- **H5 (full path law on generic targets, including minibatch SGD)** — directions label and rand, size s₁, learners
  GD(0.25), GD(1.5), SGD(0.25, B = 8). Stop when ‖F − t‖₂ ≤ 1e-3·s or 20 000 steps (no convergence requirement:
  everything is measured on the learner's own final outputs). With C± from the two signs, C^e = ½(C₊ + C₋):
  predicted v = LAW(C^e); compare with the measured v_L^e: cosine and norm ratio.
  Rule: cosine ≥ 0.98 and norm ratio ∈ [0.9, 1.1] in ≥ 90% of cases.
  Also for GD only (report + rule H5b): C_lin := the path matrix of the LINEARISED run — iterate
  r_{k+1} = (I − ηK₀/N) r_k from r₀ = F₀ − t for the same number of steps, F_k − F₀ = r_k − r₀, accumulate C as in 3.6.
  H5b rule: ‖C₊ − C_lin,₊‖_F/‖C₊‖_F ≤ 0.05 in ≥ 90% of GD cases.
  Report also ρ, cos, nr of these learners (no rule: for a curved path ρ need not equal a closed form).
- **H6 (proximity penalty reaches the nearest point)** — PROX(γ̃), γ̃ ∈ {1e-3·λ₁, 1e-2·λ₁}, all six directions, size s₁,
  sign + only (single-sign ρ). Rule: ρ ∈ [0.98, 1.02] and cos ≥ 0.999 in ≥ 90% of cases. (This is close to a
  consistency check of the NP solver and is labelled as such in the report.)
- **H7 (one exact jump is the zero point)** — the endpoint of DJ(1.0) and of SL(1) ≡ TH: nr ≤ 0.05 in ≥ 90% of cases
  (covered by H2's c = 1 cases; print it separately).

Report-only (no verdicts):
- R1 ADAM (stop ‖F − t‖₂ ≤ 1e-3·s or 5 000 steps) and SGD(0.25, 8) on label and rand at s₁: ρ, cos, nr, ‖v_L‖/‖θ_L−θ₀‖.
- R2 size ladder (does it survive large moves?): direction label, sign + only, s/S₀ ∈ {0.05, 0.1, 0.25, 0.5, 1.0},
  learners SL(32), DJ(0.25), GD(0.25), GD(1.5), HB(0.1, 0.9) (GD/HB stop at ‖F − t‖₂ ≤ 1e-3·s or 50 000 steps):
  single-sign ρ, cos, nr, R_λ₁, R_trK, NP status. Stated expectation (not a rule): ρ of SL(32) stays in [0.4, 0.6] up to s/S₀ = 0.25.
- R3 scalar version: R_λ₁ and R_trK next to ρ for every H1–H4 case (median |R_g − ρ|).
- R4 size convergence: for eigA and label, |ρ − f| at s₁ and at s₂ for H1–H2 learners (the error should shrink with s).
- R5 distances: ‖θ_L − θ₀‖/‖θ_NP − θ₀‖ per learner (must be ≥ 1).
- R6 spectrum and conditioning of every start; eigen-index chosen for eigA–D.

Linear control (once, in every MODE): model 196 → 10 linear (no hidden layer), same data, start = init, directions
label and rand, size s₁, learners SL(8), DJ(0.5), GD(1.5), HB(0.1): ‖v_L‖/‖θ_L − θ₀‖ ≤ 1e-9 and ‖v_NP‖/‖θ_NP − θ₀‖ ≤ 1e-9
for all (the vertical part does not exist in a linear model). Rule H0: all pass.

---------------------------------------------------------------------------------------------------
## 6. Unit tests (a TESTS cell; ALL must pass before any grid cell runs; print PASS/FAIL with numbers)

Ground truth exists for this pipeline, so the tests check the WHOLE measurement, not only parts of it.
Implement the building blocks generically (they take callables F(θ), J(θ)) so the same code runs on:

- U1 Jacobian: against central finite differences on 20 random entries, relative error ≤ 1e-6 (MLP, each activation).
- U2 Projector: ‖V₀J₀ᵀe‖/‖J₀ᵀe‖ ≤ 1e-9 for 5 random e; V₀V₀w = V₀w to 1e-9.
- U3 JUMP: residual ≤ 1e-11 and ‖V_b(θ − θ_b)‖/‖θ − θ_b‖ ≤ 1e-9.
- U4 NP: residual ≤ 1e-10, stat ≤ 1e-8, and ‖θ_NP − θ₀‖ ≤ distance of TH, SL(8), DJ(0.5) endpoints on the same fibre.
- U5 **Exactly solvable model** F(u, v) = u² − v², θ₀ = (1, 1), N = 1, target F₀ ± 0.02, run through the SAME
  measurement function (even part): SL(2) → 0.25, SL(8) → 0.4375, DJ(0.5) → 0.3333, DJ(1.5) → −1, DJ(1.9) → −9,
  GD(0.5) → 0.3333, GD(1.5) → −1, HB(0.05, 0.9) → −1.9067, HB(0.0105, 0.9) → 0.000 (|ρ| ≤ 0.01), PROX → 1.
  Tolerance |ρ − value| ≤ 0.01·max(1, |value|). (These values were verified independently.)
- U6 Small tanh network (inputs 8, two hidden layers of 32, scalar output, n = 16 Gaussian inputs, seed 0, start =
  weights after DJ(0.5)-fitting random Gaussian targets; s = 0.05·‖t_fit‖, direction = top eigenvector and one random
  direction): SL(8) within 0.01 of 0.4375, DJ(1.5) within 0.02 of −1, GD(1.9) on the top eigenvector within 0.1 of −9.
- U7 Linear control (sec. 5): H0 holds on a 20-parameter linear model.
- U8 Step-factor mapping: on the linear control, after ONE GD(c) step the residual component along q_i equals
  (1 − cλ_i/λ₁) times its initial value (relative 1e-9) for c = 0.7 and i ∈ {1, 2, 5}; and the HB residual sequence along
  q₁ obeys r_{k+1} = (1 + μ − c)r_k − μr_{k−1} with r_{−1} = r₀ for 5 steps (relative 1e-9).
- U9 Bookkeeping identity C + Cᵀ = ΔYΔYᵀ − Q (relative 1e-9) for a GD and an SGD run.
- U10 LAW on a straight path: in the U6 network, cos(LAW(ΔYΔYᵀ), v_NP^e) ≥ 0.999 and norm ratio ∈ [0.97, 1.03] at s = 0.05·‖t_fit‖.
- U11 LAW index order (antisymmetric part): in the U6 network take two orthonormal directions p, q (eigenvectors 2 and 12),
  ε = 0.05. Loop A: four exact jumps to F₀+εp, F₀+εp+εq, F₀+εq, F₀. Loop B: the same with p and q exchanged.
  w := θ_end(A) − θ_end(B) (outputs are equal at the end, so w is vertical up to higher order).
  Path matrices: C_A − C_B = 2ε²(q pᵀ − p qᵀ) (verify numerically from the accumulated C to 1e-9).
  Test: cos(V₀w, LAW(C_A − C_B)) ≥ 0.99, AND cos(V₀w, LAW((C_A − C_B)ᵀ)) ≤ −0.99. (The second line proves the test
  can tell the two index orders apart.)
- U12 Determinism: one H1 case run twice gives identical ρ (difference ≤ 1e-10).
- U13 SVD truncation in LAW: for a random rank-3 C, LAW via SVD equals the direct column-by-column evaluation
  Σ_j V₀Dξ_{Ce_j}[ξ_{e_j}] (relative 1e-6) on the U6 network.

---------------------------------------------------------------------------------------------------
## 7. Notebook structure and engineering

Cells: (1) title + how to run; (2) CONFIG (MODE and every constant of this spec in one place); (3) library
(`%%writefile part8lib.py`, appended with `%%writefile -a` if split; the kernel and any worker import this one file);
(4) environment check (python/torch/CUDA/GPU names, float64 smoke test, libraries present; loud message if no GPU;
never installs anything); (5) TESTS; (6) starts (build or load every θ₀, spectra); (7) grid run; (8) analysis: H-table
and R-tables; (9) save.
- Units of work: one "case" = (act, start, seed, direction, size, sign, learner) → one JSON file
  `part8_run/done/<name>.json` written atomically, holding every stored scalar of sec. 3.5 plus settings and timing.
  The vectors v_L and v_NP of each case are saved to `part8_run/vecs/<name>.pt` (float64) so even parts and any
  later re-analysis can be done without re-running; C matrices likewise for H5. θ₀, J₀ are NOT saved.
- Resumable: a case with an existing done-file and matching configuration signature is skipped. Failures go to
  `part8_run/failed/<name>.json` with the traceback and are listed loudly; analysis runs on what finished.
- Several GPUs: if `torch.cuda.device_count() ≥ 2`, start one worker subprocess per GPU (static split of the
  (act, start, seed) combinations by index modulo the number of GPUs, each pinned with CUDA_VISIBLE_DEVICES);
  otherwise run in the kernel. Workers must produce bit-identical numbers to the in-kernel path (test this in
  TESTS when ≥ 1 GPU is present by running one case both ways). Print a progress line at least every 60 s.
  You may reuse the worker / claim-file design of `plasticity_dose_tests.ipynb` in this folder.
- Memory: J is N×P float64 (N = 320, P ≈ 119k → 0.3 GB); never hold more than three Jacobians at once; free
  promptly; `torch.no_grad()` wherever autograd is not needed.
- Share work inside a combination: J₀, K₀ factor and eigen-decomposition once per start; NP started from θ_L.
- Final cell writes `results_part8.json` (all case scalars + H/R tables + config + environment) and
  `part8_outputs.zip` (results file, the done/failed folders, the executed notebook is saved separately by nbconvert),
  and prints the verdict table and where the files are. The vecs folder is NOT put in the zip (size).
- Print, for every rule, the measured numbers behind the verdict (never a bare PASS/FAIL), and a table per learner of
  median ρ, predicted f, min/max over cases.
- Timing: print the time of one Jacobian, one NP solve and one LAW evaluation before the grid, and an estimate of the
  total. If the "full" estimate exceeds 8 GPU-hours, do NOT change the grid: print the estimate and continue.

---------------------------------------------------------------------------------------------------
## 8. What to hand back

1. The TESTS output verbatim from a real execution of the delivered notebook in MODE="tiny" on this machine
   (CPU; if MNIST is unavailable the synthetic fallback is fine and must be stated).
2. The full H-table and R-tables of that tiny run (these are real results on a small network; label them as such).
3. Evidence that resume, failure listing and (simulated with CPU workers) the multi-worker path give identical numbers.
4. Every deviation from this spec and every suspected spec error, with evidence.
5. Time estimates for "quick" and "full" on one and on four RTX 3090s.
6. Confirmation that the delivered file has MODE="quick", outputs cleared, no install commands, and that both copies are identical.

---------------------------------------------------------------------------------------------------
## 9. Mistakes that would make the test meaningless (check each before delivering)

1. Computing the nearest point for the nominal target t instead of the learner's actual final outputs F_L.
2. Building the vertical projector at the endpoint instead of at the start θ₀ (only the NP stationarity check uses the
   projector at the current point).
3. Using an iterated Gauss-Newton with a line search in place of JUMP inside SL(m), or solving JUMP loosely.
4. Wrong learning-rate ↔ step-factor mapping (must be η = c·N/λ₁(K₀) with L = ‖F − t‖²/(2N)); momentum's c is the raw factor.
5. float32 anywhere; loose reference tolerances; reusing a stale J₀ after θ₀ changed.
6. Transposed C in LAW (increment side vs position side), or accumulating C with F_{k+1} instead of F_k as the position.
7. Using single-sign values where the rule says even part, or pairing v_L(+) with v_NP(−).
8. Dropping, retrying with different settings, or silently excluding cases; tuning any tolerance after seeing results.
9. Evaluating minibatch SGD's path matrix on the minibatch outputs instead of the full training set.
10. Reporting ρ without the cosine (ρ alone can look right when the vectors are not parallel).

---------------------------------------------------------------------------------------------------
## 10. AMENDMENT A1 (written 2026-11-21, BEFORE any run on the GPU machine; after the tiny validation run on synthetic data)

Reason. The law is a statement about the limit of small moves. The tiny validation run showed (and a size-halving
diagnostic confirmed, error falling by a factor 4 per halving) that at the fixed size s1 the even-part error is O(s^2) and
is largest along weak kernel directions, where the same output move needs a larger weight move. Rules H1-H4 as first
written compare the value at ONE finite size with the limit value, so they partly test "is s1 small enough", not the law.
The thresholds are NOT changed. What changes is that every small-move case is run at both sizes and the limit is estimated.
The literal rules stay in the output, unchanged, next to the new ones. Also the NP halving rule of sec. 3.4 is corrected
(it was a spec error: the solve starts with zero output residual, so "halve if the output residual increases" fires at once).

A1.1 Grid. Every small-move case (groups of H1-H4: SL, DJ, GD, HB; all six directions for SL and DJ, eigA-eigD for GD and HB)
     is run at BOTH sizes s1 = 0.02 S0 and s2 = 0.01 S0, both signs. PROX stays at s1, sign + only. H5, R1 unchanged.
A1.2 Extrapolated value. For a (learner, direction) with usable even parts at both sizes:
        rho_x = (4*rho(s2) - rho(s1)) / 3          (removes the O(s^2) error of the even part; s2 = s1/2)
     and the companion quantities are those of the SMALLER size: cos_x = cos(s2), nr_x = nr(s2).
     A case "matches (extrapolated)" when matches(rho_x, cos_x, nr_x, f, tol) holds with the SAME function and the SAME tol as before.
A1.3 Rules. H1x, H2x, H3x, H4x: >= 90% of (learner, direction) pairs match (extrapolated); H3x/H4x keep the convergence
     requirement (both sizes, both signs CONVERGED) and the 75% planned-converged condition. H7x: nr(s2) <= 0.05 in >= 90%.
     PRIMARY verdicts for the law: H1x-H4x, H5, H5b, H6, H7x, H0. The literal H1-H4, H7 (each size counted separately, as
     in the first version) are printed alongside as "literal (finite size)". The verdict summary prints both and labels them.
A1.4 Order check (report, R4 replaced): for every (learner, direction) print |rho(s1) - f|, |rho(s2) - f|, their ratio
     (expected about 4) and rho_x; flag ratios outside [2.5, 6] when |rho(s1) - f| > 1e-4.
A1.5 NP solver (sec. 3.4 corrected): the step-halving merit is the length of the iteration's own step ||theta_iter - theta||
     (as implemented, NP_MERIT = "step"); the convergence criteria are unchanged.
A1.6 R2 (report only): sizes s/S0 in {0.05, 0.1, 0.25, 0.5} (1.0 removed: in the validation run the reference points
     themselves could not be computed there), and the nearest-point solve of R2 cases is capped at 80 iterations
     (NP_CAP_R2 = 80); a capped solve is reported as NP-FAILED as before.
A1.7 Everything else is unchanged. All cases that existed before and are not affected by A1.6 must produce bit-identical
     numbers to the previous version of the notebook (check this on the tiny run).

---------------------------------------------------------------------------------------------------
## 11. AMENDMENT A2 (written 2026-11-21, before any GPU run; found in the line-by-line review)

Problem. H3/H4 (and H3x/H4x) require GD / HB runs toward a kernel eigen-direction to be CONVERGED, with the stop rule
||F - t||_2 <= 1e-6 s (and a small velocity for HB). In a nonlinear network a move of size s along q_i also creates output
residuals of order s^2 in OTHER directions; gradient descent removes those at the rate of the weakest kernel modes,
c_N = c*lambda_N/lambda_1 per step. With cond(K0) ~ 1e4-1e6 (expected for the real network; the tiny network has cond 20)
this needs millions of steps, so most runs would hit the 300 000-step cap, be marked UNCONVERGED and excluded, and H3/H4
would fail their "75% converged" condition for a bookkeeping reason. The prediction f only needs the TARGET mode to have
finished: the leftover residual in other modes is O(s^2), enters the path matrix at O(s^3), and everything is measured on
the learner's own final outputs anyway.

A2.1 Stop rule for GD(c) and HB(c, mu) in the small-move eigen-direction cases (the cases used by H3, H4, H3x, H4x) only:
     with d = (t - F0)/||t - F0|| (the unit target direction) the run is CONVERGED at the first step k at which
         |d . (F_j - t)| <= 1e-6 * s    for the last 100 consecutive steps j = k-99..k   (DIR_STOP_REL = 1e-6, DIR_WINDOW = 100)
     (the window covers momentum ringing; no separate velocity condition). Cap 300 000 steps as before -> UNCONVERGED.
     Also record at the stop: the full residual ||F - t||_2 / s and ||u||_2.
     All other uses of GD / HB / SGD / ADAM (H5, R1, R2, linear control, unit tests U5-U8) keep their existing stop rules.
A2.2 New unit test U14 (must pass): build a start with cond(K0) >= 1e4 (for example the U6 network with two of its 16
     inputs replaced by near-copies of two others, x_j' = x_j + 1e-3 * noise; print cond(K0)). For GD(1.0) and HB(0.1, 0.9)
     toward q_2 at s = 0.05*||t_fit||, even part:
       (a) with the OLD stop rule and a cap of 20 000 steps the run is UNCONVERGED (shows the problem is real; if it does
           converge, lower the perturbation to 1e-4 and say so);
       (b) with the A2.1 rule the run is CONVERGED in < 20 000 steps and |rho - f| <= 0.03*max(1,|f|) with |cos| >= 0.99.
A2.3 R2 nearest-point cap: NP_CAP_R2 = 250 (not 80), so that the s/S0 = 0.5 rung can converge.
A2.4 Everything else unchanged. In MODE="tiny" every case that is not a GD/HB eigen-direction small-move case and not an
     R2 s/S0 = 0.5 case must be bit-identical to the A1 run; for the GD/HB eigen-direction cases report the largest change
     in even-part rho relative to the A1 run (expected tiny, since those runs were converged under both rules).
