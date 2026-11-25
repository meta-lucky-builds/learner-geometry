# Part 5 — Plasticity dose tests (Tests 1 and 2), Colab GPU notebook

Deliverable: ONE self-contained Jupyter notebook, `plasticity_dose_tests.ipynb`, saved to
`the working folder`. It will be uploaded to Google Colab (GPU runtime) and run there.
It must not depend on any file from this project. Allowed libraries: torch, torchvision,
numpy, scipy, matplotlib, json, time, math (all preinstalled on Colab). No pip installs.

## 0. What is being tested

Setting: a network repeatedly memorises fresh random targets on the SAME inputs ("loss of
plasticity" benchmark). Let F = network outputs on the training inputs (a vector of length
n·C), and K = J Jᵀ the tangent kernel on those inputs (J = ∂F/∂θ).

Claim under test ("jitter dose"): every optimizer step moves the outputs by ΔF. The quantity

    dose  D = Σ_steps  ΔFᵀ (K_ref + ρ λ_max I)⁻¹ ΔF

measures how much the output path "jumps" in directions the network finds hard. The claim:
loss of trainability is driven by cumulative dose — not by the number of tasks, the amount of
weight movement, or the optimizer as such. Three consequences are tested:

- Test 1 (squared-jump law): for a learner that jumps straight to each target, splitting a
  label change into N equal steps divides the dose, and the damage, by about N; and scaling a
  label perturbation by ε scales them by ε².
- Test 2 (optimizer filter): gradient descent moves outputs by ΔF = −(η/(nC)) K r, so its jumps
  are suppressed in weak kernel directions; an exact (Gauss–Newton) learner moves by ΔF = −r,
  unfiltered. Optimizers should lose plasticity in the order of their dose.
- Collapse: across ALL optimizers, granularities and seeds, damage plotted against cumulative
  dose should fall on one curve, and dose should predict damage better than the alternatives.

A FAIL is a valid result. Nothing may be tuned after seeing damage numbers.

## 1. Data and model

- Data: MNIST training images, the first n = 64 images of a fixed per-seed random permutation
  (`torch.Generator().manual_seed(1000+seed)`), average-pooled 2×2 to 14×14 (196 inputs),
  pixel values scaled to [0,1] then standardised with the mean and std of those 64 images.
  If MNIST cannot be downloaded, fall back to synthetic inputs ~ N(0,1) of the same shape and
  PRINT A LOUD WARNING; record `data_source` in the results.
- Model (main): MLP 196 → 256 → 256 → 10, ReLU, biases, PyTorch default `nn.Linear` init,
  seeded by `torch.manual_seed(seed)`. C = 10 outputs. float32 for training.
- Model (arm "tanh"): same with tanh.
- Model (control "linear"): fixed random features φ(x) = ReLU(W₀x+b₀) with a frozen
  196→2048 layer (default init), trainable linear readout 2048→10 only. Its kernel cannot
  change, so every damage measure must stay at its initial value (§7 test 5).
- Loss: L(θ) = (1/(2nC)) ‖F − Y‖². Residual r = F − Y. "Fit" means mean squared error
  (1/(nC))‖r‖² ≤ τ = 1e-3.

## 2. Target sequences

All targets live in R^{n×C}. Y_0 = one-hot of the true MNIST labels.

- Sequence R (random relabelling; main): for task m = 1..M, Y_m = one-hot of labels drawn
  uniformly at random per example from `Generator(2000+seed)`, independent across tasks.
- Sequence E(ε) (small perturbations, for the ε² law): Y_m = Y_0 + ε·G_m for odd m and
  Y_m = Y_0 for even m, with G_m entries ~ N(0,1) from `Generator(3000+seed)` (an
  out-and-back excursion every two tasks).
- Granularity N: the change from Y_{m−1} to Y_m is presented as N micro-targets
  Y_{m−1} + (j/N)(Y_m − Y_{m−1}), j = 1..N. The learner trains on each micro-target until Fit
  or until a cap of S_cap = 4000 optimizer steps (GN: 30 iterations); a micro-target that hits
  the cap is recorded as UNFIT and training moves on.

## 3. Learners (all full-batch on the n training inputs)

- GD: θ ← θ − η∇L.
- MOM: heavy ball, momentum 0.9, buffer carried across micro-targets and tasks.
- ADAM: torch.optim.Adam, betas (0.9, 0.999), eps 1e-8, state carried across everything.
- ADAM-R: as ADAM but optimizer state reset at the start of every TASK (not every micro-target).
- GN: damped Gauss–Newton in float64 for the solve: Δθ = −Jᵀ (K + ρ_gn λ_max(K) I)⁻¹ r with
  ρ_gn = 1e-6, step-halving line search on ‖r‖ (≤ 10 halvings). J by `torch.func`
  (functional_call + vmap/jacrev, chunked to fit GPU memory).
- Learning-rate rule (fixed before any plasticity measurement; record what was chosen):
  for GD and MOM the grid is {0.01, 0.03, 0.1, 0.3, 1.0}; for ADAM and ADAM-R {1e-4, 3e-4, 1e-3,
  3e-3}. Using seed 0, sequence R, N = 1, train ONLY task 1 from the initial weights with each
  value; choose the largest value that reaches Fit within S_cap with a loss that never exceeds
  10× its starting value. Use that value for every seed, N and sequence. ADAM-R uses ADAM's value.
- GD-half, GD-quarter: GD with η/2 and η/4 (used only at N = 1, sequence R): a direct
  intervention on dose at fixed everything else.

## 4. What is recorded

### 4.1 Every optimizer step (cheap; all on the training inputs)
With F_k the outputs before the step and F_{k+1} after: ΔF = F_{k+1} − F_k, Δθ.
Accumulate per task (reset at each task boundary, and also keep running totals):

    D_K    += ΔFᵀ (K_ref + ρ λ_ref I)⁻¹ ΔF          (ρ = 1e-4; also ρ = 1e-2 as D_K_hi)
    D_raw  += ‖ΔF‖²
    W_move += ‖Δθ‖₂
    band doses: with K_ref = Σ λ_i q_i q_iᵀ (sorted descending), split the nC directions into
        TOP (largest 10%), MID (middle 80%), BOT (smallest 10%);
        D_band += Σ_{i∈band} (q_i·ΔF)² / (λ_i + ρ λ_ref)
    filter statistics: S_jump[i] += (q_i·ΔF)²,  S_res[i] += (q_i·r_k)²   (r_k = residual before the step)

K_ref, its eigen-decomposition (float64) and λ_ref = λ_max(K_ref) are computed at the START OF
EACH TASK (after the previous task is fitted) and held fixed for all steps of that task.

### 4.2 Every task boundary (after task m is fitted; m = 0 is the state after fitting Y_0)
- Kernel K on the training inputs (float64 eigen-decomposition). With K̂ = K·(nC)/tr(K)
  (trace-normalised): **primary damage** A = log tr[(K̂ + ρI)⁻¹] − (its value at m = 0), ρ = 1e-4.
  Also record: λ_max, tr K, λ at the 10th/50th/90th percentile, A with ρ = 1e-2, effective rank
  exp(entropy of normalised eigenvalues), number of eigenvalues above 1e-6·λ_max.
- Standardised probe trainability (same probe for every checkpoint of a seed): on a CLONE of the
  network, run 300 steps of plain GD on L with target Y_probe (random one-hot labels from
  `Generator(4000+seed)`) at step size η_p = nC/λ_max(K); record ψ = L_300/L_0. Also compute the
  linearised prediction ψ_lin = Σ_i (q_i·r_0)² (1 − λ_i/λ_max)^{600} / ‖r_0‖². Discard the clone.
- Unit statistics on the training inputs: fraction of hidden units never active (ReLU) or with
  |activation| > 0.99 on every input (tanh), per layer; number of unit pairs in the last hidden
  layer whose activation vectors have |cosine| > 0.99; stable rank of last-hidden features;
  ‖θ‖₂.
- Training cost of the task just finished: total optimizer steps, number of UNFIT micro-targets.
- Per-task totals of §4.1, and the filter exponent α = OLS slope of log(S_jump[i]/S_res[i]) on
  log λ_i over directions with λ_i > 1e-6 λ_ref.

Write results to `results_part5.json` after every condition (so a disconnect loses nothing),
and skip conditions already present when re-run.

## 5. Grids

MODE = "full": seeds {0,1,2}, M = 20 tasks.
MODE = "quick": seed {0}, M = 8 tasks, and only the conditions marked [q].

| grid | model | sequence | learners | N |
|------|-------|----------|----------|---|
| G-main | ReLU | R | GD, MOM, ADAM, ADAM-R | 1 [q], 2, 4 [q], 8, 16 [q], 32 |
| G-gn | ReLU | R | GN | 1 [q], 2, 4 [q], 8, 16 [q] |
| G-lr | ReLU | R | GD-half, GD-quarter | 1 [q] |
| G-eps | ReLU | E(ε), ε ∈ {0.05, 0.1, 0.2, 0.4} | GN, GD, ADAM | 1 ([q] for ε ∈ {0.1, 0.4}) |
| G-tanh | tanh | R | GD, ADAM, GN | 1, 8 |
| G-lin | linear control | R | GD, ADAM, GN | 1 [q] |

Print an estimated time before starting each grid and elapsed time after each condition.
A CONFIG cell at the top holds MODE, the grids, τ, S_cap, ρ values and an optional
`DRIVE_DIR` (if set, results and figures are also copied there).

## 6. Pre-registered predictions

Let A_final = primary damage at the last task; D_cum = cumulative D_K up to that task.
Slopes are OLS in log-log, per seed, then median over seeds. In "quick" mode report the
numbers but mark every verdict PRELIMINARY.

| id | prediction | rule |
|----|-----------|------|
| H1 | GN, sequence R: per-task dose falls as 1/N | slope of log(mean per-task D_K) on log N in [−1.3, −0.7] |
| H2 | GN, sequence R: damage falls with N | A_final strictly decreasing in N for the median over seeds, and A_final(N=16) ≤ 0.5·A_final(N=1) |
| H3 | GN, sequence E(ε): dose and damage scale as ε² | slope of log(mean per-task D_K) on log ε in [1.6, 2.4]; slope of log A_final on log ε in [1.2, 2.8] |
| H4 | filter exponents | GD: α in [1.5, 2.5]; GN: α in [−0.5, 0.5]; MOM, ADAM, ADAM-R: reported, no rule |
| H5 | optimizer ordering at N = 1 follows dose | Spearman correlation between D_cum and A_final across the five learners (median over seeds) ≥ 0.8 |
| H6 | collapse | pooled over every (learner, N, seed, task) record of G-main and G-gn: Spearman(D_cum, A) ≥ 0.8, AND it exceeds Spearman(A, task index), Spearman(A, cumulative W_move), Spearman(A, cumulative D_raw) and Spearman(A, ‖θ‖) |
| H7 | dose intervention | GD, GD-half, GD-quarter at N = 1: D_cum decreases with η, and A_final decreases with η (monotone in the median) |
| H8 | damage means lost trainability | pooled over G-main and G-gn: Spearman(A, log ψ) ≥ 0.6; and Spearman(ψ, ψ_lin) ≥ 0.8 |
| H9 | linear control | G-lin: |A| < 1e-3 at every task for every learner |

Report-only (numbers and figures, no verdict):
- R1: band doses (TOP/MID/BOT share of D_K) per learner: where each optimizer's jitter falls.
- R2: ADAM vs ADAM-R: dose, damage, first-step size after a target change.
- R3: unit statistics against D_cum and against A (dead / saturated / duplicate units, stable rank).
- R4: steps needed to fit each task against task index, per learner and N (the classic signature).
- R5: G-tanh against the ReLU results for the same learners and N.
- R6: sensitivity of H5 and H6 to ρ (repeat with the ρ = 1e-2 versions of dose and damage).
- R7: anything anomalous (divergence, UNFIT counts, non-monotone curves), as measurements only.

## 7. Built-in tests (a "TESTS" cell that must pass before any grid runs; also run locally)

1. Jacobian: rows of J match central finite differences of F on 5 random parameters and 5
   random outputs (relative error < 1e-2 in float32, < 1e-5 if computed in float64).
2. GN step: on the ReLU model, one GN step from the initial weights reduces ‖r‖ by at least 90%.
3. Dose bookkeeping: for one GD step, D_K computed by the accumulator equals
   ΔFᵀ(K+ρλI)⁻¹ΔF computed directly (relative 1e-4).
4. Filter exponents on the linear control: GD gives α in [1.9, 2.1]; GN gives α in [−0.1, 0.1].
5. Linear control: after 3 tasks with GD, |A| < 1e-4.
6. Granularity: with N = 4 the last micro-target equals Y_m exactly.
7. Reproducibility: running the same tiny condition twice gives identical A to 1e-6.

## 8. Notebook structure

1. Title + one-paragraph explanation in plain language + how to run (Runtime → GPU → Run all).
2. CONFIG. 3. Imports, device, seeding helpers. 4. Data. 5. Models. 6. Kernel/Jacobian utilities.
7. Learners. 8. Measurement utilities. 9. Runner (one condition). 10. TESTS (assert).
11. Learning-rate selection (prints the chosen values). 12. Run grids (resumable).
13. Analysis: one table per H-rule with PASS / FAIL / INCONCLUSIVE (PASS needs the rule to hold
    for the median and for at least 2 of 3 seeds; INCONCLUSIVE if median only) and the numbers.
14. Figures (matplotlib, saved as PNG and shown): (a) A vs task index per learner at N = 1;
    (b) A_final vs N per learner; (c) the collapse plot A vs D_cum, all conditions, coloured by
    learner; (d) filter plot S_jump/S_res vs λ per learner; (e) band-dose shares; (f) ψ vs A;
    (g) dose and damage vs ε. 15. A final cell that prints a compact text summary and zips
    `results_part5.json` + figures into `part5_outputs.zip` and, on Colab, offers it for download.

## 9. Local validation before delivery

The notebook must have been executed end-to-end on this machine (CPU) in a MODE = "tiny"
configuration (n = 16, width 32, M = 2, N ∈ {1, 2}, one seed, synthetic data allowed if MNIST
is unavailable, S_cap = 400) with all §7 tests passing. "tiny" exists only for this check and
is not offered as a science mode. Deliver the notebook with outputs cleared and MODE = "quick".
