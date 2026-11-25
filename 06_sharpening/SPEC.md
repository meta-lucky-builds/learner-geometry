# SPEC7 — Tests 10–12: where does the rise in sharpness during training come from?

Part 7 of the task-cycle-rotation project. CPU only, float64, deterministic.
Read SPEC4.md (path law, C bookkeeping, `Law`) and SPEC6.md §0 and Test 5 first; this spec reuses
their objects and conventions without repeating them. Read `tcr4.py` and `tcr6.py` before coding.

## 0. Ground rules

- New files only: `tcr7.py` (machinery), `test_part7.py` (unit tests), `run7.py` (grids, resumable,
  writes `results7.json`), `report7.py` (writes `REPORT7.md`). Do NOT modify any existing file.
- `import tcr4` first (fixes threads to 2). float64 everywhere. No plotting. No package installs.
- Every pass rule below is fixed now. **A FAIL is a valid result. Tune nothing** (no changing
  thresholds, step sizes, seeds, grids, exclusions) to turn a FAIL into a PASS.
- If you believe a formula or rule in this spec is wrong: run it literally, report it as FAIL or
  "spec error" with your argument and evidence, and add a clearly labelled SUPPLEMENTARY version.
- Report every non-converged / diverged / skipped cell by name. Never silently drop a cell.
- Time budget ≈ 3 h wall. If the full grid will not fit, cut in this order and say so in the report:
  (1) h = 128 in Part C, (2) h = 128 in Part B, (3) seeds 3–4 in Part C. Never cut Part A.

## 1. Objects

`S = tcr6.setup(h, seed)` gives model, X (n = 16 training inputs), `S.theta_init` (random init),
`S.theta_A` (GN fit of the teacher labels `S.yA` from `theta_init`), `S.lam`, `S.Q` (eigenvalues,
descending, and eigenvectors q_1..q_16 of K(θ_A)).

- F(θ) = model outputs on X (n-vector). J(θ) = ∂F/∂θ (n × P). K = J Jᵀ.
- λ₁(θ) = largest eigenvalue of K(θ). Also tr K(θ), λ_min(θ).
- Loss L(θ; y) = (1/2n)‖F(θ) − y‖². r = F − y.
- S_H(θ; y) = largest eigenvalue of the Hessian of L (the true sharpness). Compute with
  `scipy.sparse.linalg.eigsh(LinearOperator, k=1, which='LA')` where the Hessian-vector product is
  the central difference of the exact gradient ∇L = (1/n)Jᵀr: Hv = [∇L(θ+δv̂) − ∇L(θ−δv̂)]·‖v‖/(2δ),
  δ = 1e-5. (At r = 0, S_H = λ₁/n.)
- GD(c; θ_in, y): θ ← θ − η∇L, **η = c·n/λ₁(θ_in) fixed for the whole run** (the tcr4 convention,
  `fo_learn_C(kind="GD")`). Local step factor of kernel direction i: c_i = η λ_i/n.
- GN learner = `tcr6.gn_C` / `tcr4.gn_learn_C` (iterated min-norm Gauss–Newton).
- C bookkeeping (SPEC4): C = Σ_steps (F_{k+1} − F_k) ⊗ (F_k − F_start), entry [a,b] = ΔF_a·(F_k−F_start)_b,
  exactly as `tcr4.c_increment` does. Also accumulate Q = Σ_steps (F_{k+1}−F_k)(F_{k+1}−F_k)ᵀ and
  ΔY = F_end − F_start. Identity to assert (rel. 1e-10): C + Cᵀ = ΔYΔYᵀ − Q.
  Three pieces of C: travel = ½ΔYΔYᵀ; jitter = −½Q; area = ½(C − Cᵀ).
- LAW at a base point θ_b: `L = tcr4.Law(model, θ_b, X)`, `L(C)` = P_v Σ_ij C_ij T_ij (P-vector).
- Dλ₁[V] at θ_b: `tcr6.dderiv(lambda th: tcr6.lam1(model, th, X), θ_b, V)` (central FD, step 1e-4).
  "Predicted λ₁ change from path matrix C" := **pred(C; θ_b) = Dλ₁[LAW_θb(C)]**. Linear in C.

### 1.1 New learner: nearest point (NP)

NP(θ_b, t) = the point closest to θ_b in weight space among points with outputs t:
minimise ‖θ − θ_b‖² subject to F(θ) = t. Stationarity: θ − θ_b ∈ row space of J(θ).
Iteration from θ⁰ = θ_b:
  d_k = θ_b + J_kᵀ K_k⁻¹ [ (t − F(θ_k)) + J_k(θ_k − θ_b) ] − θ_k,   θ_{k+1} = θ_k + s·d_k,
with J_k, K_k at θ_k; s = 1, halved (down to 2⁻²⁰) until the merit
M(θ) = ‖F(θ) − t‖² + 1e-6·‖P_v(θ)(θ − θ_b)‖² decreases (if this merit proves unworkable, use plain
residual backtracking and say so). Stop when ‖F − t‖∞ ≤ 1e-10 AND
‖P_v(θ)(θ − θ_b)‖ ≤ 1e-8·max(‖θ − θ_b‖, 1e-12); cap 200 iterations → status NOT CONVERGED (report;
fall back to continuation: solve for targets F(θ_b) + (j/8)(t − F(θ_b)), j = 1..8, each started from
the previous solution but always with the same θ_b in the formula; if still not converged, record
the cell as NP-FAILED and exclude it from rules that need NP, listing it).
Record for every NP solve: iterations, final residual, stationarity ratio, ‖θ_NP − θ_b‖.

### 1.2 Straight-line transport (SL)

SL(θ_b, t; m) = weights reached by moving the outputs along the straight segment from F(θ_b) to t in
m equal sub-targets, each fitted by the GN learner started from the previous point. Default m = 64.
(Second-order theory: SL − NP is vertical and equals LAW(½ΔYΔYᵀ)(1 − 1/m).)

## 2. Unit tests (`test_part7.py`; all must pass before any grid; print PASS/FAIL with numbers)

U1. NP on the linear control (`tcr6.setup(32, 0, linear=True)`): equals θ_b + JᵀK⁻¹(t − F(θ_b)) to 1e-10
    for 3 random targets with ‖t − F‖ = 1.
U2. NP on the MLP (h = 32, seed 0, θ_b = θ_A, 3 random unit directions d, t = yA + 0.3·d): converged;
    stationarity ratio ≤ 1e-8; ‖θ_NP − θ_b‖ ≤ ‖θ_GN − θ_b‖ + 1e-12 and ≤ ‖θ_GD(0.25) − θ_b‖.
U3. Small-move agreement: for t = yA + s·d, s ∈ {0.02, 0.04, 0.08}: ‖θ_NP − θ_GN‖ ∝ s^p with fitted
    p ≥ 2.5 (they agree through second order).
U4. S_H routine: at h = 32, seed 0, θ = θ_init, y = yA: eigsh result vs largest eigenvalue of the dense
    Hessian built column by column from the same central differences: rel. diff ≤ 1e-6. At θ_A with
    y = F(θ_A): S_H = λ₁/n to rel. 1e-5.
U5. Bookkeeping identity C + Cᵀ = ΔYΔYᵀ − Q (rel. 1e-10) on one GD(1.5) run and one GN run.
U6. Regression against Test 5: h = 32, seed 0, target yA + 0.025·q₁, GD(c = 1.5) vs GN:
    [λ₁(θ_GD) − λ₁(θ_GN)] / [0.025²·φ(1.5)·G₁] ∈ [0.9, 1.1], with φ(c) = (1 − c)/(2 − c) and
    G_i := pred(q_i q_iᵀ; θ_A). (Equivalent to SPEC6's pred2.)
U7. SL second-order check: h = 32, seed 0, t = yA + 0.05·d (3 random d), m = 64:
    cos(θ_SL − θ_NP, LAW_θA(½ΔYΔYᵀ)) ≥ 0.99 and norm ratio ∈ [0.9, 1.1].
U8. Twin target: for y⁻ = 2F(θ_b) − y the initial residual is exactly −r₀ (1e-14).
U9. Linear control: for 3 random targets, λ₁ changes of GD(0.25), GD(1.9), GN, NP, SL are all ≤ 1e-9
    in absolute value, and pred(C) ≤ 1e-9.

## 3. Test 10 (Part A) — small moves, one kernel direction at a time (exact predictions)

Grid: h ∈ {32, 64, 128}; seeds 0–4; base point θ_A.
Directions: q_i for i ∈ {1, 2, 4, 7, 10, 13, 16}, plus 8 random unit vectors d (generator seed
7000 + seed) — "rand1..rand8". Sizes s ∈ {0.025, 0.05, 0.1}. Signs σ ∈ {+1, −1}.
Target t = yA + σ·s·e (e = q_i or d). Learners: NP, GN, SL(m = 64), GD(c) for c ∈ {0.25, 1.0, 1.5, 1.9}.
For every run store λ₁(end), tr K(end), C, Q, ΔY, steps, status.

Definitions (per h, seed, e, s):
- Δ_L(σ) = λ₁(θ_L,σ) − λ₁(θ_A). Even part E_L = ½[Δ_L(+) + Δ_L(−)], odd part O_L = ½[Δ_L(+) − Δ_L(−)].
- position part  = E_NP.
- path part of learner L = P_L(σ) = λ₁(θ_L,σ) − λ₁(θ_NP,σ)   (L = SL, GD(c)); its even part EP_L.
- prediction     = pred(C_L,σ; θ_A)   (C of NP taken as 0).
- closed form for GD along q_i:  s²·φ(c_i)·G_i,  c_i = c·λ_i/λ₁.  For SL: s²·½(1 − 1/m)·G_e with
  G_e = pred(e eᵀ; θ_A).

Pre-registered rules (each evaluated per width; "cells" = (seed, direction) unless stated):
- H10a (law predicts the path part): at s = 0.025, P_L(σ)/pred ∈ [0.8, 1.25] in ≥ 90% of
  (seed, e, σ, L) with L ∈ {SL, GD(0.25), GD(1.0), GD(1.5), GD(1.9)}, excluding cases with
  |pred| < 1e-3 × (median |pred| over that width at s = 0.025) — list the excluded count.
- H10b (closed form): at s = 0.025, EP_GD(c)/(s²φ(c_i)G_i) ∈ [0.67, 1.5] in ≥ 90% of (seed, i, c) with
  |φ(c_i)| ≥ 0.05; and EP_SL/(s²·½(1−1/64)·G_e) ∈ [0.8, 1.25] in ≥ 90% of (seed, e).
- **H10c (the claim: travel sharpens)**: G_e > 0 for ≥ 90% of all (seed, e) (15 directions × 5 seeds),
  AND EP_SL > 0 and EP_GD(0.25) > 0 for ≥ 90% of (seed, e) at every s.
- H10d (weak directions matter more): per (h, seed), OLS slope of log G_i on log λ_i over the seven
  eigen-directions; rule: slope ≤ −1 in ≥ 4 of 5 seeds. Report the slope and also G_i·λ_i² for each i.
- H10e (step size cannot remove it in weak directions): for i ∈ {7, 10, 13, 16}:
  EP_GD(1.9)/EP_GD(0.25) ≥ 0.8 in ≥ 90% of (seed, i, s); and for i = 1: EP_GD(1.9) < 0 in 5/5 seeds at every s.
- R10 (report only, NO prediction — this is the open question): sign and size of the position part.
  Table per width: fraction of (seed, e, s) with E_NP > 0; median E_NP/s²; median EP_GD(0.25)/s²;
  median of the ratio EP_GD(0.25)/(E_NP + EP_GD(0.25)) and its quartiles; the same for tr K in place of λ₁.
  Also Pearson correlation across (seed, e) between E_NP/s² and G_e.
- R10b (report only): odd parts O_NP, O_GD(0.25) vs the first-order prediction s·Dλ₁[ξ_e]
  (ξ_e = JᵀK⁻¹e at θ_A): median ratio.

## 4. Test 11 (Part B) — real training from the random init (not perturbative)

Grid: h ∈ {32, 64, 128}; seeds 0–4; target scale κ ∈ {0.5, 1, 2, 4}; sign σ ∈ {+, −}.
Start θ_b = θ_init, F₀ = F(θ_init). Target y⁺ = F₀ + κ·(yA − F₀); twin y⁻ = 2F₀ − y⁺.
Learner: GD(c = 0.1) (flow-like), run until ‖r‖∞ ≤ 1e-6 or 400 000 steps (status recorded).
Checkpoints: the first step at which ‖r‖₂/‖r₀‖₂ ≤ ρ, for ρ ∈ {0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4},
plus the final point. At each checkpoint with weights θ_t and outputs F_t:
- GD:  λ₁, tr K, λ_min, S_H(θ_t; y), ‖θ_t − θ_init‖, C, Q, ΔY accumulated from the start.
- NP:  θ_NP = NP(θ_init, F_t); SL: θ_SL = SL(θ_init, F_t; 64). Same scalars for both (S_H with the same y).
- Three-way split of the rise (exact identity, assert to 1e-12):
  λ₁(θ_t) − λ₁(θ_init) = position + travel + route,
  position = λ₁(NP) − λ₁(init);  travel = λ₁(SL) − λ₁(NP);  route = λ₁(GD) − λ₁(SL).
- Second-order predictions at θ_init: travel_pred = pred(½ΔYΔYᵀ(1−1/64)); route_pred = pred(C − ½ΔYΔYᵀ(1−1/64)).
Even/odd over the twin (matched by ρ): for X ∈ {total, position, travel, route}: X_even = ½(X⁺ + X⁻).

Pre-registered rules (per width, cells = (seed, κ)):
- H11a (sharpening exists and is even in the residual): total_even at the final point > 0 in ≥ 80% of cells.
  Report also raw total⁺, total⁻, and the relative rise λ₁(final)/λ₁(init) − 1 for each cell.
- **H11b (the claim)**: at the final point, (travel + route)_even > 0 in ≥ 90% of cells AND the median over
  cells of (travel + route)_even / total_even ≥ 0.5 (median taken over cells with total_even > 0).
- H11c (scaling): slope of log total_even(final) between κ = 0.5 and κ = 1 ∈ [1.6, 2.4] in ≥ 4/5 seeds.
  Report the slope for 1→2 and 2→4 as well (no rule).
- H11d (second order holds early): at ρ = 0.9, κ = 0.5: travel/travel_pred ∈ [0.67, 1.5] in ≥ 80% of
  (seed, σ); route vs route_pred: report ratio and sign agreement (no rule).
- R11 (report only): for each width a table over ρ of the median (over seeds, κ = 1 and κ = 4 separately)
  of position_even, travel_even, route_even, total_even, and the same four for tr K and for S_H;
  fraction of cells with position_even > 0; ‖θ − θ_init‖ for GD, NP, SL at the final point;
  S_H(final) vs λ₁(final)/n.
- R11b (report only, a cross-check on published claims): Spearman correlation across all (h-fixed) cells
  between total⁺(final) (not the even part) and the "difficulty" D = (y⁺ − F₀)ᵀ K(θ_init)⁻¹ (y⁺ − F₀),
  and between total⁺(final) and ‖y⁺ − F₀‖².

## 5. Test 12 (Part C) — a run that reaches the stability edge

Grid: h ∈ {32, 64, 128}; seeds 0–4; κ = 4 only; sign + only.
Let λ_i = λ₁(θ_init), λ_f = λ₁ at the final point of the Part-B run (κ = 4, +).
A cell is ELIGIBLE iff λ_f ≥ 1.10·λ_i. Report the eligible count; for a non-eligible cell record
"no threshold crossing possible" and skip. (If fewer than 3 cells are eligible in total, Part C's
rules are reported as NOT TESTABLE — that is a valid outcome; do not change κ. As SUPPLEMENTARY only,
then also try κ = 8 and report it separately.)
For eligible cells and f ∈ {0.25, 0.5, 0.75}: threshold λ* = λ_i + f(λ_f − λ_i); **η = 2n/λ*** fixed.
Run GD from θ_init to y⁺ for at most 60 000 steps; stop early if ‖r‖∞ ≤ 1e-6, or if
‖r‖∞ > 1e3·‖r₀‖∞ / non-finite (DIVERGED).
Every step record: loss, λ₁(K_t), c_eff = η λ₁(K_t)/n, residual component along the current top
eigenvector of K_t. Every 50 steps: S_H(θ_t; y), c_H = η·S_H.
Phases (by step): PRE = steps with c_eff < 1.8; EDGE = steps with c_H ≥ 1.9 measured at the two
bracketing S_H samples (if c_H never reaches 1.9, there is no EDGE phase: report it).

Windows: consecutive windows of W = 16 steps (also repeat the whole analysis with W = 64).
Analyse at most 150 windows per phase per run, evenly spaced over that phase. For a window from
θ_a (outputs F_a) to θ_b: accumulate C_w (relative to F_a), Q_w, ΔY_w;
- measured: position_w = λ₁(NP(θ_a, F_b)) − λ₁(θ_a);  path_w = λ₁(θ_b) − λ₁(NP(θ_a, F_b)).
- predicted at θ_a (Law built at θ_a): travel_w = pred(½ΔY_wΔY_wᵀ), jitter_w = pred(−½Q_w),
  area_w = pred(½(C_w − C_wᵀ)); pathpred_w = their sum = pred(C_w).
- top-mode share of the jitter: q₁ᵀQ_w q₁/tr Q_w with q₁ = top eigenvector of K(θ_a).

Pre-registered rules (pooled over eligible cells and f, per width, W = 16; W = 64 reported alongside):
- H12a (the law holds inside windows): Pearson r(path_w, pathpred_w) ≥ 0.9 and median
  path_w/pathpred_w ∈ [0.67, 1.5], separately in PRE and EDGE.
- H12b (overshoot flattens): in EDGE, jitter_w < 0 in ≥ 95% of windows.
- H12c (balance at the edge): in EDGE, with means over windows: mean(position + travel + area) > 0,
  mean(jitter) < 0, and |mean(position + travel + area + jitter)| ≤ 0.3·|mean(jitter)|.
- H12d (before the edge the flattening term is small): in PRE, |mean jitter_w| ≤ 0.2·|mean(position_w + travel_w)|.
- H12e (the run sits at the edge): median c_H over the EDGE phase ∈ [1.9, 2.1].
- R12 (report only): per run: steps, status, final loss, max and median c_eff and c_H per phase,
  fraction of the run in each phase, mean of each of position/travel/jitter/area per window per
  phase, top-mode share of jitter, and the final λ₁ compared with λ* and with the Part-B flow value λ_f
  (does the large step end flatter than flow, and by how much).

## 6. Linear control

Run Part A (seed 0, h = 32, all directions, s = 0.05) and one Part-B cell on the linear model:
every λ₁ change, every prediction and every piece must be ≤ 1e-9 in absolute value. Report max.

## 7. REPORT7.md

1. Unit-test outputs verbatim. 2. H-table: rule, width, the measured numbers, PASS/FAIL (never a bare
verdict). 3. The R tables. 4. Every skipped / failed / non-converged cell. 5. Deviations from this
spec and spec errors, each with evidence. 6. Timing. 7. A short plain-language "what the data say"
section that separates what a rule established from what is only suggested, and lists anything
surprising (in particular: the sign of the position part, whether the route part matters, any cell
where sharpness falls). Keep raw per-cell numbers in results7.json so every table can be re-derived.
