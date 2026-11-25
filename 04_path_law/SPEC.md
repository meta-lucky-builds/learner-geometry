# Part 4 — the path law: weight displacement from the path the outputs trace

Conventions as before: float64, CPU, deterministic, seeded generators. Reuse `tcr.py`,
`tcr2.py`, `tcr3.py` by import. Do NOT modify any existing file. New files only (§8).

## 0. The claim

Inputs fixed at X = X_A (n = 16). A learner moves the weights θ; the outputs F = f(X;θ) ∈ R^16
trace a path. Claim: to second order in the size of the path, the part of the weight
displacement that does not change the outputs ("vertical" part) depends on the learner ONLY
through a 16×16 matrix C of iterated sums of that output path:

    V  =  Σ_{j=1}^{16}  P_v [ Dξ_{C e_j} · ξ_{e_j} ] (θ_A)                       (LAW)

where e_j is the j-th standard basis vector of R^16, C e_j is column j of C, and ξ, Dξ·w, P_v
are exactly the objects of SPEC3.md §1 evaluated at θ_A (finite-difference step
s = 1e-4/‖w‖ with w = ξ_{e_j}(θ_A)). The map C ↦ V is linear. The same law is claimed for
every learner below; only C changes.

Associated scalar prediction: for u of SPEC3.md §1 and c_u, φ_u of SPEC3.md §1,

    φ_u(θ_end) − φ_u(θ_A)  =  − c_u(θ_A) · V          (for paths that end back on labels y_A)

## 1. Objects

Network, teacher, X_A, y_A, θ_A = GN-learn(θ_init, (X, y_A)), label directions u, v, probe set:
exactly SPEC3.md §1. Widths and seeds in §6. F_A = f(X; θ_A).

## 2. Learners (each "learn" starts from the current θ and a target y; residual r = F − y)

- **GN**: `tcr.gn_learn` (SPEC.md §3.1). Its iterations are the "steps".
- **GD(c)**, c ∈ {0.25, 1, 1.8}: θ ← θ − η g, g = Jᵀ r / n, η = c·n/λ_max(K(θ_in)) fixed for
  the learn (θ_in = weights at the start of that learn). Stop when ‖r‖_∞ ≤ 1e-8.
  Cap 400,000 steps. If ‖r‖_∞ exceeds 1e3 × its initial value, stop and mark DIVERGED.
- **MOM**: heavy ball, μ = 0.9, c = 0.25: v ← μ v − η g; θ ← θ + v; v = 0 at the start of each
  learn; η = c·n/λ_max(K(θ_in)). Same stop rule and cap.
- **ADAM** (exploratory, §5 only): torch-style Adam, lr = 1e-3, β = (0.9, 0.999), eps = 1e-8,
  state reset at the start of each learn, full-batch gradient g. Stop at ‖r‖_∞ ≤ 1e-6 or
  200,000 steps; record which.

## 3. The path matrix C

C starts at 0 at θ_A and accumulates over EVERY optimizer step of EVERY learn in the sequence.
Let F_k be the outputs before step k and F_{k+1} after it.

- GN and GD(c) ("measured" C):      C += (F_{k+1} − F_k) ⊗ (F_k − F_A)
- MOM ("measured" C):               C += ẽ_k ⊗ (F_k − F_A),   ẽ_k = − (η / ((1−μ) n)) K(θ_k) r_k
  (K(θ_k) = J Jᵀ at the weights before step k; this is the total eventual output move
  committed by the gradient taken at step k)

(a ⊗ b is the 16×16 matrix a bᵀ.) Report for each C: ‖(C+Cᵀ)/2‖_F ("symmetric part") and
‖(C−Cᵀ)/2‖_F ("area part"), and V split the same way (V_sym, V_area from the two parts of C).

**A-priori C** (no training run; uses only K_A = K(θ_A) and the learner's rule): simulate in
label space, starting at F = F_A, for each learn with target y, r = F − y:
- GN: one step e = −r:  C += e ⊗ (F − F_A); F += e.
- GD(c): η' = c/λ_max(K_A); repeat while ‖r‖_∞ > 1e-8: e = −η' K_A r; C += e ⊗ (F − F_A); F += e.
- MOM: η' = 0.25/λ_max(K_A), w = 0; repeat while ‖r‖_∞ > 1e-8: w ← μ w − η' K_A r;
  ẽ = −(η'/(1−μ)) K_A r; C += ẽ ⊗ (F − F_A); F += w.

## 4. Paths and measured quantities

All label targets are y_A + ε·(…) with u, v of SPEC3.md and, for walk8, seven vectors w_1..w_7
from `Generator(777 + seed)` (16 entries ~ N(0,1) each, drawn in order, each scaled to ‖y_A‖).

| path | sequence of targets after θ_A | measured vector |
|------|------------------------------|-----------------|
| oab (out-and-back) | y_A + εu, then y_A | V_meas = P_v(θ_A)(θ_end − θ_A) |
| tri+ | y_A + εu, y_A + εv, y_A | same |
| tri− | y_A + εv, y_A + εu, y_A | same |
| walk8 | y_A + εw_1, …, y_A + εw_7, y_A | same |
| oneway | y_A + εu only | D_meas = P_v(θ_A)(θ_end^{learner} − θ_end^{GN}) |

Predictions: V_pred = LAW(C) for the closed paths; for oneway, D_pred = LAW(C^{learner} − C^{GN}).
For each (path, learner, mode ∈ {measured C, a-priori C}) report

    cos = ⟨meas, pred⟩/(‖meas‖‖pred‖),   ratio = ‖meas‖/‖pred‖,   rel_err = ‖meas − pred‖/‖meas‖

and for closed paths Δφ_meas = φ_u(θ_end) − φ_u(θ_A) against Δφ_pred = − c_u(θ_A)·V_pred.

## 5. Predictions and rules

Verdict rule (as SPEC3.md §8): PASS if the rule holds for the median over seeds AND for at
least 4 of 5 seeds (2 of 3 where only 3 seeds are run); INCONCLUSIVE if median only; else FAIL.
Give per-seed values wherever the verdict is not PASS. A FAIL is a valid result; tune nothing.

| id | statement | rule (at ε = 0.025, per width, per learner ∈ {GN, GD(0.25), GD(1), GD(1.8), MOM}) |
|----|-----------|------|
| S1 | LAW with measured C holds on every closed path (oab, tri+, tri−, walk8) | cos ≥ 0.98 and ratio ∈ [0.9, 1.1] on each of the four paths |
| S2 | LAW with a-priori C holds on every closed path | cos ≥ 0.97 and ratio ∈ [0.85, 1.15] on each |
| S3 | learner-vs-learner (oneway), measured C, learner ∈ {GD(0.25), GD(1), GD(1.8), MOM} | cos ≥ 0.98 and ratio ∈ [0.9, 1.1] |
| S4 | stiffness: Δφ_meas vs Δφ_pred on closed paths (measured C) | same sign, and Δφ_meas/Δφ_pred ∈ [0.8, 1.25], on at least 3 of the 4 paths |
| S5 | remainder is third order: rel_err ∝ ε | slope of log rel_err on log ε over ε ∈ {0.1, 0.05, 0.025, 0.0125}, for oab and tri+, learners GN and GD(1): in [0.7, 1.5] |
| S6 | linear control: with `tcr.LinearModel(seed)` (labels y_A = model.labels(X)), for GN and for GD(1) run for exactly 2,000 steps per learn (no convergence required), ‖P_v(θ_end − θ_A)‖ < 1e-10 on oab and tri+ at ε = 0.1 | must hold, all 5 seeds |

Pre-check from a separate implementation (h = 32 seeds 0,1 and h = 64 seed 0; ε = 0.025; oab and
oneway; measured C): cos 0.9990–0.9999 and ratio 0.990–1.025 for GN, GD(0.25), GD(1), GD(1.8),
MOM; Δφ_meas/Δφ_pred between 0.98 and 1.17 with matching sign in all 15 cases (the sign of Δφ
itself varied between cases for GD). tri±, walk8, a-priori C and S5 were NOT pre-checked.

### Report-only items (numbers only, no interpretation)
- R1: for every (learner, path): ‖V_meas‖, the symmetric and area parts of C, ‖V_sym‖, ‖V_area‖.
- R2: learning-rate dependence — for oab and tri+, the relative difference
  ‖V_meas[GD(c)] − V_meas[GD(1)]‖/‖V_meas[GD(1)]‖ for c ∈ {0.25, 1.8}, next to the same for the
  symmetric part of C.
- R3: ε-dependence of cos and ratio for all learners on oab and tri+ at
  ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125} (shows where the law stops holding).
- R4: order-odd part: a = (V_meas[tri+] − V_meas[tri−])/2 per learner, its cos with
  (ε²/2)·b_uv(θ_A) and length ratio (SPEC3.md L2), for every learner.
- R5 (ADAM, exploratory): cos, ratio, rel_err of LAW with measured C (plain rule
  C += (F_{k+1} − F_k) ⊗ (F_k − F_A)) on oab and tri+, h ∈ {32, 128}, ε = 0.025, seeds {0,1,2};
  and whether each learn converged.
- R6: anything anomalous in the data, stated as measurements only.

## 6. Grid

- Main: h ∈ {32, 64, 128}, seeds {0,1,2,3,4}, ε = 0.025, all five learners, all five paths,
  both C modes.
- ε-sweep (S5, R3): h ∈ {32, 128}, seeds {0,1,2,3,4}, paths oab and tri+, all five learners,
  ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}, measured C.
- Wide: h = 512, seeds {0,1,2}, ε = 0.025, learners GN and GD(1), paths oab, tri+, tri−, oneway.
- MOM needs K(θ_k) every step; if MOM at h = 128 would exceed 40 minutes in total, run it at
  h ∈ {32, 64} only and say so.
- Overall budget 3 hours wall. Run order: tests, S6, Main (h = 32, 64, then 128), ε-sweep, Wide,
  ADAM. If the budget would be exceeded, stop, report what finished, and list what did not run.

## 7. Tests before any grid (`test_part4.py`)

1. C bookkeeping on a hand-made path: outputs F_A=(0,0), then (1,0), (1,1), (0,0) in R² give,
   with the GN/GD rule, C = [[0,0],[1,0]] + (−1,−1)⊗(1,1) = [[−1,−1],[0,−1]]. Assert exactly.
2. GD steps are horizontal: for GD(1), h = 32, seed 0, the first 20 steps of the oab path satisfy
   ‖P_v(θ_k) Δθ_k‖/‖Δθ_k‖ < 1e-10 (P_v at the weights before the step).
3. LAW is linear in C: LAW(2C₁ − 3C₂) = 2 LAW(C₁) − 3 LAW(C₂) to relative 1e-6 for two random C.
4. LAW reproduces SPEC3 formulas: LAW(−ε² u uᵀ) = −ε² c_u(θ_A) and
   LAW((ε²/2)(v uᵀ − u vᵀ)) = (ε²/2) b_uv(θ_A)·κ with κ = ±1 — determine κ numerically and
   REPORT it; both to relative 1e-5 (h = 32, seed 0, ε = 0.05).
5. Reproduce the pre-check: h = 32, seed 0, ε = 0.025, oab, measured C: GN cos > 0.999,
   GD(1) cos > 0.999, MOM cos > 0.999; oneway GD(1) vs GN cos > 0.999.
6. S6 linear control.

If 4 or 5 fails, the bug is in the code until proven otherwise (verified independently).
Do not loosen a test; report the evidence instead.

## 8. Deliverables (in `task_cycle_rotation/`)

`tcr4.py`, `test_part4.py`, `run4.py` (writes `results4.json` incrementally), `report4.py`
(writes `REPORT4.md`: one table per S-rule with verdicts, then R1–R6). Logs and debug scripts
go in your scratchpad, not in that folder.

Final message back: `test_part4.py` output verbatim (including κ from test 4); every table;
verdicts with per-seed values where not PASS; DIVERGED / NOT CONVERGED / not-run conditions;
anomalies; wall time; every ambiguity or decision.
