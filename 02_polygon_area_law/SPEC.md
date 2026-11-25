# Task-cycle rotation, part 2 — the area law (F) and anticipatory recovery (R)

Same conventions as SPEC.md: float64, CPU, deterministic, every random object from a seeded
`torch.Generator`. Reuse `tcr.py` unchanged (import it; do not edit it, do not edit `run.py`,
`report.py`, `results.json` or `REPORT.md`). New code goes in new files (§5).

Notation from SPEC.md: student MLP f(x;θ) (§2.1), teacher g, task inputs X_A, directions U, U⊥
(§2.3, per seed), probe set, GN learner (§3.1), θ_A = Learn(θ_init, A), displacement
δ_s = θ_s − θ_A, function displacement φ_s on the probe set.

---

## Part F — the area law

### F.1 Statement being tested (derived for a model linear in θ)

Let P be the orthogonal projector onto the null space of J_A (so learning A to convergence is
θ − θ* ↦ P(θ − θ*) for any joint solution θ*). For a task with inputs X_A + ε(x U + y U⊥) the
projector is P + ε(x D_u + y D_v) + O(ε²), and P D P = 0. For a closed polygon of tasks
p_0 = (0,0) = A, p_1, …, p_{N−1}, learned in the order p_1, …, p_{N−1}, A, the operator on
A's solution set is

    R = P − (ε²/2) Σ_{k=0}^{N−1} P (ΔD_k)² P  +  ε² · 𝒜 · P [D_v, D_u] P  + O(ε³)

    ΔD_k = D(p_{k+1}) − D(p_k) (indices mod N, D(p_0) = 0),
    𝒜 = signed polygon area by the shoelace formula = ½ Σ_k (x_k y_{k+1} − x_{k+1} y_k).

Consequences: the order-antisymmetric displacement is proportional to the signed enclosed
area and has a fixed direction; the order-symmetric displacement is governed by the sum of
squared edge lengths.

### F.2 Polygon families (all with p_0 = A at the origin of the (x,y) task plane)

Task at plane point (x,y): inputs X(x,y) = X_A + ε (x·U + y·U⊥), labels g(X(x,y)).

- **tri**: p_1 = (1,0), p_2 = (0,1). Area ½. (This is the generic γ=π/2 loop of SPEC.md.)
- **circ(N)**, N ≥ 3: p_k = (cos(2πk/N) − 1, sin(2πk/N)), k = 1..N−1. A regular N-gon inscribed
  in the unit circle centred at (−1,0), passing through the origin. Area (N/2)·sin(2π/N).
- **rect(c)**: p_1 = (1,0), p_2 = (1,c), p_3 = (0,c). Area c. Used with c ∈ {0.25, 0.5, 1, 2}.
- **fig8**: the vertex list is exactly [(1,0), (0,1), (0,0), (−1,0), (0,1)], where the vertex
  (0,0) means "learn A". So the forward sequence after θ_A is: (1,0), (0,1), A, (−1,0), (0,1),
  then the final A. The first lobe (0,0)→(1,0)→(0,1)→(0,0) has signed area +½; the second lobe
  (0,0)→(−1,0)→(0,1)→(0,0) has signed area −½. Net signed area 0. The reverse sequence is the
  same list reversed, then the final A: (0,1), (−1,0), A, (0,1), (1,0), A.

For a vertex list `pts`, forward sequence = learn pts in order then A; reverse sequence =
learn pts in reversed order then A. With δ₊, δ₋ the two displacements:

    a = (δ₊ − δ₋)/2,   s = (δ₊ + δ₋)/2          (also in function space from φ₊, φ₋)

### F.3 Predictions

Let a_tri be `a` for **tri** at the same (seed, h, ε).

| id | prediction | linear-model check already done (random-feature model, ε = 0.01) |
|----|-----------|------|
| F1 | ‖a_circ(N)‖ / ‖a_tri‖ = N·sin(2π/N) | N=3: 2.5986 vs 2.5981; N=8: 5.633 vs 5.657; N=64: 6.2525 vs 6.2731 |
| F2 | cos(a_circ(N), a_tri) ≈ 1 for every N (fixed direction) | 0.998 |
| F3 | ‖s_circ(N)‖ ∝ 1/N for large N, i.e. N·‖s_circ(N)‖ → constant; slope of log‖s‖ on log N over N ∈ {16,32,64} ≈ −1 | N‖s‖ = 0.01715, 0.01738, 0.01755 |
| F4 | ‖a_rect(c)‖ / ‖a_tri‖ = 2c, and cos(a_rect(c), a_tri) ≈ 1 | (not pre-checked) |
| F5 | fig8: ‖a_fig8‖ / ‖a_tri‖ → 0 as ε → 0 (slope of log ratio on log ε > 0.5) | (not pre-checked) |
| F6 | accumulation: with m forward cycles of circ(32) vs m reverse cycles, ‖a_m‖ / (m‖a_1‖) ≈ 1 for small m, decaying slowly | 1.000, 0.9987, 0.9960, 0.9901, 0.9766 for m = 1,2,4,8,16 at ε=0.02 |

For F4 and F5 there is no pre-check: report whatever is measured. A FAIL is a valid result;
never tune anything to make a prediction pass.

### F.4 Grid

Seeds {0,1,2,3,4}; widths h ∈ {32, 128, 512}; GN learner throughout.

- FG1: ε ∈ {0.1, 0.05, 0.025}; tri and circ(N) for N ∈ {3,4,6,8,12,16,24,32,48,64}. Record
  ‖a‖, ‖s‖ (weight and function space), ratio to tri, cos to a_tri.
- FG2: ε ∈ {0.1, 0.05, 0.025}; rect(c) for c ∈ {0.25, 0.5, 1, 2}.
- FG3: ε ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}; fig8.
- FG4: ε = 0.05, circ(32), m ∈ {1,2,4,8,16} cycles. One "cycle" = learn p_1..p_31 then A.
  a_m = (δ₊^{(m)} − δ₋^{(m)})/2 where δ₊^{(m)} is the displacement after m forward cycles and
  δ₋^{(m)} after m reverse cycles. Also record ‖s_m‖ and the A-loss residual at the end.
- FG5 (GD validation): h = 128, ε = 0.05, seeds {0,1,2}, tri and circ(N) for N ∈ {4, 8, 16},
  GD learner of SPEC.md §3.2. Report ‖a_circ(N)‖/‖a_tri‖ under GD. 40-minute wall cap as before.

If any single grid would exceed 60 minutes, stop it, report what finished, and say so.

---

## Part R — anticipatory recovery in plain cyclic projections

### R.1 Model (linear regression, no neural network)

    θ* ~ N(0, I_d);  θ_0 = 0
    task t ∈ {0..T−1}:  X_t ∈ R^{n×d}, entries ~ N(0,1);  y_t = X_t θ*
    learn task t to convergence:  θ ← θ − X_t⁺ (X_t θ − y_t)        (X_t⁺ = pseudo-inverse)
    loss  L_t(θ) = (1/n) ‖X_t (θ − θ*)‖²

Draw order from `Generator(seed)`: θ*, then X_0, …, X_{T−1}.
Overparameterisation ratio κ = d / (T·n).

### R.2 Schedules

- **cyclic**: E epochs, each epoch learns tasks 0,1,…,T−1 in that order.
- **shuffle**: E epochs, each epoch learns all T tasks in a fresh uniformly random permutation
  (drawn from `Generator(10_000 + seed)`).
- **partial(k)** (cyclic order, but each "learn" is k steps of gradient descent on L_t with
  step size n / λ_max(X_t X_tᵀ) instead of the exact projection), k ∈ {1, 3, 10}.

After EVERY learning step record the full vector (L_0, …, L_{T−1}).

### R.3 Measures

For the cyclic and partial schedules, epochs are 1-indexed and global learning steps are
0-indexed, so task s is learned in epoch e at global step τ(e,s) = (e−1)·T + s. Let
L_s[i] be the loss of task s recorded immediately after global step i. For e ≥ 2 the window is

    W(e,s) = ( L_s[τ(e−1,s)], L_s[τ(e−1,s)+1], …, L_s[τ(e,s)−1] )        (exactly T values)

Its first entry is right after task s was learned in epoch e−1; its last entry is just before
task s is learned again in epoch e.

    peak position   j*(e,s) = argmax_j W_j  ∈ {0..T−1}
    recovery ratio  ρ(e,s)  = W_{T−1} / max_j W_j         (1 = no recovery before the revisit)
    recovered frac  1 − ρ

Report, per epoch e ∈ {2..E}: mean and standard deviation over all tasks s and all seeds of
ρ(e,s) and of j*(e,s)/(T−1).

For all schedules also report the prequential loss: the mean over epochs 2..E, tasks and
seeds of L_s evaluated just before task s is learned.

Also, for cyclic only, the one-cycle operator M = P_{T−1} ⋯ P_0 with P_t = I − X_t⁺ X_t:
number of nonzero eigenvalues (|λ| > 1e-9), how many are non-real (|Im λ| > 1e-9), and the
median and maximum of |arg λ| over the non-real ones. (Skip this for d > 1600.)

### R.4 Grid (10 seeds each: seed ∈ {0..9}; E = 5 unless stated)

- RG1 (overparameterisation): T = 25, n = 4, κ ∈ {1.05, 1.25, 1.5, 2, 3, 4, 8, 16}
  (d = round(κ·T·n)); schedules cyclic and shuffle.
- RG2 (number of tasks): κ = 4, n = 4, T ∈ {10, 25, 50, 100}; cyclic and shuffle.
- RG3 (task size): κ = 4, T = 25, n ∈ {1, 4, 16}; cyclic.
- RG4 (many epochs): κ = 4, T = 25, n = 4, E = 20; cyclic. Report ρ per epoch for all 19.
- RG5 (partial learning): κ = 4, T = 25, n = 4; partial(1), partial(3), partial(10), and
  cyclic (exact) for reference.

### R.5 What is being asked (no pass window is pre-set except RH1)

- RH1: at κ ≥ 3 (RG1), mean ρ over epochs 2–5 is below 0.8; at κ ≤ 1.25 it is above 0.9.
  A single earlier one-seed run gave, for task 0 only: κ=4 → ρ ≈ 0.34, 0.29, 0.47 (epochs
  3–5) with peak at positions 12, 12, 17 of 24; κ=1.5 → 0.85–0.99; κ=1.1 → 0.86–0.91.
  That run is the thing being replicated; it may not replicate.
- RH2..: report the dependence of ρ and j* on κ, T, n, E and k as measured, and the
  cyclic-vs-shuffle prequential loss. No expected values are given. Report numbers only.

---

## 4. Required tests before running (`test_part2.py`)

Using the linear-in-θ model of SPEC.md §7 (class `LinearModel` in tcr.py), seed 0, ε = 0.01:

1. tri reproduces the generic γ=π/2 loop of SPEC.md: a from pts [(1,0),(0,1)] equals a_loop
   computed by the existing sequence runner to < 1e-12.
2. F1 on the linear model: |‖a_circ(N)‖/‖a_tri‖ − N sin(2π/N)| / (N sin(2π/N)) < 0.01 for
   N ∈ {3,4,6,8,16,32,64}.
3. F2: cos(a_circ(N), a_tri) > 0.99 for those N.
4. F3: slope of log‖s_circ(N)‖ on log N over N ∈ {16,32,64} in [−1.1, −0.9].
5. Part R: one exact projection step gives L_t < 1e-20 for the task just learned; and for
   T = 2 the cyclic iterates equal the closed form (P_1 P_0)^e applied to (θ_0 − θ*) to < 1e-10.
6. Part R: window bookkeeping — on a hand-built loss table (T = 3, E = 3, known numbers)
   the code returns the right W(e,s), j* and ρ.

If test 2, 3 or 4 fails, the bug is in the code until proven otherwise (these were verified
independently). Do not loosen them; report the evidence if you believe the expectation is wrong.

## 5. Deliverables (in `task_cycle_rotation/`)

- `tcr2.py` — polygon families, forward/reverse runner, part-R model, schedules, measures.
- `test_part2.py` — §4, prints PASS/FAIL per test.
- `run2.py` — runs FG1–FG5 and RG1–RG5, writes `results2.json` incrementally.
- `report2.py` — reads `results2.json`, writes `REPORT2.md`: one table per prediction F1–F6
  (per width and ε: median and min–max over seeds, next to the predicted value) and one
  table per R grid (mean ± sd), plus PASS/FAIL/INCONCLUSIVE for F1–F6 and RH1 with the rule
  you used stated explicitly. No plots unless matplotlib is already installed; install nothing.

Final message back: test output verbatim; the F1–F6 tables with predicted values alongside;
the R tables; every INVALID / NOT CONVERGED condition; wall time; and every ambiguity or
decision. In particular say how you resolved the fig8 definition and the window definition
in R.3 if you found either unclear.
