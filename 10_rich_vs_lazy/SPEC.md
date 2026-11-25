# SPEC10 — Does a larger alignment fraction help on unseen inputs once the start function is removed?

Written 2026-11-22, before any result of this test exists. A FAIL of a rule is a valid result. Tune nothing after seeing numbers.
The coder implements exactly this text. If something here is ambiguous or seems wrong, STOP and ask; do not guess.

## 1. What is being tested (read this first)

Known from the SPEC9 run (real MNIST): learners that reach the same training outputs differ on unseen inputs along one
function, with coefficient f (alignment fraction; 0 = exact Gauss-Newton jump, 1/2 = continuous path, 1 = nearest point).
The follow-up run `g4_ends.py` split the test-loss difference between learners and found, in 32 blocks:
(a) larger f makes the function the network had at initialisation larger on unseen inputs, which always costs test loss;
(b) larger f changes everything that was learned toward the true labels (32 of 32 blocks).

**Claim under test.** If the start function is removed from the model, cost (a) loses its first-order part and a larger f
should give a LOWER test loss at equal training outputs.

**The intervention, exactly.** Network outputs F(theta, x) (10 numbers per input). theta0 = the stored start weights.
- PLAIN model: output = F(theta, x). (This is what SPEC9 measured.)
- ZERO model: output g(theta, x) = F(theta, x) - F(theta0, x). Same weights, same Jacobian; its output at theta0 is 0 everywhere.

Both models are trained by the SAME learners acting on theta through F. Only two things differ between PLAIN and ZERO:
1. the training target handed to the learner (sec. 3), and
2. the subtraction of F(theta0, x_test) before the test loss and accuracy are computed (sec. 4).
Nothing else may differ. The code must make this visibly true (one code path, one `kind` switch used in exactly those two places).

Not tested here: ReLU, cross-entropy, other data, more seeds. No new learner, no change to `part9lib.py`.

## 2. Objects (all from the SPEC9 library, used unchanged)

`import part9lib as L`, configured from the CONFIG cell of `evaluation_point_tests.ipynb` exactly as `g4_ends.py` does
(env overrides `PART9_NB`, `PART9_MODE`, `PART9LIB_DIR`, `PART9_RUN_DIR`, `PART9_NO_DOWNLOAD`, `PART9_DEVICE`).
Per combination `cb`: `ctx = L.Ctx9(cb, run_dir)`; `prob, st = ctx.prob, ctx.st`; `st.th0`, `st.F0` (training outputs at theta0, flat,
length n*C); `ctx.Y` (training one-hot labels, flat); `td = ctx.test()` with `td["X"]`, `td["Y"]` (flat one-hot test labels),
`td["labels"]`, `td["n_test"]`; `F0t, l0, a0 = ctx.test0()` (F_test(theta0)); `L.test_outputs(prob, td["X"], theta)` = F_test(theta) (flat).
C = CFG["N_CLASSES"]; N = n_test*C; matrices below are (n_test, C) reshapes; "centred" = minus the per-column mean over test inputs.
Combinations: the 8 of `L.combos("G4")` (tanh/gelu, seeds 0/1, n = 32 and 128, init start), plus `L.linear_combo()` for V5.
Learners are built with `L.L_from_spec(spec)` and run with `L.run_learner9(prob, st, t, s, Ld)`, `s = L.vnorm(t - st.F0)`.
f_nominal of a learner = the number paired with its spec in `CFG["G4_LEARNERS"]`.

## 3. Training targets

For tau (may be negative) and kind:
- PLAIN: `t = st.F0 + tau * (ctx.Y - st.F0)`   (identical to SPEC9 G4 and, for |tau| = S2_FRAC, to Part A direction 'true')
- ZERO:  `t = st.F0 + tau * ctx.Y`             (the ZERO model's training outputs go from 0 to tau*Y)

Exception, for bit-level reproduction of the stored Part A cases: in the SMALL-move block the PLAIN target is built exactly as
`run_A_case` builds it, `t = st.F0 + sign * (CFG["S2_FRAC"] * ctx.S["true"]) * ctx.dirs["true"]` (mathematically the same as the PLAIN
line with tau = sign*S2_FRAC). In the LARGE-move block the PLAIN target is the line above, exactly as in `run_G4_case`.

## 4. Model output, test loss, accuracy

`out(theta) = F_test(theta) - kappa * F0t`, kappa = 0 (PLAIN), 1 (ZERO).
`loss = |out - Y_test|^2 / (2N)`, `acc = mean(argmax_c out == label)`; compute with `L.test_loss_acc(out, td)`.
Also compute the loss by the explicit formula; both must agree to 1e-12 relative (validation V6).

## 5. What runs

Order per combination: small-move block, then large-move block (tau ascending; cheap learners first).

### 5.1 Small-move block (leading order; uses the Part A measurement)
For kind in (PLAIN, ZERO) and sign in (+1, -1): tau = sign * CFG["S2_FRAC"]; learner AJ(1) (`("AJ", 1.0)`); then
`sc, vL, vNP, thNP, thTH = L.measure_ext(prob, st, thL)`.
Record `rho = sc["rho"]`, `np_status`, `th_status`, `loss_NP`, `loss_TH`, `acc_NP`, `acc_TH` (model output of this kind, sec. 4),
and keep F_test(thNP), F_test(thTH).
`b_kind = 1/2 * sum over the two signs of (loss_NP - loss_TH)` = test-loss change per unit f, even part, at |tau| = S2_FRAC.
A sign is usable iff AJ(1) status is CONVERGED, `np_status` and `th_status` are CONVERGED and all numbers are finite;
b_kind exists only if both signs are usable.

### 5.2 Large-move block
tau in CFG["G4_TAUS"] (0.1, 0.25, 0.5, 1).
- ZERO: cheap learners `("DJ",1.5)`, `("DJ",1.0)`, `("SL",4)`, `("DJ",0.25)` always. Implicit learners `("TJ",1.5,1.0)`,
  `("SLa",4,2.0)`, `("AJ",1.0)`: run in block (cb, tau) iff the STORED SPEC9 G4 record of the same (cb, tau, learner) exists and is
  usable (rule below). This selection uses only old records, never a result of this run. Print every skip.
- PLAIN: only `("DJ",1.0)` and `("DJ",1.5)` (for validation V1 and as reference for the split). All other PLAIN numbers are
  read from the stored G4 records (`loss_L`, `acc_L`, `train_res`, `learner_status`).
Stored G4 record of (cb, tau, Ld): file `done/<L.safe_name(name)>.json` in the run directory with
`name = L._case9(L.cname9(cb), "G4", "G4", Ld, "true", f"tau{tau:g}", tau, 1, f_nominal=fnom)["name"]`.
Per end point record: `learner_status`, `train_res = L.vnorm(prob.F(thL) - t) / s`, `loss`, `acc`, `dist = L.vnorm(thL - st.th0)`,
`lam1, trK = L._lam1_tr(L.gram(prob.J(thL)))`, `a_star`, `loss_star` (sec. 6), `f_nominal`, wall seconds; keep F_test(thL).
USABLE (identical to the SPEC9 G4 report): `learner_status not in ("DIVERGED","AJ-FAILED","AJ-INCONSISTENT")` and `train_res` finite
and `train_res <= CFG["G4_RES_MAX"]`. Non-finite weights -> status "DIVERGED", nothing else computed.
A learner that raises an exception is recorded as status "ERROR" with the message and is not usable; the run continues.

## 6. Split of the difference between two end points (same formulas for both blocks and both kinds)

Reference end point `lo`, other end point `hi`, both of the same kind and block. Large block: lo = DJ(1), hi = every other
usable learner. Small block: lo = thTH, hi = thNP, per sign; every piece is then averaged over the two signs.
```
Nm = F0t (matrix), Nv = centred Nm
S  = F_test(lo) - F0t, Sv = centred S
r  = out(lo) - Y_test                      (out of this kind, sec. 4)
D  = F_test(hi) - F_test(lo)               (the same for both kinds), Dv = centred D, Dc = D - Dv
loss(hi) - loss(lo) = <r, D>/N + |D|^2/(2N)          (exact)
least squares on centred matrices: Dv = eN*Nv + eS*Sv + rest   (2x2 normal equations)
const = <r, Dc>/N,  gN = <r, Nv>/N,  gS = <r, Sv>/N,  dN = eN*gN,  dS = eS*gS,  drest = <r, rest>/N,  quad = |D|^2/(2N)
R2 = 1 - |rest|^2/|Dv|^2
```
Check (V4): |const + dN + dS + drest + quad - (loss(hi) - loss(lo))| <= 1e-7 * max(|each term|, |difference|).
Per end point: `outv` = centred out; `a_star = <Y_test, outv>/|outv|^2`; `loss_star` = loss of `a_star*outv + (out - outv)`.
(If |outv| = 0, a_star and loss_star are NaN.)

## 7. Validations (the run is INVALID for interpretation unless all pass; the report prints each one)

- V1 PLAIN reproduction: for every block, PLAIN DJ(1) and DJ(1.5): |loss - stored loss_L|/|stored| and |dist - stored dist|/|stored|
  both <= 1e-9 (GPU, same machine as SPEC9: expected 0).
- V2 small-move PLAIN reproduction (n = 32 combinations only, where the Part A record exists): the stored record
  `L._case9(cname, "A", "A", L.L_AJ(1.0), "true", "s2", CFG["S2_FRAC"], sign)["name"]` has `loss_NP`, `loss_TH`, `rho`;
  relative difference of each <= 1e-9, both signs.
- V3 same training outputs: every usable end point has train_res <= G4_RES_MAX (true by definition; print the maximum per kind),
  and for ZERO additionally `L.vnorm((prob.F(thL) - st.F0) - tau*ctx.Y) / L.vnorm(tau*ctx.Y) <= G4_RES_MAX` computed by this explicit
  formula (it proves the ZERO model's training outputs are tau*Y).
- V4 the sum check of sec. 6 in every split.
- V5 linear control (`L.linear_combo()`, worker 0 only, ZERO kind, tau = 0.5, learners DJ(1.5), DJ(1), DJ(0.25)):
  max over learners of |F_test(hi) - F_test(DJ(1))|_2 / |F_test(DJ(1)) - F0t|_2 <= 1e-7. (In a model linear in the weights all
  learners coincide; a non-zero difference would mean the bookkeeping, not the network, creates the effect.)
- V6 loss two ways (sec. 4), every end point.
- V7 ZERO model at theta0: loss of out = 0 equals 1/(2C) to 1e-12 (one-hot labels), every combination.

## 8. Pre-registered rules (ZERO model; 32 blocks = 8 combinations x 4 tau; a block counts only if both learners are usable)

Let lossZ[L] be the ZERO test loss of learner L in a block, lossP[L] the PLAIN one (stored record).
- **HZ1 (primary).** lossZ[DJ(0.25)] < lossZ[DJ(1)] in at least 24 of 32 blocks. (f = 0.43 beats f = 0.)
- **HZ2.** lossZ[DJ(1.5)] > lossZ[DJ(1)] in at least 24 of 32 blocks. (f = -1 loses to f = 0.)
- **HZ3 (paired).** [lossZ[DJ(0.25)] - lossZ[DJ(1)]] < [lossP[DJ(0.25)] - lossP[DJ(1)]] in at least 28 of 32 blocks.
  (Removing the start function moves the effect of alignment in the favourable direction.)
- **HZ4.** lossZ[SL(4)] < lossZ[DJ(1)] in at least 24 of 32 blocks. (Second learner with f > 0, different mechanism: exact
  sub-jumps instead of damped steps.)
If fewer than 32 blocks count, the threshold is the same fraction rounded up (75%, 87.5%), and the report says so.
Blocks are not independent (4 starts, 2 training-set sizes, 4 nested move sizes): the report also prints each count per
combination and per tau, and the number of combinations in which the majority of tau values agrees with the rule.

## 9. Stated expectations (report only, written before the run; each printed with its count)

- E1 small moves: b_ZERO < 0 in the four n = 128 combinations; sign mixed at n = 32 (a chance term eN*gN of either sign remains).
  b_PLAIN > 0 in 8/8. Also print b_ZERO / b_PLAIN.
- E2 split, ZERO, DJ(0.25) vs DJ(1): dS + drest < 0 in at least 28/32; gN has no fixed sign (count both); eN > 0 in at least 28/32.
- E3 with the implicit learners (f up to 1) where usable: Spearman(f_nominal, lossZ) over usable learners < 0 in at least 70% of
  the blocks that have 6 or more usable learners; print the learner with the lowest lossZ per block.
- E4 accuracy: accZ[DJ(0.25)] > accZ[DJ(1)] in at least 20/32.
- E5 after the best one-number rescaling: loss_star of DJ(0.25) < loss_star of DJ(1) (ZERO) in at least 24/32.
- E6 kernel: trK rises with f among DJ(1.5), DJ(1), DJ(0.25) at tau = 0.1 in 8/8 (ZERO), as in PLAIN.
- E7 removing the start function alone: lossZ[DJ(1)] < lossP[DJ(1)] in 32/32.
- E8 best move size: for each learner the tau of the grid with the lowest lossZ; print whether it is larger for larger f.
Reading of outcomes: HZ1+HZ3+HZ4 pass -> "alignment helps once the start function is removed" holds on MNIST for these nets.
HZ3 passes, HZ1 fails -> only "the cost of alignment is the start function" holds. HZ3 fails -> the g4_ends split misled us.

## 10. Deliverables (new files only; do NOT modify any existing file; no package installation)

Project root:
1. `spec10_run.py K M` — one worker (combinations sorted by n descending, worker K takes indices i with i % M == K; the linear
   control V5 is done by worker 0). Device cuda if available else cpu, `PART9_DEVICE` overrides. On cuda it REFUSES to start
   (exit code 2, clear message) if a process whose command line contains `part9lib.worker_main` is alive (`pgrep -f`), unless
   `SPEC10_FORCE=1`. Writes only into `spec10_out/` next to the notebook: `records_K.json` (every record, split and validation
   value, plus mode, data_source, device) written atomically after every combination, and `bundle_K.pt` (float32 F_test of every
   end point incl. thNP/thTH, F0t, labels). Prints one readable table per block as it goes (flush). Per-learner try/except.
2. `spec10_report.py [dir]` — CPU, reads all `records_*.json`, prints and writes `spec10_out/spec10_report.txt` and
   `spec10_results.json`: first the validations (a banner INVALID if any fails or any worker file is missing: the report
   knows M from the records), then the verdict table HZ1-HZ4 with counts, then E1-E8, then the per-block tables.
3. `review10/make_tiny10.py` and `review10/test_spec10.py` — local test in MODE tiny on CPU: build a fresh tiny run directory with
   the library's own runner (starts, the Part A cases of AJ(1) for direction 'true' size s2 both signs, ALL G4 cases, the linear
   start), run the worker (1 of 1) and the report as subprocesses, and assert: exit codes 0; V1, V2 (<= 1e-12 on CPU), V4-V7 pass;
   every planned end point present; the report runs; and three NEGATIVE CONTROLS, each of which must make the stated check FAIL:
   (i) env `SPEC10_BREAK=target` makes the ZERO target equal the PLAIN target -> V3's explicit ZERO check must fail;
   (ii) `SPEC10_BREAK=kappa` computes the ZERO loss without subtracting F0t -> V7 must fail;
   (iii) `SPEC10_BREAK=repro` multiplies the PLAIN DJ(1) step target by 1.001 -> V1 must fail.
   The break hook is read once at start, printed in a loud banner, written into the records, and the report prints INVALID
   whenever it is set.
Style: plain Python, the same loading code as `g4_ends.py` (read it first; it ran correctly on the rig), float64 for all
arithmetic, tensors moved to CPU before saving. Reference files to read: `g4_ends.py`, `g3_parts.py`, `review9/v3/part9lib.py`
(functions named in sec. 2-5), `review9/v3/mk_tiny_run.py`, `review9/v3/mk_tiny_g4.py`, `review9/v3/common9.py`.

## 11. What the coder reports back
Files written; the full output of `test_spec10.py`; any place where this spec was ambiguous or where the library behaves
differently from what the spec assumes (quote the line); anything not done. No claims about the science.

## 12. Addendum A1 (2026-11-22, after code review and local tests, BEFORE any rig result)

A1.1 Review fix in `spec10_run.py` (note): `torch.cuda.set_device` was called with a device without an index, which raises on
the rig when one GPU is visible per worker. It is now called only for an explicit `cuda:K`. Nothing else in the delivered
code was changed. Local test after the fix: 26 of 26 checks pass. Independent cross-check `review10/crosscheck10.py` (own
learner loops, own targets, own loss; shares no code with the worker): single losses agree to 2e-8, the differences used by
HZ1, HZ2, HZ4 to 4e-6, same sign in 6 of 6.

A1.2 Reading caveat found in the local test, fixed here before the run. On the tiny local run the test labels are RANDOM and
HZ1-HZ4 still "pass" (2 of 2 blocks). Reason: with labels that carry nothing, the learned change is pure noise on unseen
inputs, a larger f shrinks it (eS < 0), and shrinking noise lowers the test loss. So HZ1 alone cannot separate "larger f
improves what was learned" from "larger f makes an over-fitted learned change smaller". Therefore the statement
"alignment IMPROVES what is learned" additionally requires, on the ZERO model, DJ(0.25) against DJ(1):
  - S1: loss_star[DJ(0.25)] < loss_star[DJ(1)] in at least 24 of 32 blocks (E5; the best one-number rescaling removes size), and
  - S2: drest < 0 in at least 24 of 32 blocks, and
  - S3: both S1 and S2 hold in at least 18 of the 24 blocks with tau <= 0.5 (in g4_ends the learned change was too SMALL there,
        so shrinking it cannot be the reason for a gain).
If HZ1 passes but S1-S3 do not, the result is "a larger f helps the ZERO model by making the learned change smaller", a size
effect, not better features. S1-S3 are computed by the reviewer from `spec10_results.json` / `records_*.json`.

A1.3 The linear control V5 is weak for damped learners: in a linear model their ratio equals their training residual
(0.5^24 = 5.96e-8, 0.75^57 = 7.56e-8) by construction. It still catches bookkeeping that differs between learners.
