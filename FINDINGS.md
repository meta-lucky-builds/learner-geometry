# Findings, sorted by strength

Results, theory, predictions and literature on the alignment-fraction / evaluation-point programme, strongest first. Section numbers (## 19., "sec. 22") are the original lab-notebook numbering; code comments refer to them as `FINDINGS.md sec. N`.

---

# PART 1 - RESULTS (measured), strongest to weakest

## Tier 1: exact, pre-registered or nothing fitted, real data

### 1. SPEC8 official result: 960/960 units on real MNIST

## 19. SPEC8 GPU run — OFFICIAL RESULT (2026-11-22; analysis-only pass on the stopped run, exact vectors)
PRIMARY verdicts, tanh and gelu, init and trained (4 groups): H1x 54/54, H2x 90/90, H3x 48/48, H4x 48/48, H5 18/18, H5b 12/12, H6 36/36, H7x 36/36 in EVERY group; H0 16/16 (3e-13).
 => 960/960 (learner, direction) units. Relative error of the extrapolated fraction: tanh median ~5e-7, max 3.8e-5 (DJ(1.9): 1.1e-3); gelu median ~2e-6, max 3.6e-4 (DJ(1.9): 1.4e-2,
 the single worst unit, gelu-init-s1 eigD: -8.872 vs -9, 71% of the tolerance; 99% of units use < 8.2% of their tolerance). |cos| >= 0.999993 (tanh), 0.999774 (gelu).
 H5: cos(LAW, measured) 0.999945..1.000000, ratio 0.974..0.9999; H5b <= 0.0059; H6 rho in [0.9999935, 1.0000002], cos 1.000000000; H7x nr(s2) <= 0.00019.
 Order check: 960 rows, ratio median 3.97-3.99, 5-95% [3.74, 4.02], 0 flagged.
Literal single-size rules: H1, H3, H4, H7 pass in all four groups (H3 gelu/init 94/96); H2 passes in 3 groups (94.4-98.3%) and FAILS in gelu/init (158/180 = 87.8%), all failures
 DJ(1.5)/DJ(1.9) on weak or generic directions (finite size next to the pole).
ReLU: NO DATA for every rule except H6 (3/3 at rho = 1); excluded as NP-FAILED or not run. Not a test of the law for ReLU (see section 18 for the actual ReLU picture).
Caveat: the notebook was stopped before nbconvert saved it, so the printed unit-test output of the GPU execution was not kept; the grid cell only runs if all tests pass
 (the TESTS cell raises otherwise), and 0 cases failed, but the UW worker-identity test printout itself is not in hand.

### 2. SPEC11 rig result: loop law 36/36, trainability ordered by f

RIG RESULT, SPEC11 (verdicts as pre-registered):
 Validations V1-V4 PASS; V5 PASS tanh, FAIL gelu by a hair (sign agreement 0.9894 < 0.99 for one direction at s1: gelu is more curved).
 PART A (both networks): HP1 loop law 36/36 (extrapolated values equal 2f-1 to 4 digits, e.g. -3.0000, -0.3333, +2.0000; AJ(.5) 1e-12),
  HP2 sign 32/32, HP3 own cost 36/36, HP4 other directions slope 1.007 (tanh, R2 1.0000) and 1.058 (gelu, R2 .998), 144 points each,
  HP5 general stiffening 32/32, HID identity 4/4 (1.0000, 1.0000, 1.0003, 1.0009).
 PART B (star of 12 excursions): tanh HB1 6/6 (-1.0147, -3.123, -.5046, -.1255, 2e-12, +.988), HB2 slope 1.068, HB3 5/5.
  gelu HB1 3/6, HB2 slope 2.86, HB3 3/5 -> FAIL: effects do not add linearly on gelu (see the depletion law below).
 PART C (six full relabelling tasks; Q6 = mean log cost of 8 fresh directions; start Q0 = -2.21 tanh, -0.43 gelu):
  tanh: DJ(1.5) -2.207, DJ(1) -2.577, SL(2) -2.535, SL(4) -2.668, SL(8) -2.814, SLa(8,1) -3.213; Spearman(f, Q6) -0.94 -> HC1 PASS.
  gelu: -2.079, -2.326, -2.163, -2.239, -2.428, -3.008; Spearman -0.83 -> HC1 PASS. HC3 PASS both (implicit lowest by 0.40 / 0.58).
  HC2 FAIL both, on DJ(1) only (at large moves DJ(1) takes several corrective steps, so it is not an f = 0 learner there: the spec's error).
  TJ(1.5,1) failed to converge on task 1 in both networks. Distance from the start after 6 tasks: 7.97, 4.96, 4.16, 2.85, 2.35, 1.85 (tanh);
  9.00, 7.09, 6.05, 4.02, 3.25, 2.42 (gelu): rank correlation with f = -1. After task 1 EVERY learner is far cheaper than the start
  (-2.97..-3.08 tanh): one task builds kernel; later tasks spend it in order of 1 - 2f; the implicit learner keeps gaining.
  A (log of mean eigenvalue x mean inverse eigenvalue) at task 6, tanh: 2.378, 1.920, 1.714, 1.427 for SL(2), SL(4), SL(8), SLa(8,1) =
  about 1.53 + 1.6 (1 - 2f): the a + b/N law of the first GPU run (sec. 10-12), continued through zero to the implicit side. Start: 1.748.
  Final true-label task (SL(8) for everyone), test loss / accuracy, tanh: .0389/.518, .0403/.527, .0416/.518, .0411/.517, .0429/.507,
  .0468/.490 (rank correlation of loss with f +0.94); fresh network .0462/.498. gelu: .0382/.471, .0384/.474, .0368/.495, .0404/.461,
  .0404/.476, .0438/.482; fresh .0376/.521.

### 3. The alignment-fraction law and its verification

## 14. THE ALIGNMENT-FRACTION LAW (2026-11-21 evening) — strongest result so far
Setup: fixed start theta_0, target outputs t, fibre = all weights with outputs t. Three reference points on the fibre:
 GN = single/iterated full min-norm Gauss-Newton jump (path matrix C = 0);
 NP = Euclidean-nearest point of the fibre to theta_0 (equivalent to C = dY dY^T; P_v(NP-GN) = LAW(dY dY^T) = -P_v grad[1/2 dY^T K^-1 dY]);
 any learner L lands at GN + LAW(C_L). Define alignment fraction R_L = (g(L)-g(GN))/(g(NP)-g(GN)) for any kernel functional g (lam1, trK).
Law (second order in displacement, ALL orders in step size):
 - GN (natural gradient, full step): 0. Gradient flow / any continuous path: 1/2 (area term). NP: 1.
 - damped GN with step fraction c, or GD along kernel mode i with c_i = eta*lam_i/n: f = (1-c)/(2-c).
 - full GD matrix (eigenbasis of K): C_ij = r_i r_j * c_j(1-c_i)/(c_i+c_j-c_i c_j); sym part f_ij = 1/2 - (1/2) c_i c_j/(c_i+c_j-c_i c_j);
   area_ij = (1/2) r_i r_j (c_j-c_i)/(c_i+c_j-c_i c_j).
 - heavy ball: f = 1 - [(1-mu)^2(1+mu) + 2 mu c]/[(1-mu)^2 (2(1+mu) - c)]; zero at c* = (1-mu)^2(1+mu)/(1+mu^2) (mu=.9: 0.0105,
   i.e. effective step factor 0.105); pole at c = 2(1+mu); first order: 1/2 - (c_eff/4)(1+mu)/(1-mu) = Ghosh et al. momentum IGR factor.
 - CEILING: for learners whose steps lie in row(J), C_sym = 1/2 dYdY^T - 1/2 Q with Q >= 0, so f <= 1/2: gradient flow is the
   maximal aligner; reaching NP (f = 1) needs vertical steps (e.g. proximity penalty to theta_0).
 - First order in eta reproduces Barrett-Dherin IGR EXACTLY (algebra: -(eta/4n) r_i r_j/(lam_i+lam_j) P_v grad K_ij both ways);
   the denominator lam_i+lam_j-(eta/n)lam_i lam_j is the all-orders resummation; its zero is the edge of stability.
   => IGR and EoS self-stabilisation are one term. Same discrete-Lyapunov structure as Damian-Ma-Lee's label-noise covariance
   (2106.06530, S_ij = lam Sigma_ij/(lam_i+lam_j-eta lam_i lam_j)) with the noise covariance replaced by r_0 r_0^T.
Verification: (a) SPEC7 Part A records (h=64 x1, h=128 x4 seeds): measured R matches to 3 decimals for
 SL (0.492), GD c=0.25/1/1.5/1.9 along q1..q16 (e.g. -9.003 vs -9.000), same for trK; random directions 0.49-0.50 median (range 0.39-0.56 = area).
 (b) independent exactly solvable model beta = u^2 - v^2 (Woodworth et al.): R_flow -> 0.5000,
 damped GN and GD match (1-c)/(2-c) to 4 decimals. (c) heavy ball in that model : closed form to 4 digits, zero crossing at 0.0105.
 (d) one non-perturbative cell (h=128, kappa=4, full training from init): (SL-GN)/(NP-GN) = 0.48, GD 0.49.
Geometry: weight->output map is a submersion; symmetric part of T = fibre-gradient of the base metric K^-1 (failure of Riemannian
 submersion = feature learning); antisymmetric part = curvature of the horizontal distribution (O'Neill A-tensor).
Literature search : NOT FOUND: flow-vs-nearest leading correction in general nets; any GN-endpoint sharpness/alignment
 measurement vs GD; deterministic full-batch regulariser with (2-eta*lam) poles; statement IGR = EoS term; NTK-derivative form of finite-step bias;
 "full Newton step has no implicit bias"; data-dependent fibres with a connection; O'Neill tensors / "Riemannian submersion iff kernel constant on fibres";
 rough-path language; lam1 vs 2/lr across tasks; unit survival curves; optimizer comparison at matched step in continual learning; smooth drift IMPROVING trainability.
 PRIOR (must cite): Damian-Ma-Lee 2106.06530 (pole + pairwise denominator, label noise; momentum version); Rosca principal flow 2302.01952 (all orders in eta*lam
 but drops third-derivative terms - ours is exactly that dropped term resummed); Miyagawa 2210.15898; Cattaneo-Shigida 2509.08483 (heavy-ball modified loss);
 Ghosh et al. (momentum IGR); Woodworth 2002.09277 (flow != nearest, one model, no formula); Li-Wang-Lee-Arora 2207.04036; Chizat lazy bounds;
 Arbel 2302.02904 (GN: larger step -> worse features; "hidden learning"), Buffelli 2411.07979 (exact GN stays lazy), Jiang 2601.03162 (LM/GN -> NTK solution),
 Amari 2006.10732; Jiang-Cohen-Li 2507.12837 (larger lr -> more KTA: a TENSION to test); Wang & Wang PMLR v282, Dong & Cheng 2603.21502 (symmetry fibres);
 Lyle 2402.18762 + Liu & Mou 2602.09234 (abruptness dose-response already known); Baveja 2509.19698 (Adam effective step vs 2); Lewandowski 2312.00246.
Other checks this pass: GN's kernel growth in the GPU run is NOT weight-norm inflation (norm x1.2 => ^4 = 2.1, but trK x6.9, lam_max x10.9; GD: norm^4 ~ trK);
 it is the weak-direction travel GN does and GD (stopping at mse 1e-3) skips. Adam in the u^2-v^2 toy: R = -1.8 / -211 / -19802 at lr 1e-4/1e-3/1e-2, ends at the
 minimum-kernel point of the fibre (v = 0): perpetual jitter at convergence = sharpness minimisation along the solution set -> degenerate kernel (lead, untested in nets).

### 4. Three-function law at third order (X8): residual 0.00000, nothing fitted

X8 RESULT SO FAR (2026-11-23 noon; 3 of 6 blocks; gelu blocks and tanh-s1 tau .1 still running):
 tanh-s0 tau .03 (13 learners), tanh-s0 tau .1 (13), tanh-s1 tau .03 (12): singular values / largest of the difference vectors against DJ(1):
  explicit learners (8): 1, 0.109, 0.025, 0.0004 | 1, 0.107, 0.025, 0.0003 | 1, 0.109, 0.025, 0.0005;
  all learners: 1, 0.116, 0.075, 0.0003 | 1, 0.116, 0.074, 0.0006 | 1, 0.142, 0.077, 0.0004.
  Three-function model with weights (A2, A3, B3) from the recursion, nothing fitted: residual 0.00000 of sum |D|^2 in all three blocks,
  for explicit AND implicit learners (AJ(.5), AJ(1), SLa(2,1), SLa(4,2), TJ(1.5,1)). Two functions: (A2, A3) 0.4-0.7%, (A2, B3) 0.06-2%.
  R1 HOLDS (3/3 so far), R2 HOLDS (3/3; sigma_4/sigma_1 <= 0.0006), R3 HOLDS (tau .03 vs .1 identical to 2 digits). R4 (gelu) pending.

X8 COMPLETE (2026-11-23 afternoon; 84 learner runs, 0 failed; all 6 blocks, 13 learners each): rank exactly 3 and (A2, A3, B3) residual 0.00000 in tanh-s0 and tanh-s1 at tau .03 and .1 and in gelu-s0 at tau .03 (fourth singular value / largest <= 0.0006). gelu-s0 at tau .1: fourth singular value 0.0023 (explicit) / 0.0048 (all), residual 0.00004 / 0.00012: next-order contamination (it was 0.0003 / 0.0005 at tau .03, i.e. grows ~10x for tau x3.3). R1 HOLDS in 5/6 blocks (literal threshold 0.002 missed by gelu tau .1 at 0.0023), R2 HOLDS 6/6, R3 HOLDS for tanh (identical spectra at both sizes), R4 HOLDS (gelu: same rank, same weights).
X8 DEEPER READ (2026-11-23): the three functions (weights A2, A3, B3) extracted per block. (i) distinct shapes: mutual cosines -0.07..-0.47 (one +0.27); sizes A3/A2 0.34-0.55, B3/A2 0.23-0.40. (ii) exact tau^3: norm ratio tau .1/.03 = 36.0-36.8 (37.0 expected), shape cos 0.998-1.000 for tanh; gelu B3 function unstable at tau .1 (cos 0.71) = the faint fourth component. (iii) EFFECT ON TEST LOSS, same sign in 6/6 blocks: A2 lowers loss, shrinks the learned change and moves its pattern toward the labels; A3 and B3 raise loss, enlarge it and move the pattern away. (iv) so the alpha-point family has loss = a alpha + (b + c/2) alpha^2 with a < 0, b, c > 0: INTERIOR OPTIMUM AT THIRD ORDER, alpha* = 2.42/2.26 (tanh-s0, tau .03/.1), 1.43/1.38 (tanh-s1), 1.23/1.04 (gelu-s0): PREDICTION for a doubled-network alpha sweep (not yet run). (v) identification: A2 = sum b c, A3 = sum b a c, B3 = 1/2 sum b c^2 are the Runge-Kutta order-condition sums (flow: 1/2, 1/6, 1/6): learners differ from gradient flow by their order-condition residuals. (vi) the algebra: for learners with all-positive steps Jensen gives B3 >= A2^2/2, equality for one alpha-point jump: large A2 forces a B3 cost, so an interior optimum is forced whenever c > 0. Accuracy changes at these sizes are within 0.05 points.

### 5. Two-gradient learners at f = 1/2 and the mixed-learner loop law (exact)

 TWO-GRADIENT EXPLICIT LEARNERS (tanh E1): single-leg fraction .5012 / .5003 (s1 / s2) for both; loop displacement
  EM +0.00006 / +0.00001, HEUN +0.00026 / +0.00006; own cost -0.00006, -0.00026; SL(2) at the same cost: -0.5011 / +0.503.
  EM1, EM2, EM3 HOLD. Leading loss of trainability is a first-order-integration artefact (factor ~8,000 at 2% moves).
 MIXED-LEARNER LOOP LAW (tanh E1 s1): out SL(1)/back AJ(1): |v|/|vNP| 0.00000 (exact); DJ(1)/AJ(1) +0.00003; AJ(1)/DJ(1)
  +0.00118; SL(2)/AJ(1) +0.25063 (0.25); AJ(1)/SL(8) +0.43740 (0.4375). MX0-MX4 HOLD: displacement = (f_out + f_back - 1) vNP;
  a learner and its time-reverse cancel exactly.

### 6. First-order learners on the real network: PRE-1 closed forms 7/7, PRE-5 look-matrix law 3/3

PRE-CHECK RESULTS (tanh network, real MNIST; all predictions registered first)
 Network facts: lamN/lam1 = 1.07e-3 (condition number about 930); Jacobian 1.3 s, gradient 8 ms; exact-jump distance on the label
   direction at s1 = 0.0378; Adam's lr at kappa = 1 is 1.10e-4 (am11's 1e-5 was kappa 0.091).
 PRE-1 HOLDS 7/7 (top kernel mode, s1, legs closed with the midpoint learner; measured / predicted loop rho, |cos| = 1.0000 each):
   GD(0.5) -0.3334/-0.3333; GD(0.1) -0.0526/-0.0526; midpoint GD(0.5) +0.0769/+0.0769; midpoint GD(1) +0.3334/+0.3333;
   heavy ball(0.02,0.9) -1.9102/-1.9101; look-ahead heavy ball(0.02,0.9,9.5) -0.0771/-0.0771; look-ahead GD(0.1,0.5) -0.0053/-0.0053.

 PRE-5 HOLDS 3/3 (look-matrix law on the label direction, s1): measured / LAW(C) = 0.9995 (HB(0.02,0.9)), 0.9995 (look-ahead),
   0.9997 (GD(1)); cosine 1.0000; 11-12 law terms. Side prediction "look-ahead displacement 3x smaller" FAILS (0.2855 against 0.2880
   of |vNPe|): on a generic direction the displacement is dominated by the ANTISYMMETRIC part of the look matrix, which is the same
   for all three learners; the symmetric (wear) part IS removed by look-ahead (2.56e-3 -> 8.9e-5, 29x).

### 7. Spectral area: new closed form, measured to 0.06%

 SPECTRAL AREA (new closed form, checked the same evening): an out-and-back by a gradient-flow-like learner along
   u = sum a_i q_i encloses area because both legs do the fast modes first: antisymmetric part of the loop look matrix
   = -s^2 a_i a_j (lam_i - lam_j)/(lam_i + lam_j). Predicted norm 8.788e-3; measured 8.793e-3 (look-ahead), 8.934e-3 (GD(1)),
   9.190e-3 (HB, stopped at 3% residual). So even exact gradient flow does not return the weights after an out-and-back on a
   generic direction: displacement 0.29 |vNPe|, orthogonal to vNPe (own cost unchanged at first order). In SPEC12 as rule HG3.

### 8. SPEC10: alignment improves what is learned (all pre-registered rules pass)

## 29. SPEC10 RESULT ON REAL MNIST (2026-11-22 night) - THE STRONGEST GENERALISATION RESULT SO FAR
VALID: V1 0.0 over 64 end points; V2 0.0 over 8; V3 max 1.0e-7 (190 ZERO end points); V4 206 splits, 0 failed; V5 7.56e-8; V6 4.8e-16; V7 0.0.
PRE-REGISTERED RULES (ZERO model g = F(theta) - F(theta0); 32 blocks): HZ1 27/32 PASS (f = 0.43 beats f = 0); HZ2 31/32 PASS (f = -1 loses);
 HZ3 32/32 PASS (paired shift against PLAIN); HZ4 28/32 PASS (SL(4), other mechanism). Addendum A1: S1 loss_star 32/32, S2 drest < 0 32/32,
 S3 both for tau <= 0.5 24/24 -> ALL PASS. So per the reading fixed in advance: "alignment improves what is learned" holds on MNIST for these nets.
 The 5 HZ1 misses are all seed 0 (gelu-s0 n32 at tau .1, .25, 1; gelu-s0-n128 tau .1 by +7e-7; tanh-s0 tau .1 by +2.5e-7): the chance term
 dN = eN*gN (gN > 0 in 20, < 0 in 12 blocks; eN > 0 in 32/32, 6-10x larger for gelu), as predicted in sec. 26.
SIZES (DJ(0.25) vs DJ(1), ZERO, n = 128): relative test loss -0.02..-0.06% (tau .1), -0.2..-0.5% (.25), -1.3..-3.0% (.5), -8.4..-12.3% (1);
 accuracy +0.2 (tau .1) to +2.1 points (tau 1: gelu-s1 72.55 -> 74.61, tanh-s1 68.31 -> 70.31, gelu-s0 76.84 -> 77.91, tanh-s0 73.08 -> 73.75).
 f = -1 (DJ(1.5)) at tau = 1, n128 gelu: loss +36%, accuracy 72.3 / 68.0 vs 76.8 / 72.6. n = 32: -1.7..-2.5% at tau 1 in 3 of 4 starts.
 Accuracy better for DJ(0.25) in 28/32 blocks.
ONE NUMBER ACROSS MECHANISMS: SL(4) (f .375, exact sub-jumps) and DJ(0.25) (f .4286, damped steps) give nearly the same gain in every block
 (ratio 0.875-0.99; f ratio 0.875). n128, tau 0.1, all usable learners: loss minus loss(f=0) in 1e-6 for f = -1, .375, .43, .71, .875, 1:
 tanh-s0 +40, -14, -16, -29, -35; tanh-s1 +26, -9, -10, -16, -17; gelu-s1 +42, -23, -26, -43, -49, -57; gelu-s0 +15, +2, +1, 0, -2, -5.
 => test loss is LINEAR IN f with negative slope across 6-7 learners of different kinds (gain/f: 37, 37, 41, 40; 24, 23, 22, 19; 61, 61, 60, 56, 57),
 and continues to f = 0.9-1 (nearest-point-like learners best where they converge). tanh n128 tau .25: f = .71 gives -291 / -233 vs -170 / -154
 at f = .43, accuracy 72.6 vs 71.8 vs 70.8 (f = 0).
SMALL MOVES (tau = +-0.01, E1): b_ZERO / b_PLAIN = -0.03, +0.11, +0.41, -0.41 (n128), -0.02, -0.02, +0.23, -0.17 (n32): the start function carried
 59-98% of the small-move cost; sign of the remainder mixed (5 negative, 3 positive). The E1 expectation "negative in 4/4 at n128" FAILED (2/4).
OTHER: E3 Spearman < 0 in 11/15 blocks with >= 6 usable learners (met exactly); E6 trK rises with f 8/8; E7 lossZ < lossP for DJ(1) 27/32 (expected 32);
 E8 best tau: implicit learners only exist at small tau, so E8 is not informative.
SPLIT REGULARITIES (ZERO, DJ(0.25) vs DJ(1)): eS < 0 in 31/32 (learned change shrinks with f), gS < 0 for tau <= .5 (so dS > 0, a cost) and > 0 at
 tau = 1; drest < 0 always and 1.3-8x larger than dS for tau <= .5; quad grows ~tau^4 and equals |drest| in size at tau = 1.
READING : in the ZERO model the output IS the learned change. +f moves the weights along -V0 grad[1/2 dY^T K^-1 dY] with dY = tau*Y, i.e. it
 adapts the kernel so that the training labels become cheap to express; the test predictions then move toward the test labels. f = 0 (exact
 Gauss-Newton) is the lazy/kernel predictor, f = 1 the nearest point. So this is "rich beats lazy" measured at exactly equal training outputs,
 with the dial f given in closed form for each learner. PRIOR (sec. 24 literature searches): Arbel 2302.02904 (larger GN step -> worse features, no theory),
 Buffelli (exact GN stays lazy), Amari 2006.10732, Chizat lazy training (also uses f - f0). Not found there: the one-number family, linearity in f,
 agreement across mechanisms, the init-function cost split.

### 9. Midpoint-kernel identity on MNIST (X4)

## 35. X4 RESULT: THE MIDPOINT-KERNEL IDENTITY HOLDS ON MNIST; X3b FIRST ROWS (2026-11-23 early)
X4 (subtraction model, n = 32, real MNIST; |G| = |g_NP - g_TH| on the 10,000 test images):
  start tau |KR_mid - (g_TH+g_NP)/2|/|G| |KR_0 - same|/|G| |KR_end - g_NP|/|G| |SL(32) - KR_mid|/|G|
  tanh-s1 0.03 0.0109 0.524 0.194 0.0166
  tanh-s1 0.10 0.0354 0.525 0.192 0.0325
  gelu-s1 0.03 0.0232 0.539 0.174 0.0211
  gelu-s1 0.10 0.0789 0.552 0.202 0.0573
 All four predictions HOLD: error <= 0.15 at tau .03 (0.011, 0.023); grows 3.25x and 3.40x for tau x3.33 (first order in tau); the start kernel
 is off by ~0.52-0.55 at both sizes; the end kernel does not give the nearest point (off by ~0.19 = |R|/2|G1|, so |R| ~ 0.39 |G1| and R is nearly
 orthogonal to G1). SL(32) (f = 0.4844) sits 0.017 from KR_mid at tau .03 (0.0156 expected from f alone).
 => STATEMENT (second order in the move, verified on MNIST): the prediction of gradient flow on unseen inputs equals kernel regression with the
 tangent kernel taken at the MIDPOINT of the path; any other learner equals that plus (f - 1/2) G(x). Exact Gauss-Newton is NOT start-kernel
 regression at this order (it carries R/2), and the nearest point is NOT end-kernel regression.

### 10. Doubled network, odd/even structure (X3b)

## 36. X3b COMPLETE: DOUBLED NETWORK, 4 STARTS x 4 MOVE SIZES, REAL MNIST, n = 32 (2026-11-23 early)
 (all learners CONVERGED). Change against DJ(1) (f = 0):
 D1 third order: |D| exponents between tau .03 and .1 = 2.99, 2.99 (tanh), 2.91, 2.95 (gelu) -> HOLDS 4/4 (subtraction model: 1.5-2.0).
 Rescaled loss: SL(4) and SL(8) better in 16/16 blocks; DJ(1.5) (f = -1) worse in 16/16.
 Non-rescaling part drest: toward the labels for f > 0 in 16/16, away for f = -1 in 16/16.
 Shrink: eS < 0 for f > 0 in 16/16 (the learned change is smaller on unseen inputs for larger f), > 0 for f = -1 in 16/16.
 Raw loss: favourable in 10/16 only (all 8 blocks with tau >= 0.3; 2 of 8 with tau <= 0.1) -> D2 AS REGISTERED FAILS; it holds for the rescaled
 loss and for drest at every size.
 Accuracy at the full move: SL(4)/SL(8) +0.34/+0.44, +0.15/+0.19, +0.14/+0.18, +0.12/+0.10 points (8/8 up); DJ(1.5) -0.46, -0.43, -0.45, -0.29 (4/4 down).
 Seed dependence: tanh s0 vs s1 rescaled change of SL(8): -3.74e-7/-3.63e-7, -4.10e-6/-3.94e-6, -3.40e-5/-3.14e-5, -2.51e-4/-2.09e-4 (within 3-17%);
 gelu s0 vs s1 same sign, factor ~2.5 apart at small moves, equal at the full move.
SUMMARY OF THE STRUCTURE NOW SUPPORTED BY DATA (smooth nets, MNIST, squared error):
 learner-dependence of predictions at equal training outputs = f x [ ODD part: second order in the move, flips with the readout sign, includes the
 growth of the start function, random sign across starts, removed EXACTLY by doubling ] + f x [ EVEN part: third order, same sign and similar size
 across starts: (a) the learned change shrinks on unseen inputs, (b) its pattern moves toward the labels ] + squared terms (interior optimum).
 Consequences seen: plain model -> more alignment looks harmful at small moves (odd part + start function); subtraction model -> coin flip at small
 moves, favourable at larger ones (27/32); doubled model -> favourable in 16/16 by rescaled loss and by pattern, at every size.
X1 n = 128 first rows (tanh-s0-n128, tau 0.1, subtraction model): AJ(0) loss .0439683, rescaled .0331684, accuracy 70.34%; AJ(1) end point
 (status AJ-FAILED: treat with care) loss .0439227, rescaled .0325623 (-1.8%), accuracy 71.11% (0.77 points). Rest running.

X3b doubled network with rescaled loss and split, tanh-s0 complete, tanh-s1 at tau .03:
 rescaled-loss change against DJ(1): SL(4) -3.2e-7, -3.6e-6, -3.0e-5, -2.3e-4 at tau .03, .1, .3, 1 (NEGATIVE at every size); DJ(1.5) (f = -1)
 +6.4e-7, +7.0e-6, +5.8e-5, +3.9e-4 (positive at every size). Non-rescaling part drest: -1.8e-8, -6.6e-7, -1.65e-5, -3.7e-4 for SL(4) (toward the
 labels at EVERY size), opposite sign for f = -1. Shrink part dS: +2.7e-8, +8.4e-7, +1.1e-5, -4.8e-4 (eS < 0 always: the learned change is smaller
 for larger f; a cost while the fit is partial, a gain at the full move). Raw loss: +8.9e-9, +1.8e-7, -5.5e-6, -8.5e-4. Accuracy at tau 1:
 +0.34 / +0.44 points for SL(4) / SL(8), -0.46 for DJ(1.5).
 SEED INDEPENDENCE: tanh-s1 at tau .03 gives rescaled-loss change -3.18e-7 (s0: -3.24e-7) and drest -1.70e-8 (s0: -1.80e-8): the same sign AND size,
 where the subtraction model's small-move slope varied in sign and by 10x between seeds. This is what the symmetry argument predicts: doubling
 removes the random-sign second-order part and leaves a systematic third-order effect.
 So D2 must be read per probe: raw loss at tau <= 0.1 goes the wrong way (shrink of a still-too-small learned function), rescaled loss and the
 non-rescaling part go the favourable way at every size.

### 11. The evaluation-point principle and the alpha rule

## 18. THE EVALUATION-POINT PRINCIPLE (2026-11-22) — deeper form of the alignment-fraction law, and the ReLU answer
Core identity (summation by parts; leading order; straight/collinear output path, positions pos_k in units of the total move):
  v_L = V_0 * sum_k J(theta at step k)^T K^-1 dF_k = V_0 * INT dJ(tau)^T K^-1 dY * Phi_L(tau),
  Phi_L(tau) = sum of the learner's output increments made with the Jacobian evaluated BEYOND path point tau.
  rho_L = INT Phi_L(tau) d omega(tau), omega = normalised distribution of where along the path the Jacobian changes (as seen by NP).
  Phi_TH = 0 (pre-point), Phi_NP = 1 (post-point), Phi_flow = 1 - tau, Phi_SL(m) = 1 - ceil(m tau)/m, Phi_DJ(c)(tau) = sum_{k: pos_k > tau} dF_k.
  Smooth activation: omega uniform => rho = INT Phi = 1/2, 1/2(1-1/m), (1-c)/(2-c). => the fractions are "average evaluation point" numbers.
This is the Ito / Stratonovich / post-point structure: Gauss-Newton jump = pre-point (0), gradient flow = midpoint (1/2), nearest solution = post-point (1);
discrete GD = Stratonovich minus the Ito correction (quadratic variation Q): C_sym = 1/2 dYdY^T - 1/2 Q IS the Ito formula; area = Levy area.
NEW PREDICTION MADE FROM THIS AND THEN TESTED : the alpha-point jump theta = theta_0 + J(theta_0 + alpha(theta-theta_0))^T a,
  F(theta) = t, has rho = alpha for ANY alpha. Measured for alpha in {-1, 0, .25, .5, .75, 1, 1.5, 2}: solvable model -1.0001, 0, .2500, .5000, .7500, 1.0000,
  1.4999, 1.9998; toy tanh net and tanh/gelu MLPs (196-32-32-10) the same to 3-4 decimals, cos +-1.000. => one knob dials feature alignment, incl. beyond the nearest solution.
COROLLARY TESTED (tanh MLP, 2000 held-out outputs): predictions on unseen inputs of all learners differ along ONE function g(x), coefficient = f:
  SL(2) .2500, SL(32) .4844, DJ(.5) .3334, DJ(1.5) -.9998, DJ(1.9) -8.996, PROX 1.0000, cos +-1.00000; spread = 0.5-0.8% of the held-out change at s = 1% S0 (grows ~ s).
RELU (196-32-32-10, n=8):
  * At s <= 1e-4 S0 (eigA) / 1e-5 S0 (rand): EXACT law: .2500 .4375 .4844 .3333 0 -1.0000 -9.0000 (inside one linear region the net is a polynomial).
  * From the first activation switch on (s >= 1e-3 S0): SL(2) .507, SL(8) .888, SL(32) .983 = (1-1/m)(1+w)/2 with ONE parameter w = 1.03 for all three,
    i.e. omega is a spike at the start of the path (one early switch carries the whole Jacobian change) instead of uniform. DJ(1) .87, DJ(1.5) +.79, DJ(1.9) +.63 (not 0, -1, -9).
  * So ReLU does not have "a different formula": it obeys the same evaluation-point law with a lumpy omega (sum of spikes at switches). Dense-switch limit -> smooth fractions.
  * GPU ReLU data (n=32, W=256, s=0.02 S0) consistent: gentle learners obey the fractions among themselves within a few % (SL(8)/SL(32) 1.02, GD(.25) 1.007),
    overshooting learners give 0.3-0.6 of the predicted magnitude, f=0 learners are far from 0 (DJ(1): 0.57 of SL(32)). NP solver fails because the post-point
    condition theta - theta_0 in row J(theta) has no solution when J jumps (needs a Clarke-type / active-set formulation).
  * Open: overshoot response in ReLU beyond the single-switch picture (finite switch strength, O(dJ^2)); proper reference solver for ReLU.
Other patterns in the smooth GPU data: generic targets give rho = 0.50-0.53 > 1/2 consistently: strong-first ordering (C_ij = r_i r_j lam_j/(lam_i+lam_j)) takes the dominant
  cross term T_(weak increment, strong position) at full strength (= NP's), the negligible one at zero => spectral bias maximises alignment among orderings.
  Path matrix of flow solves a Lyapunov/Sylvester equation (controllability-Gramian structure); multiplier matrix M_L(i,j) per learner on r r^T.
Literature search : evaluation point: only the two first-order endpoints known (explicit +h/4, implicit -h/4 |grad|^2: Halmos & Hanin 2511.14827), midpoint only for bilinear
  games (2607.09950); NOT FOUND: alpha as free parameter, function-space version, "flow = Stratonovich" for deterministic GD, linear interpolation kernel<->min-distance solution,
  alpha-connection application. ReLU: Dyer-Gur-Ari say ReLU kernel evolution "more subtle ... left to future work"; central flows exclude ReLU and degrade toward it;
  Lewkowycz: ReLU tolerates lr ~12/lam0 vs 4/lam0 and momentum makes lam_t non-constant in the lazy regime, both UNEXPLAINED; no switch-per-movement or kernel-change-per-switch measurements.

### 12. SPEC9 Part A: every rule 100%; sign of the test-loss slope

## 22. SPEC9 GPU run, first table (after Part A; pasted 2026-11-22 ~09:30 rig time) and what it shows
Seen: tail of from mid HG1 tanh/init to the verdict summary (top of file cut off by the paste). G4 and Part R
sections were empty there (not run yet at that time). Part A took ~2 h 20 min on 4 GPUs, 0 failed cases.
VERDICTS (real MNIST, MODE full): H0 16/16; HA1-HA7 all 100% in all four groups (gelu/tanh x init/trained: 84,72,72,20,56,48,24 each);
HG1 272/272 x 4; HG2 12/12 x 4. Largest raw deviation seen: 5% (AJ(2), eigD, s1), ratio s1/s2 errors ~4 (quadratic), extrapolated
residual <= 1.1e-3 (tol 0.02).
G3 (open question) ANSWER IN SIGN: b > 0 in 20/20 (true 4/4, label 8/8, rand 8/8) and in all 16 visible eigen-direction rows (tanh).
  => at equal training outputs, larger f = HIGHER test loss (squared error): f=-1 learners best, exact jump (0) < flow (1/2) < nearest point (1) < AJ(2).
  E+ and b agree (two independent readouts). Size: E+/(s/S) = 0.03-0.17 (true), 0.17-3.0 (label); 10x seed spread; "rand" E+ not
  interpretable (denominator l(theta0)-l(TH) changes sign / passes near zero, e.g. tanh-trained-s0 0.46 at s1 vs 0.06 at s2).
  as written: E+ > 0 with a helpful move means the nearest point GIVES BACK part of the gain.

### 13. Explicit learner with any fraction, healing a worn network, state/heat/area

RESULTS (all on real MNIST, n = 32, this machine):
 HEALING, implicit : HL1 FAILS at once - AJ(1) does not converge on the worn gelu network. Not a test of the healing idea.
 EXPLICIT LEARNER WITH ANY FRACTION (loop): EP(alpha) = exact jump as predictor, then the full move with the Jacobian at
  th0 + alpha (th_p - th0). tanh E1 s1: single-leg fraction 1.0017 / 2.0011 and loop displacement +1.00194 / +2.99994 for alpha = 1 / 2
  (predicted 1, 2 and +1, +3), cos 1.00000, 22 s each (AJ(1): 85-170 s). PE1 HOLDS. With EM (alpha = 1/2) the whole dial is explicit.
 HEALING, explicit (heal): worn gelu network (after 20 wearing excursions), EP(1) excursions along NEW directions, 6 pairs,
  16-24 s per pair, every leg converged and closed: lamN/lamN0 .039 -> .145 .194 .269 .343 .396 .459; mean q0/q .175 -> .290 .337 .388
  .433 .467 .496; A 3.03 -> 2.06; leverage sum of squares 11.43 -> 10.69 (start 7.85); distance from the start unchanged (1.094 -> 1.092).
  PE2 HOLDS (0.070 per pair in lamN against a threshold of +0.03 and a one-clock value of +0.055). Wear is reversible by unrelated
  explicit excursions without touching the outputs.
 STATE / HEAT / AREA (tanh, triangle F0 -> F0 + s u -> F0 + s w -> F0 and its reverse): Vsum = sum of the three legs'
  nearest-point shifts. Even part (heat): EM +0.00003 (|even|/|Vsum| 0.00012), SL(2) -0.25164 (pred -0.25, cos -0.9999), AJ(1) +0.50041
  (pred +0.5, cos +0.9999). Odd part (area): |odd|/|Vsum| 0.1396, 0.1398, 0.1392; pairwise cosines 0.9995-0.9999, norm ratios 0.996-1.001.
  TR1-TR4 HOLD: heat = (f - 1/2) x route term, area is the same for every learner, the two-gradient learner keeps area only.

### 14. Part R final: ReLU obeys the same law in three regimes

## 28. PART R FINAL (ReLU + smooth controls), (2026-11-22; run complete: 5228 cases, 0 failed)
Final verdicts: H0, HA1-HA7, HG1, HG2 as before (all 100%). HR1 gelu 229/240 PASS, tanh 240/240 PASS; HR2 gelu 235/240, tanh 240/240 PASS;
 HR3 gelu and tanh 18/18 and 144/144 PASS. RELU: HR1 FAIL 708/866 = 81.8% (rule 90%); HR2 FAIL 619/866 = 71.5%; HR4 PASS 240/240; HR5 FAIL
 (smooth form misses in 487/626 = 77.8% as expected, but the general law holds only in 77.8%, rule 90%).
Smooth controls: the reference-path form (one measured profile omega per case predicts every learner at finite size) holds up to 0.1 S0;
 all 11 HR1 misses are gelu at 0.1 (nrF 0.87-1.24, cos >= 0.983). This is the "second number per learner" answered: at finite size the
 learners are organised by the whole profile omega(tau), not by a second scalar.
RELU, three regimes (the main reading):
 (1) no switch on the path: smooth fractions exact, 240/240 (|rho - f| <= 0.03).
 (2) few switches (1-10): smooth fractions wrong (one switch can carry the whole Jacobian change: median max|omega - tau| = 0.50 at 1e-4 and 1e-3),
     but the general law holds case by case: HR1-condition share 0.99 (1-2 switches), 0.96 (3-10).
 (3) many switches: the general law FAILS case by case (share 0.84 for 11-100, 0.53 for > 100; median cos 0.960), while the smooth fractions
     come back on average: median |rho - f| over the primary learners = 0.079, 0.144, 0.107, 0.082, 0.086 at 1e-4 .. 0.1 (peak at ~7 switches,
     then down), the stated expectation (rise then fall) HOLDS. At 0.1: relu 0.086 = gelu 0.087; DJ(0.5) median rho 0.4197 (relu) vs 0.4202 (gelu);
     max|omega - tau| 0.176 (relu) vs 0.183 (gelu). Tanh at 0.1 is far closer to the fractions (0.0075), so the common value is the large-move
     behaviour of relu-like activations, not kinks.
 HR1 per size: 172/191, 162/166, 125/138, 95/128, 84/122, 70/121 (the 19 misses at 1e-5 are a resolution floor: nrF up to 3200, cos 0.003).
 HR1 misses by learner: DJ(1) 62, SL(2) 51, DJ(0.5) 32, SL(8) 10, SL(32) 3 -> coarse learners fail, fine-step learners follow the reference path.
 By direction: eigA 9 (8 of them the 1e-5 floor), label 73, rand 76 (generic directions cross 450-520 switches at 0.1, eigA ~100).
 HR2 misses: DJ(1) 98, DJ(0.5) 68, SL(2) 41, AJ(1) 15, SL(8) 12, ...; the off-line response omega_perp does NOT average out
 (median max 0.49-0.60 at every size >= 1e-3; gelu 0.006-0.10). The sec. 21 prediction "HR2 misses concentrate in eigA" was WRONG in share
 (eigA 81, label 82, rand 84 of 247): the leak is general, not special to the stiff direction; "DJ learners" and "HR1 near threshold at mid
 sizes" were right.
 omega is NOT monotone with many switches (e.g. +0.632 -> +0.611 -> +0.610): single switches push both ways; omega = drift + signed noise.
 Implicit learners on ReLU (conv/24 at 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1): AJ(1) 23, 16, 6, 1, 0, 1; AJ(0.5) 24, 17, 7, 3, 1, 0; SLa(8,1) 24, 13, 5,
 4, 1, 0 (median switches 0, 1, 7.5, 56, 171, 456). So end- or mid-point learners have no solution once a few switches lie on the path
 (roughly 20-30% failure per switch) = indirect support for "the nearest point sits on activation boundaries" (sec. 24 prediction).
ANSWER to "does ReLU have a different formula": no. Same evaluation-point law; exact without switches, predictable case by case with a few,
 statistical with many. What fails for ReLU is the case-by-case prediction from a reference path when hundreds of switches are crossed, because
 which switches a learner meets depends on its own sideways position.

### 15. Section 40 results: solvable model, conservation violated in proportion to f - 1/2, exactly three functions

RESULTS OF SECTION 40 (2026-11-23):
 T1 solvable model beta = u^2 - v^2:
  D0 HOLDS: s^3 coefficient -8.01, 0.00, 24.02, 64.19, 120.8 (predicted 32 f(1+f) = -8, 0, 24, 64, 120); u v formula exact to 6 digits.
  D1 HOLDS: sparse truth, error falls f = 0 -> 1.5 in 20/20 seeds at n = 30, 60, 120; worst at f = -0.5 in 20/20; f = -1 equals f = 0.
     Stronger compounding (alpha = 0.3, 30% noise): error at f = 0 / 1.5 = .933/.912 (n 30), .859/.737 (n 60), .749/.401 (n 120): the gain from
     larger f GROWS with n (2%, 14%, 47%).
  D2 FAILS as registered (lazy regime: 10-12 of 20, a coin flip); at alpha = 0.3 the mean error for dense truth rises with f (n 120:
     .6911 -> .7004) with an optimum near f = 0-0.5.
  D3 FAILS: no interior optimum for sparse or mixed truth up to f = 1.5 (only "benefit grows with n" survives).
  D4 HOLDS: differences against DJ(1) are third order (exponent 3.00 at alpha = 1; 2.85-2.99 at alpha = .3; MNIST doubled: 2.91-2.99);
     SL(8) better and DJ(1.5) worse than DJ(1) in 4/4; mean(u v) - alpha^2 = -4.87, -1.62, -.81, -.41, -.20 (x1e-6) for f = -1, 0, .25, .375,
     .4375 = proportional to (f - 1/2) to 3 digits: THE CONSERVED QUANTITY OF GRADIENT FLOW IS VIOLATED IN PROPORTION TO f - 1/2.
  Two-phase (t1b): a large-step first half followed by a smooth second half ends WORSE than all-smooth in 12/12 (DJ(1)), 12/12 (DJ(1.5)),
     12/12 (DJ(1.9)): the idea "large steps then decay helps through a smaller effective scale" is REFUTED for deterministic steps
     (agrees with Even et al. 2302.08982: gain is concentrated on the support and hinders sparse recovery).
 T2 real MNIST (x6): M3 HOLDS 8/8 (Spearman .12-.58: units with larger readout at the start gain more with f). M1 FAILS as registered (4/8):
  gelu: participation ratio of unit contributions 137 -> 59, 91 -> 55, 116 -> 36, 66 -> 35 of 256 units for f = -1 -> .44 (4/4, the learned
  function is carried by half as many units); tanh: ~1% fall for f >= 0 in 3/4, and the overshooting learner f = -1 is MORE concentrated
  (4/4). M1b FAILS (2/8). M2: layer balance is linear in f for tanh (R2 .96-.99) but equals its start value at f = .17, .06, .11, -.11, not
  at 1/2; gelu non-monotone => the simple per-unit balance is not the conserved quantity in these nets (activations not homogeneous).
  READING: compounding is strong for the unbounded activation and damped by saturation in tanh; this one fact orders every tanh/gelu
  asymmetry logged so far (trK +0.8% vs +21% over alpha 0 -> 2; class scatter; probe curvature; extrapolation below).
 E2 extrapolated pair FAILS as registered: n = 128, best lam > 1 in 4/8, accuracy(lam 2) >= accuracy(lam 1) in 2/8. Split by activation:
  tanh tau .5: lam = 2 best in 4/4 (n128: loss -0.9% / -1.2%, accuracy +0.34 / +0.82 points; n32-s1: lam 4 best, -3.6% at tau 1); gelu: lam = 1
  best or within noise in 8/8. (First table was computed without subtracting the start function - caught by the format check,
  discarded.)
 E1 / X7 THIRD-ORDER LAW ON REAL MNIST (doubled tanh-s0, five learners against DJ(1), 100,000 test outputs):
  singular values of the five difference vectors / largest: 1, 0.109, 0.016, 0.0003, 0.0000 at tau = .03 and 1, 0.107, 0.016, 0.0001, 0.0000
  at tau = .1 (identical => genuine third-order structure): EXACTLY THREE FUNCTIONS. With weights computed from each learner step
  sequence and no fitted coefficient - e2 (z^2 coefficient of the stability function), e3 (z^3 coefficient = chain tree), b (bushy tree) -
  the residual is 0.0000 of sum |D|^2 (two functions e2, e3: 0.0028; e2, b: 0.0003; one function e2: 0.012; e2 + e3: 0.019).
  E1a HOLDS (4.4x), E1b HOLDS (0.28%), E1c: sigma_3/sigma_1 = 0.016 <= 0.05 HOLDS but it does NOT fall with tau: the third function is real.
  => second order: ONE number (f = e2). Third order: THREE numbers (e2, e3, b), all computable from the step sizes. Weakness: 5 learners,
  1 start, 3 functions = 2 spare degrees of freedom; add SL(3), DJ(1.9), SLa, TJ and a gelu start to firm it up.

## Tier 2: strong but mined after the fact, fewer cases, or pre-checks

### 16. One clock for kernel depletion; common direction of all excursions

MINED FROM THE RAW RECORDS (not pre-registered):
 * ONE CLOCK: per unit (1-2f) the whole kernel change of one excursion is the same for all 8 learners to 3 digits (tanh E1: dtrK -4.57,
   dlam1 +0.089, dlamN -0.00199). With t = (1-2f) x excursions, pooled over 5 learners x 6 checkpoints (30 points) a straight line through
   (0, 1) fits lamN (R2 .9988 tanh, .9938 gelu), mean inverse cost (.9994, .9968), trK (.9997, .9997); zero crossings t = 376 / 36.5 (lamN),
   758 / 60.5 (inverse cost), 2974 / 118 (trK). The cost itself is NOT linear on gelu (R2 .888): it is the inverse that is.
   Every per-point estimate of the lamN crossing agrees across learners (gelu 31-40, tanh 351-442 with the same direction-to-direction pattern).
 * COMMON DIRECTION: all 91 pairwise cosines between the 14 nearest-point shifts are positive (tanh .39-.62, mean .50; gelu .49-.68, mean
   .59); the top eigenvector of the cosine matrix carries 54% / 62% and is uniform over directions: excursions in any direction push the
   weights largely the same way. This is why loss of trainability is general.
 * Signs of the kernel change for explicit excursions: tanh trK down, lam1 up, lamN down (spectrum spreads); gelu all three down
   (the kernel shrinks: -30% trK and lamN / 10 after 12 excursions of DJ(1.5)).

### 17. Collapse run and the worn network

TESTS RUN AFTERWARDS ON THIS MACHINE (each prediction registered above before its run):
 COLLAPSE (gelu, DJ(1.5), 10 pairs = 20 excursions of 2% size; reproduces the rig's first six values exactly):
  lamN/lamN0 = .816 .621 .446 .332 .203 .097 | .039 .061 .044 .039; mean q0/q = .914 .799 .694 .600 .490 .384 | .291 .265 .214 .175;
  four smallest eigenvalues at the end .039 .049 .069 .078 of the start's smallest; trK .564, lam1 .679; every leg CONVERGED.
  K1 HOLDS (.039 <= .05 at pair 7). K2 FAILS: the learner never stops; the very smallest eigenvalue floors near 4% while the rest of the
  bottom keeps sinking; the inverse cost turns from a straight line into roughly x0.82 per pair. => linear depletion down to ~10-40% of
  the start, then a tail; NO finite-time singularity at this size. Cost of fresh directions x5.7 after 20 excursions.
  THE WORN NETWORK : no dead units (0), no duplicated units (0 with |corr| > .98), feature rank 26.3 -> 25.0 and
  26.1 -> 24.4, weight norm DOWN 0.8% (13.24 -> 13.14); weights moved 9% / 5% / 31% (layers 1, 2, readout). Bottom eight kernel
  eigenvalues .122-.187 -> .0047-.024 (8-26x lower); the weakest end eigenvectors lie 97% inside the 16 weakest start eigenvectors.
  Parameter leverage (diagonal of J^T K^-1 J, sum = 320): sum of squares 7.85 -> 11.43 (46%), participation 13,045 -> 8,963 parameters,
  max .52 -> .66. => trainability lost by a factor ~6 with NONE of the field's usual symptoms (dead units, weight growth, clones);
  what changes is that the weak kernel directions sink and leverage concentrates on fewer parameters (the "polarisation" of the one
  budget law in the literature, Elmeligy et al. 2611.22070, there for linear physical networks).

### 18. Two-gradient learner at large moves (tanh and gelu)

 LARGE MOVES WITH THE TWO-GRADIENT LEARNER (tanh, Part C protocol): Q per task
  EMS(1): -3.181 -3.215 -3.239 -3.201 -3.179 -3.313 (cost of SL(2), whose Q6 is -2.535); EMS(2): ... -3.098 (SL(4): -2.668);
  EMS(4): ... -3.022 (SL(8): -2.814). A6 1.393 / 1.459 / 1.524 (start 1.748), lamN6 2.92 / 2.32 / 2.09 (SL(8) 1.64, implicit 2.44),
  distance 1.96 / 1.91 / 1.99. EC1 HOLDS 3/3, EC2 HOLDS (-3.022). EMS(1) keeps MORE trainability than the implicit learner at a fraction
  of its cost and never needs an implicit solve. EC4 HOLDS: final test loss .0467 / .0460 / .0451, accuracy .498 / .494 / .495 = the
  fresh network's (.0462 / .498), worse than every explicit history (.039-.043): a time-symmetric learner comes out as if the six random
  tasks had not happened; explicit learners carry them, as lost trainability AND as a gain on the next task (tanh).

 LARGE MOVES, gelu : Q6 EMS(1) -3.544, EMS(2) -2.964, EMS(4) -2.735 against SL(2) -2.163, SL(4) -2.239, SL(8) -2.428 and
  implicit -3.008: EC3 HOLDS 3/3. Caveat: EMS(1) makes the kernel 35x larger (lam1 x63, A 2.31 > start 1.77). Final test loss .0490,
  .0425, .0418 (fresh .0376): the trainability-preserving learners are the worst on the next task, as on tanh.

### 19. Reversible learning

LATE ADDITIONS TO SECTION 44 (2026-11-23, about 14:45):
 REVERSIBLE LEARNING (tanh; three tasks of size 0.03 S0 learned with the exact jump; only the outputs before each task stored):
  undoing in reverse order with the implicit jump AJ(1) returned the weights to 5.3e-13 of the distance moved (4.2e-13, 4.9e-13,
  5.3e-13 after each undo; output error 1e-14). Undoing with the exact jump itself left 9%, 19%, 24%. RV1, RV2 HOLD.
  320 stored numbers per step restore 118,794 weights. (Known relative, from memory, unverified: exactly reversible SGD with momentum,
  Maclaurin et al. 2015; implicit Euler inverts explicit Euler. Ours is for target changes with exact re-fits.)

### 20. SPEC12 cross-check at real size: reset-Adam odd part is real, keeping either moment breaks the mirror, GD leaves trainability untouched

### 21. Adam on the real network: kept moments double the wear, sign of a_u changes twice, hover wear scales as lr^2, own metric

 PRE-2, second half HOLDS: moments KEPT across the two legs (kappa .1): loop odd part exponent 0.99, odd / even = 2.96 at s1 (reset:
   0.27), size of the odd part = 10% of the jump distance; the even part still scales as s^2 (2.02) with a_u +0.486 / +0.482 and
   gauge ratio +0.547 / +0.545 (reset: +0.257, +0.294): keeping the moments roughly doubles the wear. The odd (first-order)
   displacement itself changes the probe costs only 2-5% as much as the even part: it points in fibre directions the costs do not
   feel at first order; its effect on trainability is second order (curvature of the cost along the fibre).
 ADAM AT kappa = 1 (lr 1.1e-4, 149 steps): a_u -1.76 / -1.83, gauge ratio -1.72 / -1.79, exponent 1.94: the loop makes the network
   EASIER to train, by 1.8 x what an exact jump costs (consistent with am11's -1.36 at kappa .91). So a_u(kappa) = -0.05, +0.26,
   +0.41, -1.8 at kappa .03, .1, .3, 1: the sign changes twice. Not predicted in advance; fits "two lags of opposite sign" only loosely.
 HOVER WEAR SCALES AS lr^2: beta1 = 0 at kappa .3: a_u +67 / +57, gauge ratio +111 / +102 (kappa .1: 5.7 / 5.3 and 10): factor 10-12
   for 3x the rate (9 predicted for steps x lr^2).
 Look-ahead Adam at kappa .03: settles slowly (1400-6000 steps), a_u -0.06 / +0.14, exponents off: not a clean regime.
 OWN METRIC (kappa .1): D spans 4e4..1.7e7 (no coordinate at the eps cap); cond(K_D) = 1008 against cond(K) = 931 and nearly the same
   normalised spectrum: on this network Adam's scale factors do NOT flatten the kernel spectrum ("flattened spectrum" half of the
   mechanism is wrong here). Covariance control (exact-jump loop in the D-geometry): rho -1.0029, cos -1.0000. Real Adam loop read in
   its own geometry: rho -0.30, cos -0.47 (ordinary geometry: -0.26, -0.26): only loosely along its own nearest-point shift.

### 22. Adam pre-checks PRE-2 to PRE-4 (mixed) and hover wear

 PRE-2 (Adam, moments reset per leg, label direction, lr scaled with the move): loop even part exponent 2.03, 1.99, 1.98 at kappa .03,
   .1, .3 HOLDS; "odd part <= 10% of even" FAILS (20-27%, exponent 2.1-2.5; probably the stopping/closing discreteness: SPEC12 V7
   tests it with a tighter tolerance). Single-leg vertical displacement is FIRST order: 0.82-0.95 of the jump distance, exponent 1.01.
 PRE-3 a_u(ADAM) > 0: FAILS at kappa .03 (-0.054 / -0.052 at s1 / s2), holds at .1 (0.257 / +0.254) and .3 (0.408 / +0.406).
   Probe-gauge ratio against the exact jump: -0.044, +0.294, +0.481. The loop displacement is 0.5-2.8 x |vNPe| but nearly orthogonal
   to it (cos +0.07, -0.26, -0.14). a_u = -rho to 3 digits in every row (universal identity).
   "RMS <= 0.25 ADAM" and "look-ahead Adam <= 0.25 ADAM": NOT TESTABLE AS WRITTEN: neither learner settles.
   beta1 = 0 at kappa .1: 6000 steps, residual 0.2-0.4 of the move, a_u +5.66 / +5.28, gauge ratio +9.95 / +9.89.
   look-ahead 9.5 at kappa .1: 6000 steps, residual 0.01-0.03, a_u +5.59 / +5.40, ratio +9.89 / +9.91.
   beta1 = 0 with look-ahead 0.5: residual 0.1-0.3, a_u +12.3 / +11.8, ratio +21.5 / +21.0.
   => HOVER WEAR (new claim K8 of SPEC12); the look-ahead cure works for plain momentum, not for fixed-rate Adam.
 PRE-4 HOLDS: closing step 0.2-0.3% of the jump distance for the converged Adam legs.

### 23. History run: earlier tasks change later test loss

 HISTORY (tanh; four earlier tasks then the true labels with SL(8); final test loss, fresh network 0.04622):
  structured history: SL(2) .04370 (-5.5%), EMS(1) .04594 (-0.6%). Shuffled history (3 permutations): SL(2) .04426 .04419 .04567 (mean
  -3.3%), EMS(1) .04836 .04859 .04799 (mean +4.5%). => explicit wear lowers later test loss in 4/4 histories (HS1 first half HOLDS);
  structure beats shuffle for both learners, 6/6 (HS2 HOLDS); the explicit-minus-symmetric gap is LARGER on shuffled history
  (-7.5% against -4.9%), so "about equal" FAILS; HS4 FAILS (shuffled is not better than structured for the explicit learner);
  HS5 FAILS for shuffled history (the symmetric learner is 4-5% worse than fresh, not within 3%): at large moves the route does matter
  for a time-symmetric learner; the "function of current outputs only" statement is a small-move statement.

### 24. X5 result: better own function, worse features

X5 RESULT (2026-11-23, 09:30; run was cut once, gelu blocks rerun; all learners CONVERGED):
 F1 flip symmetry HOLDS: tanh-s0 tau .03 cos hidden +0.9953, readout -0.9973, test-output -0.9982; gelu-s0 +0.9701, -0.9696, -0.9894; at tau .1
    +0.949/-0.971/-0.980 and +0.734/-0.726/-0.902 (1 - |cos| shrinks ~9x for tau/3.3, i.e. second order); size ratios 0.97-1.01.
    => odd/even structure confirmed directly: hidden part of the f-shift is readout-sign-blind, readout part and output effect flip.
 N1 FAILS (5/8): class scatter ratio on the training images falls with f in tanh 4/4 but RISES in gelu 3/4.
 N2: tanh test ratio falls 3/4 (predicted 4/4: fails by one), gelu rises 4/4. Both between- and within-class scatter grow with f in 8/8;
    which grows faster depends on the activation. => "v_NP is the neural-collapse direction" is REFUTED as a general statement.
 K1 FAILS (2/8): with 30 labels the probe error also RISES with f (tanh +0.5 to +2.6% per unit f). K2 HOLDS (7/8 at 5,000 labels). K3 FAILS (3/8).
    Probe error rises with f at k = 30/100/300/1000/5000 in 6, 8, 6, 7, 7 of 8 blocks. => no label count at which larger-f features are better
    for a fresh linear readout; "features specialised for few labels" is REFUTED.
 W1 FAILS (0/8) and its premise was wrong: the features-only and readout-only changes are each much larger than the total and
    opposite, because both learners have the same training outputs, so the two parts must cancel on the training set and nearly cancel on test
    inputs. Interaction negative in 16/16 rows (both f > 0 and f = -1) = this forced cancellation, not a finding. W2a 6/8, W2b 6/8: not
    interpretable for the same reason (gelu tau 1: features-only +7e-3 against a total of 2e-4).
 SAME RUNS: own test loss falls with f in 8/8 blocks (R2 .81-1.00), own accuracy rises in 8/8, while the 5,000-label probe error rises in 7/8.
 => CLEAN DISSOCIATION at exactly equal training outputs: more alignment = better own function, worse features for any fresh linear readout.
    The gain is not in the representation; it is the small remainder, on unseen inputs, of a feature change and a readout change that cancel
    on the training set. What decides the sign of that remainder is OPEN (deciding object: G = G_hidden + G_readout on test inputs; compute
    both parts from v_NP directly, no learners needed, and ask which input directions escape the cancellation).
 SIDE CHECK (x1 data): "distance^2 x trace of kernel" (a training-only complexity product) picks the best alpha exactly in 2/6 starts, adjacent in
    the rest; rank order right for gelu, wrong for tanh. Hunch only. Also: with tanh, trK rises with alpha while lambda_1 FALLS; gelu both rise.

### 25. Early results ledger (what held)

## 4. Results ledger (what held / failed)

Held (pre-registered where noted):
- Part 1: loop ∝ ε² (1.88–1.91), retraced star ∝ ε⁴ (3.86–3.92), collinear ∝ ε³, composition.
- Part 2: polygon ratios N·sin(2π/N) within ~2%; rectangles 2c within 2%; figure-eight → 0.
- Part 3: bracket law cos 0.9986–0.99996; out-and-back law d = −ε²·c_u cos −0.9997…; path
  refinement law; linear control 1e-13. GN stiffens: 128 random out-and-backs → GD needs
  6–7× more steps on fresh targets.
- Part 4: all S1–S6 PASS (GN, GD at c = 0.25/1/1.8, momentum; widths 32–512). GD's shift is
  learning-rate independent to ~1% (zigzag sits in strong directions where T is small).
- Steering: designed label loop moved one chosen unseen input 12–58% of output scale, training
  outputs restored to 1e-13; 256 = n² weight directions movable at second order (120 by loops).
- Label noise: drift matches LAW (cos 1.0000) and −∇tr K (0.996).
- Three-force checks: GD softer than exact jump along target in 3/3; runaway stiffening under
  repeated exact relearning (λ_min ↓ 30–10,000× in 400 excursions; erratic after onset).
- Anticipatory recovery: explained in a LINEAR model — needs a shared feature direction + strong
  fit per visit + fixed cyclic order (20 seeds: shared 0 → none; 0.5–0.8 → end/peak 0.31–0.53).
- Tests 5–9 (REPORT6): T5 all PASS incl. closed form Δλ₁ = ε²(1−c)/(2(2−c))·‖P_v∇λ₁‖²/λ₁²
  (ratio 0.95–1.03; ×19 between c=1.5 and 1.95). T7: order effect at convergence = ε²·bracket,
  weak pairs 53–62× strong. T8: three drift forms agree, simulation with common random numbers
  cos 0.998+, bootstrapping-specific part 0.986–0.990; stop-gradient coefficient fits **2.000**
  (DR3 as transcribed has 1 — read their theorem before claiming a discrepancy). T9: steady
  drift doubles with cycles (1.996–2.013); aimed input moved 0.99–1.01× prediction.
- Post-T5–9 checks: GD sharper than exact jump (λ_max and tr K) in 36/36 random targets at two
  widths; predicted drift per step under rotation peaks at period ≈ slowest kernel timescale
  (period 1024 vs τ 817–1131).

### 26. SPEC11 pre-check and smoke run

THE PRE-CHECK before finalising the spec (real MNIST tanh-s0, direction label, size 0.02 S0, no extrapolation):
 loop rho -1.0018, -0.5011, -0.1252, +0.0000, +1.0000 for DJ(1), SL(2), SL(8), AJ(.5), AJ(1) (predicted -1, -.5, -.125, 0, +1), cos +-1.00000;
 own-cost ratio +1.0062, +.5030, +.1256, -.0000, -1.0014; cross-cost ratio (direction rand) +1.0084, +.5042, +.1259, -.0000, -1.0044;
 identity 1.0029. AJ(.5) returns to the start to rounding (the midpoint rule is exactly reversible). The implicit learner AJ(1) makes
 directions EASIER by the amount the single jump makes them harder.
 From stored rig vectors: cosines between nearest-point shifts of generic directions (label, rand) +0.41 to +0.63 in all 8 starts; gelu
 all pairs positive (30/30), tanh top-eigen directions weakly negative with the rest => excursions should stiffen unrelated generic
 directions (general loss of trainability), by a known Gram matrix.

SMOKE RUN ON REAL MNIST FROM THE BUNDLE (QUICK, this CPU, 12.3 min): environment identical
 (rel <= 5e-14), linear control nd/n1 3e-14, references 8/8, Part A rho_loop -1.0018 / -0.5011 / +1.0000 and identity 1.00290 = the
 independent pre-check to every printed digit. Part B (2 directions x 2 signs): rho_B -1.0083 (DJ(1)), -0.5032 (SL(2)), cos -1.0000;
 all 8 probe costs rise, DJ(1) exactly twice SL(2). Part C first two tasks: Q(DJ(1)) > Q(SL(2)) in 2/2 (by 0.032, 0.004).
 Full run estimate: 6-7 h on this CPU; Colab GPU unknown (the smoke cell will show).

### 27. SPEC9 pre-checks: new closed forms

## 20. SPEC9 — three follow-up tests (written 2026-11-22)
SPEC9.md (12 sections). The pre-checks (small nets + solvable model) BEFORE writing it:
 NEW CLOSED FORMS, all verified to 3-4 decimals (tanh MLP 196-32-32-10 and u^2-v^2 model):
  SLa(m, alpha) (m sub-steps, alpha-point rule): f = 1/2 + (alpha - 1/2)/m e.g. (2,1) .7500, (2,2) 1.2499, (8,2) .6876, (4,-1) .1250
  TJ(c, alpha) (theta-method in OUTPUT space): f = (1+k)/(2+k), k = c(2 alpha - 1) e.g. (.5,1) .6001, (1.5,1) .7143, (8,1) .9000, (3,.25) -1.0003
  TM(c, alpha) (theta-method in WEIGHT space: alpha=0 GD, 1/2 implicit midpoint, 1 proximal point), per kernel mode k_i = c_i(2 alpha - 1):
     proximal point (1+c)/(2+c): .6001, .7143, .8750; IMPLICIT MIDPOINT = 1/2 AT ANY STEP SIZE (.5001 at c = .5 and 1.5); eigB values .5353, .5927, .7382 as predicted.
  => one parameter k = c(2 alpha - 1) unifies GD (k = -c), midpoint (0), proximal point (c); pole at k = -2.
 PP reference (post-point on a reference path) ~ NP in smooth nets: cos .99993+, ratio .995-.999.
 ReLU reference-path prediction (no fitted parameter, no NP): on the small ReLU net with 0-6 switches the measured rho of SL(2), SL(8), SL(32), DJ(.5), DJ(1), DJ(1.5)
  is reproduced to 2-3 decimals in 40/42 non-overshoot cases (e.g. .501/.501, .878/.878, .972/.972, .924/.923, .868/.868, .812/.814 where the smooth law says .25,.44,.48,.33,0,-1);
  overshooters (DJ(1.9)) are sensitive (grid resolution + switch position differs between fibres) -> reported separately. Kink positions shift with the fibre position
  (first-order vertical offsets after a switch): the open question for the real net is how accuracy degrades with the number of switches.

### 28. SPEC8 pre-checks and tiny validation

Pre-checks: reproduces all 14 ground-truth values in the u^2-v^2 model (rho even part, cos +-1);
 on the toy MLP (h=64): heavy-ball closed form matches in 16/16 cases to 3 decimals (e.g. -63.959 vs -63.964, 0.062 vs 0.063);
Adam (lr 1e-3) measured R = -408 (q1), -31 (q2), -10.7 (q4), -2.6 (q10), -2.4 (rand): strongly ANTI-aligning.
CORRECTION to an earlier statement: "no row-space learner can exceed 1/2" is exact only for straight output paths (C_sym <= 1/2 dYdY^T is a matrix
inequality; for curved paths the projection rho can exceed 1/2 through the area term - toy random directions gave 0.39-0.56). SPEC8 does not test a scalar ceiling.

Tiny validation (n=8, width 32, tanh, trained, synthetic data, 414 cases): PRIMARY H0 16/16, H1x 18/18, H2x 30/30, H3x 16/16, H4x 16/16, H5 6/6
 (cos 1.000000), H5b 4/4, H6 12/12, H7x 12/12 all PASS; literal H2 53/60 FAIL (finite size on weak directions), others pass.
 Adam (report only): |v_L|/|dtheta| = 0.98, nr ~200, cos -0.26..-0.38: Adam's displacement is almost entirely vertical. SL(32) rho vs size: 0.480, 0.477, 0.462, 0.403 at 0.05..0.5 S0.

### 29. ReLU details: cross-checks, kink finding, why HR2 fails, real-size cases

 - Independent cross-check (own forward, autograd Jacobian, numpy solvers) on small ReLU net, eigA, s = 1e-2 S0:
     sign -1 (2 switches in the first 1/16): omega profile = step 0 -> 1.006; lib vs mine rho_PP/rho_field/cosF/nrF/rho_omega equal to 5 decimals
       for SL(8) .87757, SL(2) .50097, DJ(.5) .92364, AJ(1) .97854, AJ(.5) .97974, DJ(1.5) .81231; tau ranges identical ([0, .875] for SL(8)).
     sign +1 (no switch on [0,1], 2 on [1,2]): smooth fractions exact (.43751, .25000, .33334, 1.0001, .50004); DJ(1.5) crosses the switches
       beyond tau = 1: rho_PP -1.552 vs smooth -1, field prediction -1.708 (cosF .937) -> overshooters are the hard case (report R-c).
 - NEW UNDERSTANDING (why the scalar form HR2 fails on ReLU while the vector law HR1 holds): DJ(.5) sign -1: rho_PP .9236, rho_field .9234 but
   rho_omega .5024. b_k is not parallel to d after a switch; K^-1 weights soft directions lambda_1/lambda_i more than the stiff direction eigA,
   so the off-line part of b_k contributes as much as the on-line part. HR2 kept as registered; A3.4 adds bperp_max, rho_perp and a note.
 - REAL SIZE (n=32, width 256, tanh-trained, synthetic inputs, CPU, through the library's own case runner): AJ(2), direction rand:
     rho(s1) 1.998861, rho(s2) 1.999715 -> rho_x = 2.000000; held-out gamma(s1) 1.999185, gamma(s2) 1.999796 -> gamma_x = 2.000000 (f = 2).
     9 / 7 outer iterations, no damping, 13-16 Jacobians per case.

 KINK FINDING : with ONE switch at tau~0.27 the midpoint jump AJ(0.5) has no self-consistent solution (outer iteration
 oscillates across the kink; neither activation pattern is consistent with its own evaluation point) = same structure as the NP failure on ReLU.
 Expect implicit-type learners (AJ, SLa) to drop out of many ReLU cases on the GPU; HR1 then rests on SL/DJ.

WHY HR2 (scalar omega form) FAILS ON RELU - why-chain taken to measured ground (small nets, eigA, s=1e-2 S0, sign -1, DJ(0.5)):
 rho_PP .9236 = on-line part .5024 (= rho_omega) + off-line part .4210 (= rho_perp) [sum .9234 = rho_field].
 (1) first DJ step crosses the switch and lands 2.6% off the line (tanh: 0.10%, curvature only); DJ shrinks all residual components alike, so
     every later b_k keeps a 2.57% off-line share.
 (2) after ONE switch the field W(tau) = V0 J(tau)^T K^-1 is RANK ONE: singular values 1, .004, .003, ... (tanh: 1, .88, .80, .69, .63).
     Every output direction moves the weights along the SAME vector v_PP (cos .995-.999 for random directions; tanh: ~.05).
 (3) the path direction d = eigA is the WEAKEST driver of that vector: random unit output directions give 4-9x more displacement than d
     (tanh: 1.0-1.2x). So 2.57% off-line -> |K^-1 b_off|/|K^-1 b_on| = .54 -> off-line contribution = 84% of the on-line one, fully aligned.
 Ground: ReLU derivative is a step => Jacobian jump of one unit on one input is low rank; V0 J0^T = 0 exactly => ONLY the jump produces vertical
  motion; the response is a covector l(tau) in output space (rho = sum_k <l(tau_k), b_k>), and HR2 replaced it by its component along d.

 Free evidence from that GPU test output (real MNIST, small ReLU net, 4 / 17 switching cells): general law cosF .994-1.000, nrF .959-1.001, while the
 smooth fractions are far off (SL(8) .5699 and .8703 vs .4375; DJ(1) .4832 and .9653 vs 0; DJ(.5) .5511 / .9411 vs .3333).

REAL-SIZE LOCAL CASES ON THE AMENDED LIBRARY (n=32, width 256, trained, SYNTHETIC inputs, CPU; R with G_REF=64, 5 learners):
 Part A: TM(6,1) eigA: rho_x .875000, gamma_x .875000 (f .875); 107 steps, ~40 inner iterations per run (floor rule), 0 capped.
         TM(1.5,.5) eigB s2: .500035 / .500023 (implicit midpoint = 1/2); TJ(.5,.5) eigD s2 .500641, status CONVERGED (4 jumps would have been flagged by the old rule).
 Part R tanh control (eigA 1e-2 +, label 1e-3 -): everything exact: SL(8) .43748, DJ(.5) .33332, AJ(1) 1.00005, AJ(.5) .49999, SLa(8,1) .56248; cosF 1.00000; omega = tau.
 Part R RELU AT REAL SIZE - three things to expect on the GPU:
  (1) switches are DENSE: eigA 1e-2 +: 62 switches by tau=1 (20 of 32 cells); label 1e-3 -: 38 switches (every cell). "One linear region" only at the smallest sizes.
      omega(tau) is a curve, not tau: 0 .338 .599 .690 .716 .872 .992 .995 1.0 (eigA); non-monotone for label (.568 -> .407 -> .710).
  (2) ALL implicit-type learners AJ(1), AJ(.5), SLa(8,1) are AJ-FAILED (outer cap 100, 100 Jacobians each) in both cases -> they will be missing from ReLU
      Part R almost everywhere; HR1 rests on SL(2,8,32), DJ(.5), DJ(1). Their last iterates still give sensible rho_PP (.93, .74) = approximate solutions exist
      down to the chatter amplitude. IDEA (not in the notebook): accept "chatter-converged" jumps at the chatter level as SUPPLEMENTARY data.
  (3) the general vector law is good but no longer exact: SL(8) eigA: .65185 vs predicted .65185 (cosF 1.00000; smooth form says .4375);
      DJ(.5) eigA: .4712 vs .4683 (cosF .9947, nrF 1.004); SL(8) label: .3845 vs .3685 (cosF .985, nrF 1.041); DJ(.5) label: .3234 vs .2857 (cosF .955, nrF 1.157 -> FAILS HR1's nr band).

### 30. g4_ends: the turn-round is not size

## 25. g4_ends ON THE RIG (2026-11-22 night): THE TURN-ROUND IS NOT SIZE
128 rows (bit-identical to the stored G4 records), no SUM CHECK FAILED, all CONVERGED. Bundles (13.7 MB each).
Pre-registered 'size' reading (sec. 23) against the data (SL(4) and DJ(0.25) rows agree in sign everywhere quoted):
 * eN > 0 for f > 0 (start function grows with f): HOLDS 30/32 (fails only tanh n128 tau = 1, tiny). gN > 0: HOLDS 32/32 (growing the random start
   function always costs test loss). eN is 6-30x larger for gelu than tanh (n32: tanh 1.6e-3, 9e-3, 3e-2, 8e-2; gelu 1.3e-2, 6e-2, 0.15, 0.32 at tau 0.1, 0.25, 0.5, 1).
 * eS > 0: FAILS, eS < 0 in 28/32 (as the local tiny test had warned): the first-order learned change SHRINKS with f on unseen inputs.
 * "turn-round where gS < 0 and |dS| > dN": FAILS. gS < 0 in 24/24 blocks with tau <= 0.5, > 0 in 8/8 at tau = 1, but with eS < 0 this makes
   dS > 0 (a cost) below tau = 1 and a gain only at tau = 1.
 * The turn-round is carried by drest: drest < 0 for f > 0 in 30/32 blocks (the two exceptions are +5e-7 at tanh n32 tau = 0.1) and opposite for f = -1
   in most blocks. So the part of the function difference that is NOT a rescaling of the start function or of the learned change moves TOWARD the
   true test labels as f grows.
 * dS + drest < 0 in 32/32 blocks (SL(4)): net effect of larger f on everything that was learned = toward the true labels, every block.
   Sizes (SL(4), units 1e-6): tanh n32: -5, -18, -78, -750 (s0), -2, -7, -64, -690 (s1); tanh n128: -10, -99, -740, -3800 / -6, -93, -720, -4500;
   gelu n32: -9, -51, -170, -300 / -3, -17, -120, -580; gelu n128: -8, -120, -510, -4300 / -19, -160, -780, -4800 (tau .1, .25, .5, 1).
   Against it: dN (12..+420) and the squared term quad (0.03..+4400, grows ~tau^4).
 * loss* (test loss after the best one-number rescaling) improves with f in tanh n128 8/8 blocks (tau = 0.5: 0.033324 -> 0.032872; accuracy 64.3 ->
   65.1%), i.e. more than size. It worsens with f in gelu n32 8/8 and tanh n32 6/8 (there the growth of the start function is not removable by one
   overall factor because the learned part needs the opposite correction).
 => ANSWER: larger f does three things on unseen inputs: (1) grows the random function the network had at initialisation (always a cost),
    (2) changes what was learned toward the true labels (a gain in 32/32 blocks), (3) a squared term that punishes any large departure (this is why
    f = -1 is worst at full moves). More data and larger moves raise (2) faster than (1): drest/dN at tau = 0.1 goes from ~0 (n = 32) to -1.9 (n = 128, tanh).
 Other regularities: best factor for the learned part alone = c/tau with c = 0.49-0.55 (tanh) and 0.66-0.74 (gelu) at every tau and both n -> the
  learned change is linear in tau even at full moves, and the full fit is 2x (tanh) / 1.5x (gelu) too large on test inputs; same numbers as optS of
  the trained starts in g3_parts (0.51, 0.68). cos(Nv, Sv) = -0.13..-0.41 (the learned change contains minus the interpolated start function).
  R2 of the two-part fit: 0.03-0.32, falls with n and tau (the difference between learners is mostly NOT a rescaling).
FAILED PREDICTIONS / CORRECTIONS: the 'size' reading of sec. 22-23 is refuted as the explanation of the turn-round. What survives of it: the label-independent growth of
 the start function is real (52/52, 30/32) and is the reason b > 0 at small moves.

LOCAL TEST (teacher-student, synthetic inputs, tcr2.FastMLP h = 64, n = 32, 2000 test inputs, 10 seeds, damped GN with
 c = 1.5, 1, 0.5, 0.25 -> f = -1, 0, 1/3, 0.43, all to residual 1e-10; 40 blocks per model):
 plain model: test loss RISES with f in 36/40 blocks (9, 9, 8, 10 of 10 at tau .1, .25, .5, 1) - the MNIST small-move result reproduced.
 start function removed: rises in 17/40 (4, 5, 5, 3) - the systematic cost is GONE, but the predicted systematic GAIN is not there (23/40 fall;
 median relative change -1.6e-5, -4.2e-5, +7.8e-4, -1.7e-2). VERDICT: half of the prediction holds (the cost of alignment is the start function);
 "larger f always helps once it is removed" is NOT shown on this small synthetic problem. Open: same test on MNIST (needs MNIST locally or the rig).

### 31. g3_parts, vector bundle, long loops

## 24. g3_parts ON MNIST, THE VECTOR BUNDLE, THREE LITERATURE SEARCH, LONG LOOPS (2026-11-22 evening)
g3_parts (rig, real MNIST; checks: theta_init difference 0.0 in 4/4, chk 1.000 in 20/20). Predictions of sec. 23:
 (i) trained starts eN > eS: HOLDS 24/24 (both > 0; eN/eS = 1.8..6.2): the part of the function present before fitting grows 2-6x faster than the
     fitted part. optS = 0.51 (tanh), 0.68 (gelu): the fitted part is too LARGE on test inputs. bN, bS, brest all > 0 in 24/24; R2 of the two-part
     fit only 0.02-0.24, brest = 15-60% of b => at trained starts most of G is NOT a rescaling of what is there, and that remainder moves away
     from the true labels in 24/24 rows.
 (ii) cS1 < 0 for "true" at init: HOLDS in sign 4/4 (-0.026, -0.062, -0.049, -0.047) and in order (tau* gelu 2.2, 1.1 < tanh 15.7, 5.6), but the
     magnitudes are 3-5x larger than predicted (0.3-0.6 / 1.5-3.5): the first-order shift of the residual explains only a small part of the G4
     fall-off. label/rand: cS1 mixed.
 (iii) |z_lab| > 2 in 47/52. free (label-independent part of b) > 0 in 52/52 and >= 65% of b in every row; lab < 0 in 36/52 (gelu-trained 12/12).
     Pattern toward the labels at init for label/rand/true (z -6..-28) is ONE sign per start (4 starts) => p = 1/8, weak.
Bundle (33.8 MB): per-test-input analysis.
 * Trained starts: growth of the start function is larger on MISCLASSIFIED test images than on correct ones in 24/24 rows (mean gain ratio
   1.2..30x; corr(gain, correct) -0.02..-0.32; corr(|G_i|, correct) < 0 in 24/24). Forced part of the reason: G(x) = J_x V0 v_NP vanishes on the
   training inputs and is bounded by sqrt(kernel posterior variance at x) |v_NP| (Cauchy-Schwarz) -> concentrated where the fit is least reliable.
   NOT yet checked directly (needs the test inputs: corr(|G(x)|, posterior sd(x)) on the rig).
 * Per-input-gain model (G_i parallel to F0v_i) explains only 11-43% of |Gv|^2. Top eigen-directions: G is nearly rank one across the 10 outputs
   at init (s2/s1 0.10-0.26 for eigA, eigB) and G(eigA) ~ G(eigB) (cos 0.76-0.88 after removing F0v). 10% of the test inputs carry 26-45% of |G|^2.

LONG LOOPS (tcr2.FastMLP h = 32/64, n = 16, rectangle of side eps |y_A| in two output directions,
 RK4 lift + GN corrector, 200-1000 cycles; control "retrace" drift 1.7e-6 per cycle vs 0.14-1.7 for loops). Prediction before the run: no settling
 (60%). Transplant tested: "repeated loops settle into a drift-free loop" (3R arm).
 * NO settling in 10/10 runs: drift per cycle stays 0.07-3.8x its first value, its direction decorrelates from the first cycle within 10-50 cycles
   while successive cycles stay aligned (cos 0.98-0.999); displacement grows like cycles^0.35..0.81 (sub-ballistic).
 * trK and lam_max RISE in 10/10 (trK x1.13..2.8, lam_max x1.2..2.3), for BOTH senses of rotation (rand: 31.7 -> 39.2 forward, 39.7 reversed;
   weak: 41.5 / 40.1 at 200 cycles) => an EVEN effect, not the bracket sign. Delta trK ~ displacement^1.8 across loop sizes (4.0, 10, 33 at
   displacement 7.9, 14.3, 25). |theta| changes only 3-8%. lam_min collapses in some runs (0.067 -> 0.0007 at h = 64), one run broke the corrector.
 * Costs u^T K^-1 u of the practised directions: no rule (down for weak loops = regression to the mean of a selected extreme; up for random loops
   forward, down reversed). "Practice makes the loop cheap" is FALSE as a law.
 * Unseen-input function moves by rms 0.2-2.2 (output scale ~0.5) while the training outputs are restored every cycle.
 READING: vertical excursions raise the kernel trace at second order whatever causes them (forced positive piece |D_w J|^2); the same convexity gives
 the U-shape of trK in f in G4. Break point of the robotics mapping: fibre dimension (1 for the 3R arm, 1361 here) - no attractor is forced.
 NOT done: more seeds/widths, scaling of the growth rate with eps (between eps^1 and eps^2 at fixed cycles), GD instead of exact transport.

### 32. g3_split on MNIST and the G4 table

## 23. g3_split ON MNIST AND THE G4 TABLE (2026-11-22, rig still running Part R)
g3_split (52 rows = 8 combinations x 6 or 7 directions, real MNIST): b_vec = b_rec in all rows; b > 0 52/52; e > 0 52/52 (the start function
 grows on unseen inputs in EVERY case); cos(G, F0v) +0.07..+0.56.
 Init starts (28 rows): scale piece = 57%..610% of b (>= 81% in 26/28); pattern NEGATIVE (toward the true test labels) in 12/12 label/rand/true
  rows (-16%..-34% of b for 'true') but only 9/16 eigen rows. CAUTION: G of label/rand/true are correlated (cos 0.5-0.73) -> this is ~4 independent
  starts, not 12. Label-dependent total -<Yv,Gv>/N has mixed sign at init (true: +1%, +7%, +16%, -54% of b).
 Trained starts (24 rows): pattern POSITIVE 24/24 and as large as the scale piece (scale 18-79% of b). The local synthetic check (sec. 22) did not
  show this. Reading to test (part 2): the part of the function present before fitting grows more than the fitted part (eN > eS).
G4 (large moves toward true labels, 32 blocks; pre-registered predictions of sec. 22):
 (a) Spearman(f, test loss) > 0 at tau = 1: FAILED, 2/8 (0.89, +0.24; six negative, -0.18..-0.90).
     Holds 16/16 for n = 32 with tau <= 0.5 and n = 128 with tau = 0.1 (0.85..1.00). Of the other 16 blocks, 4 positive.
     The G3 coefficient predicts the finite-move difference loss(AJ(1)) - loss(DJ(1)) = b (tau/0.01)^2 at n = 32:
     observed / predicted at tau = 0.1: 81%, 107%, 105%, 106% (gelu s0, s1, tanh s0, s1); tau = 0.25: 53%, 84%, 90%, 98%; tau = 0.5: 36-63%.
 (b) trK increases with f: holds 8/8 at tau = 0.1. lambda1: holds 6/8, FAILS for tanh n = 32 (815.6 -> 811.9, -0.2% per unit f).
     Where theta0 values are in the table (rows of AJ learners that never moved): at tau = 0.1 the exact jump leaves trK almost unchanged
     (gelu n32 +0.4%, tanh n128 +0.1%, gelu n128 +3..4%) while f = +1 raises it (12%, +6%, +35..39%) and f = -1 LOWERS it (-8%, -4%, -10%).
     Not new in kind (sec. 14: R_trK = rho; Buffelli: exact GN stays lazy), new as MNIST numbers and the shrink below the start for f < 0.
 (c) weight distance: |th_L-th0|^2 - |th_NP-th0|^2 proportional to (1-rho)^2 with the MEASURED rho holds to ~4% at tau = 0.1 (gelu-s0: 3.56/0.523/
     0.345/0.286/0.222/0.211/0.058 observed vs 3.40/0.535/0.355/0.295/0.233/0.208/0.061); DJ(1.5) is farthest in 32/32 blocks.
 (d) "accuracy need not follow": at n = 32 it DOES follow (monotone falling with f, e.g. gelu-s0 tau 0.25: 38.67% -> 37.75%); at tanh n = 128
     tau = 0.5 it RISES with f (64.1% -> 65.3%). Accuracy turns round where the loss turns round.
 Shape at large moves: U in f. f = -1 (DJ(1.5)) is the best learner in 15/16 small-move blocks and the worst converged one in 5/8 blocks at tau = 1, never the best there (largest distance, largest trK);
  the minimum moves from f <= -1 (small moves) to f ~ 0..0.5; for tanh n = 128, tau >= 0.25 the loss FALLS with f on [0, 0.5].
  loss vs trK ranking: identical or one swap in gelu tau = 1 (n32 and n128) and tanh n32 tau = 1; opposite in tanh n128 tau = 0.25-0.5
  (there loss follows weight distance instead). No single scalar orders test loss in all blocks.
 Confounds: AJ-type learners (AJ(1), SLa(4,2), TJ(1.5,1)) fail at large moves, so rankings at large tau use 5-8 learners with f <= 0.5;
  GD(0.25) stops at residual 1e-3 (sec. 14 note: skips the soft directions) and is the best row at n128 tau = 1 for that reason, or UNCONVERGED;
  SL(2) JUMP-FAILED in 4/4 n128 tau = 1 blocks; 2 seeds per cell. DJ step factor alone changes test accuracy at gelu n128 tau = 1:
  77.65% (c = 0.25) vs 77.51% (c = 1) vs 73.37% (c = 1.5); seed 1: 74.62 / 73.26 / 69.53. Tanh: no such order.
 FAILED PREDICTIONS / CORRECTIONS: prediction (a) wrong at tau = 1; lambda1 part of (b) wrong for tanh n32; sec. 22 "scale piece 79-108%" was a synthetic-data
  number and missed the trained-start pattern piece.

### 33. SPEC9 interpretation checks and other patterns

INTERPRETATION CHECK (local): 
  * Local tiny run (n=8, width 32, SYNTHETIC inputs, RANDOM test labels): b > 0 in 5/5 and the scale piece is 79-108% of b; cos(G, F0v) +0.34..+0.51,
    cos(G, labels) ~0. So b > 0 does not need real labels: it is the function getting LARGER on unseen inputs at fixed training outputs.
    G of different directions are nearly orthogonal to each other (cos 0.08-0.17) apart from that common inflation component.
  * (small tanh MLP, 6 directions): v_NP changes layer weight norms with sign pattern (-) in 6/6; last layer grows most;
    cos(v_NP, projected last-layer scaling) +0.12..+0.30; projected scaling of any layer moves held-out outputs along the existing function
    (cos +0.77). The last-layer-growth component carries 28-52% of b; the three layer scalings together 35-62%.
    Forced piece behind the sign: d/d(scale of last layer) of 1/2 a^T K a = a^T K_earlier-layers a >= 0 for every a.
  => Do NOT report G3 as "alignment hurts generalisation". Report: more alignment = sharper (sec. 14) = larger function off the training set
     = higher squared-error test loss, all linear in f. Whether a label-related (pattern) part exists on real MNIST: g3_split on the rig decides.

OTHER PATTERNS: b is 14-127x larger in eigD than eigA and label/rand sit at the eigD level (soft kernel directions carry the test-loss sensitivity),
  while plain GD has f != 1/2 only in stiff directions => hypothesis: step size changes test loss via this route mainly for preconditioned / GN-type
  methods, hardly for plain GD (untested). Second-order corrections: AJ(alpha) error ~ alpha(1-alpha)(a + b alpha) s^2; learners with equal f differ
  at this order (f=-1: AJ(-1) -0.032 vs DJ(1.5) +0.007 vs TJ(3,0.25) +0.012, eigD s1) => f is complete at leading order only.
  lam_B/lam_1 from TM rows: 0.231, 0.268 (tanh init s0, s1), 0.255, 0.280 (trained).

### 34. X1 (n = 32) result and first X3 rows

## 34. X1 RESULT (n = 32 part) AND FIRST X3 ROWS (2026-11-22/23 night)
X1, ZERO by subtraction, real MNIST, n = 32, tau = 0.1, AJ(alpha) (f = alpha), all CONVERGED. Change against alpha = 0 (raw loss / loss after best
 one-number rescaling / accuracy):
  tanh-s0: a .5 +6.4e-7 / -3.4e-6 / .5024; a 1 +1.5e-6 / +2.0e-6 / .5009; a 2 +3.9e-6 / +3.5e-5 / .4999; a 3 +6.8e-6 / +9.1e-5 / .4991 (acc at 0: .5020)
  tanh-s1: a .5 -4.3e-7 / -1.5e-5 / .5309; a 1 -9.2e-7 / -2.6e-5 / .5311; a 2 -2.0e-6 / -3.1e-5 / .5307 (.5308)
  gelu-s0: a .5 +5.8e-6 / -3.1e-5 / .5108; a 1 +1.2e-5 / +3.2e-5 / .5101; a 2 +2.6e-5 / +3.2e-4 / .5079 (.5087)
  gelu-s1: a .5 -2.5e-6 / -9.3e-5 / .5442; a 1 -2.6e-6 / -1.2e-4 / .5440; a 2 +3.5e-6 / -2.2e-5 / .5425 (.5437)
 VERDICTS: P1 (rescaled loss falls monotonically to alpha = 2 in >= 3/4) FAILS, 1/4. P2 half (tanh-s1 yes, gelu-s1 optimum ~0.75). P3 (accuracy at
 alpha = 2 >= at 0 in >= 3/4) FAILS, 0/4. => there is an INTERIOR OPTIMUM of alignment at n = 32: rescaled-loss optimum alpha* ~ 0.4 (tanh-s0),
 ~0.35 (gelu-s0), ~1.0 (gelu-s1), ~2.5 (tanh-s1). "Beyond the nearest point is better" is FALSE at n = 32; in 2 of 4 starts the nearest point is
 already over-aligned and the optimum is near gradient flow's 1/2. Raw loss is linear in alpha (slope sign per start as in SPEC10); rescaled loss
 is clearly quadratic. trK rises linearly with alpha in all four. (n = 128 part with alpha 0, 1, 2 still running when written.)
X3 doubled network, tanh-s0, n = 32 (first start): |F_test(SL(8)) - F_test(DJ(1))| rms = 2.24e-8, 6.05e-7, 2.21e-5, 5.35e-4 at tau .01, .03, .1, .3
 -> factors 27.0, 36.5, 24.2 for tau x3, x3.33, x3 = exponents 3.00, 2.99, 2.90. D1 CONFIRMED in this start: in the doubled network the
 learner-dependence of predictions is THIRD order (subtraction model, rig bundles: exponents 1.5-2.0). Differences are proportional to f
 (SL(2):SL(4):SL(8) = .585:.865:1 vs f ratios .571:.857:1; DJ(1.5) at tau .1 has the opposite sign, ratio -2.7 vs -2.3).
 Raw loss against DJ(1), SL(8): +4.2e-10, +1.0e-8, +2.1e-7, -6.3e-6 at tau .01, .03, .1, .3: POSITIVE (~tau^3) at small moves, negative from 0.3.
 So D2 as registered (negative at small tau) FAILS in this start. Fit: +A tau^3 - B tau^4. Reading: larger f makes the learned change SMALLER on
 unseen inputs (eS < 0, third order); while the learned function is still too small (partial fit, tau < its best factor ~0.5) that costs raw
 loss, later it helps; the pattern gain comes on top at order tau^4. Raw loss at partial fit is therefore the wrong probe; x3 did not store the
 rescaled loss (to be added on a rerun).

### 35. X2 feature-probe result; X1 n = 128 rows

## 38. X2 RESULT: FEATURE PROBE COMPLETE (64 learners, all CONVERGED); X1 n = 128 INTERRUPTED (2026-11-23, 04:45)
X2 design as run (reduced from sec. 31): 4 starts (tanh/gelu x seed 0/1, n = 32) x tau {0.5, 1} x {ZERO, PLAIN} x learners f = -1, 0, .375, .43
 = 16 blocks (8 per target kind). Registered thresholds "12 of 16" are read as 6 of 8 per target kind.,
 . Probe = nearly unregularised ridge readout fitted on test images 0-4999, scored on 5000-9999.
VERDICTS:
 Q1 (probe accuracy f = .43 above f = 0, ZERO): 3/8 -> FAILS.
 Q2 (probe accuracy f = -1 below f = 0, ZERO): 6/8 -> meets the threshold, but most gaps are inside the paired noise (~0.15 points); the only
    clear cases are gelu-s0 tau 1 (f = -1 is 0.5-0.8 points lower). Not robust.
 Q3 (same for PLAIN): 3/8 and 6/8 -> same as ZERO. The part "PLAIN and ZERO now agree" HOLDS (same sign of every slope in 16/16 block pairs).
 Q4 (CKA with labels ordered as f): 4/8 and 6/8 -> FAILS as registered; it is activation-dependent (below).
WHAT THE DATA DO SHOW (not registered, found after the run - treat as a new observation to be re-tested):
 (a) Squared error of the refitted readout RISES with f, in both hidden layers: f = .43 above f = 0 in 16/16 (last hidden), f = -1 below f = 0 in
     15/16; straight-line slope in f positive in 16/16 for the last hidden layer and 16/16 for the first. Straight line R2: tanh .97-1.00 (8/8),
     gelu-s1 .88-1.00, gelu-s0 .61-.71 (there f = -1 does nothing and f > 0 hurts).
 (b) Size: +-1 to 2.5% of the probe error at tau = 1. Against the gain from training itself (f = 0 minus start): slope per unit f is 0.05-0.11
     of it (gelu-s0), 0.2-0.5 (gelu-s1), 0.45-0.77 (tanh-s0), 1.4-1.6 (tanh-s1). All 64 trained networks still have lower probe error than their start.
 (c) Move-size scaling of the slope, tau .5 -> 1: factors 2.2-7.3, exponents 1.1-2.9 (median ~1.8): compatible with the f-dependent weight
     change f tau^2 v_NP acting on the features at first order, but noisy.
 (d) PLAIN slope = 1.03-1.96 x ZERO slope, same sign always: the start function does not change the direction here (in SPEC10 output loss it did).
 (e) CKA with labels: tanh rises with f in 8/8 blocks in both layers (R2 .89-1.00) while the refit error also rises; gelu last hidden FALLS for
     f > 0 in 8/8. So for tanh the features become more label-like in their large-variance directions yet carry slightly less linearly
     decodable label information; CKA is not a measure of decodability here.
 (f) Accuracy of the refit readout has no consistent direction in f.
READING: the weight-saving direction v_NP (larger f) makes hidden features slightly WORSE for a readout that has 5,000 labels, linearly in f,
 in both layers, both activations, with or without the start function. So the SPEC10 gain (network's own test loss falls with f, ZERO model) is
 NOT "better reusable features". As registered in sec. 31: "larger f gives better features" is wrong at the level of the hidden layers by this
 probe. Compatible with Geiger 1906.08034. OPEN, not decided: whether the gain is a few-label effect (features concentrated on what 32 labels
 need). DECIDING TEST (x2b, not run): same learners, refit the readout on k = 32, 100, 320, 1000, 5000 images (ridge chosen on held-out data):
 prediction to register before running - the slope of probe error in f is negative at small k and positive at large k, with a crossover k*.
 If the slope is positive at every k, the gain is in how the trained readout and features combine, not in the features.
X1 n = 128: the run was killed at 00:20 when the previous session ended (log stops inside gelu-s0 AJ(2)). Valid rows:
 tanh-s0: alpha 0/1/2 rescaled loss .0331684/.0325623/.0324516, raw .0439683/.0439227/.0438808, accuracy 70.34/71.11/71.52%. Flag AJ-FAILED on
   alpha 1 and 2 = iteration cap reached with last steps 3.3e-11 and 1.1e-7 (training residual 1.3e-13, 4.8e-11): effectively converged.
 tanh-s1: alpha 0 .0335807 (raw .0441526, 68.10%); alpha 1 .0332612 (raw .0441340, 68.31%) BUT implicit iteration still moving (step 1.6e-2):
   same training outputs (res 2.5e-12), f not guaranteed to be 1. alpha 2: training residual 2.9e-2 -> INVALID, discard.
 gelu-s0: alpha 0 .0282936 (raw .0436029, 76.31%); alpha 1 CONVERGED .0281324 (raw .0435939, 76.31%).
 So far: alpha 1 beats alpha 0 on raw AND rescaled loss in 3/3 starts (2 fully trusted); alpha 2 beats alpha 1 in 1/1 valid. At n = 32 the
 seed-0 starts went the other way. (skips done rows and the two tanh starts): gelu-s0 AJ(2), gelu-s1 AJ(0, 1, 2).

### 36. ReLU width sweep and layer structure

## 33. RECORDS OF THIS ROUND (2026-11-22 late night): width sweep, layer structure, two literature searches
ReLU WIDTH SWEEP (local synthetic, n = 8, widths 32-512, sizes 0.01 / 0.03 S0, 2-4 cases each): the premise was WRONG:
 the number of switches at a fixed relative move does not rise with width (3, 4, 3, 4, 5 at 0.01; 10, 13, 12, 16, 15 at 0.03). Median distance
 from the smooth fractions at 0.03: 0.096, 0.116, 0.072, 0.037, 0.036 (falls about 2.6x from width 32 to 512); at 0.01: 0.18, 0.08, 0.22, 0.06,
 0.14 (no trend, 3-5 switches = the few-switch regime). max|omega - tau| and omega_perp do not shrink with width. Verdict: weak support for
 "each switch matters less in a wider net", none for "more switches"; too few cases to fit an exponent.
LAYER STRUCTURE OF THE NEAREST-POINT SHIFT on the MNIST nets (on the 208 AJ(1) vector files, even part, size s2):
 share of |v_NP|^2: first layer 29-65%, second 12-27%, readout 14-52% (readout has 2% of the parameters -> 40-100x denser per parameter).
 Init starts (28 rows): all three layers' weight norms GROW along v_NP in 28/28; relative growth of the readout is 20-60x that of the hidden layers.
 Trained tanh starts: first layer shrinks slightly in 5/12 (the (-) pattern of the small local net appears only there); gelu trained: small, mixed.
 Total |theta|^2 grows in 48/52. cos between v_NP of different directions: eigA-eigB 0.53-0.65, generic directions 0.1-0.6 (gelu higher than tanh).

## Tier 3: failed, negative, or superseded

### 37. a-meter for ordinary optimisers (AM1 holds, AM2 fails)

 a-METER FOR ORDINARY OPTIMISERS (tanh E1 s1, full batch, legs stopped at 1e-3 of the move):
  GD(1) a = -0.016, GD(1.8) -0.014, GDmid(1) -0.019, GDmid(1.8) -0.019; single-leg fractions 0.506-0.508; 2,300-4,200 steps per leg.
  => AM1 HOLDS in substance (|a| <= 0.02; the registered interval [0, 0.15] is missed by sign at the 0.02 level): full-batch gradient
  descent is already time-symmetric on a generic direction. The midpoint version changes nothing because there is nothing to remove.
  ADAM(1e-5): converges in ~213 steps; single-leg fraction 0.29 but cosine with the nearest-point shift only 0.53; loop displacement
  mostly NOT along it (cos -0.24); the travelled direction gets harder by 0.22 of the exact jump's amount. ADAM(1e-4): displacement
  almost orthogonal to the nearest-point shift (cos 0.03); the travelled direction gets EASIER (-1.36).
  => AM2 FAILS as registered (0.22 < 0.5 at one rate, opposite sign at the other). Adam does not move along our fibre direction at
  all (different metric), so the one-number reading does not apply to it; its effect on trainability has to be read from the gauge on
  many directions, which this run did not record. The midpoint-Adam did not converge (leg residual 0.6 and 4): the design is wrong
  (Adam's step size does not shrink with the gradient, so a half-way re-evaluation near the target oscillates). AM3 untested for Adam.

### 38. GPU run of the dose law (cross-optimizer law not supported)

## 10. GPU run of Tests 1-2 (dose law) — DONE 2026-11-21, full mode, 156/156, 4x RTX 3090, 33 min

Setup: 64 MNIST images, MLP 196-256-256-10, K is 640x640, 20 tasks, 3 seeds. LR chosen: GD 1, MOM 1, ADAM 3e-3.

Verdicts (pre-registered): H1 PASS, H2 PASS, H3 FAIL, H4 FAIL, H5 FAIL, H6 FAIL, H7 FAIL, H8 PASS, H9 PASS.
=> The dose law as a cross-optimizer law is NOT supported. Say so plainly.

What the raw data show (the analysis, post hoc, not pre-registered):
1. Dose works only inside the exact learner (GN): Spearman(A, D) = 0.92 over 27 GN conditions (N and eps sweeps),
   0.875 over 540 task records vs 0.695 for task index; A ~ 0.35 ln D + 0.25 (sd 0.29); rough collapse of the
   N-sweep and the eps-sweep at equal dose. Across optimizers it fails (GD and GN have equal dose ~58, damage 0.6 vs 2.0).
2. EDGE-OF-STABILITY LOCK (strongest pattern): for every GD run, eta*lambda_max(K)/(2*640) sits at 1.00:
   GD lr=1 medians 0.998-1.04 for all N; GD-half starts 0.75-0.88 and climbs to 1.005; GD-quarter starts 0.38-0.45
   and climbs to 0.99; tanh 1.02-1.06. Momentum climbs toward its own limit 1.9 (= 1+mu): medians 1.35-1.73, max 1.86-2.03
   for N>=8. Large target jumps (sequence E, eps>=0.1) knock GD below the edge (final 0.86 -> 0.62 as eps grows).
   GN has no cap: lambda_max x6-11, effective rank x0.25, at every N.
   => H7 was mis-specified: halving lr cuts dose 2-6x but lets lambda_max rise 1.2x / 2.4x; damage A unchanged, psi worse at lr/4.
3. Unit death is the best cross-condition correlate of damage in ReLU: Spearman(A, dead frac layer 1) = 0.887 (n=129),
   dose 0.70, bottom-band dose 0.77. Death is jump-size driven for MOM (dead1 0.82, 0.53, 0.31, 0.21, 0.13, 0.12 for N=1..32)
   and GN (0.47 -> 0.11), ~N-independent for Adam (0.96 -> 0.77), absent for GD (0.03). Adam/MOM N=1 end with the
   bottom of the spectrum exactly zero (p10 -> 1e-11) and Adam hits the 4000-step cap from task ~12; no single dead-fraction threshold.
4. tanh arm (no dead units) matches the travel-minus-jitter picture qualitatively: smooth travel raises weak eigenvalues,
   jumps do not. p10 ratio: Adam N=8 x8.7 (psi x0.013-0.037 = 30-80x MORE trainable), GD x2.7-3.2, GN N=8 x1.7, GN N=1 x0.7-0.8,
   Adam N=1 x0.5-0.9. A is literally log tr K^-1 (trace-normalised), the travel potential for isotropic targets: GD A = -0.33..-0.53.
5. GD filter exponent is 0.9, not 2 (H4): GD trains each target to the threshold so strong modes finish. Order of
   exponents: GN 0.1 < Adam 0.2-0.56 ~ MOM 0.47-0.55 < GD 0.9.
6. GD is not immune: dead2 creeps 0.04 -> 0.44 and A drifts from -0.17 to +0.7 over 20 tasks.

### 39. Why the dose law failed; audit of the GPU run

## 12. Why the dose law failed + candidate replacement (post hoc) 2026-11-21
Diagnosis: in the exact learner (GN) three different quantities are collinear (all ~1/N): kernel-weighted squared jump (dose),
per-step weight kick size, and the jitter term. Other optimizers decouple them. Break points of the "dose" analogy:
(1) open loop vs feedback: GD's top-mode jitter is regulated by the stability edge (eta*lam_max/2n = 1.00 in all runs);
(2) one damage kind vs two: smooth spectrum reshaping (reversible, can be negative) and ReLU unit death (absorbing);
(3) omitted travel term (our own identity C_sym = travel - jitter); (4) Adam's metric.
Evidence: unit survival in layer 1 is exponential in task count (R2 0.91-0.995, >= linear fit); hazard per task:
GD 0.0014 flat in N and lr; GN 0.030,0.030,0.0135,0.0091,0.0054 (N=1..16; Spearman with weight path per task 0.97);
MOM 0.087,0.038,0.018,0.011,0.0065,0.0058 (~0.003+0.083/N); Adam 0.156 -> 0.067 (nearly N-independent; kills at dose 0.03).
A_final regression (129 ReLU runs): ln dose R2 0.53; -ln(survival L1) 0.72; + L2 0.80; + eff rank 0.82. log psi: 0.34 vs 0.63/0.70.
GN: A = 0.96 + 1.08/N (R2 0.974; log fit 0.935). tanh: ln p10 ratio vs A -0.957, vs log psi -0.965 (dose 0.47/0.32).
No single per-task variable orders hazard across ALL learners (best 0.68).

## 13. Law vs test vs code audit of the GPU run (2026-11-21)
Code: read DoseAcc, damage_raw, probe, train_gn, train_first_order, run_condition - faithful to SPEC5; the independent
scripts reproduce H5 (0.7/0.3/0.9) and H6 (0.486) exactly. No bug found (analysis cells for H1-H3, H7 not re-derived line by line).
Test-design faults : (A) pre-registered jitter-only dose and dropped the travel term (opposite sign): first-task data:
GD A_1 = -0.015,-0.06,-0.22,-0.47 for eps .05-.4; GN A_1 ~ 0 (C=0 single jump, as theory says); Spearman(A_1, dose) = -0.26..-0.36.
(B) scalar D mixes bands and LR selection put GD AT the stability edge (top band 19-45% of its dose vs 1-2% for GN):
tanh H5 with bottom-band dose 0.94/0.94/0.94 vs 0.54/0.37/0.54 with total D. (C) ReLU + 20 full relabel tasks = unit death
regime, outside the smooth theory; after collapse dose explodes (reverse causation) - ReLU H5 with D_bot looks like a pass
(1.0/0.7/0.9) but is negative pre-collapse. (D) A is a log and saturates; R-jumps are O(1): H3 exponents at m=20 not meaningful
(GN m=2 slope 0.94). (E) H4's alpha is 0.93-1.3 even from init at lr/4: statistic invalid in nonlinear nets, not evidence about the filter.
(F) H7 contradicted our own Part-4 result (GD shift lr-independent). (G) damage = GD-probe view; by own-optimizer steps GD and GN
lose nothing, MOM N=1 x3, Adam x20-90. (H) Adam included although the path law excludes it.
Genuine failures that survive: no dose variant unifies ReLU learners (GD+GN bottom-band 0.755 < 0.8; all five ~0.5 < task index 0.64);
dose is not causal for GD (6-10x less dose in every band, same damage); eps exponent ~1 not 2 at m=2.
Verdict: the run fairly killed "scalar jitter dose unifies optimizers"; it did NOT fairly test the three-force theory, whose

### 40. Implicit healing failed to run

HEALING RESULT (heal11): HL1 FAILS at once: the implicit learner AJ(1) does not converge on the worn network (first leg AJ-FAILED).
 So "heal with implicit excursions" is not available in the state where it is needed, at this size. Not a test of HL2/HL3.

### 41. Early failed or withdrawn claims

Failed or withdrawn :
- "Plain cyclic projections reproduce anticipatory recovery" — one-task fluke + biased ratio.
- Eigen error-threshold as a replay threshold in the form first stated (kept as open idea).
- "Cyclic order selectively raises kernel similarity of adjacent items" — no (equal change).
- "Back-and-forth drive encloses no area" — false: kernel modes lag differently (LINE area up to
  128% of rotating odd part). Same reason GD's out-and-back is a hidden loop.
- "Symmetric part independent of discount" — only in continuous time; discrete:
  Sym(AΣ) = ½(BBᵀ + AΣAᵀ), off by 0.1–11%.
- Consolidation (stiffening protects the practised direction): weak, not specific (ν fell to
  0.77–0.93; rank correlation 0.50 vs 0.9 required).
- Fixed-budget order effects are not confined to weak directions at very short budgets.
- Overshoot speeds cyclic learning: only in a tight regime; hurts for similar tasks.
- Spec errors caught in review: 1% tolerance at ε=0.01 (Part 2); step size at edge of stability
  (partial-learning schedule); GN "90% in one step" test; fit threshold cutting off fine steps;
  H6d sampling; H7d zero-prediction pair; H9a/b literal formula for reversed rotation.

### 42. SPEC7 spec error

## 11. SPEC7 (Tests 10-12, sharpening) — IN FLIGHT, and a spec error of mine
Spec error : the nearest point NP is NOT the C=0 reference.
P_v(NP - GN) = LAW(dY dY^T): NP is equivalent to C = dY dY^T; GN (single min-norm jump) is C = 0.
So ordering of lambda_1 when G>0: GN < SL ~ GD < NP. First cell (h=128, seed 0, kappa=4): total +3.94 = position(NP) +6.33
+ travel -2.43 + route +0.04; in the GN frame SL-GN = +2.23, GD-GN = +2.27. Reading: the rise is carried by WHERE the
solution is (any nearby point with those outputs is sharper; the nearest is sharpest), and flow lands flatter than the
nearest point. "Travel causes progressive sharpening" survives only relative to the algorithm-defined GN jump. Wait for full tables.

### 43. Early partial X1 / X2 rows (superseded)

X1 so far (ZERO by subtraction, AJ(alpha), tanh-s0, n = 32, tau = 0.1): raw loss rises with alpha (this start has the unfavourable chance term);
 loss after rescaling: -3.4e-6, +1.9e-6, +1.5e-5, +3.5e-5, +9.1e-5 at alpha .5, 1, 1.5, 2, 3 relative to alpha = 0 -> an INTERIOR optimum near
 alpha ~ 0.4-0.5 (P1 "monotone to alpha = 2" fails in this start). Optimum f* looks n-dependent (SPEC10 n = 128: gain still linear up to f ~ 0.9-1).
X2 partial (stopped after 6 rows to free the CPU, to be rerun): tanh-s0, tau .5, ZERO: last-hidden probe mse .03419, .03430, .03436, .03437 and
 accuracy .8942, .8944, .8932, .8928 for f = -1, 0, .375, .43 (slightly WORSE with f), CKA with labels .4411 -> .4479 (more aligned), first hidden
 layer probe accuracy .8902 -> .8928 (better). Reading to test: alignment concentrates the last-layer features on the 32 training labels, which
 helps the network's own prediction from few examples and slightly hurts a readout refitted on 5000 examples (fits Geiger et al.: fully connected
 nets better lazy at 10k points).

### 44. SPEC8 interim result (superseded by the official result)

## 17. SPEC8 GPU run — INTERIM RESULT (2026-11-22, 5,115 of 7,180 cases, real MNIST, 0 failed cases)
APPROXIMATED from single-sign scalars; H5/H5b/H6/H0 are exact in-run values). Official verdicts need the notebook's Analysis with the vectors.
Complete: all 12 tanh and gelu combinations (398/398 each) + linear control. ReLU: 323 of 2,388 cases.
SMOOTH NETWORKS (tanh, gelu; init and trained; 3 seeds; MNIST, 196-256-256-10, N=320 outputs, P=118,794):
 H0 linear: 16/16, max 3e-13. H1x 216/216, H2x 360/360, H3x 192/192, H4x 192/192 (approx. even part): 960/960 (learner, direction) units match.
 Largest relative error of the extrapolated fraction: tanh 3e-5 (DJ(1.9): 1.2e-3), gelu 3e-4 (DJ(1.9): 1.4e-2); |cos| >= 0.9993.
 Examples tanh-trained-s0: SL(2) 0.2500, SL(32) 0.4844, DJ(1.5) -1.0000, DJ(1.9) -9.0000 (eigD -8.9996), GD(1.9) eigA -8.9998, HB(1,0.9) eigA -63.9643/-63.9643.
 Order check: error ratio s1/s2 = 4.00 (10-90%: 3.73-4.01), 100% inside [2.5, 6].
 H5 full path law incl. minibatch SGD: 72/72, cos 0.999945-1.000000, ratio 0.974-0.9999 (single size s1, not extrapolated). H5b max 0.0059.
 H6 PROX: 144/144 rho = 1.000000 (min 0.999993), cos 1. H7x: DJ(1) nr(s2) <= 0.0076; TH 1e-10.
 Scalar readouts (R3): R_lam1 and R_trK equal rho to ~1e-3 median -> the law predicts top-eigenvalue and trace shifts too.
 Generic targets (label/rand): GD and SGD even rho 0.50-0.53, cos 0.93-0.97 (area term); SGD within 0.005 of GD.
 Size ladder (label): SL(32) rho 0.48-0.51 at 0.05 S0; at 0.5 S0: 0.49-0.66 with cos 0.86-0.98 (init starts drift up to 0.62-0.66). Expectation "[0.4,0.6] up to 0.25" fails for gelu/init (0.618).
 Adam (lr 1e-3): |v_L|/|dtheta| = 0.99+ (displacement almost entirely vertical), 126-758x the NP shift, cos -0.22 (gelu) / ~0 (tanh): unrelated to the alignment direction.
RELU: the nearest-point solver does not converge (300 iterations, ~1,800 halvings, stat 1e-3..6e-2) in 286 of 323 cases; each such case costs ~400 s
 (33.5 GPU-h spent on 323 cases; ~50 more hours would be needed). The 37 converged cases (eigA, + sign, s1) do NOT follow the smooth law
 (SL(32) 0.73/0.58 vs 0.484, DJ(1) 0.31 vs 0, cos 0.5-0.94). => ReLU needs a different reference solver and probably a modified law (kinks); open, not refuted properly.

---

# PART 2 - THEORY AND DERIVATIONS, best checked first

### 1. Ground formula: f from the stability function; third order

ADDED BEFORE RUNNING E1/E2 (exotic-mathematics pass, 2026-11-23):
 GROUND FORMULA (the algebra; reproduces every closed form of the project): a learner that makes steps delta_k (fractions of the move, sum 1)
  with evaluation points alpha_k has f = 1/2 + sum_k (alpha_k - 1/2) delta_k^2 / (sum_k delta_k)^2 = coefficient of z^2 in its "stability
  function" R(z) = prod_k (1 + (1-alpha_k) delta_k z)/(1 - alpha_k delta_k z) (gradient flow: e^z, coefficient 1/2).
  Checks by hand: SL(m) 1/2(1-1/m); SLa(m,a) 1/2+(a-1/2)/m; DJ(c) (1-c)/(2-c); TJ(c,a) (1+k)/(2+k) with k = c(2a-1): all exact.
 THIRD ORDER in the doubled diagonal model: beta-shift proportional to e2 + e3 (z^2 and z^3 coefficients of R). Predicted ratios with NO fit:
  SL(2):SL(4):SL(8) = .457:.800:1 and DJ(1.5)/SL(8) = -2.438; T1 measured .456:.799:1 and -2.44 (run before this line was written; the formula
  was derived after seeing them, so this is an explanation, not a prediction).
 TREE COUNT (B-series): one rooted tree with 2 nodes -> ONE learner number at second order (the law). Two trees with 3 nodes (chain, bushy) ->
  at third order learner-dependence should need at most two or three functions, weighted by e3 (chain) and b = 1/2 sum_j delta_j (sum_{i<j} delta_i)^2 (bushy), plus e2.

### 2. Look-position rule, closed forms for GD / momentum / look-ahead, look matrix, heat, hover (sec. 45)

NEW MATHEMATICS (derived while writing the spec)
 LOOK-POSITION RULE. For a learner whose weight change on a leg is a sum of gradients, sum_j J(y_j)^T c_j, on one kernel mode:
   f = sum_j w_j e_j, e_j = position along the journey (0..1) of the outputs at the look point y_j, w_j = share of the journey paid by
   gradient j. Loop law then gives rho = 2f - 1. Reproduces every earlier closed form (exact, implicit, midpoint, SL(m), GD, and the
   library's heavy-ball formula, the last one algebraically).
 CLOSED FORMS (a = 1 - 2f): GD c/(2-c); two-gradient midpoint -c^2/(c^2-2c+4); heavy ball c(1+mu)^2/[(1-mu)^2(2(1+mu)-c)]
   (= (1+mu)/(1-mu) times GD at equal long-run rate, to leading order); look-ahead by g of the last step: leading order removed in every
   mode at g* = (1+mu)/(2(1-mu)), sign flips beyond; stable for c < 2(1+mu)/(1+2g).
 LOOK MATRIX (any direction): displacement = LAW(C), C = K0 sum_j c_j E_j^T (payment x look position), closing steps included.
 HEAT = IMPLICIT GRADIENT REGULARISATION ON THE FIBRE (hand check at leading order). HOVER: for explicit steps a = sum of squared step
   fractions, so jitter around a target adds wear without end.

### 3. Teardown to ground and the solvable model

## 40. TEARDOWN TO GROUND (2026-11-23): "COMPOUNDING" - derivation and predictions written BEFORE the runs
WHY-CHAINS (every finding followed down; where each one stops):
 * Why do learners with equal training outputs differ? P > N: the fibre has dimension P - N. Ground: finite data underdetermine the function.
 * Why does fibre position matter off the training set? J_x is not in the row space of J. Ground: same.
 * Why do GN / flow / nearest point differ? The kernel changes along the fibre (weight->output map is not a Riemannian submersion). Why does it
   change? Outputs are PRODUCTS of weights (layers): the sensitivity of the output to one weight is proportional to the other weights on its
   path. Ground: multiplicative structure. (Linear-in-parameter model: no products, all learners coincide - our linear control, 1e-13.)
 * Why one number? Straight output path + second order: the only freedom is WHERE along the step the training-side gradients are evaluated;
   the test-side gradient is always integrated along the path (the common term R/2). Ground: Taylor / quadrature.
 * Why 0, 1/2(1-1/m), 1/2, 1? Growth-on-growth: (1 + r) simple, (1 + r/m)^m = 1 + r + 1/2(1-1/m) r^2, e^r = 1 + r + r^2/2, 1/(1-r) = 1 + r + r^2.
   The alignment fraction IS the second-order compounding coefficient. Ground: arithmetic of compounding.
 * Why is the leading effect odd under a readout flip and removed by doubling? Two-weight feedback (readout <-> incoming weight) is a
   hyperbolic rotation: first-order growth is odd (sinh-like), growth-on-growth is even (cosh-like); the output multiplies one of each.
   Ground: exact symmetry theta'(tau) = S theta(-tau) (sec. 39, checked numerically).
 * Why does midpoint = flow at any step size, explicit below, implicit above? The implicit midpoint rule conserves quadratic invariants
   exactly; explicit Euler shrinks them, implicit Euler grows them (two-factor model: Q+ = Q(1 - eta^2 g^2), Q, Q/(1 - eta^2 g^2)).
   So f - 1/2 = signed violation of the layer-balance conservation laws, EXTENDED to fibre directions that have no symmetry behind them.
   Ground: geometric numerical integration.
 * Why loops/holonomy, and why zero in diagonal models? Compounding in different output directions does not commute when layers are matrices.
   Ground: matrix non-commutativity.
 * Why are weak kernel directions 50-60x more active? xi = J^T K^-1 e is large there: more weight motion per unit output = more compounding.
SOLVABLE MODEL (the algebra, to be checked numerically): diagonal network beta = u^2 - v^2 (itself a doubled model), u = v = alpha at start,
 X beta = y underdetermined. alpha-point jump with evaluation point f: with s = X^T nu,
   u = alpha (1 + 2(1-f)s)/(1 - 2fs), v = alpha (1 - 2(1-f)s)/(1 + 2fs),
   beta/alpha^2 = 8 s + 32 f(1+f) s^3 + ... u v = alpha^2 (1 - 4(1-f)^2 s^2)/(1 - 4 f^2 s^2).
 (i) SECOND order: only u v changes (= the quantity gradient flow conserves; a symmetry direction, invisible in the function): rises with f.
 (ii) THIRD order: the function changes: the link s -> beta is linear for f = 0 (exact minimum-L2 interpolator), super-linear for f > 0
      (large coefficients amplified = sparser solution; the learner minimises a penalty that grows more slowly than L2), sub-linear for
      -1 < f < 0 (denser than L2), back to L2 at f = -1.
 => the alignment dial is a dial of the implicit penalty between dense and sparse AT FIXED INITIALISATION SCALE (Woodworth's dial is the
    initialisation scale). This would explain: third-order systematic effect in doubled nets; "shrink + pattern toward labels"; interior
    optimum that moves up with n (sparse recovery needs enough samples); worse generic features (concentration).

### 4. What is now the ground; mathematical homes

WHAT IS NOW THE GROUND (the reading): outputs are products of weights -> learning compounds -> learners differ in how they approximate
 compound growth (their stability function; flow = e^z). f - 1/2 is at once: the compounding deficit, the violation of the conserved
 quantities of flow (exact in solvable models), the rate at which unused kernel directions lose trainability (1 - 2f = 1/N for N sub-steps:
 the a + b/N law of the plasticity run, sec. 10-12), and the leading learner-dependence of predictions. Higher orders: rooted trees.
MATHEMATICAL HOMES (structural, each checked against a number we have): (1) signature / Chen-Fliess series: path matrix C = level-2
 signature, antisymmetric part = Levy area; free-Lie-algebra count N(N-1)/2 = 120 for N = 16 = the "120 by loops" of sec. 4; loop scalings
 eps^2, eps^3, eps^4 = ball-box theorem of sub-Riemannian geometry. (2) B-series / Butcher trees: 1 tree at order 2 (one number), 2 at
 order 3 (e2: three numbers, confirmed). (3) Pade approximants of e^z: explicit = (1,0), implicit = (0,1), midpoint = (1,1); the pole of
 the stability function = where the small-move law fails (DJ(1.9) worst unit; alpha-point pole at 2 f s = 1). (4) symmetric composition:
 doubling = averaging a method with its mirror image, which cancels even orders (retraced star eps^4; figure-eight -> 0).
 HUNCHES (not tested): heavy-tailed weight spectra from compounding (Kesten / Martin-Mahoney); log-normal synapses and multiplicative
 plasticity; Amari alpha-connections (pre/post-point = dual projections); post-point = isothermal (Hanggi-Klimontovich) convention.

### 5. Second-order derivation and the readout-flip symmetry

## 32. THEORY STEP + X3 "DOUBLED NETWORK" (local, real MNIST) - written BEFORE the X3 run (2026-11-22 late night)
DERIVATION):
 v_NP = tau^2 V0 H_a xi (the nearest-point shift is the Hessian of the dual-weighted training function applied to the first-order step, projected).
 Learner L on unseen x: g_L(x) = tau J_x xi + tau^2 { 1/2 R(x) + f_L G1(x) }, R(x) = xi^T H_x xi - J_x J^+ (xi^T H_X xi), G1(x) = J_x V0 H_a xi.
 Kernel regression with the tangent kernel taken at fraction s of the path: tau J_x xi + tau^2 { s R(x) + s G1(x) }.
 => gradient flow (f = 1/2) IS kernel regression with the MIDPOINT kernel to this order; every other learner = that + (f - 1/2) tau^2 G1(x).
SYMMETRY : flipping the sign of the readout layer (theta -> theta', F -> -F) leaves K, a and the first-order predictor unchanged and flips
 the sign of every second-order term (R, G1). So over random initialisations the second-order learner-dependent term has NO preferred sign;
 the growth of the start function (eN > 0 always) is exactly such an odd term. A sign-definite effect of f can only enter at order tau^3
 (through the second-order, label-aligned part of the kernel change, which is even).
 CONSISTENT WITH DATA ALREADY IN HAND: small moves (tau = 0.01) b_ZERO has mixed sign (5 negative, 3 positive of 8); HZ1 holds in 5/8 at tau = 0.1 and
 7/8, 8/8, 7/8 at larger tau; drest grows faster than tau^2.
 COROLLARY: in a DOUBLED network F_D = (F(theta_a) - F(theta_b))/sqrt2 with theta_a = theta_b at the start (Chizat's unbiasing trick), the
 second-order shift is (H xi0/2, H xi0/2): the same in both copies, hence invisible in the function at EVERY input (G1 = 0 identically). The
 learner-dependence of predictions then starts at order tau^3 and should be sign-definite.

### 6. The physics, compactly (early)

## 3. The physics, compactly

Fixed training inputs X. F(θ) = outputs on X, J = ∂F/∂θ, K = J Jᵀ (tangent kernel).
"Vertical" = weight directions that leave F unchanged (P_v = I − JᵀK⁻¹J).
ξ_e = JᵀK⁻¹e = min-norm weight step moving F by e.

- **Path law (Part 4, verified):** for any learner whose steps lie in row(J) (GN, GD, momentum),
  the vertical weight shift is, to second order, V = LAW(C) = Σ_ij C_ij T_ij with
  T_ij = P_v[Dξ_{e_i}·ξ_{e_j}] and C = Σ_steps (F_{k+1}−F_k) ⊗ (F_k−F_start)
  (momentum: "committed increment" ⊗ position). Cosine ≈ 0.999, error ∝ loop size.
  Fails for Adam as written.
- **T is known:** J_x·T_ij = K⁻¹K⁻¹[dNTK with one test index − kernel-explained part]
  (Roberts–Yaida–Hanin dNTK); identity checked to 1e-9.
- **Three forces (verified as exact identity, 1e-9):** C = ½ΔYΔYᵀ − ½Q + Area, where ΔY = net
  output displacement, Q = Σ squared jumps (jitter), Area = antisymmetric part.
  1. displacement → descent on ¼ΔYᵀK⁻¹ΔY (kernel aligns with travel);
  2. jitter → ascent on ¼tr(QK⁻¹) (wiggled directions stiffen);
  3. area × Lie bracket [ξ_u, ξ_v] → non-gradient (curl measure 1.0).
  Symmetric part = −P_v∇[½tr(C_sym K⁻¹)].
- **Jitter rule:** outcome depends on where Q falls. Label noise through GD: Q ∝ K² → shrinks
  tr K (flatness; reproduces Li–Wang–Arora, cos 0.996). Top-mode overshoot → lowers λ_max.
  Uneven label noise → noise-weighted trace (Xu et al.). Exact (unfiltered) target jumps → hit
  weak directions → collapse of λ_min (plasticity loss).
- **Kernel filter:** GD moves outputs by −(η/n)K r (suppressed in weak directions); GN by −r.
- **Input loops (Part 1–2, linear-model theorem + nets):** for a closed polygon of tasks made by
  shifting the INPUTS, rotation = signed polygon area × P[D_v,D_u]P·(θ−θ*); shrinkage ∝ Σ
  squared edge lengths; exists even in linear models.
- **Label loops (Part 3):** order-odd shift = (ε²/2)[ξ_u,ξ_v]; zero in linear models; ∝ width^−0.47;
  bracket 50–60× larger between weak kernel directions than strong ones.

### 7. Why subtracting the start function cannot stop its growth

 KEY POINT understood while writing it: subtracting the start function does NOT stop it from growing with f (the weights still carry it; D is the
 same for both models); it removes the |Nv|^2 term from gN, leaving gN^Z = (<Sv,Nv> - <Y,Nv>)/N, a chance-sign term. That is exactly why the local
 teacher-student test turned 36/40 into a coin flip. The gain must come from dS + drest.

---

# PART 3 - PREDICTIONS AND HYPOTHESES

## 3A. Predictions already tested, best-holding first

### 1. SPEC11 follow-up predictions (collapse, EM, mixed loops, EMS, healing, history, triangle, EP, a-meter)

## 43. SPEC11 RIG RESULT (2026-11-23, RTX 3090, 174 units, 15.2 min, 0 errors) AND THE LINEAR-DEPLETION LAW
PREDICTION WRITTEN BEFORE THE COLLAPSE RUN (gelu-init-s0, learner DJ(1.5), the rig's star continued with new
 directions E7.. = label_dir(8000+7), gauss_dir(9000+8), label_dir(8000+9), gauss_dir(9000+10), size s1):
 From the rig data the smallest kernel eigenvalue of every learner lies on ONE straight line lamN/lamN0 = 1 - t/36.5 with
 t = (1 - 2f) x (number of excursions) (30 points, 5 learners, R^2 0.994). DJ(1.5) has t = 3 per excursion and stood at 0.097 after 12
 excursions (t = 36). K1: after pair 7 (t = 42) lamN/lamN0 <= 0.05; K2: the learner stops (a leg not converged, not closed, or a
 numerical exception) at pair 7, 8 or 9. If instead lamN levels off above zero, the straight line holds only away from zero (an avoided
 crossing) and there is no finite-time collapse at this size.
SECOND PREDICTION, WRITTEN BEFORE ITS RUN : TWO-GRADIENT EXPLICIT LEARNERS WITH f = 1/2.
 Reasoning: the alignment fraction is the Runge-Kutta sum A2 = sum b_i c_i; any second-order Runge-Kutta rule has A2 = 1/2 with NO implicit
 solve. EM (explicit midpoint): th_h = exact jump from the start to the half-way target (start Jacobian); then the full move from the
 start using the Jacobian AT th_h (jump_a(prob, th0, t, Jb = J(th_h))). HEUN (trapezoid): th_p = exact jump to the full target; then the
 full move from the start using 1/2 (J(th0) + J(th_p)). Both cost two Jacobians and two solves, like SL(2) (f = 1/4, loop -0.5).
 EM1 single-leg fraction within 0.02 of 0.5 at s2 for both. EM2 loop: |rho_loop| <= 0.05 at s1 and smaller at s2, for both (not exactly
 zero like the implicit midpoint; a higher-order remainder). EM3 own-cost ratio |dq_u / M_uu| <= 0.05. If these hold, the leading loss of
 trainability under back-and-forth targets is an artefact of FIRST-order integration and disappears for any second-order explicit rule.
THIRD PREDICTION, WRITTEN BEFORE ITS RUN : THE MIXED-LEARNER LOOP LAW (from time-reversal symmetry).
 Reasoning: out leg contributes f_out s^2 uu^T to the path matrix, back leg (-1 + f_back) s^2 uu^T, so the loop displacement is
 (f_out + f_back - 1) vNP. The implicit jump is the ADJOINT (time-reverse) of the explicit jump (f -> 1 - f), so going out with one and
 back with the other must be exactly reversible. Predictions (tanh-s0, direction E1, size s1, even part):
 MX1 out DJ(1) / back AJ(1): |rho_loop| <= 0.03. MX2 out AJ(1) / back DJ(1): |rho_loop| <= 0.03.
 MX3 out SL(2) / back AJ(1): rho_loop = +0.25 +- 0.03. MX4 out AJ(1) / back SL(8): rho_loop = +0.4375 +- 0.03.
 If these hold: loss of trainability = broken time-reversal symmetry of the learner, and alternating explicit and implicit phases
 (a symmetric composition) costs nothing.
 MX0 (added before the run): out SL(1) (the exact jump, f = 0) / back AJ(1): the implicit jump is the exact adjoint of the exact jump,
     so the return must be exact to rounding at ALL orders: |v| / |vNP| <= 1e-6 (not merely small).
FOURTH PREDICTION, WRITTEN BEFORE ITS RUN : THE TWO-GRADIENT LEARNER AT LARGE MOVES (SPEC11 Part C protocol:
 six full relabelling tasks, same task seeds, same probes W1..W8, final true-label task with SL(8)). Learner EMS(m): m sub-targets, each
 reached by the explicit-midpoint rule (2 Jacobians and 2 solves per sub-target, so EMS(m) costs the same as SL(2m)). Rig values of
 Q6 = mean log cost of fresh directions after task 6 (lower = more trainable): tanh SL(2) -2.535, SL(4) -2.668, SL(8) -2.814,
 implicit SLa(8,1) -3.213; gelu SL(2) -2.163, SL(4) -2.239, SL(8) -2.428, SLa(8,1) -3.008.
 EC1 (tanh): Q6(EMS(m)) < Q6(SL(2m)) for m = 1, 2, 4. EC2 (tanh): Q6(EMS(4)) <= -3.0. EC3 (gelu): as EC1.
 EC4 (trade-off; tanh): final true-label test loss of EMS(4) is ABOVE SL(8)'s 0.04289.
 Confidence: EC1 for m = 2, 4 about 70%; m = 1 (one midpoint step for a full relabelling) about 50%: the move is far outside small-move theory.
FIFTH PREDICTION, WRITTEN BEFORE ITS RUN : HEALING A WORN NETWORK WITHOUT CHANGING ITS OUTPUTS.
 Start from the gelu network left by the collapse run (after its last completed pair of DJ(1.5) excursions). Apply out-and-back
 excursions with the implicit learner AJ(1) along NEW random directions (label_dir / gauss_dir seeds 8000/9000 + 21..), both signs each,
 size s1. Outputs return to F0 each time. The one-clock law says each AJ(1) excursion is t = -1.
 HL1 every leg converges and closes (close <= 1e-6) for the first 4 pairs. HL2 lamN/lamN0 rises by at least +0.03 per pair on average
 over the first 4 pairs (law: +0.055). HL3 mean q0/q rises by at least +0.02 per pair (law: +0.033).
 If HL2/HL3 fail while HL1 holds, the recharge works only on unworn networks and "healing" is wrong as stated.
SIXTH, WRITTEN BEFORE ITS RUN : WHY DID EARLIER TASKS CHANGE LATER GENERALISATION?
 Four earlier tasks on the same 32 images, then the true 10-class labels learned with SL(8); test loss and accuracy of the final network.
 STRUCTURED history: labels y mod 2, y div 5, y mod 3, y div 2 (coarse versions of the true classes, one-hot in the 10 outputs).
 SHUFFLED history: the same four label vectors with the images permuted (same label counts, no relation to the images); 3 permutations.
 Learners for the history: SL(2) (explicit) and EMS(1) (two-gradient, time-symmetric at leading order, same cost). Baseline: fresh network.
 Two readings: WEAR (explicit steps stiffen weak directions; label-independent) predicts HS1: explicit history lowers the final test loss
 below fresh for structured AND shuffled histories on tanh, and the explicit-minus-symmetric gap is about equal for both.
 IMPRINT (the kernel keeps the structure of what was travelled) predicts HS2: structured beats shuffled for both learners.
 Both can hold. No confident prediction for which is larger; this is a measurement with the alternatives stated.
 ADDED WHILE hist11 WAS RUNNING, BEFORE READING ANY OF ITS OUTPUT (from the "state, heat, area" reading below):
 For a time-symmetric learner the symmetric part of the path matrix is 1/2 dY_net dY_net^T, a function of the NET output change only,
 so the route should not matter: HS5: EMS(1) histories, structured or shuffled, end within 3% of the fresh network's final test loss.
 For an explicit learner the extra term is -1/2 Q (sum of squared jumps): it stiffens most the directions that were wiggled.
 HS4 (risky, counter-intuitive): with SL(2), the SHUFFLED history gives a LOWER final test loss than the STRUCTURED history, because its
 wear falls on label-noise directions while the structured history wears the class directions the final task needs.
SEVENTH PREDICTION, WRITTEN BEFORE ITS RUN : STATE, HEAT, AREA SEPARATED ON MNIST.
 Triangle of targets F0 -> F0 + s u -> F0 + s w -> F0 (u = E1, w = E2, s = s1) and the same triangle in the opposite order.
 v_fwd, v_rev = vertical displacement after each. EVEN part = (v_fwd + v_rev)/2 = "heat" (route length, learner-dependent);
 ODD part = (v_fwd - v_rev)/2 = "area" (orientation-dependent, should not depend on the learner).
 Heat prediction: (f - 1/2) x Vsum, Vsum = vNPe(leg 1) + vNPe(leg 2) + vNPe(leg 3) (nearest-point shifts of the three legs).
 TR1 EM (two-gradient, f = 1/2): |even| / |odd| <= 0.05, or |even| / |Vsum| <= 0.01 if the odd part is itself tiny.
 TR2 SL(2): even part = -0.25 x Vsum within 0.03, cosine <= -0.99. TR3 DJ(1)... not run; AJ(1): even part = +0.5 x Vsum within 0.05.
 TR4 the odd parts of EM, SL(2) and AJ(1) agree pairwise (cosine >= 0.95, norms within 15%).
 TR5 (measurement): size of the area part relative to |Vsum|.

EIGHTH PREDICTION, WRITTEN BEFORE ITS RUN : AN EXPLICIT LEARNER WITH f > 1/2 ("predicted end point").
 EP(alpha): th_p = exact jump from the start to the target (start Jacobian); then the full move from the start using the Jacobian at
 th0 + alpha (th_p - th0). Two Jacobians, two solves, no fixed-point iteration. By the evaluation-point law f = alpha at second order
 (alpha = 1/2 is the EM learner already tested; alpha = 1 imitates the implicit jump; alpha = 2 goes beyond it).
 PE1 (tanh E1 s1, even part): loop displacement rho = 2 alpha - 1 within 0.05 max(1, |2 alpha - 1|): EP(1) -> +1, EP(2) -> +3.
 PE2 (healing, worn gelu network after the 20 wearing excursions, new random directions, size s1, EP(1)): every leg converges and closes
     for 4 pairs; lamN/lamN0 rises by >= +0.03 per pair on average (one-clock law: +0.055); mean q0/q rises by >= +0.02 per pair (0.033).
 If PE1 holds and PE2 fails: recharge works on fresh networks only, and wear is not simply reversible by unrelated excursions.
NINTH PREDICTION, WRITTEN BEFORE ITS RUN : THE TIME ASYMMETRY OF ORDINARY OPTIMISERS ("a-meter", ladder rung 1b).
 Out-and-back excursion (tanh-s0, direction E1, size s1, both signs) carried out by full-batch first-order optimisers on the squared
 loss, each leg run until |F - t| <= 1e-3 s (or a step cap). Read-out a = -rho_loop (1 for the exact jump, 0 for a time-symmetric learner).
 GD(c): step = c N / lambda_1. GDmid: each step uses the gradient at the half-way point of a trial step. ADAM(lr): standard Adam, state
 reset at each leg. ADAMmid: trial Adam step, gradient re-evaluated half-way, moments updated once with that gradient, then the step.
 AM1 gradient descent is nearly time-symmetric on a generic direction: a(GD(1)) and a(GD(1.8)) in [0, 0.15] (per-mode asymmetry
     c_i / (2 - c_i) is tiny in the weak kernel directions that dominate the shift).
 AM2 Adam is strongly asymmetric: a(ADAM) >= 0.5 at both learning rates (it takes near-full steps in weak directions; in the solvable
     model its fraction was -1.8 to -211).
 AM3 the midpoint versions cut the asymmetry: a(GDmid) <= 0.5 a(GD), a(ADAMmid) <= 0.5 a(ADAM).
 If AM2 fails (Adam about symmetric here) the Adam-versus-SGD plasticity gap needs another explanation in this regime.

### 2. X8 predictions (three-function law)

## 41. TEST 1 (X8): THREE-FUNCTION LAW WITH MORE LEARNERS AND STARTS - predictions written BEFORE the run (2026-11-23)
WEIGHTS, computed for every learner from its step sizes d_k and evaluation points a_k by the recursion (c = sum of earlier steps, q = A2 of
 earlier steps): A2 = sum d_k (c_k + a_k d_k); A3 = sum d_k [q_k + a_k d_k (c_k + a_k d_k)]; B3 = 1/2 sum d_k (c_k + a_k d_k)^2.
 Explicit learners: A2 = e2, A3 = e3, B3 = b (as in X7). SL(m): d = 1/m, a = 0. SLa(m, a): d = 1/m. AJ(a) = one step. DJ(c): d_k = c(1-c)^k,
 a = 0. TJ(c, a): d_k = (1-r) r^k with r = (1-(1-a)c)/(1+ac), evaluation point a.
PREDICTIONS:
 R1 explicit learners only (8): the difference vectors have exactly three significant singular values (sigma_4/sigma_1 <= 0.002) in all
    three starts at both sizes, and the three-function model with weights (A2, A3, B3) leaves <= 0.1% of sum |D|^2.
 R2 the same holds with the implicit learners added (13 learners): sigma_4/sigma_1 <= 0.005 and residual <= 0.5%. Confidence ~50%: the
    implicit learners may need a fourth function (then the count of invariants at third order is larger than three; that is a finding).
 R3 the singular-value ratios are the same at tau .03 and .1 (within 10%), i.e. the structure is of one order.
 R4 gelu obeys the same law (rank 3, same weights) although its sizes differ.

### 3. SPEC11 claims C1-C5

## 42. SPEC11 "TRAINABILITY AND THE EVALUATION POINT" (pilot for Colab) - spec, pre-check, code, review (2026-11-23)
CLAIMS (all parameter-free): C1 out-and-back excursion by a learner with fraction f displaces the weights by (2f - 1) vNP (vNP = nearest-
 point shift of one leg); C2 vNP = -(s^2/2) V grad q_u, q_u = u^T K^-1 u; C3 cost of any direction w changes by (1 - 2f) 2<vNP_w, vNP_u>/s^2;
 C4 excursions add; C5 (large moves, ordinal) larger f -> lower cost of fresh directions after full relabelling tasks.
 Derivation: path matrix of an out-and-back = (2f - 1) s^2 u u^T (out leg f s^2 uu^T, back leg -s^2 uu^T + f s^2 uu^T).

### 4. X4 prediction (midpoint kernel)

X4 (identity check, written before the run): "gradient flow = kernel regression with the midpoint kernel at second order", equivalently
 KR_mid(x) = 1/2 (g_TH(x) + g_NP(x)) + O(tau^3) for the subtraction model. Script, MNIST n = 32, tau .03 and .1. PREDICTION: the relative error
 |KR_mid - (g_TH + g_NP)/2| / |g_NP - g_TH| is <= 0.15 at tau = .03 and grows about 3.3x from .03 to .1 (first order in tau), while the same error
 with the START kernel (KR_0) is of order 1 at both sizes; also KR_end - g_NP is of order 1 (the nearest point is NOT end-kernel regression).

### 5. X3 doubled-network predictions (D1 held, D2 failed)

X3 doubled problem built on the library's Problem class from the stored start (theta_a = theta_b = theta0),
 target tau*Y, learners DJ(1) [f 0], SL(2) [.25], SL(4) [.375], SL(8) [.4375], and DJ(1.5) [-1] at tau .1 and 1; tau in {.01, .03, .1, .3, 1};
 the same learners on the subtraction model for comparison. n = 32 starts.
PREDICTIONS: D1 exponent of |F_test(SL(4)) - F_test(DJ(1))| in tau between .01 and .1: doubled 2.7-3.3, subtraction 1.8-2.2, in 4/4 starts.
 D2 doubled: loss[SL(4)] < loss[DJ(1)] in at least 17 of 20 blocks (4 starts x 5 tau), including at least 10 of the 12 with tau <= 0.1;
 loss ordered SL(8) < SL(4) < SL(2) < DJ(1) in at least 14 of 20; loss[DJ(1.5)] > loss[DJ(1)] in at least 7 of 8.
 D3 doubled, tau <= 0.1: the gain grows ~tau^3 (ratio of gains at .1 and .03 between 20 and 60).
 If D1 fails (exponent 2 in the doubled model) the symmetry argument is wrong. If D1 holds and D2 fails, the third-order term has no fixed sign either.

### 6. T1 / T2 / E1 / E2 predictions (mixed)

PREDICTIONS, T1 (d = 400, alpha = 1, 20 seeds, n in {30, 60, 120}):
 D0 numerics reproduce the two series (s^3 coefficient 32 f(1+f); u v formula) to 3 digits.
 D1 sparse truth (8 nonzeros): recovery error of the alpha-point learner is ordered by f(1+f): worst at f = -0.5, f = -1 within 2% of f = 0,
    falling monotonically for f = 0 -> 0.5 -> 1 -> 1.5 in >= 18/20 seeds at every n.
 D2 dense truth (Gaussian): error RISES for f = 0 -> 1.5 in >= 18/20 seeds.
 D3 mixed truth (sparse + dense part with 30% of the energy) with noise: the best f is interior at n = 30 and moves UP with n. (Moderate, ~55%.)
 D4 iterated learners in the same model (SL(m), DJ(c) built from Gauss-Newton jumps): end-point differences in beta against DJ(1) are THIRD
    order in the move (exponent 2.8-3.1) and for sparse truth SL(8) has lower error than DJ(1) and DJ(1.5) higher, as in X3b.
PREDICTIONS, T2 on real MNIST (n = 32, ZERO target, tau in {0.5, 1}, learners f = -1, 0, .25, .375, .4375; 8 blocks):
 M1 participation ratio of the last-hidden units' contributions to the learned output change on test images FALLS with f in >= 6/8 blocks
    (the learned function is carried by fewer units).
 M1b participation ratio of the per-unit readout change falls with f in >= 6/8.
 M3 rich-get-richer: across units, the f-slope of the readout norm correlates positively (Spearman) with the unit's readout norm at the
    start in >= 6/8.
 M2 (measurement, no prediction): f-slope of the layer-balance sum (readout norm^2 minus incoming norm^2 over last-hidden units).

 E1 PREDICTION (doubled MNIST net, tanh-s0, tau .03 and .1, learners DJ(.5), DJ(1.5), SL(2), SL(4), SL(8) against DJ(1)):
  E1a the best ONE-function model (D_L = e2 A) leaves at least 3x the squared residual of the two-function model (e2 A + e3 B);
  E1b the two-function model leaves <= 2% of sum |D|^2 at tau = .03;
  E1c singular values of the five difference vectors: sigma_3/sigma_1 <= 0.05 at tau = .03 and falls from tau .1 to .03, sigma_2/sigma_1 does not.
  Confidence ~50%: a third function (bushy weight b) may be needed.
 E2 PREDICTION ("extrapolated pair": out* = out_DJ(1) + lam (out_SL(4) - out_DJ(1)), SPEC10 bundles, ZERO model): for the n = 128 blocks with
  tau in {0.5, 1}, test loss is lowest at lam > 1 among {1, 1.5, 2, 3} in >= 6/8 blocks and accuracy at lam = 2 >= accuracy at lam = 1 in >= 6/8.

### 7. g3_parts / g4_ends predictions; zero-model prediction

PREDICTIONS WRITTEN BEFORE THESE TWO RUN:
 g3_parts: (i) trained starts: eN > eS in >= 20/24 rows. (ii) 'true' direction, init: cS1 < 0 in 4/4 (the nearest-point shift takes back part
  of the first-order change on unseen inputs), and first-order tau* smaller for gelu (0.3-0.6) than for tanh (1.5-3.5) (from the G4 n32 ratios above
  after subtracting the |G|^2 term). (iii) |z_lab| > 2 in most MNIST rows (10 000 test images make even cos 0.01 significant) - so z only says
  that G is related to the labels at all, the SIGN columns carry the information.
 g4_ends: 'size' reading = eN > 0, eS > 0 for f > 0, gN > 0, turn-round where gS < 0 and |dS| > dN, loss* then no longer prefers larger f.
  ALREADY DOUBTFUL before the run: the local tiny test (random labels) gave eN > 0 but eS < 0 (the learned change SHRINKS with f on unseen inputs).
  If eS < 0 on MNIST too, the picture is "larger f = start function larger + less change off the training points", and the n128 turn-round must
  then come from drest or from gS > 0 (learned part too large). If loss* and accuracy improve with f at tanh n128, the effect is more than size.

NEW PREDICTION from this (written before the local test below): remove the start function (model g(x) = f(theta, x) - f(theta0, x)) and larger f
 should LOWER test loss at every move size.

### 8. Predictions before G4 (mixed)

PREDICTIONS WRITTEN BEFORE SEEING G4: Spearman(f, test loss) > 0 at tau = 1; lambda1 and trK increase with f; |theta_L-theta_0|^2 excess over AJ(1)
  proportional to (1-f)^2 (DJ(1.5) farthest); test ACCURACY need not follow (argmax ignores output size).

### 9. X1 beyond the nearest point (failed)

## 30. X1 "BEYOND THE NEAREST POINT" (local, real MNIST, CPU) - predictions written BEFORE the run (2026-11-22 night)
PREDICTIONS: P1 loss after the best one-number rescaling falls monotonically with alpha from 0 to 2 in at least 3 of the 4 n = 32 starts at tau = 0.1.
 P2 raw ZERO test loss: minimum at alpha >= 1 in tanh-s1 and gelu-s1 (the two n32 starts whose slope was negative in SPEC10), not in the two seed-0
 starts (chance term). P3 accuracy at alpha = 2 >= accuracy at alpha = 0 in at least 3 of 4. P4 (b): among converged SLa(8, alpha) learners the
 raw loss at f = 1.0 is below that at f = 0.4375 in at least 3 of 4 starts; an optimum at finite f > 1 may appear (squared term).
 Uncertain: whether alpha > 1 solves converge at tau >= 0.1 and whether alpha >= 4.5 sub-jumps converge at all.

### 10. X5 predictions (mostly failed)

## 39. X5 "IS ALIGNMENT SPECIALISATION?" - predictions written BEFORE the run (2026-11-23 morning)
WHY: X2 (sec. 38) says hidden features get slightly worse for a 5,000-label readout as f rises; X3b (sec. 36) says the network's own predictions
 get better (doubled model). Candidate reading that would make both true: v_NP is ascent on a^T K a inside the fibre = it separates the training
 classes in feature space and co-adapts features with the network's own readout ("specialisation to the training labels", the direction of
 neural collapse). Three independent probes of that reading in one run, plus the exact symmetry behind the odd/even structure.
SYMMETRY (derived on paper, exact): S = negate the last layer (weights and bias). F(S theta) = -F(theta). A network started at S theta_0 that
 learns +dY is the S-image of the original network learning -dY: theta'_L(tau) = S theta_L(-tau). So every term of even order in tau keeps its
 hidden-layer part and flips its readout part; in particular the f-dependent shift f tau^2 v_NP: hidden part EVEN, readout part ODD, and its
 effect on any output G(x) ODD. (This is why doubling cancels G and why a readout-blind feature probe sees one sign in every start.)
PREDICTIONS (slope = straight-line slope against f over the five learners):
 F1 flip check (2 starts, tau .03 and .1, D = theta_SL(8) - theta_DJ(1)): cos(D'_hidden, D_hidden) > +0.9, cos(D'_readout, D_readout) < -0.9,
    cos(test-output difference', test-output difference) < -0.9 at tau = .03; departures from +-1 shrink from tau .1 to .03.
 N1 within-class / between-class scatter of last-hidden features on the 32 TRAINING images falls with f in >= 7/8 blocks.
 N2 the same ratio on the 10,000 TEST images falls with f in the 4 tanh blocks; gelu not predicted (leaning: rises).
 K1 probe error of a readout fitted on k = 30 labels (best ridge strength, chosen on the f = 0 learner) FALLS with f in >= 6/8 blocks.
 K2 at k = 5,000 it RISES with f in >= 7/8 (replication of X2). K3 slope at k = 30 is below the slope at k = 5,000 in >= 7/8.
    The confidence in K1 is moderate (~55%). If the slope is positive at every k, "features specialised for few labels" is wrong and the
    gain must sit in the co-adaptation of features with the trained readout (W2).
 W1 swap test (readout of learner a on hidden layers of learner b; pair DJ(1) / SL(8)): total change = features part + readout part to within
    20% of |total| in >= 6/8 (interaction is higher order).
 W2 the features part has the sign of the total change in >= 6/8 and is the larger of the two parts in >= 6/8 (i.e. the same feature shift
    that hurts a data-rich readout helps the network's own readout).

### 11. X2 feature-probe predictions (failed)

## 31. X2 "FEATURE QUALITY BY LINEAR PROBE" (local, real MNIST) - predictions written BEFORE the run
Literature search : Arbel 2302.02904 measures feature quality by test loss after refitting the last layer and
 says the GN step-size effect "lacks a theory, no formula"; nothing found on implicit/midpoint Gauss-Newton for networks, on centring changing
 which optimizer generalises best, on catapult for centred models, or on moving past the nearest-to-initialisation solution in nonlinear nets
 (linear analogues: Freeman 2510.19206 inflating the min-norm interpolator; negative optimal ridge, Kobak 1805.10939, Wu-Xu 2006.05800).
 COUNTER-EVIDENCE to keep in view: Geiger et al. 1906.08034 (fully connected nets do better LAZY on Fashion-MNIST with 10k points, gradient flow);
 Yeom et al. 2602.00827 (interior optimum of feature-learning strength: "over-alignment").
PREDICTIONS: Q1 probe accuracy of DJ(0.25) > DJ(1) in at least 12 of 16 (start, tau) blocks at n = 32 for ZERO targets; Q2 DJ(1.5) < DJ(1) in at
 least 12 of 16; Q3 the same two orderings for PLAIN targets in at least 12 of 16 each (the probe removes the start-function cost, so PLAIN and
 ZERO should now agree); Q4 CKA ordered the same way in at least 12 of 16. If Q1-Q2 fail, "larger f gives better features" is wrong at the level
 of the last hidden layer and the SPEC10 gain lives elsewhere (e.g. in the readout).

### 12. HR2 failure predictions (share wrong)

 Predictions for the GPU run: HR2 failures concentrate in (a) eigA, (b) learners taking linearised steps across a kink (DJ; SL passes because exact
  jumps return to the line), (c) few-switch cases; with many switches W is a sum of many rank-one pieces and the leak should average out
  (HR2 pass fraction non-monotone in size). Check with rho_perp, R-a/R-b/R-d tables per size and direction.

### 13. ReLU width prediction (premise wrong)

Local width sweep started : prediction written before it: at fixed relative size the deviation from the
 smooth fractions falls with width (more, smaller switches), roughly width^-1/2. Result: see sec. 29 when written.

## 3B. Open hypotheses, best-supported first

### 14. Equivalence audit: SAM, midpoint, extragradient; Adam

EQUIVALENCE AUDIT (reasoned, not yet tested): two-gradient rules form one family indexed by where the second
 gradient is taken: backwards along the step = sharpness-aware minimisation (alpha < 0), start = plain, half-way = midpoint, end =
 extragradient / proximal / pull-toward-start regularisers. Our law then says: SAM amplifies wear (it is known to seek flat minima),
 midpoint is neutral, extragradient recharges. PREDICTION: SAM accelerates loss of trainability under changing targets.
 Adam: near-full steps in weak kernel directions = large asymmetry there (solvable model earlier: fraction -1.8 to -211); plain GD's
 asymmetry c_i/(2 - c_i) is tiny in weak directions. This would explain the field's "Adam loses plasticity much faster than SGD".
 a-meter run in progress when this was written.

### 15. Open predictions P-A to P-E

OPEN PREDICTIONS WITH DECIDING TESTS:
 P-A implicit or midpoint learners do not lose trainability in continual learning where explicit ones do (sign of 1 - 2f); test: the
     plasticity protocol of sec. 10 on tanh with SLa / proximal steps.
 P-B a two-stage Gauss (fourth-order) implicit learner matches the predictions of gradient flow with error ~ tau^5.
 P-C deeper networks: larger f-sensitivity (more factors compound). P-D ReLU: larger still.
 P-E large-step benefit in ordinary vs doubled networks (sec. 39 B), unchanged.

### 16. Open connections B, C, D

OPEN CONNECTIONS KEPT (each with its deciding test):
 (B) Large-learning-rate / flatness benefit = damping of the random start function (odd channel). Large step = low f (DJ(c): f = (1-c)/(2-c)).
     Symmetry part now verified; claim about real training untested. Test: step-size sweep of plain GD, ordinary vs doubled network, same data.
     Prediction: the large-step gain shrinks or reverses when doubled. Kill: same gain in both.
 (C) Interior optimum = bias-variance trade-off of learning the kernel from n labels; optimum rises with n (2 starts so far). Test: resample the
     training set at fixed start, alpha in {0, .5, 1, 2}; variance of predictions should rise with alpha, bias fall. Read 2602.00827 in full.
 (D, wildcard) Grokking = slow drift along the fibre after fit: train loss fixed, test loss linear in the fibre coordinate, weight decay pushes
     toward the nearest point. Test: in a small grokking run, project theta(t) - theta_fit on v_NP and compare test-loss change with f x <r, G>.

### 17. SPEC10 consequences to test

CONSEQUENCES TO TEST (not yet done): (i) f > 1 (AJ(1.5), AJ(2)) should be better still until the squared term wins (an optimum beyond the
 nearest point); (ii) large-step GD/catapult gains should vanish or reverse on a ZERO model (they come from shrinking the start function/kernel,
 the dN channel); (iii) implicit Gauss-Newton (end-point Jacobian) = second-order speed with f = 1; (iv) more seeds, larger n, cross-entropy,
 a second dataset, SGD/Adam.

### 18. Next steps in order of value (as of sec. 37)

NEXT STEPS, in order of value:
 a. Finish x1-128 and x2; record counts against P1-P4 and Q1-Q4.
 b. Doubled network at n = 128 (needs ~5 GB RAM per Jacobian pair; 7.7 GB free when checked) and with AJ(alpha) to locate the optimum without the
    odd part; then cross-entropy, a second dataset, more seeds (needs a rig; write a spec first, as for SPEC10).
 c. Direct test of the readout-flip symmetry (negate the last layer of a stored start: the small-move slope must flip sign).
 d. Large-step GD on plain vs doubled models (prediction: the catapult gain comes from the odd/start-function channel and vanishes when doubled).
 e. ReLU: active-set nearest-point solver to test "the nearest point sits on activation boundaries"; width sweep with more cases.
 f. Write-up candidates: (i) the law + one-function family + midpoint-kernel identity; (ii) SPEC10 + odd/even structure + interior optimum.
 Prior work that must be cited is listed in sec. 24, 29, 33 (Arbel 2302.02904, Buffelli 2411.07979, Geiger 1906.08034, Yeom 2602.00827,
 Woodworth 2002.09277, Lobo 2605.17718, Freeman 2510.19206, Klein-Huang / Shamir-Yomdin, Sinha-Druga 2610.07253, Roberts-Yaida-Hanin).

### 19. ReLU nearest point sits on activation boundaries

   PREDICTION : the nearest point of a ReLU fibre sits ON activation boundaries of training inputs (a unit exactly at zero on a
   training input; Clarke stationarity theta - theta0 = [lam J+ + (1-lam) J-]^T a) - that is why AJ/NP solvers oscillate; it is the high-dimensional
   form of "knots at the data". Test: active-set nearest-point solver (pin the oscillating unit, free lam in [0,1]); count pinned (unit, input) pairs.

### 20. Self-consistent field law for dense kinks; open G4 questions

      Reading: with dense kinks the Jacobian depends on the learner's own vertical position, not only on tau (learner and reference sit on the same fibre but in
      different activation regions). NEXT-ORDER IDEA: self-consistent field law v' = W(tau; v) b (an ODE in (tau, v)) instead of the quadrature sum W(tau_k) b_k.
      HR1 for relu on the GPU may land near its 90% threshold or fail at mid sizes; R-d (accuracy vs number of switches) is the table to read.

OPEN (not graveyard): what orders test loss at equal training outputs at large moves; whether the f = -1 learner's advantage at small moves and
 disadvantage at large moves is one quadratic in the vertical displacement; why tanh and gelu differ at n = 128.

### 21. Edge of stability: a formula for alpha; open/next after the law

Upgraded EoS hypothesis (replaces "balance of alignment vs jitter" wording):
  equilibrium oscillation size sigma^2 = 2 alpha / beta (known); our contribution = alpha predicted from
  the output path: alpha ~ sum over sub-threshold kernel directions of (residual travel rate) x grad(1/lambda_i-weighted potential) . grad S.
Deciding test (cheap, CPU): on a small net trained by GD into the edge of stability, measure alpha = -grad L . grad S
  directly each step and compare with the alignment-force prediction computed from K, residual and T. Kill: correlation
  near zero or wrong sign. Second test: sharpening rate vs initial residual size per kernel direction (Agarwala found
  prop. to residual variance; ours predicts the per-direction 1/lambda weighting, as in Yoo et al.'s Q).

Open/next: (1) BGN on all Part B cells -> non-perturbative midpoint; (2) momentum/Adam R on the MLP (was running);
 (3) real-net test: damped GN / NGD step-fraction sweep, GD lr sweep, proximity penalty -> measure R on a real dataset; (4) resolve tension with
 "larger lr -> more alignment" (which quantity); (5) write the theorem cleanly with the O(displacement^3) remainder.

### 22. Three-channel law, right next test, dose follow-ups

Candidate "three-channel" law: (a) top eigenvalue = optimizer's stability limit; (b) bulk/weak spectrum: smooth travel raises,
squared jumps lower (a + b/N form); (c) ReLU units: exponential survival, hazard set by kick size (floor + beta/N).
Better source analogy than plain dose: linear-quadratic cell-survival model with fractionation (PARTIAL: no repair analogue).
Kill tests: LeakyReLU arm (death channel off => should behave like tanh, Adam+gradual gains trainability); Adam lr sweep at
fixed N (hazard ~ lr, not dose); GN with step-norm cap at fixed dose; GD lr switch mid-run (lam_max must re-lock to new 2n/eta);
longer runs (100 tasks) to test exponential survival properly.

sign predictions the first-task data support. Right next test: smooth activation, small one-way moves and loops, 1-3 tasks,
predict the SIGNED change of log tr Khat^-1 from the full path matrix C (travel - jitter) per band, GD below and at the edge
separately, damage also by own-optimizer steps.

Open follow-ups: (a) is the edge lock in continual learning known? (literature searches); (b) intervention test:
cap lambda_max for GN/Adam (or lr warm restarts for GD) and see if damage follows; (c) Adam + gradual targets + smooth
activation as a plasticity-GAIN recipe: replicate on a second dataset; (d) redo dose with full path matrix C (travel + jitter), not Q only.

### 23. Next GPU tests after the evaluation-point principle

Next tests (GPU, cheap): alpha-rule on the real net; held-out one-dimensional family on MNIST test images incl. whether larger f lowers test loss; ReLU: measure omega along the path
  and predict each learner's rho from its own evaluation positions (no fitted parameter); replace NP for ReLU by a kink-aware solver.

### 24. Early open hypotheses

## 7. Open hypotheses (each with its deciding test)

- **Dose law (Tests 1–2, the notebook):** loss of trainability follows cumulative dose
  D = Σ ΔFᵀ(K+ρλI)⁻¹ΔF across optimizers and granularities; GN dose ∝ 1/N and ∝ ε²; filter
  exponent ≈ 2 for GD, ≈ 0 for GN; collapse plot. Decided by the GPU run (rules H1–H9 in SPEC5).
- **Edge of stability = balance of two forces:** sharpening from travel in sub-threshold modes
  (alignment force) vs flattening from top-mode overshoot ((1−c)/(2−c)). Evidence: Test 5, the
  36/36 check, unstable run settling at c_eff 1.95–1.98. Needs: literature searches' evidence, then a real
  edge-of-stability run comparing measured sharpening rate with the force prediction.
- **Alternation resonance:** order effects per step peak when block length ≈ slowest kernel
  timescale. Only predicted beyond period 256. Needs measurement across periods.
- **Realistic path law (Tests 3–4):** minibatches, real data; loop vs retraced path ending on
  the same task (reverse-order comparison is confounded by first-order recency when tasks
  conflict).
- Still-untested robustness: ReLU/GELU, depth, cross-entropy, more training inputs, Adam with
  its own metric, weight decay timescale, large moves (third order).
- DR3 coefficient: read the theorem, compare with κ* = 2.

### 25. Paused ideas (kept, not dropped)

## 8. Paused ideas (kept, not dropped)

Complex-plane regularisation / analytic-continuation extrapolation law ((ε/Q)^α(x); map complex
singularities; Lee–Yang zeros of softmax). Correctable vs uncorrectable forgetting (Schur
complement of old-task Hessian w.r.t. head). Action-Jacobian error as world-model metric
(shadowing). Replay equipartition and learning-coefficient link; Kramers escape. Sequential
learning = projective measurement (QQ = Pythagoras; replay as Zeno). LLM order effects from
feature-subspace angles. Coding-theory (memory with defects / write-once) for continual
learning; dead units (1−p)². Lie-bracket reachability by task switching; bracket to reduce
forgetting; AdamW bracket. Unscouted: Benna–Fusi cascades, Blackwell garbling order, free
probability for merging. Holonomic steering as a control tool; Dido/isoperimetric optimal loops;
test prediction as a signature series (kernel regression = level 1).

---

# PART 4 - OVERALL SUMMARIES, newest first

### 1. Where the picture stands (sec. 44)

WHERE THE PICTURE STANDS
 (1) Loss of trainability under changing targets is, at leading order, the learner's failure to be its own time-reverse. Held: loop law
     36/36, mixed pairs 5/5 (one exact), two-gradient learners at f = 1/2, 1, 2, triangle decomposition, both networks.
 (2) History enters the weights through three terms: STATE (net output change), HEAT ((f - 1/2) x route length; wear, and a bias that
     lowered later test loss in 4/4 histories), AREA (task order; learner-independent).
 (3) The dial is fully explicit and cheap: wear (plain steps), neutral (midpoint), recharge (predicted end point). Recharge heals a worn
     network without changing its outputs.
 (4) NOT established: any of this for minibatch SGD or Adam on a standard benchmark; cross-entropy; ReLU; more than two networks and
     one seed each; a derivation of the straight-line depletion; the collapse (it floors instead).

 CONSEQUENCE FOR THE PLAN: the learners for which today's results apply directly are those that take near-full steps in every kernel
  direction (Gauss-Newton, natural-gradient, K-FAC-type, exact re-fitting). Plain gradient descent does not wear at this order.
  Whether Adam wears, and by how much, is OPEN; the a-meter must use the gauge (mean inverse cost over probe directions), not the
  projection on the nearest-point shift. The planned Adam test needs a redesigned time-symmetric adaptive step before it is run.

### 2. What is established (sec. 37)

WHAT IS ESTABLISHED (each with its section):
 1. Evaluation-point law on real MNIST, smooth nets: every rule 100% (sec. 22, 28). General finite-size form holds on smooth nets to 0.1 S0.
 2. ReLU: same law; exact without activation switches, case-by-case with a few, statistical with hundreds; pre-set ReLU rules failed (sec. 28).
 3. On unseen inputs all learners differ along one function with coefficient f; test loss linear in f (sec. 22-23).
 4. SPEC10 (pre-registered, valid): start function removed -> larger f gives lower test loss at equal training outputs, 27/32, 31/32, 32/32, 28/32,
    anti-size conditions 32/32, 32/32, 24/24; up to -12% loss and +2.1 accuracy points at n = 128 (sec. 29).
 5. Midpoint-kernel identity on MNIST: gradient flow = kernel regression with the tangent kernel at the path midpoint (error 0.011-0.023 of |G| at
    3% moves, first order in the move); learner L = that + (f - 1/2) G(x) (sec. 32, 35).
 6. Odd/even structure: second-order learner-dependence flips with the readout sign (random sign across starts, contains the start-function
    growth); doubling removes it exactly; what remains is third order (exponents 2.91-2.99) and systematic: 16/16 by rescaled loss, 16/16 by the
    non-rescaling part, raw loss 10/16 (shrink of a still-too-small learned change at small moves) (sec. 32, 34-36).
 7. Interior optimum of alignment at n = 32 (rescaled loss best at alpha ~0.35-1 in 3 of 4 starts, ~2.5 in one): "beyond the nearest point is
    better" FAILED at n = 32 (sec. 34).

### 3. Honest status line (early)

## 9. Honest status line

A synthesis, not yet a breakthrough: one equation (weights driven by the output path; three
forces) that reproduces several separately published results and yields a few predictions not
found in the literature. Everything verified is on tanh networks with 16 training inputs,
full-batch, small moves. Each deeper literature search has found more prior work, so say "not
found in N searches", never "new". The GPU run and the realistic tests decide whether it matters.

---

# PART 5 - LITERATURE EVIDENCE FROM SCOUTS, newest first

### 1. Scout, first-order learners and Adam (sec. 45)

LITERATURE SEARCH (scout45.md): leading-order GD term = Barrett & Dherin 2021; momentum factor = Ghosh et al. 2023; look-ahead family
 = Kovachki & Stuart 2021 (zero at g* derivable, not treated, nobody found who built it); Adam = Cattaneo et al. 2024 (two opposite
 lags, neutral at beta1 = beta2, valid for runs >> 1/(1-beta2) steps); Asadi et al. 2023 (resetting the second moment helps);
 Dherin 2023 (bracket term for task order); Singh 2026 / DePavia 2025 (Adam not gauge-equivariant).

### 2. Scouts, continual-learning round

LITERATURE SEARCH THIS ROUND : NOT FOUND anywhere: a rate law for loss of plasticity; explicit vs implicit / two-gradient
 / time-symmetric updates in continual learning or plasticity work ("a real gap"); regularisers read as implicit steps; any method that
 moves along the identical-output set to restore trainability; a closed-cycle inequality for gradient learners (one attempted
 "dissipation to forgetting" claim was WITHDRAWN: Okanohara 2602.07950); fatigue descriptions for networks; thermodynamic length with the
 learner's trainability as the dissipated quantity. FOUND, must cite: Liu-Mou 2602.09234 (gradual change preserves plasticity,
 monotone in step count, Adam, no law; numbers too sparse to test 1/N); Kumar 2308.11958 (L2 toward init beats weight decay on 3/5; no
 "toward start of task" comparison; distance from init not measured); Lyle 2303.01486, 2402.18762 (each classic symptom fails as a
 cause; Adam's stale moments); Nikishin 2205.07802 (primacy bias grows with replay ratio); Mahdaviyeh 2506.04377 and Lanzillotta
 2512.07400 (replay can hurt; buffer overfitting); Elmeligy 2611.22070 (capacity + polarisation = constant in linear physical learners,
 offline repair that erodes memories); Crooks 0706.0559 / Salamon-Berry (N-step dissipation L^2/N); 2603.12324 (thermodynamic geodesics
 for RL curricula); Joudaki 2510.00304 (trapping manifolds; simplicity bias also drives plasticity loss); 2605.09044 (NTK rank can fail
 to predict trainability); Ash-Adams 1910.08475 and Berariu 2106.00042 (warm-start gap); Achille 1711.08856 (critical periods);
 Winter-Janssen 2611.27647 (cyclic weight perturbation: return-point memory; different protocol, no kernel measured).

### 3. Scout, section 40

LITERATURE SEARCH : KNOWN: Woodworth 2002.09277 (flow: hypentropy, initialisation scale dial); Even et al. 2302.08982
 (GD/SGD: effective initialisation alpha_inf^2 = alpha^2 exp(-sum q(step x gradient)), q(x) = -1/2 ln((1-x^2)^2); GD large steps hinder
 sparse recovery, SGD helps); Mind the Drift 2610.06120 (per-step factor (1 - z^2) on u^2 - v^2; Gain and Drift); Hofmann 2607.04993,
 Wang et al. 2310.17087 (two-factor balance contraction); Kunin 2012.04728, Medeiros 2604.07405 (per-step drift of conserved quantities;
 link to generalisation listed as OPEN); Andriushchenko 2210.05337, Li-Wei-Ma 1907.04595 (large step then decay = noise mechanisms).
 NOT FOUND: implicit / proximal / midpoint / Gauss-Newton updates on diagonal or two-factor models; "midpoint conserves balance" in ML;
 the optimiser at fixed initialisation scale as a dense <-> sparse dial; large-then-decay via an inherited conserved quantity;
 conservation breaking <-> generalisation.

### 4. Scout, X5

LITERATURE SEARCH : Kornblith 2010.16402 (better own loss -> worse transferable features, class separation; no formula, no
 matched-output control); over-alignment 2602.00827 (interior optimum of feature-learning strength; sample-size dependence not in abstract);
 Atanasov 2212.12147 (variance from initialisation, reduced by ensembling and by feature learning); Zhang 1905.07777 (antisymmetric
 initialisation), Hu 1905.11368 (difference trick); Musat 2511.01938 (after memorisation GD minimises weight norm on the zero-loss manifold);
 Lyu 2311.18817, Kumar 2310.06110. NOT FOUND: optimizer/step size varied at matched training outputs with both own-test and probe measured;
 probe ranking as a function of probe label count; large vs small learning rate on centred/doubled networks; test loss linear in a
 post-interpolation coordinate.

### 5. Scouts D and E

LITERATURE SEARCH D (second-order methods, centring; read bodies of 12 papers): Buffelli 2411.07979 (exact GN lazy, generalises poorly; damping, small steps,
 weight decay did NOT help in their mini-batch setting); Wadia 2008.07545 (whitening / unregularised GN lose information; intermediate
 regularisation lambda ~ 0.5 beats GD); Arbel 2302.02904 (GN: larger step -> worse features; "hidden learning" at small steps; says a theory of the
 step-size effect is missing, no formula); Geiger 1906.08034 (centred model alpha[f - f0]; fully connected nets better LAZY; init-induced part of
 the error ~1/h removable by ensembling; end-of-training NTK regression reproduces the net); Woodworth 2002.09277; Yeom 2602.00827 (interior
 optimum of feature-learning strength, +6 points; "over-alignment + over-fitting" split; c* ~ n^-2 h^-1); Lewkowycz 2003.02218; Jiang 2601.03162
 (LM/GN lazy; switch to AdamW recovers); Yoshida-Nitanda 2509.25637 (preconditioner ranking reverses with where the signal sits); Cayci 2511.03972
 (antisymmetric init); Lobo 2605.17718 (one GD step: kernel = k0 + target-aligned term scaled by eta^2). NOT FOUND: implicit / midpoint / end-point
 Gauss-Newton or natural gradient for networks; centring changing which optimizer or learning rate generalises best; catapult on centred models;
 "one correction function scaled by an algorithm-dependent number"; a formula for feature learning vs GN step factor.
LITERATURE SEARCH E (minimum norm and beyond; mostly abstracts): Freeman 2510.19206 (linear: inflating the min-norm interpolator by c > 1 lowers risk under
 anisotropic covariance with signal in the top directions; c_opt formula; breaks interpolation); negative optimal ridge: Kobak 1805.10939, Wu-Xu
 2006.05800, Tsigler-Bartlett 2009.14286, Patil 2404.01233; model extrapolation theta1 + alpha(theta1 - theta0): ExPO 2404.16792 (optimum alpha
 shrinks as training proceeds: 8.0 -> 0.3), Extra-Merge 2605.26484; silent alignment Atanasov 2111.00034. NOT FOUND: any nonlinear-network result
 on moving past the nearest-to-initialisation solution at fixed training outputs; any study changing kernel-target alignment with training outputs
 held fixed; a one-parameter family between the kernel predictor and a feature-learning predictor with the location of the test-error minimum.

### 6. Scouts A, B, C

Literature search :
 A (optimizers at equal training outputs): NOT FOUND: one-parameter family / common function on test inputs; bound by kernel posterior variance;
   any study matching training outputs exactly; closed-form test-point correction beyond gradient flow; post-hoc one-number rescaling of optimizer
   gaps; large learning rate HURTING squared-error test loss or a reversal with sample size. PRIOR to cite: Amari et al. 2006.10732 (linear:
   preconditioners -> different interpolants, bias/variance; early stopping flips the ranking); Arbel et al. 2302.02904 (LARGER GN step -> worse
   features/test loss, no theory); Zhu et al. 2205.11787 + Lewkowycz 2003.02218 (catapult: top kernel eigenvalues shrink, test loss better;
   "why that helps" stated open); Cortesi et al. 2603.02620 ("Same Error, Different Function": optimizers differ 1.3-7.8x more than seeds, no
   structure given); Geiping 2109.14119; Dyer-Gur-Ari 1909.11304 (corrections on training data, flow only); Calvo-Ordonez 2502.01556.
   READING : our small-move result (larger GN step factor = lower test loss, 15/16) is the catapult side, our full-move result (DJ(1.5) worst in
   5/8) is the Arbel side; the U-shape in f joins the two literatures. Untested as a claim about their settings.
 B (ReLU): central flows 2410.24206 exclude ReLU (needs third derivatives; error grows toward ReLU; "not clear how to quantify smoothness");
   Cohen 2103.00065 (instability below 2/eta with ReLU; GD leaves the flow path from the start for an FC ReLU net); Meltzer 2301.07737 (ReLU catapult
   to eta*H ~ 12 vs proven 4, open); Zhu 2205.11787 (frozen-pattern quadratic model, rank-one kernel updates); Du 1810.02054 (flip fraction
   <= 2R/sqrt(2 pi), falls with width); Williams 1906.07842 / Sahs 2008.01772 (1-D: knots attracted to data points; kernel regime = spline).
   NOT FOUND: switches per unit movement, kernel change per switch, any implicit/proximal/nearest-point method on ReLU nets failing at activation
   boundaries, Clarke/Filippov treatment of sharpness. (Savarese et al. 1902.05040, from THE memory, not literature searches-verified: min-norm 1-D ReLU
   interpolant = linear spline with knots AT the data.)

 C (mathematics outside ML; snippets only): NOT FOUND: endpoint difference as a scalar times one vector with the scalar = evaluation point; link of the
   alpha-prescription to deterministic schemes on constraint manifolds; lift endpoint vs nearest point via O'Neill tensors. FOUND: robotics twin of our
   label loops: pseudoinverse control of redundant arms is not repeatable (Klein & Huang 1983; Shamir & Yomdin 1988 Lie-bracket condition; Luo & Ahmad
   1992 drift density, bracket of J^+ replaceable by bracket of J^T; Klein 1992/93 drift-free limit trajectories for the 3R arm; Baillieul extended
   Jacobian = repeatable by construction). MUST CITE next to Sinha & Druga. Also Zeng 2101.09180 (pseudoinverse Newton converges "approximately" to
   the nearest point, no formula), Pes & Rodriguez 2101.07560 (min-norm GN).

### 7. Third scout and synthesis (sec. 8c)

## 8c. Third literature searches (self-stabilisation, step factor 1-2) and the synthesis (2026-11-21)

Literature search read 17 papers (Damian 2209.15594, Arora-Li-Panigrahi 2205.09745, central flows 2410.24206,
Ma 2204.11326, Ahn 2204.01050, Chen-Bruna 2206.04172, Kreisler 2305.13064, Barrett-Dherin 2009.11162,
Beneventano-Woodworth 2501.09137, Lewkowycz 2003.02218, Zhu 2210.03294, Li-Wang-Li 2207.12678,
Agarwala 2210.04860, Litman 2604.20446, Edge Flow 2606.18080, Rod Flow 2602.01480 + 2611.08358,
Berthier 2609.01034, Jiang-Cohen-Li 2507.12837, MacDonald-Vidal 2607.08380, Hofmann 2607.04993).

What our Test-5 formula actually is (the algebra, exact for a geometric zigzag r_k = (1-c)^k eps):
  path coefficient = (1-c)/(2-c) = 1 - 1/(2-c)
  = travel term (1/2 eps^2, independent of c) minus jitter term (1/2 * sum of squared jumps = 1/2 eps^2 c/(2-c)).

Consequences:
1. NO conflict with Barrett-Dherin / Beneventano-Woodworth. d/dc[(1-c)/(2-c)] = -1/(2-c)^2 < 0 always:
   larger step = flatter, monotonically. Gradient flow (c->0) is the SHARPEST (coefficient 1/2);
   the exact jump (c=1) is the zero point; c>1 is flatter than the exact jump. The "sign flip at c=1"
   is only a change of reference (exact jump instead of flow). Say it that way from now on.
   Small c: 1/2 - c/4 - ..., i.e. first-order-in-step flattening, the implicit-gradient-regularisation shape.
2. The 1/(2-c) pole is NOT new mathematics: it is the geometric sum of squared jumps.
   MacDonald-Vidal's subcritical drift is prop. to sum y1_t^2 = y0^2/(c(2-c)) (same sum; they did not write it).
   Summing Lewkowycz's exact kernel recursion over a decaying trajectory gives (c-4)/(2-c) = -1 - 2/(2-c):
   same pole, same monotone direction (total change, not the output-preserving part, so constants differ).
   Hofmann has a noise-driven rate prop. to 1/(2-c). Nobody wrote (1-c)/(2-c) or stated the pole for sharpness.
3. All cubic-term self-stabilisation theories (Damian, central flow, Rod Flow, Edge Flow, Berthier) need c >= 2.
   Flattening from a DECAYING zigzag at 1<c<2 appears only in MacDonald-Vidal (near a manifold of flat minima,
   least squares) and in exactly solvable two-factor models. Ours is the same effect in output-path form, general net.
4. What is still not found anywhere: the travel term. Damian's alpha = -grad L . grad S and the central flow's
   <grad S, -grad L> are ASSUMED positive. Our law gives a formula for it: travel in each kernel direction times
   the slope of the K^-1 potential (alignment force). Cohen: "we do not know why progressive sharpening occurs".
   So the open niche is a formula for alpha, not the self-stabilisation half.

### 8. Edge-of-stability / alternation scouts (sec. 8b)

## 8b. Evidence from the edge-of-stability / alternation literature searches (2 of 3 back when written)

Progressive sharpening (PS) literature searches:
- Cohen et al. 2103.00065: PS occurs under gradient FLOW; weaker with width, stronger with
  depth, dataset size, task difficulty; "We do not know why progressive sharpening occurs."
- Central flows (Cohen, Damian et al. 2410.24206): dS/dt = η⟨∇S,−∇L⟩ − ½ησ²‖∇S‖². The second
  term (sharpness reduction ∝ oscillation variance × ‖∇S‖²) is the SAME structure as our jitter
  force on the top mode → self-stabilisation side is known in weight-space form. The first term
  (PS) is taken as given, not explained (also an assumption in Damian–Nichani–Lee 2209.15594).
- Supports our "sharpening comes from travel": Agarwala et al. 2210.04860 (PS rate at init ∝
  variance of the initial residual; no PS in models linear in parameters); Yoo et al. 2506.06940
  (final sharpness set by "dataset difficulty" Q = Σ d_i²/σ_i², i.e. squared distance each mode
  must travel weighted by inverse spectrum — the same shape as our potential ΔYᵀK⁻¹ΔY; corr 0.99
  / 0.81 with measured sharpness); Roulet et al. 2312.00209 / 2407.06183 (step-size tuners that
  stay below the edge give EVER-INCREASING sharpness — what our balance account predicts when
  there is no overshoot to flatten).
- Not found: any GD-vs-Newton/Gauss–Newton comparison of solution sharpness at equal fit; any
  sign-definite "GD sharper than exact jump" statement; any per-direction step-factor-below-1
  account; a rate for PS under flow in general.
- Models to compare against: Wang–Li–Li 2207.12678 (output-layer norm), Rosenfeld–Risteski
  2311.04163 (opposing signals), Liu et al. 2503.02809, Kalra et al. 2311.02076 (UV model; no PS
  at large width in standard parametrisation), MacDonald et al. 2305.14683 (Jacobian must grow).

Alternation-speed literature searches:
- Interior optima in switching period exist empirically: Skriloff 2608.23889 (joint performance
  best at an intermediate reach ηT; no kernel spectrum measured; he proposes the Lie bracket
  over a period as the driver — future work); Xu & Harish 2608.04927 (linear regression along a
  task path: progress Θ(τ) for too-short and Θ(1/τ) for too-long per-task time, explicit
  eigenvalues; "a time-scale matching problem"; one pass, no cycles); Dynamical Batching
  2610.26156 (class-cycling: failure region at short period, benefit band at longer).
- Minibatch order is second order in step size for memoryless optimizers (Sweeney 2606.29554).
- Not found: order-dependence peak tied to the tangent-kernel spectrum; frequency-response /
  phase-lag analysis of training under periodic targets; hysteresis-area scaling laws for
  training; seasonal drift vs period.

Third literature searches (self-stabilisation equations, behaviour for step factor between 1 and 2, Barrett–
Dherin comparison) had not reported when this was written.

### 9. Prior-work map

## 5. Prior-work map (what is NOT ours)

- Ordered-double-sum expansions: Smith et al. 2101.12176; Dherin 2311.00235, 2502.01557;
  Shaw & Whalley 2504.04274. Task-order brackets: Rukhovich 2501.15556, Sweeney 2606.24993 /
  2609.34348 / 2606.29554, Kwapong 2611.20087.
- Algorithm dependence at finite width via dNTK and "algorithm projectors": Roberts–Yaida–Hanin
  2106.10165 (ch. 11, ∞); Dyer & Gur-Ari 1909.11304; Huang & Yau 1909.08156.
- **Label/output loop with min-norm transport, area scaling, reversal, bracket prediction:**
  Sinha & Druga 2610.07253 (Oct 2026) — they were first; their open boundary: SGD/Adam do not
  reproduce it, no unseen-input test.
- Ehresmann connection / curvature / holonomy on transformer weights (symmetry fibres only):
  Wang & Wang PMLR v282.
- Label-noise drift as Itô correction: Blanc 1904.09080; Damian–Ma–Lee 2106.06530; Li–Wang–Arora
  2110.06914 (also rotating "k-phase motor" drift). Uneven noise: Xu et al. 2602.05065.
- Mirror-flow iff output-gradient fields commute: Li–Wang–Lee–Arora 2207.04036 (yes/no only).
- TD extra implicit regulariser: Kumar et al. DR3 2112.04716; rank collapse 2010.14498.
- Noise → degenerate units: Chen et al. 2306.04251. LoP as trapping in those sets, and
  "generalisation mechanisms drive it": Joudaki et al. 2510.00304. Gradual vs abrupt targets:
  Liu & Mou 2602.09234. eNTK conditioning and plasticity: Rosseau 2606.09762; Lyle 2303.01486,
  2402.18762. Anticipatory recovery: Yang et al. 2403.09613 (their toy = shared + private weights).
- Counter-evidence to keep in view: Koster & Kukla 2409.17300 (sharpness regularisation does not
  change plasticity loss; snippet only); Jiang–Cohen–Li 2507.12837 (larger LR lowers yᵀK⁻¹y more).

Not found in any search so far (≈literature searches): the output-path form of the law for
arbitrary learners/target sequences; the three-force split with the K⁻¹ potential; squared-jump
/ dose law for plasticity; runaway law; input-loop area law; steering a chosen unseen input by
a label loop; area term for time-irreversible targets in general; alternation-speed resonance;
the (1−c)/(2−c) formula (pending literature searches).

