# SPEC11 - Trainability and the evaluation point (pilot; one GPU or CPU)

Written 2026-11-23. Read every section before writing code. Where this text and your intuition disagree, this text wins;
where this text is ambiguous or seems wrong, STOP and report the ambiguity in your final message instead of guessing.

## 0. What is being tested (and what is not)

Setting (unchanged from SPEC8-10): fixed training inputs; F(theta) = network outputs on them (vector of length N = n*C);
J = dF/dtheta; K = J J^T; "vertical" = weight directions that leave F unchanged to first order. A learner moves the
outputs from F0 to a target t. Its alignment fraction f is known in closed form (library function `predict_f9`).

Claims under test. u, w are UNIT vectors in output space; s is a move size; theta0 is the start; F0 = F(theta0).

* C1 LOOP LAW. Move the outputs from F0 to F0 + sigma*s*u with a learner (leg 1), then from there back to exactly F0 with
  the SAME learner (leg 2). The weights do not return: theta2 - theta0 is a vertical displacement equal, at second order
  in s, to (2f - 1) * vNP, where vNP is the nearest-point shift of ONE leg of the same size, sign and direction
  (section 4). So: f < 1/2 (explicit learners) -> displaced opposite to vNP; f = 1/2 -> no displacement at this order;
  f > 1/2 (implicit learners) -> displaced along vNP.
* C2 POTENTIAL IDENTITY. With q_u(theta) = u^T K(theta)^-1 u (the squared weight distance needed, to first order, to move
  the outputs by u): vNP = -(s^2/2) * (vertical part of grad_theta q_u), at second order in s.
* C3 COST CHANGE. After the out-and-back of C1 along u, the cost of ANY unit direction w changes by
  dq_w = (1 - 2f) * M_wu with M_wu = 2 <vNP_w, vNP_u> / s^2 (vNP_w, vNP_u: nearest-point shifts of single legs of size s
  along w and u). M is a Gram matrix, so the travelled direction itself always gets costlier for f < 1/2 and cheaper
  for f > 1/2; other directions follow the sign of <vNP_w, vNP_u>.
* C4 ACCUMULATION. Several out-and-back excursions in a row add up: displacement = (2f-1) * sum of the vNP's, cost change
  = (1-2f) * sum of the M's, while the kernel has not changed much.
* C5 LARGE MOVES (the practical question; ordinal only). In a sequence of full relabelling tasks, learners with larger f
  end with LOWER cost on fresh probe directions than learners with smaller f.

NOT tested here: generalisation (except one exploratory number in Part C), ReLU, other losses, stochastic learners,
Adam, n other than 32. Do not add any of these.

The thing that must be true of the code: every learner in a comparison visits the SAME sequence of training outputs; the
only thing that differs between learners is where on the fibre they end. Any shortcut that breaks this breaks the test.

## 1. Files, environment, what you may touch

You write NEW files only, all inside `task_cycle_rotation/spec11/` of the project folder (create it). Never modify or
copy-edit `part9lib.py` or any other existing file. Read-only inputs:

* `$TCR_DATA/keep\part9lib.py` - the library used on the rig (reviewed, unchanged). All learners, the exact jump,
  the nearest-point solver and the measurement come from it. Do NOT re-implement any of them.
* `$TCR_DATA/keep\evaluation_point_tests.ipynb` - holds the CONFIG cell (the one containing `S1_FRAC, S2_FRAC`).
* `$TCR_DATA/keep\part9_run\starts\` - stored start weights (`<combo>.pt`, `<combo>.json`).
* `$TCR_DATA/keep\mnist_data\MNIST\raw\` - the four MNIST idx files (uncompressed).
* `$TCR_OUT/loc.py` - a working example of loading library + config + MNIST from idx files on CPU. Read it.

Deliverables (all in `task_cycle_rotation/spec11/`):

1. `spec11_common.py` - shared helpers (loading, directions, q, units, file I/O). Imported by run, report and tests.
2. `spec11_run.py` - computes and stores results (section 8).
3. `spec11_report.py` - reads stored results, prints and writes the report (section 9).
4. `make_bundle11.py` - run on this machine; builds `spec11_bundle.zip` (section 10).
5. `spec11_colab.ipynb` - notebook for Colab (section 11).
6. `test_spec11.py` - small tests on a tiny network (section 12).
7. `NOTES11.md` - every decision you made that this spec did not fix, every deviation, and the test output.

Environment: Python 3, torch (float64 everywhere, `L.DT`), numpy. No other packages. NOTHING may install packages
(no pip, no apt, no `!pip` in the notebook). Local machine: Windows, CPU only, another experiment is using most cores:
run only the tiny tests of section 12 locally and call `torch.set_num_threads(2)` in the tests. Do not run full-size
MNIST computations yourself; I will.

## 2. Loading (identical data and settings to the rig)

`spec11_common.load(bundle_dir, device, threads=None)` must:

1. Read `cfg11.json` from the bundle (section 10) -> dict CFG. It is the rig's CONFIG in MODE "full".
   Set `CFG["MNIST_DOWNLOAD"] = False`, `CFG["RUN_DIR"] = <out dir>`.
2. `sys.path.insert(0, bundle_dir)`; `import part9lib as L`; `L.configure(CFG)`; `L.set_device(device)`.
   `device` is the string "cpu" or "cuda:0". If it contains ":" also call `torch.cuda.set_device(int(index))`;
   NEVER call `torch.cuda.set_device` with a plain "cuda" string or a torch.device (it raises).
3. Read the four idx files from `<bundle>/mnist/` exactly as `loc.py` does and install them:
   `L._MNIST_RAW = (train_images uint8 (60000,28,28), train_labels int64)`,
   `L._MNIST_TEST_RAW = (test_images, test_labels)`, `L.DATA_SOURCE = "mnist"`.
   torchvision must not be needed and nothing may be downloaded.
4. Return `(L, CFG)`.

Combinations (dicts as the library expects):
`TANH = dict(act="tanh", start="init", seed=0)`, `GELU = dict(act="gelu", start="init", seed=0)`,
`LIN = dict(act="linear", start="init", seed=0)`. Names from `L.cname9(cb)`: "tanh-init-s0", "gelu-init-s0",
"linear-init-s0". Context: `ctx = L.Ctx9(cb, run_dir)` where `run_dir` is a directory that contains `starts/` with the
stored start files (the bundle provides `<bundle>/run/starts/`; pass `run_dir = <bundle>/run`). Then
`prob = ctx.prob`, `st = ctx.st` (start object: `st.th0`, `st.F0`, `st.J0`, `st.V0(vec)`), `ctx.dirs`, `ctx.S0`.
n = 32, C = 10, N = 320; P = 118,794 for tanh/gelu, 1,970 for the linear model.

## 3. Directions, sizes, probes (all deterministic)

All vectors are float64, length N, on the library device. Random numbers ALWAYS come from `L.gen(seed)` (a CPU
generator) and are moved to the device afterwards, exactly as `L.make_directions` does.

* S0 = `ctx.S0`. Sizes: s1 = CFG["S1_FRAC"] * S0 (= 0.02 S0), s2 = CFG["S2_FRAC"] * S0 (= 0.01 S0).
* Helper `label_dir(st, seed)`: `lab = torch.randint(0, C, (n,), generator=L.gen(seed))`;
  `y = one_hot(lab, C).to(L.DT).reshape(-1).to(device)`; return `(y - st.F0) / L.vnorm(y - st.F0)`.
* Helper `gauss_dir(st, seed)`: `v = torch.randn(N, generator=L.gen(seed), dtype=L.DT).to(device)`; return `v / L.vnorm(v)`.
* With sd = combination seed (0 here):
  excursion directions  E1 = `ctx.dirs["label"]`, E2 = `ctx.dirs["rand"]`,
  E3 = label_dir(st, 8000 + 100*sd + 3), E4 = gauss_dir(st, 9000 + 100*sd + 4),
  E5 = label_dir(st, 8000 + 100*sd + 5), E6 = gauss_dir(st, 9000 + 100*sd + 6);
  probe directions  W1..W4 = label_dir(st, 7000 + 100*sd + j), j = 1..4;  W5..W8 = gauss_dir(st, 7500 + 100*sd + j), j = 5..8.
  Names: "E1".."E6", "W1".."W8". They are built once per combination from the START (st.F0) and never rebuilt.
* Cost: `q(prob, theta, vecs)`: `K = L.gram(prob.J(theta))`; `Lc = L.chol(K)`; return `[float(x @ L.csolve(Lc, x)) for x in vecs]`.
  One Jacobian per call; pass all needed vectors in one call.
* Kernel summary `ksum(K)`: eigenvalues lam (ascending, `torch.linalg.eigvalsh`): trK = sum, lam1 = max, lamN = min,
  inv_mean = mean(1/lam), A = log(mean(lam) * mean(1/lam)).

## 4. Reference: nearest-point shift of one leg

`reference(L, prob, st, x, s, sigma)` for unit direction x, size s, sign sigma in {+1, -1}:

    t = st.F0 + sigma * s * x
    thT = L.jump(prob, st.th0, t, Jb=st.J0)["theta"]            # exact jump from the start
    sc, vL, vNP, thNP, thTH = L.measure_ext(prob, st, thT)      # nearest point of the same outputs
    usable  <=>  sc["np_status"] == "CONVERGED" and sc["th_status"] == "CONVERGED"
                 and not sc["invalid"] and not sc["below_resolution"]

Keep `vNP` (vector, = st.V0(thNP - st.th0)), `sc` (scalars) and the jump status. EVEN PART of a reference:
`vNPe(x, s) = 0.5 * (vNP(x, s, +1) + vNP(x, s, -1))`, usable only if both signs are usable.

References needed per combination: E1, E2 at s1 and s2; E3..E6 at s1; W1..W8 at s1; both signs each -> 32 references.
Compute them once, save each vNP to disk (section 8), and reuse.

Gram matrices to store per combination (small, needed by the report): for size s1 the 14 x 14 matrix
`G1[a][b] = <vNPe(a, s1), vNPe(b, s1)>` over a, b in (E1..E6, W1..W8); for size s2 the 2 x 2 matrix over (E1, E2).
Also store the 4 numbers `<vNP(x,s,+1), vNP(x,s,-1)> / (|.||.|)` for x in (E1, E2), s in (s1, s2) (sign agreement).

## 5. Part A - single out-and-back excursions (tests C1, C2, C3)

Learners LA (constructor -> name -> f from `L.predict_f9(Ld)`; check the f values in a unit test):
`L.L_DJ(1.0)` DJ(1) 0;  `L.L_DJ(1.5)` DJ(1.5) -1;  `L.L_DJ(0.5)` DJ(0.5) 1/3;  `L.L_SL(2)` SL(2) 0.25;
`L.L_SL(8)` SL(8) 0.4375;  `L.L_AJ(0.5)` AJ(0.5) 0.5;  `L.L_AJ(1.0)` AJ(1) 1;  `L.L_AJ(1.5)` AJ(1.5) 1.5;
`L.L_SLa(2, 1.0)` SLa(2,1) 0.75.   Name strings: use `L.learner_name9(Ld)`.

EXCURSION `excursion(L, prob, st, Ld, u, s, sigma)`:

    r1  = L.run_learner9(prob, st, st.F0 + sigma * s * u, s, Ld);  th1 = r1["theta"]
    st1 = L.Start(prob, th1)                                      # leg 2 starts from the learner's own end point
    r2  = L.run_learner9(prob, st1, st.F0, s, Ld);                th2 = r2["theta"]
    # the leg-2 target is the ORIGINAL st.F0 tensor itself (not F(th1) - sigma*s*u, not a recomputed copy)
    d = th2 - st.th0;  v = st.V0(d)                               # vertical part at the START
    status "OK" <=> r1["status"] == "CONVERGED" and r2["status"] == "CONVERGED" and all finite
                    and close <= CLOSE_TOL, close = L.vnorm(prob.F(th2) - st.F0) / s,  CLOSE_TOL = 1e-6
    else status = first of: r1 status, r2 status, "NONFINITE", "NOT-CLOSED"

Return scalars: statuses of both legs, close, nd = |d|, nv = |v|, horiz = |d - v| / |d|, steps/iterations the library
reports, seconds; `v` (vector); `v1 = st.V0(th1 - st.th0)` (vector; single-leg displacement); `th2` (vector).
`del st1` and `L.free_memory()` before returning. A failed first leg skips the second.

UNIT of Part A = (combination, excursion direction u in {E1, E2}, size in {s1, s2}, learner in LA); it runs BOTH signs.
For a unit with both signs "OK" compute and store (ref = references of the same direction, size and signs):

* LOOP:  `es = L.even_stats(v(+), vNP(+), v(-), vNP(-))` -> rho_loop = es["rho"], cos_loop = es["cos"], nr_loop = es["nr"].
  Also the single-sign values rho_sig = <v(sig), vNP(sig)> / |vNP(sig)|^2 and cosines, for sig = +1, -1.
* SINGLE LEG (validation V2):  `e1 = L.even_stats(v1(+), vNP(+), v1(-), vNP(-))` -> rho_leg, cos_leg, nr_leg (expected: f).
* COSTS: q at th2 for the 15 vectors (u first, then W1..W8, then E1..E6; u appears twice - keep it simple) for each sign;
  dq_x(sig) = q_x(th2(sig)) - q_x(theta0);  dq_x = 0.5 * (dq_x(+1) + dq_x(-1));  store all, plus q_x(theta0).
* KERNEL: ksum at th2 for each sign and at theta0; store the differences averaged over the two signs.

If one sign is not "OK" the unit is stored with its statuses and no derived numbers (the report excludes it and lists it).

C2 IDENTITY check, per combination and direction u in {E1, E2}: vh = vNPe(u, s1) / |vNPe(u, s1)|, h = |vNPe(u, s1)|;
`Dq = (q_u(theta0 + h*vh) - q_u(theta0 - h*vh)) / (2h)`; store Dq and, for s in (s1, s2), pred(s) = -2 |vNPe(u, s)| / s^2
and iota(s) = Dq / pred(s).

## 6. Part B - a star of excursions (tests C4)

Learners LB: DJ(1), DJ(1.5), SL(2), SL(8), AJ(0.5), AJ(1). Size s1 only.
UNIT of Part B = (combination, learner). Sequence: for m = 1..6: excursion along +E_m, then along -E_m, each one
STARTING FROM THE END POINT OF THE PREVIOUS ONE, each one returning to the original st.F0:

    th = st.th0
    for m in 1..6:
        for sigma in (+1, -1):
            stc = L.Start(prob, th)
            r1 = L.run_learner9(prob, stc, st.F0 + sigma * s1 * E_m, s1, Ld);  st1 = L.Start(prob, r1["theta"])
            r2 = L.run_learner9(prob, st1, st.F0, s1, Ld);                     th = r2["theta"]
        CHECKPOINT m (after both signs)

Note the leg-1 target is built from the ORIGINAL st.F0 (not from stc.F0); leg-2 target is st.F0.
At each checkpoint m store: `vc = st.V0(th - st.th0)` norms and its dot products with all 14 even references of size s1
(`<vc, vNPe(a, s1)>`, a in E1..E6, W1..W8); close = |F(th) - st.F0| / s1; q at th for the 14 vectors; ksum at th;
statuses. If any leg is not "CONVERGED" or close > CLOSE_TOL, the unit stops there (status recorded, later checkpoints
absent). Keep the final `vc` vector on disk.

## 7. Part C - full relabelling tasks (tests C5), and the linear control

PART C. Learners LC: DJ(1.5), DJ(1), SL(2), SL(4), SL(8), `L.L_SLa(8, 1.0)` SLa(8,1) (f 0.5625), `L.L_TJ(1.5, 1.0)`
TJ(1.5,1) (f 5/7). UNIT = (combination, learner). Tasks m = 1..6: `lab_m = torch.randint(0, C, (n,), generator=L.gen(6000 + 100*sd + m))`,
target `Y_m = one_hot(lab_m)` flattened (the outputs must EQUAL the one-hot labels: a full relabelling).

    th = st.th0
    for m in 1..6:
        stc = L.Start(prob, th);  s_m = L.vnorm(Y_m - stc.F0)
        r = L.run_learner9(prob, stc, Y_m, s_m, Ld);  th = r["theta"]
        fit = L.vnorm(prob.F(th) - Y_m) / s_m          # must be <= FIT_TOL = 1e-6, and r["status"] == "CONVERGED"
        CHECKPOINT m: q at th for W1..W8; ksum at th; |th - st.th0|; status, fit, seconds

A unit stops at the first task that is not CONVERGED or has fit > FIT_TOL (status recorded). After task 6, for units
that completed: FINAL TASK, the same for every learner: `stf = L.Start(prob, th)`;
`rf = L.run_learner9(prob, stf, ctx.Y, L.vnorm(ctx.Y - stf.F0), L.L_SL(8))` (true labels, learner SL(8) for everyone);
then test loss and accuracy of the PLAIN network: `td = ctx.test()`; `Ft = L.test_outputs(prob, td["X"], rf["theta"])`;
`loss, acc = L.test_loss_acc(Ft, td)`. Store with status and fit. (Exploratory; no rule.)

LINEAR CONTROL (validation V4): combination LIN, direction `ctx.dirs["label"]`, size s1 = 0.02 * ctx.S0, sign +1,
learners DJ(1), SL(2), AJ(1): run `excursion`; store nd = |th2 - th0| and n1 = |th1 - th0|. No references needed.
(The model is linear in its weights, so every learner must return to the start: nd / n1 ~ 1e-9 or smaller.)

## 8. `spec11_run.py`: stages, storage, resume

Command line / environment (environment variables, with defaults):
`SPEC11_BUNDLE` (default: directory of the script), `SPEC11_OUT` (default `./spec11_out`),
`SPEC11_DEVICE` (default "cuda:0" if `torch.cuda.is_available()` else "cpu"), `SPEC11_THREADS` (optional int, CPU threads),
`SPEC11_PARTS` (default "V,R,A,B,C": comma list; V = environment check + linear control, R = references),
`SPEC11_COMBOS` (default "tanh-init-s0,gelu-init-s0"), `SPEC11_QUICK` ("1" = smoke run, below), `SPEC11_BREAK` (below).

Order of work: for each combination in order: V-check (section 13 V1), R, A, B, C; the linear control once, first.
Every unit writes ONE json file `SPEC11_OUT/units/<part>__<combo>__<key>.json` atomically (write temp, rename) when it
is complete; vectors go to `SPEC11_OUT/vecs/<same name>__<what>.pt`. On start, a unit whose json exists and whose
stored `signature` equals the current one is SKIPPED (resume). signature = md5 of (sorted science settings of this
spec: learner lists, seeds, sizes, CLOSE_TOL, FIT_TOL, QUICK flag, BREAK value, md5 of part9lib.py, md5 of cfg11.json).
References are units too (`R__<combo>__<dir>__<size>__<sign>`), their vNP saved as vectors; Gram matrices written to
`SPEC11_OUT/units/G__<combo>.json` after all references of the combination exist.
Print one line per finished unit (flush=True) with its key numbers and seconds, and a final line `SPEC11 RUN COMPLETE`
only if every requested unit exists. Catch exceptions per unit: store status "ERROR" with the message, continue.
Never let one failed unit stop the run. Call `L.free_memory()` after every unit.

QUICK = "1": combination tanh only; Part A: direction E1, size s1, learners DJ(1), SL(2), AJ(1); Part B: learners
DJ(1), SL(2), m = 1..2; Part C: learners DJ(1), SL(2), tasks 1..2, no final task; references only those needed
(E1, E2 at s1; W1, W2 at s1). Output to the same structure (the signature differs, so QUICK results are never mixed
with full results).

BREAK (negative controls; default ""): "noreturn" = leg 2 of every excursion is SKIPPED (th2 := th1; close is still
computed and the "NOT-CLOSED" status is NOT applied, so the rules see the wrong object); "swap" = the report-facing
f of every learner is replaced by 1 - f; "repro" = Part A runs the unit (tanh, E1, s1, SL(2)) twice and stores both.
With "noreturn" or "swap" the report's primary rules must FAIL; with "repro" the two results must be bit-identical.

## 9. `spec11_report.py`: derived numbers and rules

Reads `SPEC11_OUT`, writes `spec11_report.txt` (everything printed) and `spec11_results.json` (all verdicts + tables).
Works on partial output: missing units are listed as "NOT RUN" and excluded. Every rule prints: the units used, the
units excluded with the reason, the count that passed, the threshold, PASS/FAIL (or NO DATA if fewer than half of the
planned units are usable). Extrapolation everywhere: x_x = (4 x(s2) - x(s1)) / 3 (as SPEC9 Part A).

Predictions written before any full-size run (2026-11-23). g = 2f - 1.

PART A (unit for the rules = (combination, direction, learner) with both sizes usable):
* HP1 loop law: rho_x = (4 rho_loop(s2) - rho_loop(s1))/3 satisfies |rho_x - g| <= 0.03 * max(1, |g|); AND if |g| >= 0.1:
  cos_loop(s2) * sign(g) >= 0.99; if |g| < 0.1 (AJ(0.5)): nr_loop(s2) <= 0.05. PASS if >= 90% of usable units.
* HP2 sign: sign(rho_loop(s2)) = sign(g) for every usable unit with |g| >= 0.1. PASS if 100%.
* HP3 own cost: kappa(s) = dq_u(s) / M_uu(s) with M_uu(s) = 2 |vNPe(u, s)|^2 / s^2; kappa_x extrapolated;
  |kappa_x - (1 - 2f)| <= 0.05 * max(1, |1 - 2f|). PASS if >= 90% of usable units. (AJ(0.5): |kappa_x| <= 0.05.)
* HP4 other directions (size s1 only): over all (combination, direction u, learner, probe w in W1..W8) points with
  usable data, least-squares line through the origin of measured dq_w against predicted (1 - 2f) * 2 G1[w][u] / s1^2:
  slope in [0.9, 1.1] and R^2 >= 0.95 (R^2 of the through-origin fit: 1 - SSres/SStot with SStot = sum y^2), separately
  for tanh and gelu. PASS if both.
* HP5 general stiffening: for every usable unit at s1, m = mean over W1..W8 of dq_w / q_w(theta0); sign(m) = sign(1 - 2f)
  for learners with |g| >= 0.1. PASS if >= 90%. (Expected from stored rig data: the nearest-point shifts of generic
  directions have cosine +0.4 to +0.6 with each other.)
* HID identity (per combination and direction): iota_x = (4 iota(s2) - iota(s1))/3; |iota_x - 1| <= 0.03. PASS if 4/4.
PART B (per combination; checkpoints m = 1..6; Vpred(m) = 2 * sum_{k<=m} vNPe(E_k, s1), built from the stored Gram
 matrix and dot products, never from recomputed vectors):
* HB1 accumulated displacement at m = 6: rho_B = <vc, Vpred(6)> / |Vpred(6)|^2 within 0.05 * max(1, |g|) of g, and
  cosine * sign(g) >= 0.98 for |g| >= 0.1; AJ(0.5): |vc| / |Vpred(6)| <= 0.05. PASS if >= 5 of 6 learners in each combination.
* HB2 accumulated cost at m = 6: through-origin regression over (learner, probe w in W1..W8) of dq_w(m=6) against
  (1 - 2f) * 2 * sum_{k<=6} 2 G1[w][E_k] / s1^2: slope in [0.85, 1.15], R^2 >= 0.9. (Factor: two excursions per m.)
* HB3 additivity: for each learner with |g| >= 0.1, rho_B(m) for m = 1..6 stays within 0.1 * max(1, |g|) of g. PASS if
  >= 5 of 6 learners per combination. (Report the trend of rho_B(m) - g with m: it measures how the kernel change
  feeds back.)
PART C (per combination; only learners that completed all 6 tasks; Q(L) = mean over W1..W8 of log q_w at task 6):
* HC1 ordering: Spearman correlation of f with Q over the completed learners <= -0.8 (needs >= 5 learners, else NO DATA).
* HC2 explicit ladder: Q(DJ(1.5)) > Q(DJ(1)) > Q(SL(2)) > Q(SL(4)) > Q(SL(8)) (all four inequalities).
* HC3 implicit side: every completed learner with f > 1/2 has Q < Q(SL(8)).
EXPLORATORY TABLES (no rule; print them): per learner the kernel summary changes in Part A (trK, lam1, lamN, A);
 the 14 x 14 cosine matrix of even references; Part B: dq_w(m) for every m; Part C: Q, A, trK, lam1, lamN, |theta - theta0|
 per task, and the final true-label test loss / accuracy per learner.
VALIDATION (section 13) results first, then rules, then tables, then a final block:
 "DECISION INPUT: HP1 .. HP2 .. HB1 .. HC1(tanh) .. HC1(gelu) .." (verdicts only; the decision itself is mine).

With SPEC11_BREAK = "swap" the report uses 1 - f in place of f everywhere (f only enters the report).

## 10. `make_bundle11.py` (run by the authors on this machine)

Builds `task_cycle_rotation/spec11/bundle/` and zips it to `spec11_bundle.zip`:
`part9lib.py` (byte copy), `cfg11.json`, `run/starts/{tanh-init-s0,gelu-init-s0,linear-init-s0}.{pt,json}` (byte
copies), `mnist/` with the four uncompressed idx files (byte copies), `spec11_common.py`, `spec11_run.py`,
`spec11_report.py`, `ref11.json`, `MANIFEST.json` (md5 and size of every file).
`cfg11.json`: take the CONFIG cell exactly as `loc.py` does (exec the cell source, take `CFG`, assert MODE == "full"),
dump with `json.dump(CFG, f, sort_keys=True)` (tuples become lists; the library accepts lists).
`ref11.json` (reference numbers computed HERE on CPU, per combination tanh, gelu, linear): S0, |F0|, sum(F0), lam1, lamN,
trK of the start; for tanh and gelu also loss_0 and acc_0 from `ctx.test0()`, and `float(ctx.dirs["label"] @ ctx.dirs["rand"])`.
This is the only full-size computation the script does (three Jacobians and two test passes; about a minute).

## 11. `spec11_colab.ipynb`

Plain notebook, no installs. Cells: (1) markdown: what to upload (`spec11_bundle.zip` into the session's files), what
each cell does, that the run is resumable (re-run the run cell after a disconnect if the files survived; otherwise
download `spec11_out.zip` regularly); (2) environment print: python, torch version, cuda available, GPU name, free
disk; unzip the bundle to `/content/spec11` if not already there; verify MANIFEST md5s; (3) QUICK smoke run
(`SPEC11_QUICK=1`, expected a few minutes) followed by the report on it; (4) full run, output streamed line by line,
plus a periodic zip of `units/` (not `vecs/`) to `/content/spec11_out_partial.zip`; (5) report; (6) zip
`spec11_out/units`, `spec11_report.txt`, `spec11_results.json` (and the final Part B vectors) into
`spec11_results.zip` and print its path and size (it is downloaded from the files panel; do not call any
download API that needs a browser prompt inside a long cell).
The notebook must also run unchanged on a CPU-only machine (device auto-detected).

## 12. `test_spec11.py` (tiny network; run it; all must pass)

Build a tiny problem with the library (`L.layered_problem(X, [6, 8, 8, 2], "tanh", seed)` with n = 5 Gaussian inputs,
N = 10; and a linear one `[6, 2]`), `st = L.Start(prob, th)`, with a CFG copied from `cfg11.json`. Unit directions:
Gaussian. Use sizes s1 = 0.02, s2 = 0.01 times the norm of (random one-hot - F0). Tests (print PASS/FAIL each, exit
code 1 on any failure):
T1 `predict_f9` values of all learners in LA, LB, LC equal the numbers in sections 5-7.
T2 loop law on the tiny net for DJ(1), SL(2), SL(8), AJ(0.5), AJ(1), DJ(1.5): |rho_x - (2f-1)| <= 0.03 max(1,|g|).
T3 identity iota_x within 0.03 of 1.  T4 own-cost kappa_x within 0.05 max(1,|1-2f|) of 1 - 2f for DJ(1), AJ(1).
T5 cross cost: for one other direction, measured dq_w within 10% of (1-2f) * 2<vNPe_w, vNPe_u>/s1^2 for DJ(1) and AJ(1)
   (skip with a printed note if |cos(vNPe_w, vNPe_u)| < 0.1).
T6 linear model: nd / n1 <= 1e-8 for DJ(1), SL(2), AJ(1).
T7 resume: run a QUICK-like tiny run twice into a temp dir; second run recomputes nothing and files are unchanged.
T8 BREAK "noreturn" and "swap" make the tiny-net loop-law check FAIL; "repro" gives bit-identical numbers.
T9 the leg-2 target is the original F0 object: after an excursion |F(th2) - F0| / s <= 1e-6 for SL(2) and AJ(1).
T10 star additivity on the tiny net: after +E1, -E1, +E2, -E2 with DJ(1): rho_B within 0.1 of -1.
The tiny-net helpers must call the SAME functions of `spec11_common.py` that the real run uses (excursion, reference,
q, unit computations), so the tests exercise the real code path. If a test fails because the PHYSICS disagrees with
this spec (not because of a bug), do not tune tolerances: report it.

## 13. Validations the run stores and the report prints first

V1 environment: for each combination, recompute S0, |F0|, sum(F0), lam1, lamN, trK (and loss_0, acc_0, the label-rand
   dot product) and compare with `ref11.json`: relative difference <= 1e-8 each (1e-6 for lamN). FAIL -> the report says
   "ENVIRONMENT MISMATCH" at the top and every rule is marked INVALID.
V2 single-leg law: rho_leg extrapolated within 0.02 * max(1, |f|) of f for every usable Part A unit (>= 90% -> PASS).
   (This is the established law; if it fails here the code path is wrong.)
V3 closure: max close over all OK excursions <= 1e-6, and max horiz <= 1e-2; report both maxima.
V4 linear control: nd / n1 <= 1e-8 for the three learners.
V5 references: all 32 per combination usable; sign agreement cosines >= 0.99.
V6 excluded units: list every unit not "OK" with its status. More than 20% excluded in a part -> that part is "NO DATA".

## 14. Expected cost (for your information)

One Jacobian is 320 x 118,794 (about 300 MB, float64). On this CPU a learner run takes 5-60 s, a reference about a
minute. Part A is about 290 legs, Part B about 290, Part C about 90 large moves, references 64. Do not optimise
anything in a way that changes numbers (no float32, no approximate solves, no batching of different learners).

## 15. Amendment A1 (2026-11-23, after the pre-check on the real tanh network; overrides the sections it names)

Pre-check result ($TCR_OUT/precheck11_tanh.log; direction "label", size s1, one size, no extrapolation):
loop rho -1.0018, -0.5011, -0.1252, +0.0000, +1.0000 for DJ(1), SL(2), SL(8), AJ(0.5), AJ(1) (predicted -1, -0.5, -0.125,
0, +1); own-cost ratio +1.0062, +0.5030, +0.1256, -0.0000, -1.0014; cross-cost ratio +1.0084, +0.5042, +0.1259, -0.0000,
-1.0044; identity 1.0029. For AJ(0.5) the weights return to the start to rounding (|d| ~ 0), so horiz = |d - v|/|d| is
~1 there and means nothing.

A1.1 (section 13, V3): the bound "max horiz <= 1e-2" applies only to excursions of learners with |2f - 1| >= 0.1.
     For learners with |2f - 1| < 0.1 report instead nd / |vNP| (same direction, size, sign) and require <= 0.05.
A1.2 (section 5): in `excursion`, compute horiz as |d - v| / max(|d|, 1e-300) (no division by zero).
A1.3 (section 9, HP1/HB1 for AJ(0.5)): unchanged (nr_loop(s2) <= 0.05; |vc| / |Vpred(6)| <= 0.05); the cosine is not
     used for that learner anywhere.

## 16. Amendment A2 (2026-11-23, after reviewing the delivered code)

A2.1 (section 13, V6): the rule "more than 20% excluded -> NO DATA" does not apply to Part C. Part C learners may stop
     for physical reasons (not converged, kernel not positive definite); HC1 needs >= 5 completed learners, HC2 the
     five ladder learners, HC3 SL(8) and at least one implicit learner.
A2.2 (section 12): the tiny test network is much more curved than the MNIST network (one excursion at the spec's test
     size changes q by 36%); the tiny tests use sizes 1/16 of those written in section 12. Tolerances unchanged.
A2.3 (sections 6, 7): an exception raised inside a learner run in Part B or Part C (for example a Cholesky failure)
     ends the unit with status "EXC:<type>" and KEEPS the checkpoints already stored.
A2.4 (section 8): a unit stored with status "ERROR" (an unexpected exception) is run again on the next start;
     every other status counts as done.
A2.5 (section 4): a reference is usable only if its first jump also converged (implementation's reading, accepted).
