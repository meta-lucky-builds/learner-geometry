# SPEC12 - Gradient descent, momentum and Adam: where the learner looks, and what that does to trainability

Written 2026-11-23 (evening). Read every section before writing code. Where this text and your intuition disagree, this
text wins; where this text is ambiguous or seems wrong, STOP and report the ambiguity in your final message instead of
guessing. Sections 0-1 say what is tested and give the mathematics; sections 2-12 say exactly what to build; section 13
lists numbers I measured with an independent scripts that your code must reproduce.

## 0. What is being tested (and what is not)

Setting (unchanged from SPEC8-11): fixed training inputs; F(theta) = network outputs on them (length N = n*C);
J = dF/dtheta (N x P); K = J J^T; "vertical" = weight directions that leave F unchanged to first order;
V0 = I - J0^T K0^-1 J0 is the vertical projector at the start theta0 (`st.V0`). A LEG moves the outputs from where they
are to a target T. SPEC11 established, for learners that solve each leg exactly with Jacobians (Gauss-Newton type):

* loop law: out to F0 + sigma*s*u and back to F0 with the same learner displaces the weights by (2f - 1) * vNP, where
  vNP is the nearest-point shift of one leg and f is the alignment fraction of the learner;
* cost change: the cost q_x(theta) = x^T K(theta)^-1 x of the travelled direction changes by (1 - 2f) * M_uu,
  M_uu = 2 |vNPe(u,s)|^2 / s^2. We call a = 1 - 2f the TIME ASYMMETRY of the learner: a > 0 wears (costs rise), a = 0 is
  neutral, a < 0 recharges.

SPEC12 asks the same questions of FIRST-ORDER learners: gradient descent, momentum, and the Adam family. First-order
learners minimise loss(theta; T) = |F(theta) - T|^2 / (2N); their gradient is g(theta; T) = J(theta)^T (F(theta) - T) / N.

Claims under test (u, w unit output vectors; s a move size; theta0 the start; F0 = F(theta0)):

* K1 LOOK-POSITION RULE (one kernel mode). Let u be an eigenvector of K0. For any learner whose weight change is a sum
  of gradients, sum_j J(y_j)^T c_j (y_j = the weights where gradient j was taken, its "look point"), the alignment
  fraction is f = sum_j w_j e_j, where e_j = how far along the journey the outputs at the look point are (0 = start,
  1 = target) and w_j = the share of the journey that gradient pays for (sum_j w_j = 1). In words: f is the average
  place where the learner looked, weighted by how much each look paid for. This gives closed forms (section 1.4) for
  gradient descent, two-gradient midpoint descent, heavy-ball momentum and look-ahead momentum, and the loop law
  rho = 2f - 1 must hold with them.
* K2 MOMENTUM AMPLIFIES, LOOK-AHEAD CANCELS (consequences of K1, tested through its numbers). Heavy ball with momentum mu
  has a = c (1+mu)^2 / [(1-mu)^2 (2(1+mu) - c)]: at the same long-run rate it is (1+mu)/(1-mu) times more asymmetric than
  plain descent (19 times at mu = 0.9). Taking the gradient at theta_k + gamma * (last step) removes the leading-order
  asymmetry in EVERY mode at once when gamma = gamma* = (1+mu) / (2(1-mu)) (0.5 without momentum, 9.5 at mu = 0.9), with
  one gradient per step; beyond gamma* the sign flips.
* K3 LOOK-MATRIX LAW (any output direction). For a closed path made of such learners, the weight displacement is
  LAW(C) with C = sum_j W_j E_j^T, W_j = K0 c_j (what look j pays for, in output units), E_j = F(y_j) - F0 (where it
  looked). `LAW` is the library function `law`.
* K4 ADAM, SECOND ORDER: TWO LAGS PULL IN OPPOSITE DIRECTIONS. Adam is not a sum of gradients (each weight has its own
  scale factor), so K1-K3 do not apply to it directly. On an out-and-back excursion with its moments reset at the start
  of each leg its loop displacement is second order in the move (in the linearised model a reset Adam leg is exactly
  odd in the task). Hypothesis: its own-direction wear a_u is the sum of a POSITIVE part from the lag of the first
  moment (momentum; as in K2; grows like (1+b1)/(1-b1)) and a NEGATIVE part from the lag of the second moment (the
  scale factors are built from older, larger gradients; grows with b2). Tested by a b1 series, a b2 series, a look-ahead
  series and frozen-metric learners (momentum without any second-moment lag).
* K5 ADAM MOVES ALONG THE FIBRE AT FIRST ORDER. One Adam leg has a vertical weight displacement of FIRST order in the
  move (sums of gradients have none). It is odd in the task, so it cancels on an out-and-back with reset moments, but
  NOT (a) when the second moment is kept across legs, (b) on a triangle of tasks. Prediction: in those two cases the
  loop displacement scales as s^1; in all other cases as s^2.
* K6 ACCUMULATION. Twelve excursions in a row change the trainability gauge at a rate that is ordered like, and roughly
  proportional to, the single-loop numbers of K4.
* K7 (exploratory) ADAM IN ITS OWN METRIC: read in the geometry of its own frozen scale factors, the loop displacement
  of Adam points along the nearest-point shift of that geometry.
* K8 (exploratory) HOVER WEAR. A learner whose steps do not shrink to zero keeps changing the weights along the fibre
  while its outputs only jitter around the target (every explicit jitter step adds asymmetry, whatever its sign).
  Prediction: Adam left running after it has reached the target keeps losing trainability; gradient descent does not.

NOT tested here: minibatch noise, cross-entropy, ReLU, weight decay, learning-rate schedules, n other than 32, any
benchmark. Do not add any of these.

The thing that must be true of the code: every learner in a comparison is taken to EXACTLY the same outputs at the end of
every leg (section 5.2: each first-order leg is closed onto its target with the explicit-midpoint learner, which is
time-symmetric and so adds no asymmetry of its own). The only thing that differs between learners is where on the
fibre they end. Any shortcut that breaks this breaks the test.

## 1. The mathematics (for understanding; the code implements sections 2-12)

1.1 Sum-of-gradients learners. A leg from theta_a to theta_b belongs to this class if
theta_b - theta_a = sum_j J(y_j)^T c_j for look points y_j (weights) and coefficient vectors c_j (length N). Members:
exact jump (one look at the start), implicit jump (one look at the end), explicit midpoint EM (one look half-way),
SL(m), gradient descent (look at theta_k, c = -(eta/N) r_k), heavy ball (same looks; the momentum makes look j pay
again in every later step, total factor 1/(1-mu)), look-ahead and two-gradient variants. Adam is not a member.

1.2 Second-order law. Expand J(y_j) = J0 + DJ[y_j - theta0]. Because every look point of such a learner is itself reached
by sums of gradients, y_j - theta0 = J0^T K0^-1 E_j + (second order), E_j = F(y_j) - F0. Then
V0 (theta_end - theta0) = V0 sum_j (DJ[J0^T K0^-1 E_j])^T c_j + (third order) = LAW(C),  C = sum_j (K0 c_j) E_j^T.
(`law(prob, st, C)` computes V0 sum_k S_k D xi_{a_k}[xi_{b_k}] for C = sum_k S_k a_k b_k^T; LEFT index = payment,
RIGHT index = look position; same convention as the `C += outer(dF, F - F0)` of the library.)

1.3 One mode. If u is an eigenvector of K0 and the journey is F0 -> F0 + s u, everything stays along u at leading order:
E_j = s e_j u, K0 c_j = s w_j u, so C = s^2 f u u^T with f = sum_j w_j e_j, and LAW(C) = f * vNP: f is the alignment
fraction. On the way back the looks sit at 1 - e_j with payments -w_j, so the loop gives (2f - 1) * vNP.

1.4 Closed forms (c = per-step contraction of the mode: c = eta * lambda / N; all follow from 1.3; checked numerically
by the authors in a one-number model and on the real tanh network, section 13):

    GD(c):               f = (1 - c) / (2 - c)                              a = c / (2 - c)
    GDmid(c):            f = 1/2 + c^2 / (2 (c^2 - 2c + 4))                 a = -c^2 / (c^2 - 2c + 4)
    HB(c, mu):           f = 1 - [(1-mu)^2 (1+mu) + 2 mu c] / [(1-mu)^2 (2(1+mu) - c)]
                         a = c (1+mu)^2 / [(1-mu)^2 (2(1+mu) - c)]
    HBLA(c, mu, gamma):  no closed form used; computed by the scalar recursion `f_scalar` below (exact for one mode).

Reference implementation (copy VERBATIM into `spec12_common.py`; it is the single source of every Part M prediction):

    def f_scalar(c, mu=0.0, gamma=0.0, mid=False, tol=1e-14, cap=2_000_000):
        """Alignment fraction of a sum-of-gradients learner on ONE kernel mode (linearised model).
        x = position of the outputs along the journey (0 start, 1 target); u = last step (same units).
        Step: look y = x + gamma*u  (mid=True: y = x + (c/2)*(1-x); needs mu == 0 and gamma == 0);
              pay = c*(1-y); u = mu*u + pay; x = x + u.
        The gradient taken at y pays for the share w = pay/(1-mu) of the journey. f = sum of w*y.
        Returns (f, steps); f = nan if the recursion diverges or does not converge within cap steps."""
        assert not (mid and (mu != 0.0 or gamma != 0.0))
        x = 0.0; u = 0.0; f = 0.0; k = 0
        while k < cap:
            y = x + (0.5 * c * (1.0 - x) if mid else gamma * u)
            pay = c * (1.0 - y)
            f += (pay / (1.0 - mu)) * y
            u = mu * u + pay
            x = x + u
            k += 1
            if abs(1.0 - x) < tol and abs(u) < tol:
                return f, k
            if abs(x) > 1e6:
                break
        return float("nan"), k

Stability in one mode (for information): GD and GDmid need c < 2; HB needs c < 2(1+mu); HBLA needs
c < 2(1+mu) / (1 + 2 gamma) (the look-ahead acts on the outputs like a momentum reduced to mu - c*gamma); at gamma* this
is c/(1-mu) < 1 + mu. The code does not use these formulas: `f_scalar` returning nan means "not convergent in the
linear model" and such a learner/direction pair is not run (listed in the plan as SKIPPED-UNSTABLE).

1.5 Why momentum amplifies and look-ahead cancels. With momentum, the gradient taken at position x_j keeps paying in later
steps, when the outputs have already moved on: the looks are stale. In the small-c limit the staleness is
(1/2 + mu/(1-mu)) steps = (1+mu)/(2(1-mu)) steps, against 1/2 step for plain descent: ratio (1+mu)/(1-mu). Looking ahead
by gamma* = (1+mu)/(2(1-mu)) of the last step puts the look where its payment is centred. gamma* does not depend on c,
so one setting serves every mode (to leading order in c).

1.6 A fixed diagonal metric changes nothing essential. A learner theta <- theta - eta * D * g with a FIXED positive diagonal
D is plain gradient descent on the re-parametrised problem F_D(phi) = F(D^(1/2) phi) (phi = D^(-1/2) theta; its Jacobian
is J D^(1/2), its kernel K_D = J D J^T). All laws hold in phi-space. Used for the frozen learners and for K7.

1.7 Adam. Step: m <- b1 m + (1-b1) g; v <- b2 v + (1-b2) g*g; k <- k+1;
step = -lr * (m / (1 - b1^k)) / (sqrt(v / (1 - b2^k)) + eps)  (all element-wise). Three things separate it from 1.1:
(i) its scale factors D = 1/(sqrt(v-hat)+eps) are not the identity, so even at first order its displacement is not in the
row space of J0: it has a vertical part of order s (K5); (ii) D depends on the gradients seen, i.e. on the task, so the
displacement is not linear in the task: closed polygons need not cancel at first order, and a second moment kept across
legs breaks the mirror symmetry between "out" and "back" (K5); (iii) both moments are stale averages (K4).
In the linearised model (J frozen) a leg run with RESET moments is exactly odd in the task (g -> -g gives m -> -m,
v -> v, step -> -step), so on an out-and-back the first-order parts cancel and what remains is second order.

1.8 Universal reading (no law assumed). For ANY small vertical displacement d at theta0 and any unit x:
q_x(theta0 + d) - q_x(theta0) = <grad q_x, d> + O(|d|^2) = -(2/s^2) <vNP(x, s), d> + ...   (SPEC11 claim C2).
So for any learner whose loop displacement d is second order, (cost change of u) / M_uu = -rho, with
rho = <d_even, vNPe> / |vNPe|^2. This identity is a CHECK of the measurement for every loop (section 11, V4), and it
is the reason the trainability effect of a learner can be read without knowing its law.

1.9 Known relatives (so that nothing here is mistaken for new when it is not; from a literature scout, 2026-11-23):
the leading-order term of a = c/(2-c) is the "implicit gradient regularisation" of gradient descent (Barrett & Dherin
2021: loss + (h/4)|grad|^2); the factor (1+mu)/(1-mu) for heavy ball is in Ghosh et al. 2023; the look-ahead family is
in Kovachki & Stuart 2021, whose formula vanishes at gamma* but who do not treat that point; for Adam, Cattaneo et al.
2024 find a correction proportional to (1+b1)/(1-b1) - (1+b2)/(1-b2) (valid for runs much longer than 1/(1-b2) steps),
which is the origin of K4's "two lags" hypothesis. What this spec adds: exact one-mode forms, the look-position rule and
look matrix, the loop / cost (trainability) reading, and the tests on a real network.

## 2. Files, environment, what you may touch

You write NEW files only, all inside `task_cycle_rotation/spec12/` of the project folder (create it). Never modify,
move or copy-edit any existing file (not `part9lib.py`, not anything in `task_cycle_rotation/spec11/`). Read-only inputs:

* `task_cycle_rotation/spec11/bundle/` - the data bundle used on the rig: `part9lib.py` (the library), `cfg11.json`,
  `run/starts/` (stored start weights), `mnist/` (idx files), `ref11.json`. It is complete; do not rebuild it.
* `task_cycle_rotation/spec11/spec11_common.py` - reviewed helpers of the previous spec. IMPORT it (add its directory to
  `sys.path`) and reuse, unchanged: `load`, `COMBOS`, `combo_from_ctx`, `make_combo`, `label_dir`, `gauss_dir`, `kernel`,
  `q_from_kernel`, `q`, `ksum`, `finite`, `reference`, `ref_usable`, `even`, `env_unit`, `file_md5`, `signature`,
  `unit_path`, `vec_path`, `read_unit`, `unit_done`, `load_vec`, `run_unit`, `spearman`, `through_origin`, `within`,
  `sign`. Do not copy their code; do not re-implement the exact jump, the nearest-point solver, `law`, or the
  measurement (`L.jump`, `L.jump_a`, `L.law`, `L.even_stats`, `L.Start`, `L.Problem` come from the library).
* `task_cycle_rotation/spec11/test_spec11.py` - shows how a tiny problem is built for tests (`tiny_combo`). Read it.
* `task_cycle_rotation/SPEC11.md` - the previous spec (its sections 3, 4, 8 are referred to below).
* `$TCR_OUT/precheck12.py`, `precheck12b.py` - the pre-check scripts for this spec (working examples of the
  first-order leg, the closing, the loop and the look matrix on the real network). Read them; your code must reproduce
  their numbers (section 13), but follow THIS text where they differ.

Deliverables (all in `task_cycle_rotation/spec12/`):

1. `spec12_common.py` - learners, legs, closing, polygons, readings, units, predictions, plan.
2. `spec12_run.py` - computes and stores results (section 8).
3. `spec12_report.py` - reads stored results, prints and writes the report (section 9).
4. `test_spec12.py` - tests on a tiny network (section 10).
5. `NOTES12.md` - every decision you made that this spec did not fix, every deviation, every `# spec note:` comment you
   left in the code (list them all with file and line), and the full output of the tests.

Environment: Python 3, torch (float64 everywhere, `L.DT`), numpy; nothing else; NOTHING may install packages. This
machine: Windows, CPU only, 8 cores, and another job of mine may be running. Locally you run ONLY the tiny tests of
section 10 (call `torch.set_num_threads(2)` in them). Do NOT run the real-size stages: I run them. Results go to the
directory given on the command line (it will be on drive D:); never write results under drive C: or inside the project
folder (the tests use a temporary directory and delete it). Mark anything you had to decide yourself with a comment
starting `# spec note:`.

## 3. Loading, combinations, directions, constants

Loading (identical data and settings to the rig; all through the reviewed helpers):

    sys.path.insert(0, <.../task_cycle_rotation/spec11>);  import spec11_common as S
    L, CFG = S.load(bundle_dir, device, threads, out_dir)          # bundle_dir = .../spec11/bundle
    ctx = L.Ctx9(S.COMBOS[cname], bundle_dir + "/run")
    combo = S.combo_from_ctx(L, CFG, ctx)                           # prob, st, sizes s1/s2, dirs E1..E6 W1..W8, S0
    combo["eig"] = {nm: (ctx.dirs[nm], float(ctx.dinfo[nm]["lam_ratio"])) for nm in ("eigA", "eigB", "eigC")}

Combinations: "tanh-init-s0" and "gelu-init-s0" (196-256-256-10 MLP, n = 32, N = 320, P = 118,794, float64).
Sizes: s1 = combo["sizes"]["s1"] (0.02 S0), s2 = combo["sizes"]["s2"] (0.01 S0). Directions: E1..E6, W1..W8 exactly as
SPEC11 section 3 (they are in combo["dirs"]); eigA, eigB, eigC = eigenvectors of K0 with eigenvalue ratio
lam_ratio = lambda/lambda_1 (tanh: 1.0, 0.2314, 0.1007). N = prob.N, P = prob.P, st = combo["st"], F0 = st.F0,
theta0 = st.th0. A direction name is looked up in combo["dirs"] first, then in combo["eig"].

Constants (module level in `spec12_common.py`): EPS = 1e-8; B1 = 0.9; B2 = 0.999; GSTAR = 9.5 (= (1+B1)/(2(1-B1)));
CLOSE_TOL = 1e-6; TAU_MODE = 1e-5; TAU_GEN = 1e-3; CAP_MODE = 4000; CAP_GEN = 6000; CAP_ADAM = 3000; DIVERGE = 1e4;
SIGNS = (1, -1). Probe list of a unit whose first direction is u: QV = [u, W1, ..., W8] (names "u", "W1".."W8").

## 4. Learners (exact definitions)

A learner is a dict with "kind" and parameters. `learner_name(Ld)` returns the names written below (numbers with `%g`).
`g(y; T)` always means: `Fy, cache = prob.fwd(y); r = Fy - T; g = prob.vjp(cache, r) / N`.

EXACT learners (kind "X"; no learner phase, no closing; they reach the target to the library JUMP_TOL):

* SL1 (exact jump; f = 0, a = 1): `L.jump(prob, th, T)` (Jacobian at the start th of the leg). Name "SL1".
* EM (explicit midpoint; f = 1/2, a = 0): `em_close(prob, th, T)` of section 5.2 applied to the whole leg. Name "EM".

FIRST-ORDER learners. Every one runs the SAME loop (section 5.1); they differ only in the step rule. The loop carries
th (weights) and u (the last applied step of THIS leg; zero at the start of every leg).

Kind "E" (Euclidean family). Parameters c, mu (default 0), gamma (default 0), mid (default False; requires mu = 0 and
gamma = 0). Learning rate lr = eta = c * N / lambda_1(K0) with lambda_1 = `st.lam1` of the start theta0 of the UNIT;
the same for every leg of the unit. Names: "GD(c)" (mu = 0, gamma = 0, not mid), "GDmid(c)", "HB(c,mu)" (gamma = 0),
"HBLA(c,mu,gamma)" (gamma != 0, any mu including 0).

    look point y = th + gamma * u            (y = th when gamma == 0)
    not mid:  u_new = mu * u - lr * g(y; T)
    mid:      p = -lr * g(th; T);   u_new = -lr * g(th + 0.5 * p; T)

Kind "A" (Adam family). Parameters kappa, b1 (default B1), b2 (default B2), gamma (default 0), keep (default "none";
one of "none", "m", "v", "mv"). State: m, v (zero vectors) and TWO step counters km, kv (0), one per moment.
Learning rate lr = kappa * dJ / sqrt(P), where dJ = |thJ - theta0| and thJ is the exact-jump end point of the FIRST leg
of the unit with sign +1 (`L.jump(prob, st.th0, F0 + s * X1, Jb=st.J0)`; stored in the reference unit, section 6.1);
the same lr for both signs and every leg. (kappa = the fraction of the weight distance of the exact jump that one
full-size Adam step covers. Halving s halves lr.) Name: "ADAM(kappa,b1,b2,gamma,keep)", e.g. "ADAM(0.1,0.9,0.999,0,none)".

    look point y = th + gamma * u;   g = g(y; T)
    km += 1;  kv += 1
    m = b1 * m + (1 - b1) * g;       v = b2 * v + (1 - b2) * g * g
    u_new = -lr * (m / (1 - b1**km)) / (torch.sqrt(v / (1 - b2**kv)) + EPS)

With b1 = 0 this gives m = g and 1 - b1**km = 1 (km >= 1): no special case in the code. With gamma = 0 and state reset
this must equal `torch.optim.Adam(lr=lr)` with default settings (test T3).
State policy `keep` at the start of every leg AFTER the first leg of a polygon / star (the first leg always starts from
zero state): "none" -> m = v = 0, km = kv = 0; "m" -> v = 0, kv = 0 (m and km kept); "v" -> m = 0, km = 0 (v and kv
kept); "mv" -> everything kept. u is zero at the start of every leg under every policy.

Kind "F" (frozen metric). Parameters kappa, b1 (0 or B1). Name "PRE(kappa)" for b1 = 0, "PREM(kappa)" for b1 = B1.
lr as for kind "A". A fixed positive vector D is computed ONCE per unit from the first leg with sign +1 of the twin
learner ADAM(kappa, B1, B2, 0, none): run its learner phase (section 5.1; no closing) from theta0 toward F0 + s * X1;
with its final v and kv: D = 1 / (torch.sqrt(v / (1 - B2**kv)) + EPS). If the twin does not end "CONVERGED" the unit has
status "NO-TWIN". Step (state m, km; reset at the start of every leg):

    km += 1;  m = b1 * m + (1 - b1) * g(th; T);   u_new = -lr * (m / (1 - b1**km)) * D

After any step rule: `th = th + u_new; u = u_new`.

## 5. Legs, closing, polygons

### 5.1 Learner phase of one first-order leg

    def fo_leg(L, prob, th_start, T, Ld, lr, state, tau, cap, D=None, look=None, hover=0, brk=""):
        s_leg = L.vnorm(T - prob.F(th_start))          # may reuse the first forward pass of the loop
        th = th_start.clone(); u = zeros(P); kleg = 0; hovering = False; left = 0
        loop:
            Fk, cache_k = prob.fwd(th);  rn = L.vnorm(Fk - T)
            if not finite(rn) or rn > DIVERGE * s_leg:                       status = "DIVERGED";     break
            if not hovering:
                if rn <= tau * s_leg and (kleg == 0 or L.vnorm(u) <= tau * L.vnorm(th - th_start)):
                    status = "CONVERGED"
                    if hover == 0: break
                    hovering = True; left = hover; th_stop = th.clone(); steps_stop = kleg; res_stop = rn / s_leg
                elif kleg >= cap:                                              status = "UNCONVERGED";  break
            if hovering:
                [record: see below]
                if left == 0: break
                left -= 1
            y = th + gamma * u if gamma != 0 else th
            (Fy, cache_y) = prob.fwd(y) if gamma != 0 else (Fk, cache_k)       # reuse the forward pass when y is th
            g = prob.vjp(cache_y, Fy - T) / N
            u_new = step rule of section 4 (the "mid" rule takes its second gradient here)
            [look matrix, only if look is not None: section 5.4]
            th = th + u_new;  u = u_new;  kleg += 1
        return dict(theta=th, status=status, steps=kleg, res=rn / s_leg, state=..., and for hover > 0 also
                    theta_stop, steps_stop, res_stop, hover records)

The stopping test is ALWAYS made at the weights th themselves (never at the look point), BEFORE taking a step. The
second condition of the convergence test (last step small compared with the distance travelled) keeps momentum
learners from being stopped while they are passing through the target at speed. With hover = H > 0 the learner is NOT
stopped when it first converges: it takes exactly H further steps (no stopping test except DIVERGED; state and u simply
continue), and the result carries both the weights at the moment of convergence (theta_stop) and after the H extra
steps (theta). Hover records (cheap scalars kept inside the loop): Qh = sum over the hover steps of
|F_(k+1) - F_k|^2 / s_leg^2 (use consecutive Fk of the loop), res_max and res_mean = maximum and mean of rn / s_leg over
the hover steps, and snapshots: a list `hover_snaps` of step counts (given by the caller, e.g. [500, 2000]) at which a
clone of th is kept (the last one equals theta).
tau and cap come from section 7 per part. BREAK "stalelook" (section 8): the look point is th - gamma * u.

### 5.2 Closing (explicit midpoint; also the EM learner)

    def em_close(L, prob, th, T):
        F = prob.F(th);  J = prob.J(th)
        h = L.jump(prob, th, F + 0.5 * (T - F), Jb=J)                 # half-way point, Jacobian at th
        r = L.jump_a(prob, th, T, Jb=prob.J(h["theta"]))              # the whole move from th with the half-way Jacobian
        status = "CONVERGED" if h["status"] == r["status"] == "CONVERGED" else "CLOSE-FAILED"
        return dict(theta=r["theta"], status=status, pay=T - F, look=F + 0.5 * (T - F))

Every first-order leg ends with `em_close(prob, theta_of_learner_phase, T)` unless the learner phase DIVERGED.
Why this learner: it is time-symmetric at second order (measured: loop ratio 0.00006), so it closes the outputs exactly
without adding wear or recharge of its own. Record for every leg: status of the learner phase, steps, res (residual
over s_leg at the stop), close_w = |theta_closed - theta_raw| / |theta_closed - th_start| (closing size in weights),
and the closing status.

Leg status: "OK" iff learner phase "CONVERGED" and closing "CONVERGED" and |F(theta_closed) - T| / s_leg <= CLOSE_TOL.
Otherwise the first failing reason ("DIVERGED", "UNCONVERGED", "CLOSE-FAILED", "NOT-CLOSED", "NONFINITE").
An UNCONVERGED learner phase is STILL closed and the polygon continues (its numbers are stored, the status of the unit
is "UNCONVERGED"); after DIVERGED, CLOSE-FAILED, NOT-CLOSED or NONFINITE the polygon stops.
BREAK "noclose" (section 8): the closing is skipped (theta_closed := theta_raw) and "NOT-CLOSED" is not applied.

### 5.3 Polygon (a closed path of legs)

    def polygon(L, prob, st, Ld, lr, th_start, targets, tau, cap, state=None, D=None, look=None, brk=""):

targets = list of output vectors T_1..T_m; the caller always builds them from the ORIGINAL st.F0 and the last one is
st.F0 itself (the tensor, not a recomputation). Leg i runs from the closed end point of leg i-1 (th_start for i = 1)
to T_i. `state` is the Adam-family state entering leg 1 (None = zero state); before each later leg the keep policy of
section 4 is applied to the state returned by the previous leg. `polygon` returns the list of leg records, the closed
weights after leg 1 (th1) and after the last leg (thE), the state returned by the last leg (so that a star can hand it
to the next polygon, applying the keep policy itself), and status = "OK" iff every leg is "OK" (else the first non-OK
leg status). For SL1 and EM each leg is the exact learner itself (close_w = 0, steps = 0, res = 0; leg "OK" iff the
jumps CONVERGED and the outputs are within CLOSE_TOL).

### 5.4 Look matrix (only where section 7 asks for it: kinds "X" and "E")

C is an N x N float64 matrix, accumulated over ALL legs of a polygon, always relative to the ORIGINAL start outputs
st.F0 and kernel st.K0 of the polygon:

* learner phase of a kind-"E" leg, not mid: two N x N accumulators per leg, A1 = 0, A2 = 0. At every step, with the
  look-point outputs Fy and r = Fy - T:  `o = torch.outer(r, Fy - st.F0); A1 += o; A2 = mu * (A2 + o)`.
  At the end of the learner phase: `C += -(lr / N) / (1 - mu) * (st.K0 @ (A1 - A2))`.
  (Look j pays -(lr/N) r_j in every later step with weights 1, mu, mu^2, ...; A2 removes the part of the geometric
  tail that was cut off when the leg stopped.)
* mid: the applied gradient is the one taken at th + p/2: with Fh = outputs there, `o = torch.outer(Fh - T, Fh - st.F0)`;
  `A1 += o` (A2 stays 0); same final formula with mu = 0.
* closing of a leg, and every EM leg: `C += torch.outer(c["pay"], c["look"] - st.F0)` (c = result of `em_close`).
* SL1 leg from th to T: `C += torch.outer(T - F(th), F(th) - st.F0)`.

Prediction for the polygon: `Vlaw, info = L.law(prob, st, C)` (a length-P vertical vector; `info["kept"]`,
`info["kept_frac"]`). LAW is linear in C: for the even part over the two signs use C_even = 0.5 * (C(+1) + C(-1)) and
call `L.law` ONCE.

## 6. References and readings

### 6.1 Reference unit R(combination, direction x, size)

For sigma in (+1, -1), with t = F0 + sigma * s * x:

    rec, vNP = S.reference(L, prob, st, x, s, sigma);   usable = S.ref_usable(rec)
    thJ = L.jump(prob, st.th0, t, Jb=st.J0)["theta"];   dJ = L.vnorm(thJ - st.th0);   qJ = S.q(L, prob, thJ, QV)

and once: q0 = S.q_from_kernel(L, st.K0, QV); ksum0 = S.ksum(st.K0); nvNPe = |S.even(vNP(+1), vNP(-1))|;
Muu = 2 * nvNPe**2 / s**2; agree = cosine between vNP(+1) and vNP(-1). QV = [x, W1..W8].
Store the two vNP vectors (files) and every scalar/list above (per sign where it has a sign). Status "OK" iff both
signs usable. References needed per combination: eigA, eigB, eigC at s1; E1, E2 at s1 and at s2.

### 6.2 Pair unit (the work-horse): one learner, one closed polygon, both signs

Inputs: learner Ld, list of direction vectors X = [X1, ..., X_(m-1)] (m - 1 >= 1 nodes besides the start), size s,
tau, cap, the reference unit of (X1, s). For sigma in (+1, -1):

    targets = [F0 + sigma * s * Xi for Xi in X] + [F0]          # F0 is st.F0 itself
    run `polygon` from st.th0 with state None (lr and D from section 4; D is computed once, before the sign loop)

EXCURSION = X has one direction (out and back). TRIANGLE = X has two directions [u, w] (F0 -> F0 + s u -> F0 + s w -> F0).
Per sign store: status, the leg records, and (when th1 / thE exist and are finite)

    n1 = |th1 - th0|;  v1 = st.V0(th1 - th0);  d = thE - th0;  v = st.V0(d);  nd = |d|;  nv = |v|
    horiz = |d - v| / max(nd, 1e-300);  close = |F(thE) - F0| / s
    q1 = S.q(L, prob, th1, QV);   KE = S.kernel(L, prob, thE);  qE = S.q_from_kernel(L, KE, QV);  ksumE = S.ksum(KE)

Combined over the two signs (only if both polygons have th1 and thE; `e` = even = half the sum, `o` = odd = half of
[(+1) minus (-1)]); q0, qJ, Muu, nvNPe, dJ, vNP(+-1) come from the reference unit:

    ve, vo   = even / odd part of v;    nve, nvo = their norms           (loop displacement)
    v1e, v1o = even / odd part of v1;   nv1e, nv1o                       (single-leg vertical displacement)
    dq[x](sigma) = qE[x](sigma) - q0[x];   dqe[x], dqo[x]               (cost change of probe x after the loop)
    G(sigma) = mean over W1..W8 of q0[x] / qE[x](sigma);  dGe = even part of (G - 1);  dGo = odd part of (G - 1)
    h[x](sigma) = q1[x](sigma) - qJ[x](sigma);  he[x], ho[x]            (single leg: cost at the end point of the learner
                                                                         minus cost at the end point of the exact jump,
                                                                         SAME outputs)
    Ho = mean over W1..W8 of |ho[x]| / q0[x]                             (first-order effect of fibre motion on costs)
    lamN_e, trK_e, inv_e = even parts of ksumE["lamN"]/ksum0["lamN"] - 1, the same for "trK", and for "inv_mean"

EXCURSIONS only (X1 = u; vNP = the stored references of (u, s)):

    es  = L.even_stats(v(+1),  vNP(+1), v(-1),  vNP(-1))   -> rho, cos, nr        (loop: the law says rho = 2f - 1)
    es1 = L.even_stats(v1(+1), vNP(+1), v1(-1), vNP(-1))   -> rho1, cos1, nr1     (single leg: the law says rho1 = f)
    a_u   = dqe["u"] / Muu          (own-direction wear; the law says 1 - 2f; section 1.8 says a_u = -rho for ANY learner
                                     whose loop displacement is second order)
    phi_u = -he["u"] / Muu          (single-leg cost reading; the law says f)

Unit status = "OK" iff both polygons "OK"; otherwise the first non-OK status (sign +1 first). All numbers that could
be computed are stored either way. No vectors are stored by pair units.

With `want_law` (Part G): accumulate C(sigma) (section 5.4) in both polygons, C_even = 0.5 (C(+1) + C(-1)),
`Vlaw, info = L.law(prob, st, C_even)`, and store: law_ratio = <ve, Vlaw> / <Vlaw, Vlaw>, law_cos = cosine(ve, Vlaw),
n_law = |Vlaw|, kept, kept_frac, uCu = (u @ C_even @ u) / s**2, n_sym, n_asym (Frobenius norms of the symmetric and
antisymmetric parts of C_even). `want_uCu` (Part M): accumulate C the same way but do NOT call `L.law`; store uCu only
(the law says uCu = 2f - 1 on a kernel mode; this checks the accumulation against `f_scalar` without `law`).

### 6.3 Star unit (Part S): twelve excursions in a row

    th = st.th0;  state = None
    for mm in 1..6:
        for sigma in (+1, -1):
            run `polygon` from th with targets [F0 + sigma * s1 * E_mm, F0] and the incoming state
                (ORIGINAL F0; for the very first polygon state = None; before every later polygon apply the keep policy
                 to the state returned by the previous polygon, exactly as between legs)
            if its status is not "OK": stop the unit with that status (checkpoints already made are kept)
            th = thE
        CHECKPOINT mm: K = S.kernel(L, prob, th); store ksum(K), q for [W1..W8] (S.q_from_kernel), G = mean of q0/q over
            W1..W8 (q0 from st.K0), nvc = |st.V0(th - th0)|, nd = |th - th0|, close = |F(th) - F0| / s1, and the leg
            records of this mm.

Adam-family lr: kappa * dJ(E1, s1, +1) / sqrt(P), ONE value for the whole star. tau = TAU_GEN, cap = CAP_ADAM.
OPERATIONAL reading, at the end of a star with all 6 checkpoints, and in a separate unit OP0 at theta0: for x in
(W1, W5): run the learner phase (section 5.1, no closing) of ADAM(0.1, B1, B2, 0, none) (zero state, lr from
dJ(E1, s1, +1)) from the weights th toward F(th) + s1 * x with tau = TAU_GEN, cap = CAP_ADAM; store status, steps, and
r50 = |F(theta_50) - T| / s1 after exactly 50 steps (taken inside the loop when kleg == 50; nan if the leg ended
earlier).

### 6.4 Hover unit (Part H)

One learner, direction u = E1, size s1, snapshots HS = [500, 2000]. For sigma in (+1, -1), T = F0 + sigma * s1 * u:

    r = fo_leg(prob, st.th0, T, Ld, lr, None, TAU_GEN, cap, hover=2000, hover_snaps=HS)
    if r["status"] != "CONVERGED": the unit has that status (store what exists) and no hover readings
    thA = em_close(prob, r["theta_stop"], T)["theta"]                      # stopped as usual, then closed
    for each snapshot n in HS: thB[n] = em_close(prob, snapshot weights, T)["theta"]
    qA = S.q(L, prob, thA, QV);  qB[n] = S.q(L, prob, thB[n], QV)
    GA = mean over W1..W8 of q0[x] / qA[x];   GB[n] likewise;   w[n](sigma) = GB[n] / GA - 1
    wu[n](sigma) = (qB[n]["u"] - qA["u"]) / Muu;   nvh[n](sigma) = |st.V0(thB[n] - thA)|

Store per sign: steps_stop, res_stop, Qh, res_max, res_mean, w[n], wu[n], nvh[n], closing statuses; combined: even parts
we[n], wue[n]. All closings must be "CONVERGED" and closed to CLOSE_TOL, else status "CLOSE-FAILED" / "NOT-CLOSED".
(thA and thB sit on exactly the same outputs T, so w and wu are pure effects of the extra steps.)

### 6.5 Own-metric unit (Part D; exploratory)

For kappa in the list of section 7, direction u = E1, size s1:

    lr, D as for kind "F" (twin ADAM(kappa, B1, B2, 0, none), first leg, sign +1);  sq = torch.sqrt(D)
    probD = L.Problem(F=lambda ph: prob.F(sq * ph), J=lambda ph: prob.J(sq * ph) * sq, N=N, P=P, C=prob.C,
                      fwd=lambda ph: prob.fwd(sq * ph), bwd=lambda cache, r: sq * prob.vjp(cache, r), name="D")
    stD = L.Start(probD, st.th0 / sq)
    references in the D-geometry: recD, vD_ref(sigma) = S.reference(L, probD, stD, u, s1, sigma), usable as usual
    control (covariance): exact-jump loop on probD: j1 = L.jump(probD, stD.th0, stD.F0 + sigma*s1*u, Jb=stD.J0);
        j2 = L.jump(probD, j1["theta"], stD.F0);  vc(sigma) = stD.V0(j2["theta"] - stD.th0);
        ctrl = L.even_stats(vc(+1), vD_ref(+1), vc(-1), vD_ref(-1))            # must give rho = -1, cos = -1
    the ordinary ADAM(kappa, B1, B2, 0, none) excursion (polygon in theta-space, usual closing), both signs:
        vD(sigma) = stD.V0((thE - st.th0) / sq);   own = L.even_stats(vD(+1), vD_ref(+1), vD(-1), vD_ref(-1))
        v(sigma) = st.V0(thE - st.th0);            eucl = L.even_stats(v(+1), vNP(+1), v(-1), vNP(-1))  (stored refs)

Store ctrl, own, eucl (rho, cos, nr each), the statuses, and a summary of D: min, 1%, median, 99%, max (quantiles on
`D[::7]`), the share of coordinates with D > 0.5 / EPS, and the eigenvalue range of K_D = `stD.K0` (lam1, lamN).
If `L.Start` or a D-reference fails (exception, ill-conditioned, not usable) the unit has status "D-FAILED:<reason>".

## 7. Parts and grids (per combination)

Unit names: `<part>__<combo>__<dirs>__<size>__<L.safe_name(learner_name)>` (extra suffix where noted). tau / cap:
Part M: TAU_MODE / CAP_MODE; all other parts: TAU_GEN and CAP_GEN (kind "E") or CAP_ADAM (kinds "A", "F").

Part V. V__<combo>__env: `S.env_unit(L, combo, ref11[combo], with_test=False)` (ref11 = the bundle `ref11.json`).
  V__<combo>__chk: equivalence with the library on the real network, direction E1, size s1, sign +1, 60 steps each
  (tau = 0 so that nothing converges, cap = 60, compared with `L.run_first_order(prob, st, T, s1, Lib)` whose
  stop_rel = 0, cap = 60): GD(1) vs `L.L_GD(1.0, 0.0, 60)`; HB(0.02,0.9) vs `L.L_HB(0.02, 0.9, 0.0, 60)`;
  ADAM(0.1,B1,B2,0,none) vs `L.L_ADAM(lr, 0.0, 60)` with the same lr. Store |theta_mine - theta_lib| / |theta_lib - theta0|.

Part R. References of section 6.1.

Part M (K1, K2). Excursions on kernel modes, size s1, with `want_uCu`. Prediction per unit: c_i = c * lam_ratio;
  f_pred, _ = f_scalar(c_i, mu, gamma, mid); g_pred = 2 f_pred - 1. Learner list LM:
  GD(0.5), GD(1), GDmid(1), HB(0.02,0.9), HB(0.1,0.9), HB(0.1,0.5), HBLA(0.02,0.9,0.9), HBLA(0.02,0.9,4.75),
  HBLA(0.02,0.9,9.5), HBLA(0.02,0.9,14.25), HBLA(0.1,0.9,9.5), HBLA(0.2,0,0.5).
  Directions: eigA: all 12; eigB: GD(1), GDmid(1), HB(0.1,0.9), HBLA(0.1,0.9,9.5); eigC: HB(0.1,0.9), HBLA(0.1,0.9,9.5).
  A pair whose f_pred is nan is not run (plan entry "SKIPPED-UNSTABLE").

Part G (K3). Excursions with `want_law`, size s1. On E1: SL1, EM, GD(1), HB(0.1,0.9), HBLA(0.1,0.9,9.5), HB(0.02,0.9).
  On E2: HB(0.1,0.9), HBLA(0.1,0.9,9.5). (HB(0.02,0.9) is expected to stop at the cap on these directions; it is kept
  on purpose: the closing is part of the look matrix, so the law must hold for it too.)

Part A (K4, K5). Excursions. "ADAM(k)" below means ADAM(k, B1, B2, 0, none); only the changed parameters are written.
  A0 controls: SL1, EM on (E1, E2) x (s1, s2).
  A1 core: ADAM(k), k in {0.03, 0.1, 0.3, 1.0}, on (E1, E2) x (s1, s2).
  A2 first-moment series: E1, (s1, s2), k = 0.1: b1 in {0, 0.5, 0.8, 0.95}.
  A3 second-moment series: E1, s1, k in {0.03, 0.1}: b2 in {0.5, 0.9, 0.99}.
  A4 look-ahead series: E1, s1: k = 0.1 with gamma in {0.5, 1, 2, 4, 9.5}; k = 0.03 with gamma in {1, 4, 9.5}.
  A5 frozen metric: E1, (s1, s2), k in {0.03, 0.1}: PRE(k), PREM(k).
  A6 kept moments: (E1, E2) x (s1, s2), k = 0.1: keep in {"m", "v", "mv"}; and E1, (s1, s2), k = 0.3: keep = "mv".
  A7 tolerance check: E1, s1, ADAM(0.1) with tau = 1e-4 and cap = 6000 (name suffix `__tau1e-4`).

Part T (K5). Triangles X = [E1, E2], sizes s1 and s2: SL1, EM, GD(1), ADAM(0.03), ADAM(0.1), ADAM(0.3).
  (Name: dirs = "E1+E2". The reference unit used for lr / q0 is R(E1, size); QV = [E1, W1..W8].)

Part H (K8). Hover units on E1, s1: GD(1), ADAM(0.03), ADAM(0.1), ADAM(0.3), ADAM(0.1) with b2 = 0.9,
  ADAM(0.1) with b1 = 0.95.

Part S (K6). Star units: SL1, EM, ADAM(0.03), ADAM(0.1), ADAM(0.3), ADAM(0.1) with keep = "mv",
  ADAM(0.1) with b1 = 0.95, ADAM(0.1) with b2 = 0.9; plus the unit OP0.

Part D (K7). Own-metric units for k in {0.03, 0.1, 0.3}.

## 8. `spec12_run.py`: stages, storage, resume

Command line (argparse): `--bundle` (default `<this file>/../spec11/bundle`), `--out` (REQUIRED), `--combos`
(default "tanh-init-s0,gelu-init-s0"), `--parts` (default "V,R,M,G,A,T,H,S,D"), `--threads` (int, default 4),
`--device` (default "cpu"), `--quick` (flag), `--break` (default ""), `--only` (substring; run only units whose name
contains it; references they need are computed if missing), `--list` (print the plan - every unit name, and for Part M
its c_i, f_pred, g_pred - and exit without loading the network).

Order of work: for each combination in order: V, R, M, G, A, T, H, S, D. Every unit goes through
`S.run_unit(L, out, name, sig, fn)` and is skipped when `S.unit_done(out, name, sig)` (resume). References store their
vNP vectors (`S.vec_path`, loaded with `S.load_vec`). signature = `S.signature(settings, lib_path, cfg_path)` with
settings = a dict holding every constant of section 3, every grid of section 7, the quick flag and the break value.
Print one line per finished unit (flush=True) with its key numbers and seconds; print `SPEC12 RUN COMPLETE` at the end
only if every planned unit exists with the current signature. One failed unit never stops the run (`S.run_unit` stores
status "ERROR"; such units are retried on the next start). Call `L.free_memory()` after every unit. Inside the star,
hover and own-metric units catch exceptions of a learner run and end the unit with status "EXC:<type>", keeping what
was stored so far (as SPEC11 amendment A2.3).

QUICK: combination tanh only; R: eigA s1, E1 s1, E2 s1; M: eigA with GD(1), HB(0.1,0.5), HBLA(0.2,0,0.5); G: E1 with EM
and GD(1); A: E1 s1 with SL1, EM, ADAM(0.3), PREM(0.3), ADAM(0.3) keep "mv"; T: s1 with SL1 and ADAM(0.3);
H: ADAM(0.3) with hover 300 and snapshots [100, 300]; S: SL1 and ADAM(0.3) with mm = 1..2, and OP0; D: k = 0.3.
(The signature differs, so QUICK results never mix with full results.)

BREAK (negative controls; default ""): "stalelook" = every look-ahead learner looks at th - gamma*u (section 5.1);
"noclose" = closings are skipped (section 5.2); "repro" = the unit M__tanh-init-s0__eigA__s1__GD_1_ is run twice and
both results stored (second one with suffix `__repro`). With "stalelook" rule HM1 must FAIL on the HBLA rows; with
"noclose" validation V3 must FAIL; with "repro" the two records must be identical in every number.

## 9. `spec12_report.py`: derived numbers and rules

Usage: `python spec12_report.py --out <dir>`; prints the report and writes `<dir>/spec12_report.txt` and
`<dir>/spec12_report.json` (every derived number and every rule verdict). It reads stored units only (no network).
Verdicts: PASS / FAIL / NO DATA (fewer usable units than the rule needs). Never tune a threshold; print the numbers.
Definitions used below: a unit is USABLE if its status is "OK"; in Part M also if its status is "UNCONVERGED" and
every leg has res <= 1e-4 (the closing finished the job; mark such rows with `*`).
Gam(unit) = dGe(unit) / dGe(SL1 unit of the same direction(s) and size, Part A0) (wear of the probe gauge in units of
an exact-jump excursion; needs both units).
x_even = log2(nve(s1) / nve(s2)), x_odd = log2(nvo(s1) / nvo(s2)), x1_odd = log2(nv1o(s1) / nv1o(s2)) for the same
learner and direction(s) at the two sizes.

First block - validations (section 11). Then, per combination:

Part M table: learner, direction, c_i, f_pred, g_pred, status, steps (max over legs), res, close_w (max), rho, cos, nr,
  a_u, rho1, phi_u, uCu.
  HM1 LOOP LAW WITH THE LOOK-POSITION RULE: unit passes iff |rho - g_pred| <= 0.03 max(1, |g_pred|) and
      (|g_pred| >= 0.1: cos * sign(g_pred) >= 0.99;  |g_pred| < 0.1: nr <= |g_pred| + 0.05).
      PASS iff >= 90% of the usable units pass and at least 14 units are usable.
  HM2 COST READING: |a_u + g_pred| <= 0.05 max(1, |g_pred|); same PASS rule.
  HM3 SINGLE LEG: |rho1 - f_pred| <= 0.03 max(1, |f_pred|) and |phi_u - f_pred| <= 0.05 max(1, |f_pred|); same PASS rule.
  HM4 LOOK MATRIX ON A MODE: |uCu - g_pred| <= 0.02 max(1, |g_pred|); same PASS rule.
  Also print, from the eigA rows: measured a = -rho against the closed forms of section 1.4 for GD and HB, the ratio
  a(HB(0.02,0.9)) / [c/(2-c) at c = 0.2] (momentum amplification at equal long-run rate; predicted
  1.910 / 0.1111 = 17.19), and the look-ahead series a(gamma) for HB(0.02, 0.9): gamma = 0, 0.9, 4.75, 9.5, 14.25.

Part G table: learner, direction, status, steps, res, close_w, nve/nvNPe, rho, cos, law_ratio, law_cos, n_law/nvNPe,
  kept, kept_frac, uCu, n_asym/n_sym, a_u, Gam.
  In Part G a unit is USABLE if its status is "OK" or "UNCONVERGED" (every closing converged and closed).
  HG1 LOOK-MATRIX LAW: unit passes iff (n_law/nvNPe >= 0.05: |law_ratio - 1| <= 0.05 and law_cos >= 0.98;
      n_law/nvNPe < 0.05: nve/nvNPe <= 0.1). PASS iff every usable unit passes and >= 6 units are usable.
  HG2 LOOK-AHEAD ON A GENERIC DIRECTION (E1 and E2): nve(HBLA(0.1,0.9,9.5)) / nve(HB(0.1,0.9)) and the same ratio
      for a_u and Gam. PASS iff both nve ratios <= 1/3 (both units "OK").

Part A table (one row per unit): direction, size, learner, status, steps, res, close_w, nve/nvNPe, nvo/nve, rho, cos,
  a_u, Gam, lamN_e / lamN_e(SL1), rho1, phi_u, nv1o/dJ, Ho; then a table of exponents for every learner run at both
  sizes: x_even, x_odd, x1_odd, a_u(s1), a_u(s2).
  HA0 SECOND ORDER (A1): a (direction, k) pair passes iff x_even in [1.8, 2.2] and
      |a_u(s1) - a_u(s2)| <= 0.05 max(1, |a_u(s2)|). PASS iff >= 90% of the pairs with both units usable pass (>= 6 pairs).
  HA1 FIRST-ORDER FIBRE MOTION (A0, A1): every usable A1 unit has nv1o/dJ >= 0.3, every A1 pair has x1_odd in
      [0.9, 1.1], and every usable A0 unit has nv1o/dJ <= 0.02. PASS iff all hold.
  HA2 FIRST-MOMENT LAG RAISES WEAR (A2 plus ADAM(0.1) of A1; E1; each size separately): over the usable units with
      b1 in {0.5, 0.8, 0.9, 0.95}: Spearman(b1, a_u) = 1 (needs >= 3 units), and the least-squares line
      a_u = A * (1+b1)/(1-b1) + B has A > 0 and R^2 >= 0.9. Print A, B, R^2, and the same for Gam. (b1 = 0 is reported
      only: in the pre-check it never converges.)
  HA3 SECOND-MOMENT LAG LOWERS WEAR (A3 plus ADAM(k) of A1; E1, s1; each k separately): over the usable units with
      b2 in {0.5, 0.9, 0.99, 0.999}: Spearman(b2, a_u) <= -0.8 (needs >= 3 units). Print a_u and Gam per b2.
  HA4 LOOK-AHEAD LOWERS WEAR (A4 plus ADAM(k); E1, s1; k = 0.1): over the USABLE units of the gamma series (gamma = 0
      included): Spearman(gamma, a_u) <= -0.9 (needs >= 4 units, else NO DATA). Print status, steps, res, a_u and Gam
      per gamma for both k. (In the pre-check gamma = 9.5 at k = 0.1 never settles: look-ahead removes the averaging
      that lets Adam converge at a fixed learning rate. Where the series stops converging is itself a result.)
  HA5 FROZEN METRIC (A5; each k, size s1): (i) a_u(PREM) > a_u(PRE) and Gam(PREM) > Gam(PRE); (ii) a_u(ADAM(k)) <
      a_u(PREM(k)) (the drifting second moment lowers the wear). PASS iff (i) and (ii) hold at both k; print all values.
  HA6 KEPT MOMENTS (A6 plus ADAM(k) of A1): keep "v" and "mv": x_odd in [0.8, 1.2] and nvo/nve >= 1 at s1;
      keep "m" and "none": nvo/nve <= 0.5 at s1. PASS iff all usable (direction, k) cases hold (>= 4 cases per group).
  Reported without verdict: the k series of A1 (a_u, Gam, steps against k) on both directions; and a separate table
  "learners that never settle" with every UNCONVERGED Part A unit: steps, res, close_w, a_u, Gam, nve/nvNPe (they
  were closed by the midpoint learner after CAP_ADAM steps of hovering).

Part T table: learner, size, status, nve, nvo, nvo/nve, dGe, dGo, Gam (against the SL1 triangle of the same size), and
  x_even, x_odd per learner.
  HT FIRST ORDER ON A TRIANGLE: each ADAM learner: x_odd in [0.8, 1.2] and nvo/nve >= 1 at s1; SL1, EM, GD(1):
      nvo/nve <= 0.2 at s1, and x_even in [1.8, 2.2] for SL1 and GD(1). PASS iff all usable learners hold (>= 5).

Part H table: learner, steps_stop, res_stop, and per snapshot n: res_mean, Qh (at the end), nvh/nvNPe(E1, s1),
  Wg[n] = -we[n] / |dGe(SL1, E1, s1)|, Wu[n] = wue[n] (wear in units of one exact-jump excursion; positive = wear).
  HH HOVER WEAR (exploratory): "supported" iff GD(1) has |Wg[2000]| <= 0.05 and at least one ADAM row has
      Wg[2000] >= 0.1 with Wg[2000] >= 1.5 * Wg[500]; "not supported" iff every ADAM row has |Wg[2000]| <= 0.05;
      otherwise "mixed". Print everything.

Part S table: per learner and checkpoint mm: G, lamN/lamN0, trK/trK0, inv_mean ratio, nvc; then
  Rstar = (1 - G_6) / (1 - G_6(SL1)), and the operational readings (steps and r50 at the end against OP0).
  HS1 CONTROL: EM has |Rstar| <= 0.05 and SL1 has G_6 < 1.
  HS2 ORDER: over the usable reset learners (ADAM(0.03), ADAM(0.1), ADAM(0.3), b1 = 0.95, b2 = 0.9):
      Spearman(Rstar, Gbar) >= 0.9 with Gbar = mean of the single-loop Gam at s1 over the directions available in Part A
      for that learner (needs >= 4 learners).
  HS3 SIZE: for the same learners |Rstar - Gbar| <= 0.1 + 0.3 |Gbar|. PASS iff >= 80% hold.
  Reported: the keep = "mv" star against ADAM(0.1); Spearman between Rstar and ln(r50_end / r50_OP0) over all learners.

Part D table: k, status, D summary, ctrl (rho, cos), own (rho, cos, nr), eucl (rho, cos, nr).
  HD0 CONTROL: ctrl rho within 0.03 of -1 and cos <= -0.99 for every usable unit (if this fails, K7 is not read).
  HD OWN METRIC (exploratory): "supported" iff |own cos| >= 0.8 at every usable k; "not supported" iff |own cos| <= 0.3
      at every usable k; otherwise "mixed".

Last block: one line per rule and combination (HM1-4, HG1-2, HA0-6, HT, HH, HS1-3, HD0, HD) with verdict and the
numbers it used, then the list of every unit that is not "OK" with its status.

## 10. `test_spec12.py` (tiny network; run it; all must pass)

Build the tiny problems exactly as `test_spec11.py` does (`tiny_combo([6, 8, 8, 2], ...)`: n = 5, N = 10, P = 146, sizes
already scaled by 1/16; and the linear `[6, 2]`), add `combo["eig"] = {"eigA": (st.Q[:, 0].clone(), 1.0)}`. Use a
temporary output directory (deleted at the end). Print PASS/FAIL per test and exit with code 1 on any failure. The
tests must call the SAME functions of `spec12_common.py` that the real run uses.

T1  `f_scalar` against the closed forms of section 1.4 (GD, GDmid, HB at c in {0.1, 0.5, 1.0} x mu in {0, 0.5, 0.9}
    where stable): |difference| <= 1e-9; f_scalar(0.02, 0.9, 9.5) = 0.46146 +- 1e-5; f_scalar(0.02, 0.9, 14.25) > 0.5;
    f_scalar(1.2, 0.0, 0.5) is nan.
T2  Loop law on the tiny net along eigA at s1 for GD(0.5), GDmid(1), HB(0.1,0.5), HBLA(0.2,0,0.5), HB(0.02,0.9),
    HBLA(0.02,0.9,9.5): HM1 passes for each; uCu within 0.02 max(1,|g_pred|) of g_pred (HM4).
T3  Library equivalence on the tiny net (25 steps, as V__chk): GD, HB, ADAM: relative difference <= 1e-9.
T4  Closing: for ADAM(0.3) and GD(1) excursions |F(thE) - F0| / s <= 1e-6; the EM excursion has nr <= 0.05 and
    |a_u| <= 0.05; the SL1 excursion has rho within 0.03 of -1, a_u within 0.05 of 1, phi_u within 0.05 of 0; and
    |a_u + rho| <= 0.03 max(1, |rho|) for all four.
T5  Look-matrix law on the tiny net, generic Gaussian direction E1, for SL1, GD(0.5), HB(0.1,0.5), HBLA(0.2,0,0.5):
    the HG1 condition holds for each (whatever the status: a leg that stops at the cap is closed, and the closing is
    part of the look matrix).
T6  State policies on the tiny net with ADAM(0.3): after a two-leg polygon the returned counters are
    keep "none": km = kv = steps of leg 2;  "mv": km = kv = steps of leg 1 + leg 2;  "m": km = sum, kv = steps of leg 2;
    "v": km = steps of leg 2, kv = sum.
T7  Frozen metric = gradient descent in re-scaled weights: with a random positive D (exp of a Gaussian vector times
    0.5), 20 steps of the kind-"F" rule with b1 = 0 from theta0 (tau = 0, cap = 20, D given directly) equal
    sq * (20 steps of kind "E" GD with the same lr on `probD` of section 6.5 from theta0 / sq): relative difference <= 1e-10.
    (Call `fo_leg` with the lr passed directly in both cases.)
T8  Mirror symmetry on the LINEAR tiny net with ADAM(0.3): the excursion with keep "none" returns to the start
    (nd / n1 <= 1e-7 for both signs); with keep "mv" it does not (nd / n1 >= 1e-4).
T9  Resume: run a QUICK-like tiny plan twice into the temporary directory; the second run recomputes nothing and the
    files are unchanged.
T10 BREAK: "stalelook" makes T2 FAIL for the two HBLA learners and leaves the other four passing; "noclose" makes the
    closure check of T4 FAIL for ADAM(0.3); "repro" gives identical records.
T11 Star (mm = 1..2) on the tiny net for SL1, EM and ADAM(0.3): two checkpoints each; SL1 has G_2 < 1; EM has
    |1 - G_2| <= 0.2 (1 - G_2(SL1)).
T12 Hover unit on the tiny net for ADAM(0.3) with hover 100, snapshots [50, 100]: runs, thA and thB closed to 1e-6.
T13 Own-metric unit on the tiny net (k = 0.3): the control gives rho within 0.03 of -1.
If a test fails because the PHYSICS disagrees with this spec (not because of a bug), do not tune tolerances and do not
change sizes: report it in NOTES12.md and in your final message, with the numbers.

## 11. Validations the report prints first

V1 environment: `V__<combo>__env` has pass = True. FAIL -> "ENVIRONMENT MISMATCH" at the top, every rule INVALID.
V2 library equivalence: the three numbers of `V__<combo>__chk` are <= 1e-9.
V3 closure: max close over all OK polygons <= 1e-6; max close_w over all OK first-order legs (report it; flag > 0.05);
   max horiz over OK excursion units with nve/nvNPe >= 0.1 is <= 1e-2.
V4 universal identity (section 1.8): for every usable excursion unit, |a_u + rho| <= 0.03 max(1, |rho|). Report the
   pass share separately for (Parts M, G), (Part A with keep "none" or "m"), (Part A with keep "v" or "mv"; these may
   legitimately fail, their displacement is first order).
V5 controls: every SL1 excursion: rho within 0.03 of -1, cos <= -0.99, a_u within 0.03 of 1; every EM excursion:
   nr <= 0.05 and |a_u| <= 0.03.
V6 references: all usable; agree >= 0.99.
V7 tolerance independence (A7): |a_u(tau 1e-4) - a_u(tau 1e-3)| <= 0.02; report nvo(tau 1e-4) / nvo(tau 1e-3).
V8 excluded units: list every unit not "OK" with its status; a rule that has fewer usable units than it needs says
   NO DATA.

## 12. Expected cost (for your information)

One Jacobian is 320 x 118,794 (about 300 MB, float64) and takes about 1.3 s on this CPU; a gradient about 8 ms; a
reference about a minute; one `L.law` call up to a few minutes. Parts M and G: about 30 units per combination, 1-6
minutes each; Parts A, T, H, S, D: about 110 units, 20 s to 3 min each. Do not optimise anything in a way that changes
numbers (no float32, no approximate solves, no batching of different learners). Keep at most three Jacobians alive at
a time (delete them and call `L.free_memory()`); never store a Jacobian or a P-vector in a unit record.

## 13. Pre-check numbers (tanh network; the scripts; your full-size code must reproduce them when I run it)

Part M, eigA (lam_ratio 1), s1, loop rho measured / predicted (tau 1e-6 in the pre-check):
GD(0.5) -0.3334 / -0.3333; GDmid(0.5) +0.0769 / +0.0769; GDmid(1) +0.3334 / +0.3333; HB(0.02,0.9) -1.9102 / -1.9101;
HBLA(0.02,0.9,9.5) -0.0771 / -0.0771; HBLA(0.1,0,0.5) -0.0053 / -0.0053; GD(0.1) -0.0526 / -0.0526; |cos| = 1.0000 each.
Part A, E1: exact-jump loop a_u = 1.0075 (s1), 1.0019 (s2); ADAM(0.03): a_u -0.054 / -0.052 (s1 / s2), rho +0.054,
cos +0.067, steps 549; ADAM(0.1): a_u +0.257 / +0.254, rho -0.256, cos -0.262, steps 197; ADAM(0.3): a_u +0.408 / +0.406,
rho -0.407, cos -0.144, steps 127; x_even 2.03 / 1.99 / 1.98; nv1o/dJ 0.85 / 0.82 / 0.95 with x1_odd 1.01; nvo/nve at s1
0.20 / 0.27 / 0.26; closing size about 0.003 of dJ. (In the pre-check the gauge ratio was taken over 9 probes including
u: -0.044, +0.294, +0.481.) ADAM(0.1) with b1 = 0: 6000 steps, residual
0.2-0.4 (UNCONVERGED), a_u +5.66 / +5.28, gain ratio +9.95 / +9.89; ADAM(0.1) with gamma = 9.5: 6000 steps,
residual 0.01-0.03 (UNCONVERGED), a_u +5.59 / +5.40, ratio +9.89 / +9.91 (the pre-check cap was 6000 steps).

## 14. Amendment A1 (2026-11-23, late evening; after the rest of the pre-check and before any full-size run; overrides the sections it names)

Pre-check results that arrived after sections 0-13 were written (tanh, E1; $TCR_OUT/pre12_adam_tanh.log,
pre12_metric_tanh.log, pre12b_tanh.log):

* ADAM(1.0): a_u = -1.76 / -1.83 (s1 / s2), probe-gauge ratio -1.72 / -1.79, x_even 1.94, 149 steps. Together with section
  13: a_u = -0.05, +0.26, +0.41, -1.8 at k = 0.03, 0.1, 0.3, 1.0. The sign changes twice; at the largest rate the loop
  makes the network EASIER to train. Not predicted.
* Moments kept ("mv"), k = 0.1: x_odd = 0.99, nvo/nve = 2.96 at s1, a_u = +0.486 / +0.482 (reset: +0.257 / +0.254).
* b1 = 0 at k = 0.3: 6000 steps, never settles, a_u = +67 / +57 (k = 0.1: +5.7 / +5.3): hover wear grows like lr^2.
* gamma = 9.5 at k = 0.03: settles only after 1400-6000 steps; a_u = -0.06 / +0.14 (not a clean regime).
* Own metric (k = 0.1): control rho = -1.0029, cos = -1.0000; own rho = -0.30, cos = -0.47; ordinary geometry rho = -0.26,
  cos = -0.26. cond(K_D) = 1008 against cond(K) = 931: the scale factors do not flatten the kernel spectrum here.

A1.1 (section 7, Part A): A1 core additionally runs ADAM(0.01) on E1 at (s1, s2).
     A2 first-moment series: k in {0.1, 1.0} (was 0.1 only); sizes (s1, s2) for k = 0.1 and s1 for k = 1.0.
     A3 second-moment series: k in {0.1, 1.0} (was {0.03, 0.1}); size s1.
A1.2 (section 7, Parts H and S): add ADAM(1.0) to both learner lists.
A1.3 (section 9, Part A): HA2 and HA3 are evaluated separately for each k of their series (verdict per k).
     New rule HA7 SIGN PATTERN OF THE RATE SERIES (registered now, from tanh / E1 only; the test is whether it repeats on
     E2 and on the gelu network): for each (combination, direction) with the four A1 units k = 0.1, 0.3, 1.0 usable at
     s1: a_u(0.1) > 0, a_u(0.3) > 0, a_u(1.0) < 0, and Gam has the same three signs. PASS iff it holds in all (at least 3)
     usable (combination, direction) cases. Print the full k series (0.01 ... 1.0) of a_u, Gam, steps, nv1o/dJ, nve/nvNPe.
A1.4 (section 9, Part S): HS2 and HS3 use the reset learners ADAM(0.03), ADAM(0.1), ADAM(0.3), ADAM(1.0), b1 = 0.95,
     b2 = 0.9 (a negative Gbar predicts Rstar < 0, i.e. G_6 > 1).
A1.5 (section 9, Part H): the hover table also prints Wg and Wu per snapshot for ADAM(1.0); the verdict rule is unchanged.
* Look-matrix law on E1 (pre12b_tanh.log; legs stopped at 1e-3 or at 6000 steps, closed with the midpoint learner):
  measured / LAW(C): HB(0.02,0.9) 0.9995, HBLA(0.02,0.9,9.5) 0.9995, GD(1) 0.9997; cosine 1.0000 each; 11-12 law terms,
  about 40 s per `L.law` call. All three have nve/nvNPe = 0.286-0.288 and are nearly orthogonal to vNPe: on a generic
  direction the displacement is dominated by the ANTISYMMETRIC part of C (n_asym 8.8e-3 to 9.2e-3 for all three), while
  the symmetric part (the wear) is 2.6e-3 (HB), 1.1e-3 (GD(1)), 8.9e-5 (look-ahead). The antisymmetric part matches a
  closed form (below) to 0.06% for the look-ahead learner: predicted 8.788e-3, measured 8.793e-3.

A1.6 (sections 6.2, 9; Part G). SPECTRAL AREA. For a learner that follows gradient flow closely, the out-and-back along
     u = sum_i a_i q_i (q_i, lambda_i = eigenvectors / eigenvalues of K0; a = st.Q.T @ u) goes out fast modes first and
     comes back fast modes first, so the path encloses an area: the antisymmetric part of the loop look matrix is, in the
     eigenbasis, Apred_ij = -s^2 a_i a_j (lambda_i - lambda_j) / (lambda_i + lambda_j) (row = payment, column = look
     position). In every `want_law` unit additionally store, with Casym = 0.5 (C_even - C_even^T) and
     Apred = st.Q @ Apred_eig @ st.Q.T:  area_ratio = <Casym, Apred> / <Apred, Apred>, area_cos = <Casym, Apred> /
     (|Casym| |Apred|) (Frobenius), n_asym_pred = |Apred|.
     HG2 is REPLACED by: look-ahead removes the wear part, not the area part: n_sym(HBLA(0.1,0.9,9.5)) / n_sym(HB(0.1,0.9))
     <= 1/3 on E1 and on E2; print the nve ratio too (expected near 1).
     New rule HG3 SPECTRAL AREA: GD(1) and HBLA(0.1,0.9,9.5) (each usable unit): area_cos >= 0.98 and
     |area_ratio - 1| <= 0.05; printed for every other Part G unit (SL1 and EM have no area: n_asym / n_asym_pred <= 0.05).
     (If area_cos comes out near -1 the sign convention of the formula is wrong, not the code: report it, do not flip it.)

## 15. Amendment A2 (2026-11-23, late evening; after reviewing the delivered code line by line)

Review result: `spec12_common.py` (1,208 lines), `spec12_run.py`, `spec12_report.py`, `test_spec12.py` read in full. The
learner loop, the closing, the four state policies, the look matrix (with the tail correction), the readings and the
plan follow sections 3-9. the implementation's files are kept unchanged as `*.implementation`; the edits are marked `NOTE (review)`:
the grids of A1.1 / A1.2, the spectral-area fields and rules of A1.6, rule HA7 and the other report changes of A1.3 /
A1.4, and test T8 (below). After the edits all 14 tiny-network tests pass (T1-T13 and the report smoke test).

A2.1 (section 10, T8). Testing found that late in a long Adam run on the tiny linear network rounding errors are
     amplified about 1e9-fold (the two legs are mirror images to 1.7e-13 for the first 400 steps, to 5e-4 at step 480,
     in the stretch where the residual rises again), so "returns to the start within 1e-7" cannot be reached in
     float64. T8 is now: (a) 100 steps toward +T and toward -T from the same start give mirror-image displacements
     within 1e-9 (measured 1.4e-13); (b) the excursion with reset moments returns to the start within 1e-3 of the leg
     length (measured 1.5e-5, 4.7e-6) and with kept moments it does not (>= 1e-3 and at least 100 times larger;
     measured 3.6e-2). CONSEQUENCE FOR THE REAL RUN: a small odd part of a reset-Adam loop can be amplified rounding or
     stopping discreteness, not physics; V7 (tighter tolerance) and the exponent of the odd part are the checks.
A2.2 Implementation readings accepted as written (its `# spec note:` list, NOTES12.md): hover statistics over the hover steps only;
     a failed exact jump is "CLOSE-FAILED"; an UNCONVERGED leg whose closing also fails takes the closing reason; one
     reference unit holds both signs; a pair unit runs even if its reference is unusable (V6 reports it); star and
     hover units guard only the learner runs; QUICK plans Part V and reads "all checkpoints" as its own 2; BREAK
     "noclose" also skips the hover closings; unit names of hover / star / own-metric units; an UNCONVERGED Part M unit
     is usable only if every closing converged; V3 counts star checkpoints; V4 counts SL1, EM, PRE, PREM in the
     "none / m" group; HS3 needs the same 4 learners as HS2; the report takes --bundle / --quick / --break to pick the
     signature; the tests write to a temporary directory on D:.
A2.3 Differences between the pre-check scripts and this spec, found by testing (the spec stands): the "adam" stage
     took lr from the exact-jump distance of sign -1 (relative difference of order s), kept one step counter, and
     used 9 probes (u included) for the gauge ratio; the "modes" stage used tau 1e-6 and cap 6000.
A2.4 (section 11, V3) The horiz bound is a report-only validation: on the tiny network Adam excursions have horiz 0.22
     (a strongly curved toy); what it is on the real network is read from the report, and a failure there is reported,
     not tuned.
