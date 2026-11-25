# Where on the solution set does a learner land?

Train a network until it hits given target outputs on its training inputs. Many weight settings produce exactly those
outputs: together they form a surface in weight space (the *fibre*). Different learners (one exact Gauss-Newton jump,
gradient flow, gradient descent, momentum, implicit and midpoint steps, ...) land at different points of that surface.

This repository shows that, at second order in the size of the move and at all orders in the step size, **where a
learner lands is set by one number `f`, computable in closed form from its step sizes alone**, and that this number
predicts the change of sharpness, the learner's predictions on unseen inputs, its test loss at fixed training outputs,
and how fast it loses trainability under changing targets. Every claim was tested with predictions and pass rules
written before the run.

Full record, strongest result first: **[FINDINGS.md](FINDINGS.md)**. All laws in one table: **[LAWS.md](LAWS.md)**.

## Headline results

| | Result | Where |
|---|---|---|
| Law | Alignment fraction `f` (0 = exact Gauss-Newton jump, 1/2 = gradient flow, 1 = nearest point of the fibre); damped Gauss-Newton / GD per kernel mode `f = (1-c)/(2-c)`; heavy ball, proximal point and midpoint in closed form. First order in the step reproduces implicit gradient regularisation; the pole at `c = 2` is the edge of stability. | FINDINGS Part 1 #3, stages 04-06, 08 |
| Real network | MLP 196-256-256-10 on MNIST (118,794 parameters), tanh and gelu, init and trained: **960/960** pre-registered units pass; median relative error ~1e-6. | stage 08 |
| Why | `f` is the average *evaluation point* of the learner's Jacobians: pre-point (Ito) = 0, midpoint (Stratonovich) = 1/2, post-point = 1. An alpha-point jump has `f = alpha` for any alpha. `f` is the `z^2` coefficient of the learner's stability function. | stages 08-09 |
| Third order | Learner differences need exactly three functions, weighted by the Runge-Kutta order-condition sums of the step sequence; rank 3 and residual 0.00000 on MNIST with nothing fitted. | stage 11 (x7, x8) |
| Generalisation | Gradient flow = kernel regression with the tangent kernel at the midpoint of the path; every other learner = that + `(f - 1/2) G(x)`. With the start function removed, larger `f` lowers test loss at exactly equal training outputs (all pre-registered rules pass; up to -12% loss, +2.1 accuracy points). | stages 09-11 |
| Trainability | An out-and-back excursion by a learner with fraction `f` moves the weights by `(2f - 1) v_NP` (36/36 on MNIST). Kernel depletion follows one clock `t = (1 - 2f) x excursions`. Two-gradient explicit rules (f = 1/2) remove the leading wear; an explicit predicted-end-point learner restores a worn network without changing its outputs. | stage 12 |
| First-order learners | Closed forms for GD, heavy ball and look-ahead momentum (7/7 on the real network); spectral-area closed form; plain full-batch GD barely wears; Adam does not follow the one-number picture. | stage 13 |

**Scope.** Small fully connected networks, MNIST with 32-128 training points, squared error, mostly small moves, one or
two seeds per start. The algebra is general; behaviour at scale, with cross-entropy, minibatch SGD / Adam on standard
benchmarks, and deeper or convolutional networks is untested. Failed predictions are listed in FINDINGS.md next to the
ones that held.

## Stages

| Folder | What it tests | Runs on |
|---|---|---|
| `01_input_task_loops` | geometric phase of closed loops of tasks built by shifting inputs | CPU, minutes |
| `02_polygon_area_law` | rotation proportional to the signed polygon area; anticipatory recovery | CPU |
| `03_label_loops` | loops in label space, Lie-bracket law, out-and-back stiffening | CPU |
| `04_path_law` | the path law `V = LAW(C)` for GN, GD and momentum | CPU |
| `05_three_forces` | displacement / jitter / area split; tests 5-9 | CPU |
| `06_sharpening` | sharpening against the exact jump and the nearest point | CPU |
| `07_dose_law_gpu` | the "dose law" for loss of plasticity across optimizers (**did not hold**) | GPU notebook |
| `08_alignment_fraction` | the alignment-fraction law on real MNIST; alpha rule; one-function family; ReLU | GPU notebook + CPU checks |
| `09_evaluation_point` | alpha families, held-out predictions and test loss, ReLU reference-path law | GPU notebook + CPU checks |
| `10_rich_vs_lazy` | start function removed: does larger `f` improve what is learned? | GPU scripts |
| `11_mnist_follow_up_tests` | X1-X8 and T1: beyond the nearest point, feature probes, doubled network, midpoint kernel, three-function law | CPU, needs stage-09 data |
| `12_trainability` | loop law, one clock, collapse, healing, history, state/heat/area | GPU/Colab + CPU, needs stage-09 data |
| `13_first_order_and_adam` | look-position rule, GD / momentum / Adam, spectral area | CPU, needs stage-09 data |

Each stage folder holds its `SPEC.md` (test design, pass rules fixed before the run, and amendments), the code, unit
tests, and its report or results where they are small.

Shared libraries are in `lib/`: `tcr*.py` (stages 01-06), `part8lib.py` (stage 08), `part9lib.py` + `common9.py` (stages
09-11), `loc.py` (CPU loader for the stage-09 configuration on real MNIST), `spec11_common.py` / `spec12_common.py`.

## Running

```bash
pip install -e .            # makes lib/ importable; add [notebooks] for the GPU notebooks
cd 01_input_task_loops
python test_linear.py       # unit tests: 7/7
python run.py && python report.py
```

Stages 01-06 are self-contained on CPU. `02_polygon_area_law/test_part2.py` reports 5/6: test F1 uses a 1% tolerance
that is too tight at the size it runs at (a spec error recorded in FINDINGS.md); the measured ratios match the
prediction to about 4% with cosine 0.999.

The notebooks of stages 07-09 run on a GPU machine and download MNIST themselves. Stages 10-13 reuse what the stage-09
run stores (start weights, configuration, MNIST idx files). Point the scripts at that data with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `TCR_DATA` | folder holding the stage-09 run (`keep/` with `part9_run/`, `mnist_data/`, the notebook), and other stored data | `data` |
| `TCR_OUT` | where the follow-up scripts write their records | `outputs` |
| `TCR_BUNDLE` | the unpacked stage-12 bundle built by `12_trainability/make_bundle11.py` | `$TCR_DATA/spec11_bundle` |

Large raw outputs (per-case weight vectors, about 20 GB) are not included.

## License

MIT, see [LICENSE](LICENSE).
