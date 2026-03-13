# Results Mapping

This page maps repository presets and small override recipes to the paper's main empirical questions. Some entries are close to headline figures, while others are appendix-style supporting experiments that stress the same mechanisms from a different angle.

## Reader Guide

- Use [docs/reproducibility.md](reproducibility.md) for environment setup, launch workflow, and run-recording discipline.
- Use this page when the starting point is a target result, figure pattern, or appendix-style comparison.
- Where figure numbers are explicit in the current public paper/blog materials, this page follows that numbering.
- Paper presets are the closest entry points to the larger end-to-end workflow.
- Example presets isolate one mechanism at a time and are usually better for fast iteration.
- For figure-style comparisons, keep the dataset, backbone, `num_nodes`, `num_GPU`, and `k_steps` fixed while changing one communication factor at a time.
- All strict paper-style commands on this page should run with `data_sampling_mode=resample`. If a listed base preset is simulator-oriented and defaults to `fixed`, the command below adds an explicit override.

## Paper Figures and Closest Presets

| Paper figure or result family | Closest preset | Suggested command | Main metrics | Notes |
| --- | --- | --- | --- | --- |
| Figure 3: single final merge reveals strong hidden performance | configs/examples/figure1_single_merge_resnet18.yaml | `python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml` | `avg_test_accuracy`, `avg_model_test_accuracy`, `avg_model_test_accuracy - avg_test_accuracy`, `consensus_error` | Closest preset for the headline one-shot merging phenomenon |
| Temporal communication allocation result family: a late dense window is more effective than an early one under the same budget | configs/examples/figure2_dense_window_resnet18.yaml | `python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml` | `avg_test_accuracy`, `avg_model_test_accuracy`, `r`, `consensus_error` | Use `truncate_v2` and fix `window_size` while moving `point1` |
| Figure 6: persistent sparse communication preserves mergeability better than no communication | configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml | `python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=0.2` | `avg_test_accuracy`, `avg_model_test_accuracy`, `avg_model_test_accuracy - avg_test_accuracy` | Base preset is simulator-oriented, so the command switches it to paper-style resampling |
| Figure 6 ablation: no communication weakens mergeability | configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml | `python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=localtraining --set args.r_schedule=fixed --set args.r_start=0.0` | `avg_test_accuracy`, `avg_model_test_accuracy`, `avg_model_test_accuracy - avg_test_accuracy` | Pure local training should shrink the merged-model advantage |
| Figure 4-5 family: mergeability geometry and landscape intuition | configs/examples/figure1_single_merge_resnet18.yaml | `python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml` | `avg_model_test_accuracy`, `consensus_error` | Current repo can generate training checkpoints, but dedicated loss-landscape plotting is not yet packaged as a one-command workflow |
| Figure 7 / Section 4.2 family: acceleration-related interpretation | configs/paper/default_multi_gpu.yaml | `python3 scripts/run_with_config.py --config configs/paper/default_multi_gpu.yaml` | `avg_model_test_accuracy`, `consensus_error` | Use as an empirical anchor for theory-side discussion; no standalone theory visualization script is packaged yet |
| Stronger heterogeneity makes the mergeability story easier to observe | configs/examples/heterogeneous_data_alpha005.yaml | `python3 scripts/run_with_config.py --config configs/examples/heterogeneous_data_alpha005.yaml --set args.data_sampling_mode=resample` | `avg_test_accuracy`, `avg_model_test_accuracy`, `consensus_error` | Base preset is simulator-oriented, so the command switches it to paper-style resampling |
| Structured random communication on top of a sparse base topology | configs/examples/topology_exponential_random.yaml | `python3 scripts/run_with_config.py --config configs/examples/topology_exponential_random.yaml --set args.data_sampling_mode=resample` | `avg_test_accuracy`, `avg_model_test_accuracy`, `r` | Uses the simulator preset but switches it to paper-style resampling for strict comparison |
| Paper-facing larger run with pretrained backbone and post-merge rounds | configs/paper/default_multi_gpu.yaml | `python3 scripts/run_with_config.py --config configs/paper/default_multi_gpu.yaml` | `avg_test_accuracy`, `avg_model_test_accuracy`, `consensus_error` | A larger CLIP-based run that is closer to the main paper workflow |
| Post-training merging as an explicit experimental stage | configs/examples/post_merge_demo.yaml | `python3 scripts/run_with_config.py --config configs/examples/post_merge_demo.yaml --set args.data_sampling_mode=resample` | `avg_test_accuracy`, `avg_model_test_accuracy`, `consensus_error` | Uses the simulator preset but switches it to paper-style resampling for strict comparison |

## Main Paper Figures

### Figure 3: single final merge

Preset:

- configs/examples/figure1_single_merge_resnet18.yaml

Core settings:

- `gossip_topology=random`
- `r_schedule=truncate`
- `r_start=0.2`
- `r_end=num_nodes-1`
- `point1=0.99`

Interpretation:

- training remains in a sparse random-gossip regime for most of the run
- near the end, communication switches to a dense regime equivalent to connecting to all other logical nodes

Suggested variations:

```bash
python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.alpha=0.05

python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.diff_init=true

python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.optimizer_name=adamw --set args.lr=0.001
```

Figure-adjacent robustness knobs:

- `alpha`: strengthens or weakens heterogeneity
- `diff_init`: tests sensitivity to initialization mismatch
- `optimizer_name`, `lr`, `batch_size`: connect the headline effect to appendix-style optimization sensitivity checks

### Temporal communication allocation: late dense communication window

Preset:

- configs/examples/figure2_dense_window_resnet18.yaml

Core settings:

- `gossip_topology=random`
- `r_schedule=truncate_v2`
- `r_start=0.2`
- `r_end=num_nodes-1`
- `point1=0.9`
- `window_size=0.05`

Interpretation:

- training begins with sparse random gossip
- a temporary dense window is activated late in training
- after the window, communication returns to the sparse regime

For this comparison, `window_size` should be fixed while varying `point1`, because the intended question is when the dense window starts under the same `truncate_v2` communication pattern and the same window budget.

Suggested timing sweep with fixed `window_size=0.05`:

```bash
python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.1

python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.5

python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.9
```

### Figure 6: sparse communication versus no communication

Sparse-communication baseline:

```bash
python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=0.2
```

No-communication ablation:

```bash
python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=localtraining --set args.r_schedule=fixed --set args.r_start=0.0
```

The key comparison is the gap between `avg_model_test_accuracy` and `avg_test_accuracy`. Persistent sparse communication should preserve that gap far better than pure local training.

### Figure 1 (c): mergeability geometry and landscape structure

we apply the
implementation in https://github.com/crisostomi/cycle-consistent-model-merging/blob/master/notebooks/plots/plot_loss_contours_n_models.ipynb.

### Paper-facing larger workflow

Two paper presets are intended for larger runs:

- configs/paper/run_main_experiment.yaml
- configs/paper/default_multi_gpu.yaml

The first is the main wrapper target used by `scripts/run_main_experiment.sh`. The second is the cleaner starting point when a direct config launch is preferred.

Both paper presets already encode `data_sampling_mode=resample`.

## Appendix-Oriented Experiment Groups

The following comparisons are not tied to a single headline plot. They are useful supporting studies for checking robustness, understanding mechanisms, or building appendix-style ablations.

### Appendix C.1 style: near-silent training before the final merge

This group stays closest to the setup described for the sparse-gossip training phase before the one-shot merge.

| Supporting question | Base preset | Suggested overrides | Why it is useful |
| --- | --- | --- | --- |
| How late should dense communication start under `truncate_v2`? | configs/examples/figure2_dense_window_resnet18.yaml | Fix `window_size=0.05`, then vary `point1=0.1`, `0.5`, `0.9`, `0.99` | Separates early-window and late-window behavior under a fixed dense-window budget |
| How wide should the dense window be? | configs/examples/figure2_dense_window_resnet18.yaml | `window_size=0.02`, `0.05`, `0.1` | Tests whether the gain comes from timing alone or also from the duration of dense communication |
| How much sparse communication is enough to preserve mergeability? | configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml | `r_start=0.0`, `0.2`, `1`, `2` with `gossip_topology=random` | Connects mergeability to communication strength rather than only to timing |

### Appendix-style optimization sensitivity

These checks are the closest current repository equivalents to appendix discussions about whether the phenomenon survives changes in optimization regime.

| Supporting question | Base preset | Suggested overrides | Why it is useful |
| --- | --- | --- | --- |
| How sensitive is the phenomenon to heterogeneity? | configs/examples/heterogeneous_data_alpha005.yaml | `data_sampling_mode=resample` plus `alpha=1.0`, `0.1`, `0.05` | Makes it easier to compare weak, moderate, and strong non-IID regimes |
| Does the base topology matter when communication is random but constrained? | configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml | `data_sampling_mode=resample` plus `gossip_topology=random`, `exponential+random`, `ring+random` | Separates unrestricted random communication from structured random communication |
| How sensitive is the effect to optimizer choice? | configs/examples/figure1_single_merge_resnet18.yaml | `optimizer_name=sgd`, `adamw` with matched learning-rate ranges | Connects the main phenomenon to optimizer-level appendix checks |
| How sensitive is the effect to learning rate? | configs/examples/figure1_single_merge_resnet18.yaml | `lr=0.1`, `0.01`, `0.001` according to optimizer and model stability | Distinguishes a robust phenomenon from one that only appears in a narrow optimization regime |
| How sensitive is the effect to batch size? | configs/examples/figure1_single_merge_resnet18.yaml | `batch_size=64`, `128`, `256` while keeping other settings fixed | Tests whether the mergeability signal is stable across training noise scales |

### Appendix-style post-training and initialization checks

| Supporting question | Base preset | Suggested overrides | Why it is useful |
| --- | --- | --- | --- |
| How much can post-training merging still recover? | configs/examples/post_merge_demo.yaml | `data_sampling_mode=resample` plus `post_merge_rounds=0`, `4`, `16`, `32` and `end_topology=exponential`, `complete` | Clarifies whether gains are already present at training end or are amplified by extra merging |
| Does initialization choice change the single-merge story? | configs/examples/figure1_single_merge_resnet18.yaml | `diff_init=true`, `diff_init=false` | Checks whether mergeability depends strongly on shared initialization |

## Appendix-Style Command Snippets

### Timing sweep for the dense window

```bash
python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.1

python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.5

python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.9

python3 scripts/run_with_config.py --config configs/examples/figure2_dense_window_resnet18.yaml --set args.window_size=0.05 --set args.point1=0.99
```

### Optimizer, learning-rate, and batch-size sweeps

```bash
python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.optimizer_name=sgd --set args.lr=0.01

python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.optimizer_name=adamw --set args.lr=0.001

python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.optimizer_name=sgd --set args.lr=0.01 --set args.batch_size=64

python3 scripts/run_with_config.py --config configs/examples/figure1_single_merge_resnet18.yaml --set args.optimizer_name=sgd --set args.lr=0.01 --set args.batch_size=256
```

### Communication-strength sweep under persistent sparse gossip

```bash
python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=0.0

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=0.2

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=1

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random --set args.r_schedule=fixed --set args.r_start=2
```

### Topology-family sweep

```bash
python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=random

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=ring

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=exponential

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=exponential+random

python3 scripts/run_with_config.py --config configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml --set args.data_sampling_mode=resample --set args.gossip_topology=ring+random
```

### Post-merge sweep

```bash
python3 scripts/run_with_config.py --config configs/examples/post_merge_demo.yaml --set args.data_sampling_mode=resample --set args.post_merge_rounds=0

python3 scripts/run_with_config.py --config configs/examples/post_merge_demo.yaml --set args.data_sampling_mode=resample --set args.post_merge_rounds=4

python3 scripts/run_with_config.py --config configs/examples/post_merge_demo.yaml --set args.data_sampling_mode=resample --set args.post_merge_rounds=16

python3 scripts/run_with_config.py --config configs/examples/post_merge_demo.yaml --set args.data_sampling_mode=resample --set args.post_merge_rounds=32 --set args.end_topology=complete
```

## Reading the Metrics

- `avg_test_accuracy` tracks the average performance of the current local models.
- `avg_model_test_accuracy` evaluates the globally averaged model and acts as the repository's counterfactual mergeability signal.
- `avg_model_test_accuracy - avg_test_accuracy` measures how much performance is hidden if only local models are inspected.
- `consensus_error` helps distinguish useful inconsistency from complete collapse into unrelated local solutions.

When the merged-model metric stays high while the local-model metric remains much lower, the repository is observing exactly the kind of hidden mergeability emphasized by the paper.
