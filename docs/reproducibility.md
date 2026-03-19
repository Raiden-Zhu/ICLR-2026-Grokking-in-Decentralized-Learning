# Reproducibility Guide

This page is the paper-oriented entry point for the repository. Use it when the main goal is to prepare the environment, launch runs reliably, and record enough context for reproducible comparisons. For figure-oriented experiment selection and appendix-style ablations, use [docs/results_mapping.md](results_mapping.md).

## Scope

The repository currently supports:

- paper-oriented multi-GPU runs through config-driven launchers
- smaller smoke tests for environment validation
- offline or disabled W&B logging when public-cloud logging is not desired

The repository does not currently aim to benchmark real network communication overhead across physically separate machines. Its strongest use case is reproducing optimization behavior and topology effects in a controlled simulator.

## Data Sampling Policy

The repository now exposes two explicit node-data regimes:

- `data_sampling_mode=resample`: paper-facing mode. Each node keeps a fixed class distribution, but concrete sample indices are redrawn whenever the training dataloader iterator is recreated. This matches the original paper-style setup.
- `data_sampling_mode=fixed`: simulator-facing mode. Each node draws one weighted subset at startup and then keeps training on that same local pool. This better matches a persistent "data on device" interpretation.

For strict paper reproduction, keep `data_sampling_mode=resample`. If you start from a simulator preset for an appendix-style comparison, add `--set args.data_sampling_mode=resample` explicitly.

## Recommended Workflow

### 1. Prepare the environment

```bash
bash scripts/bootstrap_env.sh
cp .env.example .env
bash scripts/preflight.sh
```

The env helpers now read `.env` and `.env.local` as restricted `KEY=VALUE` files instead of sourcing them as shell scripts.

If needed, log into optional services:

```bash
bash scripts/login_wandb.sh
bash scripts/login_hf.sh
```

### 2. Validate the environment with a smoke test

```bash
python3 scripts/run_with_config.py --config configs/examples/smoke_test_1gpu_1000steps.yaml
```

This is the safest first run because it checks the dataset path, launcher path, and basic training loop before a longer experiment. The launcher also normalizes compatibility aliases such as `num_gpus`/`num_GPU` and `non_iid`/`nonIID` before dispatching to the training entrypoint.

### 3. Run the lightweight regression checks

```bash
bash scripts/run_targeted_regression_checks.sh
```

This script is a lightweight consistency check for the current workflow. It checks CLI-to-main argument compatibility, normalized runtime kwargs, shared dataset helper behavior, and the current 6-tuple structure returned by classification dataset loaders.

### 4. Launch a paper-oriented run

Use one of the current paper-facing presets:

- configs/paper/run_main_experiment.yaml
- configs/paper/default_multi_gpu.yaml

Typical entrypoints:

```bash
bash scripts/run_main_experiment.sh
```

or

```bash
python3 scripts/run_with_config.py --config configs/paper/default_multi_gpu.yaml
```

Practical note:

- On large multi-node presets, the very beginning of a run can look briefly stalled while dataset objects and dataloaders are being prepared. This startup pause is expected and does not by itself indicate a training failure.

## Current Preset Roles

### Paper-oriented presets

- configs/paper/run_main_experiment.yaml: thin wrapper target for the main experiment launcher
- configs/paper/default_multi_gpu.yaml: baseline multi-GPU preset with larger node count and post-merge rounds
- both paper presets use `data_sampling_mode=resample`
- both paper presets set `args.data_loading_workers=0` by default so large runs start more predictably across environments; changing it later is a runtime tuning choice rather than a change to the data or training objective

### Example presets

- configs/examples/smoke_test_1gpu_1000steps.yaml: short run for validation, debugging, and environment checks
- configs/examples/few_gpu_many_nodes_1gpu_8nodes.yaml: demonstrates many logical nodes on a single GPU
- configs/examples/few_gpu_many_nodes_4gpu_32nodes.yaml: demonstrates a constrained-GPU larger-node simulation setting
- configs/examples/heterogeneous_data_alpha005.yaml: emphasizes stronger data heterogeneity in the simulator-style fixed-subset setting
- configs/examples/post_merge_demo.yaml: highlights post-merge behavior with a different end topology
- configs/examples/figure1_single_merge_resnet18.yaml: figure-oriented paper preset for the single final merge phenomenon, uses `data_sampling_mode=resample`
- configs/examples/figure2_dense_window_resnet18.yaml: figure-oriented paper preset for a temporary dense communication window, uses `data_sampling_mode=resample`
- configs/examples/topology_exponential_random.yaml: isolates the behavior of structured random communication
- all heavier example presets except the smoke test and the 1-GPU/8-node demo now set `args.data_loading_workers=0` by default, prioritizing stable startup behavior over loader parallelism

## Key Logged Metrics

The most important paper-facing metrics currently logged by the repository are:

- `avg_test_accuracy`: the average global test accuracy of the current local models across nodes
- `avg_model_test_accuracy`: the test accuracy of the globally averaged model, used as a counterfactual mergeability proxy
- `avg_model_test_accuracy - avg_test_accuracy`: a compact summary of the gap between the merged-model view and the current local-model view
- `consensus_error`: a parameter-space disagreement metric across node models

How to interpret them:

- In the current implementation, training-time communication and final merged-model construction are intentionally not identical: decentralized training can still propagate floating BatchNorm statistics, while the final merged model averages learnable parameters directly and then refreshes BatchNorm statistics through a short calibration pass.
- The cleanest primary quantities are `avg_test_accuracy` and `avg_model_test_accuracy`.
- If `avg_test_accuracy` is low but `avg_model_test_accuracy` is much higher, the decentralized system may be underestimating the quality of the learned models when evaluated only through local models.
- `avg_model_test_accuracy - avg_test_accuracy` is a convenient derived summary of that gap and is useful as a compact mergeability indicator.
- This is the repository counterpart of the paper's counterfactual merged-model analysis.

W&B axis note:

- The repository logs an explicit `step` field for train, valid, test, mergeability, topology, and consensus metrics. This is the simulator's shared reference step and is the reader-facing x-axis.
- W&B also maintains its own internal `_step`, but `_step` only counts logging events. It is not the training step definition used by this repository.
- When reading curves in W&B, use the logged `step` field as the x-axis rather than `_step`.

## Reproducing Paper-Style Phenomena

### Single final merge phenomenon, Figure 1 style

Recommended preset:

- configs/examples/figure1_single_merge_resnet18.yaml

Loader note:

- This preset sets `args.data_loading_workers=0` by default to keep startup behavior predictable across environments. If you increase it later, treat that as a runtime tuning change rather than a paper-facing semantic change.
- The same default now applies to the other heavier reader-facing presets for the same reason.

Sampling note:

- this preset intentionally uses `data_sampling_mode=resample`, because the original paper-facing workflow fixes the class distribution per node but keeps redrawing concrete samples over training

Core settings:

- `gossip_topology=random`
- `r_schedule=truncate`
- `r_start=0.2`
- `r_end=num_nodes-1`, which is equivalent to dense or fully connected communication over logical nodes
- `point1` set very close to `1`, such as `0.99`, so the late stage switches to dense communication only at the very end or near the very end

Practical note:

- Because communication rounds are discrete, the exact number of dense rounds depends on `k_steps` and `max_steps`. In this codebase, `truncate` is best understood as a late-stage switch schedule rather than a mathematically exact one-step operator unless the discrete schedule is chosen carefully.

Important hyperparameters to vary when studying the effect:

- `alpha` for heterogeneity strength
- `diff_init` for initialization sensitivity
- `gossip_topology`
- `optimizer_name`
- `lr`
- `batch_size`
- `model_name`
- `pretrained`

Model-name note for ResNet-18 users:

- The paper-facing default in this repository is `model_name=resnet18_cifar_stem` (practical for small-image settings such as TinyImageNet with `image_size=64`).
- For a variant closer to official torchvision/ImageNet pretrained stem semantics, use `model_name=resnet18_imagenet_stem`.

Metrics to watch:

- `avg_test_accuracy`
- `avg_model_test_accuracy`
- `avg_model_test_accuracy - avg_test_accuracy`

### Dense communication window, Figure 2 style

Recommended preset:

- configs/examples/figure2_dense_window_resnet18.yaml

Sampling note:

- this preset also uses `data_sampling_mode=resample`; if you re-implement the same schedule under the simulator's fixed-subset regime, treat that as a separate experiment rather than a strict reproduction

Core settings:

- `gossip_topology=random`
- `r_schedule=truncate_v2`
- `r_start=0.2`
- `r_end=num_nodes-1`
- `point1=0.9`
- `window_size=0.05`

Interpretation:

- `truncate_v2` is a temporary dense-window schedule: it starts from sparse gossip, switches to a dense communication window near the end of training, and then returns to sparse communication after the window.

Practical note:

- Because the implementation clamps the training end to the final `r_end` regime at `current_iter >= end_iter`, the exact last-step behavior depends on the discretization of rounds. For paper-style experiments, the main takeaway is to place the dense window late and compare different start positions and window sizes.

## Communication Schedule Naming

The repository's schedule names are intentionally short, but the most useful interpretation is:

- `truncate`: late-stage switch schedule
- `truncate_v2`: temporary dense-window schedule

These names are helpful when translating the code settings back to the temporal communication patterns studied in the paper.

## Topology Semantics

The repository supports both plain and combined topologies.

- `random`: sample neighbors from the complete graph
- `exponential+random`: sample neighbors from the candidate set defined by the exponential topology
- `ring+random`: sample neighbors from the candidate set defined by the ring topology

This means `exponential+random` is not the same as unrestricted random communication. It preserves the exponential topology as the base candidate structure and only randomizes which of those eligible neighbors are active.

Recommended preset for this behavior:

- configs/examples/topology_exponential_random.yaml

## TinyImageNet Notes

For TinyImageNet runs, the repository now treats dataset preparation as part of the reproducibility boundary:

- downloads and archive extraction are handled through the hardened `datasets/tinyimagenet.py` path
- processed caches no longer rely on the old external pickle-based path that motivated the original review item
- `strict_loading` and `max_failure_ratio` are the main controls for how corrupted samples are handled during dataset loading

If you are reproducing older local commands or YAMLs, note that `load_pickle` is still accepted as a compatibility input but does not change current training behavior.

## Reproducibility Checklist

For runs that you want to compare carefully, record the following:

- commit hash
- exact config file used
- all CLI overrides passed through scripts/run_with_config.py
- dataset path and dataset version
- GPU model and GPU count
- Python, PyTorch, and CUDA versions
- seed value
- W&B mode, either online, offline, or disabled

## Output Artifacts

A standard run can produce the following artifacts:

- W&B metrics, if logging is enabled
- cached datasets and model downloads under local cache directories
- a final `convergence_model_multi_gpu.pth` checkpoint reconstructed from the shared target state at the end of training

## Practical Advice

- Run the smoke test before changing multiple knobs at once.
- Use disabled or offline W&B mode in restricted environments.
- Increase node count only after confirming memory headroom.
- Treat post-merge rounds as part of the experimental design, not as a universal default.
- Use scripts/gpu_monitor.sh when exploring larger many-node simulations, since CPU and GPU memory pressure can rise quickly.
- If you update launcher options or classification-dataset loading paths, rerun `bash scripts/run_targeted_regression_checks.sh` before longer experiments.
