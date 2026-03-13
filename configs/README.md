# Config Presets

The repository organizes runnable presets into two groups.

## Paper Presets

These configs are the paper-facing starting points.

They use `data_sampling_mode=resample`, which matches the original paper-style data semantics: each node keeps a fixed class distribution, but concrete training samples are redrawn as the dataloader iterator is recreated.

- paper/run_main_experiment.yaml: main experiment launcher target used by scripts/run_main_experiment.sh
- paper/default_multi_gpu.yaml: baseline multi-GPU preset with larger node count and post-merge rounds

## Example Presets

These configs are intended to demonstrate representative simulator use cases.

Unless noted otherwise, they use `data_sampling_mode=fixed`, meaning each node receives one weighted local subset at startup and keeps that subset throughout training. This is the preferred mode for simulator studies that want a stronger "data on device" interpretation.

- examples/smoke_test_1gpu_1000steps.yaml: short validation run
- examples/few_gpu_many_nodes_1gpu_8nodes.yaml: one GPU hosts multiple logical nodes
- examples/few_gpu_many_nodes_4gpu_32nodes.yaml: a larger constrained-resource many-node simulation
- examples/heterogeneous_data_alpha005.yaml: stronger non-IID heterogeneity setting
- examples/post_merge_demo.yaml: post-merge rounds with an alternate end topology
- examples/figure1_single_merge_resnet18.yaml: paper-facing figure preset, uses `data_sampling_mode=resample`
- examples/figure2_dense_window_resnet18.yaml: paper-facing figure preset, uses `data_sampling_mode=resample`
- examples/topology_exponential_random.yaml: random sampling constrained by an exponential base topology

Common starting points:

- For environment validation, start with smoke_test_1gpu_1000steps.
- For the smallest clear few-GPU / many-node example, start with few_gpu_many_nodes_1gpu_8nodes.
- For a more representative constrained-resource simulation, start with few_gpu_many_nodes_4gpu_32nodes.
- For stronger heterogeneity, start with heterogeneous_data_alpha005.
- For post-training merging studies, start with post_merge_demo.
- For the headline single-merge phenomenon, start with figure1_single_merge_resnet18.
- For late dense-window comparisons, start with figure2_dense_window_resnet18.
- For `base+random` topology behavior, start with topology_exponential_random.

Interested readers are encouraged to combine these presets with small overrides or cross-preset variations to explore additional behaviors and test how stable the paper's reported phenomena remain under changed settings.

If a simulator preset is repurposed for a paper-style comparison, add `--set args.data_sampling_mode=resample` at launch time.

All presets can be launched through scripts/run_with_config.py.
