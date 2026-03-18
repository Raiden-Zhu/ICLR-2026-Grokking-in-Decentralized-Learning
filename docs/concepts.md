# Concepts Guide

This page explains the conceptual model behind the repository. It is the best place to understand what is being simulated, how logical nodes map onto GPUs, and how to interpret the main configurations.

## Core Idea

The repository studies decentralized-learning behavior when GPU resources are limited. Instead of requiring one physical worker per logical node, it allows a small number of GPUs to host many logical nodes and then simulates the decentralized optimization process over those nodes.

A useful mental model is:

- logical nodes are the decentralized learners of interest
- GPUs are execution devices that host subsets of those nodes
- gossip is defined over the full set of logical nodes

## Node, GPU, and Worker

### Logical node

A logical node is one decentralized learner with its own local model state, local optimizer state, and local data stream.

### GPU worker

A GPU worker is a process bound to one GPU. Each worker is responsible for training a subset of the full node set.

### Mapping

If you request more nodes than GPUs, multiple logical nodes are assigned to each GPU worker. This is the main mechanism that enables resource-constrained simulation.

## High-Level Training Loop

The current implementation follows this sequence:

1. Create all logical nodes and assign subsets of them to GPU workers.
2. Run local training independently for each node for `k_steps` local updates.
3. Publish local states into shared memory.
4. Let the rank-0 control path construct the gossip matrix and apply the centralized aggregation step.
5. Reload the updated parameters back onto the GPU workers.
6. Run evaluation only when the explicit shared evaluation schedule says so; that schedule is anchored to the slowest logical node's completed local step rather than to one fixed worker-local node.
7. Repeat until the per-node training budget is exhausted.

This means the node-level learning rule is preserved, while the execution is packed onto fewer GPUs. The publish / aggregate / reload / evaluate boundaries are kept explicit in the implementation so that refactors do not silently change the synchronization protocol.

## What "Simulation" Means Here

In this repository, simulation means that the optimization behavior of many decentralized nodes is reproduced without requiring one physical machine or one physical GPU per node.

This is strong for studying:

- topology effects
- local-update frequency
- data heterogeneity
- post-merge behavior
- scaling trends in node count under limited hardware

This is not intended to directly measure:

- real network latency
- bandwidth bottlenecks across physical machines
- systems-level throughput under true distributed deployment

## Main Configuration Knobs

### `num_nodes`

The total number of logical decentralized-learning nodes.

### `num_GPU`

The canonical training key for the number of GPU worker processes. When `num_nodes` is larger than `num_GPU`, each GPU hosts multiple logical nodes.

Compatibility note:
- `num_gpus` is still accepted by the config launcher and validation layer as an alias, but the internal canonical name is `num_GPU`.

### `k_steps`

The number of local optimization steps taken by each node before the next gossip communication round.

### `gossip_topology`

The communication pattern over logical nodes. Current examples include fixed structured topology, random topology, and random links on top of a structured backbone.

Combined topology syntax is written as `base+random`.

Important examples:

- `random`: random neighbors are sampled from the complete graph, so every other logical node is a potential candidate
- `exponential+random`: random neighbors are sampled only from the candidate set induced by the exponential topology
- `ring+random`: random neighbors are sampled only from the candidate set induced by the ring topology

So `base+random` does not mean mixing two independent operators. It means using the base topology to define the candidate neighbor set, and then performing random sampling inside that candidate set.

### `r_start`, `r_end`, `r_schedule`

These control how many extra neighbors are used and how connectivity changes over training.

### `nonIID`, `alpha`, `node_datasize`, `data_sampling_mode`

Compatibility note:
- `nonIID` is the canonical training key.
- `non_iid` is still accepted as a compatibility alias by the launcher and validation layer.

These control the local data regime. Smaller `alpha` means stronger heterogeneity under a Dirichlet split.

`data_sampling_mode` answers a different question from `alpha`.

- `alpha` and `nonIID` determine the node's class preference distribution
- `data_sampling_mode` determines how that class preference is turned into concrete samples over training

The two modes are:

- `fixed`: draw one weighted subset once, then keep training on that same node-local pool. This is the more natural simulator mode when the goal is to approximate persistent data stored on a device.
- `resample`: keep the node's class distribution fixed, but redraw concrete sample indices as the dataloader iterator is recreated. This is the paper-facing mode used for reproducing the original workflow.

So `fixed` means fixed class distribution plus fixed sample identities, while `resample` means fixed class distribution plus changing sample identities.

### `strict_loading`, `max_failure_ratio`

These control the hardened TinyImageNet loading path.

- `strict_loading=true` fails fast when corrupted samples are encountered.
- `max_failure_ratio` allows bounded tolerance before the loader raises.

They are mainly relevant when reproducing TinyImageNet experiments or validating local dataset integrity.

### `post_merge_rounds`

Additional gossip-only rounds applied after decentralized training. This is especially relevant for studying how much extra merging improves agreement or performance.

### `end_topology`

An optional topology used specifically during post-merge rounds.

## Why the Few-GPU / Many-Node Story Is Legitimate

The key claim that this repository can support is:

A small number of GPUs can simulate decentralized training with many more logical nodes by changing where node computations are executed, while keeping the node-level gossip rule defined over the full simulated node set.

That is a valid and useful research claim.

A stronger systems claim would require a different implementation and a different evaluation protocol.

## Example Interpretation

Suppose you have 4 GPUs but want to study 32 decentralized nodes.

The repository can assign 8 logical nodes to each GPU worker, run local updates for those nodes, synchronize their states, apply gossip over all 32 logical nodes, and then continue training. This lets you study decentralized-learning dynamics under node counts that exceed your GPU count.

## Compatibility Notes

The current user-facing launcher keeps a small compatibility surface for older configs and commands:

- `num_gpus` and `non_iid` are normalized into the canonical keys `num_GPU` and `nonIID`, and current launcher/validation paths emit a deprecation warning when those aliases are used
- `load_pickle` is still accepted as a legacy no-op input, but it does not affect current training semantics
- `scripts/run_with_config.py` is the recommended entrypoint because it applies this normalization before dispatching to the training code

## Project Positioning

The repository is best understood in two connected ways:

- official code for the paper's main empirical story about late communication and single global merging
- a lightweight simulator for studying decentralized optimization dynamics with many logical nodes on a limited number of GPUs

The implementation is designed for optimization studies, topology comparisons, and mergeability analysis. It is not intended as a production distributed runtime or a benchmark of physical network costs.
