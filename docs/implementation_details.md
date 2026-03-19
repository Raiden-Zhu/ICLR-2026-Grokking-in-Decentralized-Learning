# Implementation Details

This page summarizes the implementation-level semantics of the repository. It is the right place to answer questions such as:

- what is actually averaged during gossip or global merging
- which states should be averaged, re-estimated, or left untouched
- how the current simulator differs from a full distributed runtime
- how to interpret the main mergeability-related metrics in practice

Use this page together with [docs/concepts.md](concepts.md) and [docs/reproducibility.md](reproducibility.md):

- [docs/concepts.md](concepts.md) explains the simulator at the conceptual level
- [docs/reproducibility.md](reproducibility.md) explains how to run experiments reliably
- this page explains the current implementation at the state and metric level

## Scope

The repository is built around a simulator-style execution model:

1. each GPU worker trains a subset of logical nodes locally
2. workers publish local node states into shared memory
3. rank 0 applies the gossip operator centrally over the full logical-node state
4. updated parameters are broadcast back to workers
5. evaluation runs only at the explicit scheduled checkpoints after reload

The current worker loop keeps these publish / aggregate / reload / evaluate stages explicit, with barriers separating them. This is why the repository is best understood as a decentralized-learning simulator for optimization behavior rather than as a production distributed runtime for measuring real network costs.

## Current Merging Semantics

There are two closely related merge operators in the repository:

- **gossip mixing during training**: a row-stochastic communication operator is applied to node states at each communication round
- **global averaging for analysis or final export**: node models are averaged into a single merged model for evaluation or saving

The operator changes, but the state taxonomy remains consistent across both settings in the current implementation.

One practical distinction is worth stating explicitly:

- during training-time communication, floating model state is still propagated across nodes, including BatchNorm running statistics
- for the final merged model, only learnable floating-point parameters are averaged directly, and BatchNorm statistics are refreshed afterward through a short calibration pass

This design keeps training-time ResNet behavior close to the original communication path while giving the final merged model a cleaner parameter-averaging semantics.

## State Taxonomy

The current implementation follows the policy below.

### 1. States That Should Be Averaged

These states participate in both gossip mixing and global averaging.

- all learnable floating-point parameters
- convolution and linear weights and biases
- attention projections and embeddings
- classifier heads
- LayerNorm affine parameters
- BatchNorm affine parameters (`weight` and `bias`)
- positional embeddings and other floating-point learned tensors

In the current code, these tensors are the primary merge object. The final merged model is constructed by averaging them across nodes.

### 2. States That Should Be Re-estimated

These states are model statistics rather than core learned parameters.

- BatchNorm `running_mean`
- BatchNorm `running_var`

In the current implementation, these states may still move during training-time communication as part of the floating model state, but they are not treated as first-class merge targets for the final merged model. Instead, the merged model runs a short calibration pass before evaluation or final export so that BatchNorm statistics are refreshed after parameter averaging.

This matters most for BatchNorm-based models such as [models/resnet_micro.py](../models/resnet_micro.py) and the BatchNorm-enabled [models/mlp.py](../models/mlp.py).

### 3. States That Should Not Be Averaged

These states do not have a stable or meaningful averaging semantics.

- BatchNorm `num_batches_tracked`
- other discrete counters or non-floating bookkeeping buffers

In the current implementation, they are excluded from ordinary averaging. They are copied as buffers for node models, and for merged-model evaluation they are allowed to update naturally during the BatchNorm re-estimation pass.

### 4. States Needed Only for Resume-Training Fidelity

These states do not belong to the main paper-facing merge semantics. They matter only if the goal is to resume optimization after a merge while preserving training dynamics more faithfully.

- optimizer state such as momentum and Adam moments
- scheduler state
- AMP `GradScaler` state
- per-node step counters and similar control state
- RNG state and sampler progress

The current implementation does not merge them.

## How This Maps to the Current Code

### Shared-State Gossip Path

The current simulator separates model state into two channels:

- floating model state is packed into shared flat buffers
- model buffers are copied through separately

The shared-state schema is instantiated from the same model-construction path as the worker models. When a model family uses `pretrained` to choose a structural branch, the schema path follows that same branch while skipping unnecessary pretrained-weight loading in the rank-0 setup process.

The centralized gossip operator is applied to the floating flat-buffer state. In practice, this means learnable floating-point parameters are always communicated, and floating BatchNorm statistics can also move during training-time communication.

- shared state pool creation: [core/shared_state.py](../core/shared_state.py)
- copying model state into shared buffers: [core/shared_state.py](../core/shared_state.py)
- flat-buffer gossip update: [core/communication.py](../core/communication.py)
- centralized round execution and rank-0 control phase: [main_multi_GPU.py](../main_multi_GPU.py)

At a high level, the current code uses floating state for communication rounds, while reserving stricter semantics for the final merged model.

### Average-Model Evaluation Path

The repository also reconstructs a global average model directly from the shared target buffer for mergeability analysis.

- average-model reconstruction: [main_multi_GPU.py](../main_multi_GPU.py)
- mean-state loading helper: [core/shared_state.py](../core/shared_state.py)

After the averaged parameters are loaded, the merged model runs a short BatchNorm re-estimation pass before evaluation. This keeps the merged-model path aligned with the repository's current BatchNorm policy. The average-model path and final convergence-model export are both reconstructed on the centralized aggregation device, which currently remains equivalent to `cuda:0` by policy.

So the current implementation intentionally distinguishes between:

- **training-time communication semantics**: communicate floating state so that ResNet weights and BatchNorm statistics continue to evolve together during decentralized training
- **final merged-model semantics**: average learnable parameters, then recalibrate BatchNorm statistics instead of treating them as final averaged quantities

## Current Implementation Summary

In implementation terms, the current repository now does the following:

1. final merged-model averaging operates only on learnable floating-point parameters
2. floating BatchNorm statistics can still be propagated during training-time communication
3. BatchNorm running statistics are refreshed on the merged model through a short calibration pass
4. discrete counters such as `num_batches_tracked` are excluded from direct averaging
5. optimizer, scheduler, AMP, RNG, and sampler state remain outside the merge path

## Why BatchNorm Needs Special Treatment

BatchNorm exposes an important difference between "state that can be numerically averaged" and "state that should be semantically averaged."

- `weight` and `bias` behave like learnable parameters and belong to the averaged set
- `running_mean` and `running_var` are data-dependent statistics and are better re-estimated after merging
- `num_batches_tracked` is a counter and should not be merged by naive averaging

This is why a blanket "average every tensor in the state dict" rule is usually too coarse for serious model-merging analysis under non-IID data.

## Compatibility Surface

For older commands and YAMLs, the repository still keeps a narrow compatibility surface at the launcher/config layer:

- `num_gpus` is normalized to the canonical `num_GPU`
- `non_iid` is normalized to the canonical `nonIID`
- `load_pickle` is accepted only as a legacy no-op input and does not participate in current training semantics

This compatibility layer exists to keep old configs runnable while making the active implementation boundary clearer.

## Runtime and Logging Notes

A few practical details are useful when reading results from the current codebase:

- training and evaluation move tensors to device with non-blocking transfers where supported; this affects execution efficiency, not the learning rule
- the worker loop uses `optimizer.zero_grad(set_to_none=True)`; this reduces redundant memory traffic without changing optimizer step order
- launcher aliases such as `num_gpus`/`num_GPU` and `non_iid`/`nonIID` are normalized into one canonical runtime view before training starts
- CIFAR-10, CIFAR-100, and TinyImageNet share the same dataset-finalization structure, while preserving split policy, transforms, calibration-view semantics, and the existing 6-tuple return contract
- same-step W&B logging is batched before upload, while preserving metric keys, aggregation rules, and step alignment

For interpretation purposes, the intended invariants are:

- the training mathematical formulation remains unchanged
- node-data semantics remain unchanged, including the `data_sampling_mode=resample` stream behavior used by paper-facing runs
- metric meaning, key names, and step semantics remain unchanged

The lightweight check `scripts/run_targeted_regression_checks.sh` is available if you want to verify launcher and dataset-path consistency in a local environment.

## Practical Note on Non-Floating State

The current shared-memory implementation gives the cleanest semantics to the final merged-model parameters. Non-floating buffers are handled separately, and non-parameter training state remains outside the merge path.

## Practical Note on Mergeability Metrics

The metric

- `avg_model_test_accuracy - avg_test_accuracy`

is conceptually important as a compact summary of hidden mergeability. In the current implementation, the logging path stores averaged local-model metrics and merged-model metrics separately by step, then records the gap only after both are available.

For careful comparisons, the cleanest primary quantities remain:

- `avg_test_accuracy`
- `avg_model_test_accuracy`

and the gap can be read as a convenient derived summary of the difference between them.

## Recommended Interpretation

For paper reading and simulator use, the cleanest interpretation of the current repository is:

- the simulator is reliable for studying optimization dynamics, topology effects, and the existence of hidden mergeability
- the most meaningful merge object is the learnable floating-point model state
- BatchNorm statistics deserve explicit treatment rather than silent inclusion in a generic averaging rule
- full resume-training fidelity is a separate engineering problem from merged-model evaluation

## When to Go Beyond the Current Policy

The current policy is enough for:

- paper-style mergeability analysis
- one-shot merged-model evaluation
- topology and communication-schedule comparisons

A more complete state-handling design becomes necessary only if the repository evolves toward:

- exact checkpoint merging for resume training
- optimizer-state-aware federated training experiments
- stronger claims about end-to-end training-state preservation after merge