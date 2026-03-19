"""Evaluation and post-merge helpers for the active multi-GPU runtime."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics produced for a single node and split."""

    split_name: str
    node_index: int
    step: int
    accuracy: float
    loss: float


def evaluate_model_metrics(network, dataloader, device):
    """Evaluate one model on one dataloader and return accuracy/loss."""
    if dataloader is None:
        return None

    network.eval()
    correct = 0
    total = 0
    cumulative_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = network(images)
            loss = F.cross_entropy(outputs, labels)
            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return 100 * correct / total, cumulative_loss / len(dataloader)


def evaluate_network(network, dataloader, step, node_index, split_name):
    """Evaluate one local node model and package its metrics for logging."""
    assert split_name in {"valid", "test"}, "split_name must be either 'valid' or 'test'"
    device = next(network.parameters()).device
    metrics = evaluate_model_metrics(network, dataloader, device)
    if metrics is None:
        return None
    accuracy, loss = metrics
    return EvaluationResult(
        split_name=split_name,
        node_index=node_index,
        step=step,
        accuracy=accuracy,
        loss=loss,
    )


def log_evaluation_result(log_queue, result):
    """Forward evaluation metrics to the central logging thread."""
    if result is None:
        return
    log_queue.put(
        {
            "type": result.split_name,
            "network_idx": result.node_index,
            "accuracy": result.accuracy,
            "loss": result.loss,
            "step": result.step,
        }
    )


def evaluate_local_models(
    local_node_indices,
    local_networks,
    valid_dataloaders_list,
    test_dataloader,
    evaluation_step,
    log_queue,
):
    """Evaluate local models after one centralized gossip/broadcast round."""
    for local_idx, global_idx in enumerate(local_node_indices):
        valid_result = evaluate_network(
            local_networks[local_idx],
            valid_dataloaders_list[local_idx],
            evaluation_step,
            global_idx,
            "valid",
        )
        log_evaluation_result(log_queue, valid_result)

        test_result = evaluate_network(
            local_networks[local_idx],
            test_dataloader,
            evaluation_step,
            global_idx,
            "test",
        )
        log_evaluation_result(log_queue, test_result)


def get_avg_model(networks, calibration_loader=None, reestimate_batch_norm_stats=None):
    """Average mergeable parameters across nodes into one consensus model."""
    convergence_model = deepcopy(networks[0])
    mergeable_parameter_keys = [
        key
        for key, value in networks[0].named_parameters()
        if value.requires_grad and value.is_floating_point()
    ]

    with torch.no_grad():
        for key in mergeable_parameter_keys:
            mean_value = sum(network.state_dict()[key] for network in networks) / len(networks)
            convergence_model.state_dict()[key].copy_(
                mean_value.to(dtype=convergence_model.state_dict()[key].dtype)
            )

    if reestimate_batch_norm_stats is not None:
        reestimate_batch_norm_stats(
            convergence_model,
            calibration_loader,
            next(convergence_model.parameters()).device,
        )
    return convergence_model


def evaluate_average_model_from_shared_state(
    config,
    shared_state_pool,
    test_dataloader,
    calibration_loader,
    step,
    log_queue,
    aggregation_device,
    create_model_from_config,
    copy_mean_state_to_model,
    reestimate_batch_norm_stats,
):
    """Evaluate the average model reconstructed directly from the shared target buffer."""
    if test_dataloader is None:
        return

    avg_model = create_model_from_config(config).to(aggregation_device)
    copy_mean_state_to_model(
        avg_model,
        shared_state_pool,
        device=aggregation_device,
        buffer_name="target",
    )
    reestimate_batch_norm_stats(avg_model, calibration_loader, aggregation_device)
    metrics = evaluate_model_metrics(avg_model, test_dataloader, aggregation_device)
    if metrics is not None:
        accuracy, loss = metrics
        log_queue.put(
            {
                "type": "avg_model",
                "test_accuracy": accuracy,
                "test_loss": loss,
                "step": step,
            }
        )
    del avg_model
    torch.cuda.empty_cache()


def run_post_merge_rounds(
    config,
    final_networks,
    test_dataloader,
    calibration_loader,
    log_queue,
    aggregation_device,
    *,
    compute_consensus_error,
    get_gossip_matrix,
    gossip_update,
    is_disabled_topology,
    reestimate_batch_norm_stats,
):
    """Run optional pure-gossip rounds after decentralized training finishes."""
    if config.post_merge_rounds <= 0:
        return

    print(f"\n{'='*60}")
    print(f"Training completed at step {config.max_steps}")
    print(f"Starting {config.post_merge_rounds} post-merge rounds...")

    post_topology = config.end_topology if config.end_topology else config.gossip_topology
    print(f"Using topology: {post_topology}")
    print(f"{'='*60}\n")

    for merge_round in range(config.post_merge_rounds):
        if not is_disabled_topology(post_topology):
            gossip_matrix, _ = get_gossip_matrix(
                config.num_nodes,
                topology=post_topology,
                r=config.r_end,
                wandb=None,
                current_iter=config.max_steps,
                end_iter=config.max_steps,
            )
            gossip_update(final_networks, gossip_matrix)

        avg_test_acc = 0.0
        avg_test_loss = 0.0
        num_evaluated_networks = 0
        for network in final_networks:
            metrics = evaluate_model_metrics(network, test_dataloader, aggregation_device)
            if metrics is None:
                continue
            accuracy, loss = metrics
            avg_test_acc += accuracy
            avg_test_loss += loss
            num_evaluated_networks += 1

        if num_evaluated_networks > 0:
            avg_test_acc /= num_evaluated_networks
            avg_test_loss /= num_evaluated_networks

        avg_model_metrics = None
        avg_model = None
        if test_dataloader is not None:
            avg_model = get_avg_model(
                final_networks,
                calibration_loader=calibration_loader,
                reestimate_batch_norm_stats=reestimate_batch_norm_stats,
            ).to(aggregation_device)
            avg_model.eval()
            avg_model_metrics = evaluate_model_metrics(avg_model, test_dataloader, aggregation_device)

        avg_model_test_acc = avg_model_metrics[0] if avg_model_metrics is not None else 0.0
        avg_model_test_loss = avg_model_metrics[1] if avg_model_metrics is not None else 0.0
        consensus_error = compute_consensus_error(final_networks)

        log_queue.put(
            {
                "type": "post_merge",
                "post_merge_round": merge_round + 1,
                "post_merge_avg_test_accuracy": avg_test_acc,
                "post_merge_avg_test_loss": avg_test_loss,
                "post_merge_avg_model_test_accuracy": avg_model_test_acc,
                "post_merge_avg_model_test_loss": avg_model_test_loss,
                "post_merge_consensus_error": consensus_error,
                "step": config.max_steps,
            }
        )

        if merge_round == config.post_merge_rounds - 1:
            log_queue.put(
                {
                    "type": "post_merge_final",
                    "avg_test_accuracy": avg_test_acc,
                    "avg_test_loss": avg_test_loss,
                    "avg_model_test_accuracy": avg_model_test_acc,
                    "avg_model_test_loss": avg_model_test_loss,
                    "avg_model_test_accuracy - avg_test_accuracy": avg_model_test_acc - avg_test_acc,
                    "consensus_error": consensus_error,
                    "step": config.max_steps,
                }
            )
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS (after {config.post_merge_rounds} post-merge rounds):")
            print(f"  Avg Test Accuracy: {avg_test_acc:.2f}%")
            print(f"  Avg Model Test Accuracy: {avg_model_test_acc:.2f}%")
            print(f"  Consensus Error: {consensus_error:.6f}")
            print(f"{'='*60}\n")
        else:
            print(
                f"Post-merge round {merge_round + 1}/{config.post_merge_rounds}: "
                f"Avg Test Acc={avg_test_acc:.2f}%, "
                f"Avg Model Test Acc={avg_model_test_acc:.2f}%, "
                f"Consensus Error={consensus_error:.6f}"
            )

        if avg_model is not None:
            del avg_model
            torch.cuda.empty_cache()

    print(f"Post-merge completed: {config.post_merge_rounds} rounds\n")
