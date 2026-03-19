"""W&B logging process for multi-GPU training."""

import queue

import wandb
from tqdm import tqdm


def _pop_mergeability_payload(step, avg_test_metrics, avg_model_metrics):
    """Return one merged payload once both averaged local and merged metrics are available."""
    if step not in avg_test_metrics or step not in avg_model_metrics:
        return None

    test_metrics = avg_test_metrics.pop(step)
    avg_model = avg_model_metrics.pop(step)
    return {
        "avg_test_accuracy": test_metrics["accuracy"],
        "avg_test_loss": test_metrics["loss"],
        "avg_model_test_accuracy": avg_model["accuracy"],
        "avg_model_test_loss": avg_model["loss"],
        "avg_model_test_accuracy - avg_test_accuracy": (
            avg_model["accuracy"] - test_metrics["accuracy"]
        ),
        "step": step,
        "k_steps": test_metrics.get("k_steps", avg_model.get("k_steps")),
    }


def _compute_average_metrics(metrics_by_network):
    avg_loss = sum(m["loss"] for m in metrics_by_network.values()) / len(metrics_by_network)
    avg_accuracy = sum(m["accuracy"] for m in metrics_by_network.values()) / len(
        metrics_by_network
    )
    first_metrics = next(iter(metrics_by_network.values()))
    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "k_steps": first_metrics.get("k_steps"),
    }


def _append_round(payload):
    """Add the communication-round view derived from the explicit simulator step."""
    step = payload.get("step")
    k_steps = payload.get("k_steps")
    if step is None or k_steps is None or int(k_steps) <= 0:
        return payload

    updated = dict(payload)
    updated["round"] = int(step) // int(k_steps)
    return updated



def _build_buffered_metrics_payload(step, metrics_by_network, prefix, *, include_average):
    """Build one W&B payload for one logical step while keeping metric keys unchanged."""
    first_metrics = next(iter(metrics_by_network.values()))
    payload = {"step": step, "k_steps": first_metrics.get("k_steps")}
    for network_idx, metrics in metrics_by_network.items():
        payload[f"{prefix}_loss/network_{network_idx}"] = metrics["loss"]
        payload[f"{prefix}_accuracy/network_{network_idx}"] = metrics["accuracy"]

    average_metrics = None
    if include_average:
        average_metrics = _compute_average_metrics(metrics_by_network)
        payload[f"avg_{prefix}_loss"] = average_metrics["loss"]
        payload[f"avg_{prefix}_accuracy"] = average_metrics["accuracy"]

    return payload, average_metrics


def logging_process(log_queue, total_steps, num_nodes):
    """Handle W&B logging from all worker processes."""
    train_metrics_buffer = {}
    valid_metrics_buffer = {}
    test_metrics_buffer = {}
    avg_test_metrics = {}
    avg_model_metrics = {}

    pbar = tqdm(total=total_steps, desc="Multi-GPU Training Progress")

    while True:
        try:
            log_item = log_queue.get(timeout=1)
            if log_item is None:
                break

            log_type = log_item["type"]

            if log_type == "train":
                network_idx = log_item["network_idx"]
                step = log_item["step"]
                pbar.update(log_item.get("k_steps", 1))

                if step not in train_metrics_buffer:
                    train_metrics_buffer[step] = {}
                train_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                    "k_steps": log_item.get("k_steps"),
                }

                if len(train_metrics_buffer[step]) == num_nodes:
                    payload, _ = _build_buffered_metrics_payload(
                        step,
                        train_metrics_buffer[step],
                        "train",
                        include_average=True,
                    )
                    wandb.log(_append_round(payload))
                    del train_metrics_buffer[step]

            elif log_type == "valid":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                if step not in valid_metrics_buffer:
                    valid_metrics_buffer[step] = {}
                valid_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                    "k_steps": log_item.get("k_steps"),
                }

                if len(valid_metrics_buffer[step]) == num_nodes:
                    payload, _ = _build_buffered_metrics_payload(
                        step,
                        valid_metrics_buffer[step],
                        "valid",
                        include_average=True,
                    )
                    wandb.log(_append_round(payload))
                    del valid_metrics_buffer[step]

            elif log_type == "test":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                if step not in test_metrics_buffer:
                    test_metrics_buffer[step] = {}
                test_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                    "k_steps": log_item.get("k_steps"),
                }

                if len(test_metrics_buffer[step]) == num_nodes:
                    payload, _ = _build_buffered_metrics_payload(
                        step,
                        test_metrics_buffer[step],
                        "test",
                        include_average=False,
                    )
                    avg_test_metrics[step] = _compute_average_metrics(test_metrics_buffer[step])
                    wandb.log(_append_round(payload))
                    merged_payload = _pop_mergeability_payload(
                        step,
                        avg_test_metrics,
                        avg_model_metrics,
                    )
                    if merged_payload is not None:
                        wandb.log(_append_round(merged_payload))
                    del test_metrics_buffer[step]

            elif log_type == "gossip_params":
                wandb.log(_append_round({"r": log_item["r"], "step": log_item["step"], "k_steps": log_item.get("k_steps")}))

            elif log_type == "consensus_error":
                wandb.log(
                    _append_round(
                        {"consensus_error": log_item["error"], "step": log_item["step"], "k_steps": log_item.get("k_steps")}
                    )
                )

            elif log_type == "avg_model":
                step = log_item["step"]
                avg_model_test_acc = log_item["test_accuracy"]
                avg_model_test_loss = log_item["test_loss"]

                avg_model_metrics[step] = {
                    "loss": avg_model_test_loss,
                    "accuracy": avg_model_test_acc,
                    "k_steps": log_item.get("k_steps"),
                }
                merged_payload = _pop_mergeability_payload(
                    step,
                    avg_test_metrics,
                    avg_model_metrics,
                )
                if merged_payload is not None:
                    wandb.log(merged_payload)

            elif log_type == "post_merge":
                wandb.log(
                    _append_round(
                        {
                            "post_merge_round": log_item["post_merge_round"],
                            "post_merge_avg_test_accuracy": log_item["post_merge_avg_test_accuracy"],
                            "post_merge_avg_test_loss": log_item["post_merge_avg_test_loss"],
                            "post_merge_avg_model_test_accuracy": log_item["post_merge_avg_model_test_accuracy"],
                            "post_merge_avg_model_test_loss": log_item["post_merge_avg_model_test_loss"],
                            "post_merge_consensus_error": log_item["post_merge_consensus_error"],
                            "step": log_item["step"],
                            "k_steps": log_item.get("k_steps"),
                        }
                    )
                )

            elif log_type == "post_merge_final":
                wandb.log(
                    _append_round(
                        {
                            "avg_test_accuracy": log_item["avg_test_accuracy"],
                            "avg_test_loss": log_item["avg_test_loss"],
                            "avg_model_test_accuracy": log_item["avg_model_test_accuracy"],
                            "avg_model_test_loss": log_item["avg_model_test_loss"],
                            "avg_model_test_accuracy - avg_test_accuracy": log_item[
                                "avg_model_test_accuracy - avg_test_accuracy"
                            ],
                            "consensus_error": log_item["consensus_error"],
                            "step": log_item["step"],
                            "k_steps": log_item.get("k_steps"),
                        }
                    )
                )

        except queue.Empty:
            continue

    pbar.close()
