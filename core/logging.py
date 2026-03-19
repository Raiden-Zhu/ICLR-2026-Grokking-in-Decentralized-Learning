"""W&B logging process for multi-GPU training."""

import queue

import wandb
from tqdm import tqdm


def _try_log_mergeability_gap(step, avg_test_metrics, avg_model_metrics):
    """Log merged-model gain once both averaged local and merged metrics are available."""
    if step not in avg_test_metrics or step not in avg_model_metrics:
        return False

    wandb.log(
        {
            "avg_model_test_accuracy - avg_test_accuracy": (
                avg_model_metrics[step]["accuracy"] - avg_test_metrics[step]["accuracy"]
            ),
            "step": step,
        }
    )
    del avg_test_metrics[step]
    del avg_model_metrics[step]
    return True


def _flush_buffered_metrics(step, metrics_by_network, prefix, *, include_average):
    """Emit one W&B record for one logical step while keeping metric keys unchanged."""
    payload = {"step": step}
    for network_idx, metrics in metrics_by_network.items():
        payload[f"{prefix}_loss/network_{network_idx}"] = metrics["loss"]
        payload[f"{prefix}_accuracy/network_{network_idx}"] = metrics["accuracy"]

    average_metrics = None
    if include_average:
        avg_loss = sum(m["loss"] for m in metrics_by_network.values()) / len(metrics_by_network)
        avg_accuracy = sum(m["accuracy"] for m in metrics_by_network.values()) / len(
            metrics_by_network
        )
        payload[f"avg_{prefix}_loss"] = avg_loss
        payload[f"avg_{prefix}_accuracy"] = avg_accuracy
        average_metrics = {"loss": avg_loss, "accuracy": avg_accuracy}

    wandb.log(payload)
    return average_metrics


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
                }

                if len(train_metrics_buffer[step]) == num_nodes:
                    _flush_buffered_metrics(
                        step,
                        train_metrics_buffer[step],
                        "train",
                        include_average=True,
                    )
                    del train_metrics_buffer[step]

            elif log_type == "valid":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                if step not in valid_metrics_buffer:
                    valid_metrics_buffer[step] = {}
                valid_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                }

                if len(valid_metrics_buffer[step]) == num_nodes:
                    _flush_buffered_metrics(
                        step,
                        valid_metrics_buffer[step],
                        "valid",
                        include_average=True,
                    )
                    del valid_metrics_buffer[step]

            elif log_type == "test":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                if step not in test_metrics_buffer:
                    test_metrics_buffer[step] = {}
                test_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                }

                if len(test_metrics_buffer[step]) == num_nodes:
                    average_metrics = _flush_buffered_metrics(
                        step,
                        test_metrics_buffer[step],
                        "test",
                        include_average=True,
                    )
                    avg_test_metrics[step] = average_metrics
                    _try_log_mergeability_gap(step, avg_test_metrics, avg_model_metrics)
                    del test_metrics_buffer[step]

            elif log_type == "gossip_params":
                wandb.log(
                    {
                        "r": log_item["r"],
                        "step": log_item["step"],
                    }
                )

            elif log_type == "consensus_error":
                wandb.log({"consensus_error": log_item["error"], "step": log_item["step"]})

            elif log_type == "avg_model":
                step = log_item["step"]
                avg_model_test_acc = log_item["test_accuracy"]
                avg_model_test_loss = log_item["test_loss"]

                wandb.log(
                    {
                        "avg_model_test_accuracy": avg_model_test_acc,
                        "avg_model_test_loss": avg_model_test_loss,
                        "step": step,
                    }
                )
                avg_model_metrics[step] = {
                    "loss": avg_model_test_loss,
                    "accuracy": avg_model_test_acc,
                }
                _try_log_mergeability_gap(step, avg_test_metrics, avg_model_metrics)

            elif log_type == "post_merge":
                wandb.log(
                    {
                        "post_merge_round": log_item["post_merge_round"],
                        "post_merge_avg_test_accuracy": log_item["post_merge_avg_test_accuracy"],
                        "post_merge_avg_test_loss": log_item["post_merge_avg_test_loss"],
                        "post_merge_avg_model_test_accuracy": log_item["post_merge_avg_model_test_accuracy"],
                        "post_merge_avg_model_test_loss": log_item["post_merge_avg_model_test_loss"],
                        "post_merge_consensus_error": log_item["post_merge_consensus_error"],
                        "step": log_item["step"],
                    }
                )

            elif log_type == "post_merge_final":
                wandb.log(
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
                    }
                )

        except queue.Empty:
            continue

    pbar.close()
