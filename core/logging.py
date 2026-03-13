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

                wandb.log(
                    {
                        f"train_loss/network_{network_idx}": log_item["loss"],
                        f"train_accuracy/network_{network_idx}": log_item["accuracy"],
                        "step": step,
                    }
                )
                pbar.update(log_item.get("k_steps", 1))

                if step not in train_metrics_buffer:
                    train_metrics_buffer[step] = {}
                train_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                }

                if len(train_metrics_buffer[step]) == num_nodes:
                    avg_train_loss = (
                        sum(m["loss"] for m in train_metrics_buffer[step].values())
                        / num_nodes
                    )
                    avg_train_accuracy = (
                        sum(m["accuracy"] for m in train_metrics_buffer[step].values())
                        / num_nodes
                    )
                    wandb.log(
                        {
                            "avg_train_loss": avg_train_loss,
                            "avg_train_accuracy": avg_train_accuracy,
                            "step": step,
                        }
                    )
                    del train_metrics_buffer[step]

            elif log_type == "valid":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                wandb.log(
                    {
                        f"valid_accuracy/network_{network_idx}": log_item["accuracy"],
                        f"valid_loss/network_{network_idx}": log_item["loss"],
                        "step": step,
                    }
                )

                if step not in valid_metrics_buffer:
                    valid_metrics_buffer[step] = {}
                valid_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                }

                if len(valid_metrics_buffer[step]) == num_nodes:
                    avg_valid_loss = (
                        sum(m["loss"] for m in valid_metrics_buffer[step].values())
                        / num_nodes
                    )
                    avg_valid_accuracy = (
                        sum(m["accuracy"] for m in valid_metrics_buffer[step].values())
                        / num_nodes
                    )
                    wandb.log(
                        {
                            "avg_valid_loss": avg_valid_loss,
                            "avg_valid_accuracy": avg_valid_accuracy,
                            "step": step,
                        }
                    )
                    del valid_metrics_buffer[step]

            elif log_type == "test":
                network_idx = log_item["network_idx"]
                step = log_item["step"]

                wandb.log(
                    {
                        f"test_accuracy/network_{network_idx}": log_item["accuracy"],
                        f"test_loss/network_{network_idx}": log_item["loss"],
                        "step": step,
                    }
                )

                if step not in test_metrics_buffer:
                    test_metrics_buffer[step] = {}
                test_metrics_buffer[step][network_idx] = {
                    "loss": log_item["loss"],
                    "accuracy": log_item["accuracy"],
                }

                if len(test_metrics_buffer[step]) == num_nodes:
                    avg_test_loss = (
                        sum(m["loss"] for m in test_metrics_buffer[step].values())
                        / num_nodes
                    )
                    avg_test_accuracy = (
                        sum(m["accuracy"] for m in test_metrics_buffer[step].values())
                        / num_nodes
                    )
                    wandb.log(
                        {
                            "avg_test_loss": avg_test_loss,
                            "avg_test_accuracy": avg_test_accuracy,
                            "step": step,
                        }
                    )
                    avg_test_metrics[step] = {
                        "loss": avg_test_loss,
                        "accuracy": avg_test_accuracy,
                    }
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
