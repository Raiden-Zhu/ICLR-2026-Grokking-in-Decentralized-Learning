from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SparseGossipMatrix:
    """Row-wise sparse gossip operator with explicit neighbor lists and weights."""

    row_indices: Tuple[torch.Tensor, ...]
    row_weights: Tuple[torch.Tensor, ...]
    num_nodes: int


def create_identity_matrix(N):
    """Create an identity matrix of size N x N."""
    matrix = np.zeros((N, N))
    for i in range(N):
        matrix[i, i] = 1
    return matrix


def create_ring_gossip_matrix(N):
    """Create uniform ring topology gossip matrix."""
    matrix = np.zeros((N, N))
    for i in range(N):
        matrix[i, i] = 1 / 3
        matrix[i, (i - 1) % N] = 1 / 3
        matrix[i, (i + 1) % N] = 1 / 3
    return matrix


def create_left_gossip_matrix(N):
    """Create uniform left-neighbor topology gossip matrix."""
    matrix = np.zeros((N, N))
    for i in range(N):
        matrix[i, i] = 1 / 2
        matrix[i, (i - 1) % N] = 1 / 2
    return matrix


def create_complete_gossip_matrix(N):
    """Create uniform complete topology gossip matrix."""
    return np.ones((N, N)) / N


def create_exponential_matrix(N):
    """Create uniform exponential topology gossip matrix."""
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(N)])
    x /= x.sum()
    topo = np.empty((N, N))
    for i in range(N):
        topo[i] = np.roll(x, i)
    return topo


def create_random_r_gossip_matrix(N, r_total, seed=None):
    """
    Create random uniform gossip matrix where each node has average total neighbors.

    Self-connection is always included.
    """
    if seed is not None:
        np.random.seed(seed)

    r_total = max(1.0, min(float(r_total), float(N)))

    matrix = np.zeros((N, N))
    for i in range(N):
        neighbors = [i]
        others = np.array([j for j in range(N) if j != i])

        if r_total > 1:
            guaranteed = int(r_total - 1)
            if guaranteed > 0:
                fixed_neighbors = np.random.choice(others, size=guaranteed, replace=False)
                neighbors = np.append(neighbors, fixed_neighbors)
                others = np.array([j for j in others if j not in fixed_neighbors])

            remaining_prob = r_total - 1 - guaranteed
            if remaining_prob > 0 and len(others) > 0:
                if np.random.random() < remaining_prob:
                    random_neighbor = np.random.choice(others, size=1)
                    neighbors = np.append(neighbors, random_neighbor)

        matrix[i, neighbors] = 1.0 / len(neighbors)

    return matrix


def create_random_r_gossip_rows(N, r_total, seed=None):
    """Create row-wise random gossip weights without materializing a dense matrix."""
    if seed is not None:
        np.random.seed(seed)

    r_total = max(1.0, min(float(r_total), float(N)))

    row_indices = []
    row_weights = []
    for i in range(N):
        neighbors = [i]
        others = np.array([j for j in range(N) if j != i], dtype=np.int64)

        if r_total > 1:
            guaranteed = int(r_total - 1)
            if guaranteed > 0:
                fixed_neighbors = np.random.choice(others, size=guaranteed, replace=False).tolist()
                neighbors.extend(fixed_neighbors)
                others = np.array([j for j in others if j not in fixed_neighbors], dtype=np.int64)

            remaining_prob = r_total - 1 - guaranteed
            if remaining_prob > 0 and len(others) > 0 and np.random.random() < remaining_prob:
                neighbors.append(int(np.random.choice(others, size=1)[0]))

        neighbor_array = np.asarray(neighbors, dtype=np.int64)
        row_indices.append(neighbor_array)
        row_weights.append(np.full(len(neighbor_array), 1.0 / len(neighbor_array), dtype=np.float32))

    return row_indices, row_weights


def get_topology_candidates(N, topology):
    """Get candidate neighbors for each node under a base topology."""
    candidates = []

    if topology == "ring":
        for i in range(N):
            candidates.append({i, (i - 1) % N, (i + 1) % N})
    elif topology == "left":
        for i in range(N):
            candidates.append({i, (i - 1) % N})
    elif topology == "exponential":
        for i in range(N):
            node_candidates = set()
            for j in range(N):
                distance = (j - i) % N
                if distance == 0 or (distance & (distance - 1)) == 0:
                    node_candidates.add(j)
            candidates.append(node_candidates)
    elif topology == "complete":
        all_nodes = set(range(N))
        for _ in range(N):
            candidates.append(all_nodes.copy())
    else:
        raise ValueError(f"Unsupported base topology for random sampling: {topology}")

    return candidates


def create_random_on_topology_matrix(
    N,
    base_topology,
    r_total,
    seed=None,
    warn=True,
    iteration=None,
    global_seed=None,
):
    """
    Creates a gossip matrix by randomly sampling total connections from base topology candidates.

    This allows combining structured topologies with randomness, e.g., 'exponential+random'
    means randomly sample r connections from exponential topology's candidate neighbors.

    Args:
        N: Number of nodes
        base_topology: Base topology type ('ring', 'exponential', 'left', 'complete')
        r_total: Number of connections per node (including self)
        seed: Random seed for reproducibility (overrides iteration/global_seed if provided)
        warn: Whether to warn if r exceeds candidate count (default: True)
        iteration: Current iteration number (for deterministic seed generation)
        global_seed: Global random seed (combined with iteration for reproducibility)

    Returns:
        A gossip matrix where each node has r connections sampled from base topology

    Notes:
        - If seed is provided, uses that seed directly
        - If global_seed and iteration are provided, uses hash(global_seed, iteration) as seed
        - Otherwise (seed=None, no global_seed), uses numpy's current random state
    """
    # Determine the actual seed to use
    if seed is not None:
        # Explicit seed provided
        np.random.seed(seed)
    elif global_seed is not None and iteration is not None:
        # Generate deterministic seed from global_seed and iteration
        # This ensures reproducibility while keeping each iteration different
        deterministic_seed = (global_seed * 10007 + iteration) % (2**31 - 1)
        np.random.seed(deterministic_seed)

    # Get candidate neighbors for each node from base topology
    candidates = get_topology_candidates(N, base_topology)
    matrix = np.zeros((N, N))

    # Check if requested total neighbors exceed maximum candidates
    max_candidates = max(len(c) for c in candidates)
    min_candidates = min(len(c) for c in candidates)

    if warn and r_total > max_candidates:
        import warnings

        warnings.warn(
            f"Warning: total neighbors={r_total} exceeds maximum candidates ({max_candidates}) in '{base_topology}' topology.\n"
            f"   Automatically capping to max available candidates.\n"
            f"   For N={N} nodes in '{base_topology}': candidate range is [{min_candidates}, {max_candidates}].\n"
            f"   Effective connections will be capped at {max_candidates} per node.",
            UserWarning,
        )

    for i in range(N):
        node_candidates = list(candidates[i])
        max_r = len(node_candidates)

        # Handle non-integer total neighbors (e.g., r_total=1.2)
        # Floor gives guaranteed connections, fractional part gives probability for one more
        r_floor = int(r_total)
        r_frac = r_total - r_floor
        # Cap requested total neighbors at the maximum number of candidates
        effective_r_base = min(r_floor, max_r)

        # Add one more connection with probability equal to fractional part
        effective_r = effective_r_base
        if r_frac > 0 and effective_r_base < max_r:
            if np.random.random() < r_frac:
                effective_r = effective_r_base + 1

        if effective_r >= max_r:
            # Use all candidates if requested neighbors exceed candidate count
            selected = node_candidates
        else:
            if i in node_candidates:
                # Remove self from candidates temporarily
                other_candidates = [c for c in node_candidates if c != i]
                # Sample (effective_r - 1) from others, then add self back
                if effective_r > 1 and len(other_candidates) > 0:
                    num_to_sample = min(effective_r - 1, len(other_candidates))
                    selected_others = np.random.choice(
                        other_candidates, size=num_to_sample, replace=False
                    ).tolist()
                    selected = [i] + selected_others
                else:
                    selected = [i]
            else:
                # Self not in candidates (unusual case), just sample effective_r
                selected = np.random.choice(
                    node_candidates, size=effective_r, replace=False
                ).tolist()

        # Assign equal weights
        matrix[i, selected] = 1.0 / len(selected)

    return matrix


def create_random_on_topology_rows(
    N,
    base_topology,
    r_total,
    seed=None,
    warn=True,
    iteration=None,
    global_seed=None,
):
    """Create row-wise random gossip weights sampled from a base topology."""
    if seed is not None:
        np.random.seed(seed)
    elif global_seed is not None and iteration is not None:
        deterministic_seed = (global_seed * 10007 + iteration) % (2**31 - 1)
        np.random.seed(deterministic_seed)

    candidates = get_topology_candidates(N, base_topology)
    row_indices = []
    row_weights = []

    max_candidates = max(len(c) for c in candidates)
    min_candidates = min(len(c) for c in candidates)

    if warn and r_total > max_candidates:
        import warnings

        warnings.warn(
            f"Warning: total neighbors={r_total} exceeds maximum candidates ({max_candidates}) in '{base_topology}' topology.\n"
            f"   Automatically capping to max available candidates.\n"
            f"   For N={N} nodes in '{base_topology}': candidate range is [{min_candidates}, {max_candidates}].\n"
            f"   Effective connections will be capped at {max_candidates} per node.",
            UserWarning,
        )

    for i in range(N):
        node_candidates = list(candidates[i])
        max_r = len(node_candidates)

        r_floor = int(r_total)
        r_frac = r_total - r_floor
        effective_r_base = min(r_floor, max_r)

        effective_r = effective_r_base
        if r_frac > 0 and effective_r_base < max_r and np.random.random() < r_frac:
            effective_r = effective_r_base + 1

        if effective_r >= max_r:
            selected = node_candidates
        else:
            if i in node_candidates:
                other_candidates = [c for c in node_candidates if c != i]
                if effective_r > 1 and len(other_candidates) > 0:
                    num_to_sample = min(effective_r - 1, len(other_candidates))
                    selected_others = np.random.choice(
                        other_candidates, size=num_to_sample, replace=False
                    ).tolist()
                    selected = [i] + selected_others
                else:
                    selected = [i]
            else:
                selected = np.random.choice(
                    node_candidates, size=effective_r, replace=False
                ).tolist()

        selected_array = np.asarray(selected, dtype=np.int64)
        row_indices.append(selected_array)
        row_weights.append(np.full(len(selected_array), 1.0 / len(selected_array), dtype=np.float32))

    return row_indices, row_weights


def _build_dense_matrix_from_rows(num_nodes, row_indices, row_weights):
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for row_idx, (neighbors, weights) in enumerate(zip(row_indices, row_weights)):
        matrix[row_idx, torch.as_tensor(neighbors, dtype=torch.long)] = torch.as_tensor(
            weights, dtype=torch.float32
        )
    return matrix


def _build_sparse_gossip_operator(num_nodes, row_indices, row_weights):
    return SparseGossipMatrix(
        row_indices=tuple(torch.as_tensor(neighbors, dtype=torch.long) for neighbors in row_indices),
        row_weights=tuple(torch.as_tensor(weights, dtype=torch.float32) for weights in row_weights),
        num_nodes=num_nodes,
    )


def _build_gossip_rows(
    num_nodes,
    topology="complete",
    r=3,
    wandb=None,
    current_iter=0,
    end_iter=1,
    global_seed=None,
):
    """Construct row-wise neighbor indices and weights for one gossip round."""
    N = num_nodes
    r_extra = max(float(r), 0.0)
    r_total = r_extra + 1.0

    if "+" in topology:
        parts = topology.split("+")
        if len(parts) == 2 and parts[1] == "random":
            base_topology = parts[0]
            candidates = get_topology_candidates(N, base_topology)
            max_candidates = max(len(c) for c in candidates)
            max_extra = max_candidates - 1

            if wandb and r_extra > max_extra:
                wandb.log(
                    {
                        "r_requested_extra": r_extra,
                        "r_actual_max_extra": max_extra,
                        "r_capped": True,
                        "topology": topology,
                    }
                )

            return create_random_on_topology_rows(
                N,
                base_topology,
                r_total,
                seed=None,
                warn=(current_iter == 0),
                iteration=current_iter,
                global_seed=global_seed,
            )
        raise ValueError(
            f"Invalid combined topology format: {topology}. Use 'base_topology+random'"
        )

    if topology == "localtraining":
        return [np.asarray([i], dtype=np.int64) for i in range(N)], [
            np.asarray([1.0], dtype=np.float32) for _ in range(N)
        ]
    if topology == "ring":
        row_indices = []
        row_weights = []
        for i in range(N):
            neighbors = np.asarray([i, (i - 1) % N, (i + 1) % N], dtype=np.int64)
            row_indices.append(neighbors)
            row_weights.append(np.full(3, 1.0 / 3, dtype=np.float32))
        return row_indices, row_weights
    if topology == "left":
        row_indices = []
        row_weights = []
        for i in range(N):
            neighbors = np.asarray([i, (i - 1) % N], dtype=np.int64)
            row_indices.append(neighbors)
            row_weights.append(np.full(2, 1.0 / 2, dtype=np.float32))
        return row_indices, row_weights
    if topology == "complete":
        neighbors = np.arange(N, dtype=np.int64)
        weights = np.full(N, 1.0 / N, dtype=np.float32)
        return [neighbors.copy() for _ in range(N)], [weights.copy() for _ in range(N)]
    if topology == "exponential":
        row_indices = []
        row_weights = []
        for i in range(N):
            neighbors = [j for j in range(N) if (j - i) % N == 0 or (((j - i) % N) & (((j - i) % N) - 1)) == 0]
            neighbor_array = np.asarray(neighbors, dtype=np.int64)
            row_indices.append(neighbor_array)
            row_weights.append(np.full(len(neighbor_array), 1.0 / len(neighbor_array), dtype=np.float32))
        return row_indices, row_weights
    if topology == "random":
        return create_random_r_gossip_rows(N, r_total)
    if topology == "ringtocomplete":
        progress = current_iter / end_iter
        point = 0.95
        next_topology = "ring" if progress < point else "complete"
        return _build_gossip_rows(N, topology=next_topology)
    if topology == "completetoring":
        progress = current_iter / end_iter
        point = 0.95
        next_topology = "complete" if progress < point else "ring"
        return _build_gossip_rows(N, topology=next_topology)
    if topology == "lefttocomplete":
        progress = current_iter / end_iter
        point = 0.95
        next_topology = "left" if progress < point else "complete"
        return _build_gossip_rows(N, topology=next_topology)
    if topology == "completetorandom":
        progress = current_iter / end_iter
        point = 0.95
        next_topology = "complete" if progress < point else "random"
        return _build_gossip_rows(N, topology=next_topology, r=r)
    raise ValueError(f"Invalid topology {topology}")


def get_gossip_matrix(
    num_nodes,
    topology="complete",
    r=3,
    wandb=None,
    current_iter=0,
    end_iter=1,
    global_seed=None,
):
    """
    Create the communication matrix for one gossip round.

    Supported topologies:
    - localtraining, ring, left, complete, exponential, random
    - ringtocomplete, completetoring, lefttocomplete, completetorandom
    - combined topology `base+random` where base is one of ring/left/exponential/complete

    - `r` is interpreted as extra neighbors (excluding self).
    - Internal sampling converts it to total neighbors by `r_total = r + 1`.
    """
    row_indices, row_weights = _build_gossip_rows(
        num_nodes,
        topology=topology,
        r=r,
        wandb=wandb,
        current_iter=current_iter,
        end_iter=end_iter,
        global_seed=global_seed,
    )
    return _build_dense_matrix_from_rows(num_nodes, row_indices, row_weights), None


def get_sparse_gossip_matrix(
    num_nodes,
    topology="complete",
    r=3,
    wandb=None,
    current_iter=0,
    end_iter=1,
    global_seed=None,
):
    """Create the communication operator as row-wise sparse neighbor lists."""
    row_indices, row_weights = _build_gossip_rows(
        num_nodes,
        topology=topology,
        r=r,
        wandb=wandb,
        current_iter=current_iter,
        end_iter=end_iter,
        global_seed=global_seed,
    )
    return _build_sparse_gossip_operator(num_nodes, row_indices, row_weights), None


def dense_to_sparse_gossip_matrix(gossip_matrix):
    """Convert an explicit dense gossip matrix into row-wise sparse neighbors."""
    if gossip_matrix.ndim != 2 or gossip_matrix.shape[0] != gossip_matrix.shape[1]:
        raise ValueError("gossip_matrix must be a square 2D tensor")

    num_nodes = gossip_matrix.shape[0]
    row_indices = []
    row_weights = []
    dense_matrix = gossip_matrix.to(dtype=torch.float32, device="cpu")
    for row_idx in range(num_nodes):
        neighbors = torch.nonzero(dense_matrix[row_idx], as_tuple=False).flatten()
        row_indices.append(neighbors)
        row_weights.append(dense_matrix[row_idx, neighbors])

    return SparseGossipMatrix(
        row_indices=tuple(neighbors.to(dtype=torch.long) for neighbors in row_indices),
        row_weights=tuple(weights.to(dtype=torch.float32) for weights in row_weights),
        num_nodes=num_nodes,
    )
