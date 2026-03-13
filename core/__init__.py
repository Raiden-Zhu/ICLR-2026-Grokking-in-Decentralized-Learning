"""Core training modules."""

from .communication import compute_r, get_gossip_matrix, get_sparse_gossip_matrix, gossip_update, gossip_update_flat_buffer
from .gossip_matrix import dense_to_sparse_gossip_matrix
from .gradients import get_gradient_vector
from .shared_state import (
	compute_consensus_error_from_buffer,
	copy_mean_state_to_model,
	copy_model_to_shared_buffer,
	copy_shared_buffer_to_model,
	copy_source_state_to_target,
	create_shared_state_pool,
)

__all__ = [
	"compute_r",
	"get_gossip_matrix",
	"get_sparse_gossip_matrix",
	"gossip_update",
	"gossip_update_flat_buffer",
	"dense_to_sparse_gossip_matrix",
	"get_gradient_vector",
	"compute_consensus_error_from_buffer",
	"copy_mean_state_to_model",
	"copy_model_to_shared_buffer",
	"copy_shared_buffer_to_model",
	"copy_source_state_to_target",
	"create_shared_state_pool",
]
