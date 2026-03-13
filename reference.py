"""
Multi-GPU Decentralized Training Script for Generic Sharded Data (e.g. FineWeb-Edu).

This script simulates Non-IID training by assigning different data shards (physical files) to different nodes.
It maintains full compatibility with the original decentralized training logic (Gossip, CDAT, etc.).
"""
import os
import argparse
import time
import random
import threading
import queue
import glob
from pathlib import Path
from copy import deepcopy
from typing import List, Dict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Barrier
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

# Import original utilities
from hessian_utils2 import estimate_precondhessian_lmax_blockdiag
from src.utils.gossip_matrix_new import create_gossip_matrix, gossip_update, gossip_accumulate
from src.utils.decentralized_utils import create_consensus_model, compute_consensus_error
from src.optim.cdat_utils import CDATManager
from src.utils.asymmetry_utils import convert_to_asymmetric_model
from src.model import build_model_from_args, get_hessian_blocks, infer_model_family_from_name
from src.optim import build_optimizers

# Import new Generic Sharded Loader
from data_loader_generic import GenericShardedDataset

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import math

def get_scheduled_value(
    step: int, 
    total_steps: int, 
    start_val: float, 
    end_val: float, 
    strategy: str, 
    alpha: float = 1.0
) -> float:
    """Universal scheduler function for dynamic parameters."""
    if strategy == 'constant':
        return start_val
    if total_steps == 0: 
        return start_val
    progress = min(1.0, max(0.0, step / total_steps))
    if alpha > 0 and alpha != 1.0:
        progress = progress ** alpha
    
    if 'linear' in strategy:
        return start_val + (end_val - start_val) * progress
    elif 'cosine' in strategy:
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return end_val + (start_val - end_val) * cosine_factor
    elif 'exp' in strategy:
        s_safe = start_val if abs(start_val) > 1e-9 else 1e-9
        e_safe = end_val if abs(end_val) > 1e-9 else 1e-9
        if s_safe * e_safe < 0:
            return start_val + (end_val - start_val) * progress
        ratio = max(1e-9, e_safe / s_safe)
        return s_safe * (ratio ** progress)
    return start_val

def get_connectivity(step: int, total_steps: int, args) -> float:
    if 'random' not in args.topology:
        return 0.0
    c_max = args.connectivity_max
    c_min = args.connectivity_min
    strategy = args.connectivity_scheduler
    
    if strategy.startswith('wsd'):
        warmup_steps = args.warmup_steps
        decay_steps = args.decay_steps if args.decay_steps is not None else 0
        wsd_total_steps = args.max_iters 
        alpha = args.scheduler_alpha
        
        if step < warmup_steps:
            progress = step / max(1, warmup_steps)
            if 'linear' in strategy or strategy == 'wsd_time':
                return c_max - (c_max - c_min) * progress
            elif 'cosine' in strategy:
                term = 0.5 * (1 + math.cos(math.pi * progress))
                return c_min + (c_max - c_min) * term
            elif 'exp' in strategy:
                return c_min + (c_max - c_min) * ((1.0 - progress) ** alpha)
        elif step >= wsd_total_steps - decay_steps:
            progress = (step - (wsd_total_steps - decay_steps)) / max(1, decay_steps)
            progress = min(1.0, progress)
            if 'linear' in strategy or strategy == 'wsd_time':
                return c_min + (c_max - c_min) * progress
            elif 'cosine' in strategy:
                term = 0.5 * (1 - math.cos(math.pi * progress))
                return c_min + (c_max - c_min) * term
            elif 'exp' in strategy:
                return c_min + (c_max - c_min) * (progress ** alpha)
        else:
            return c_min
            
    limit_1 = args.connectivity_max
    limit_2 = args.connectivity_min
    start_val = args.connectivity
    end_val = args.connectivity 
    
    if strategy == 'constant':
        return args.connectivity
    if 'increase' in strategy:
        start_val = limit_2 
        end_val = limit_1   
    elif 'decay' in strategy or strategy == 'cosine':
        start_val = limit_1 
        end_val = limit_2   
    else:
        start_val = limit_1
        end_val = limit_2
    return get_scheduled_value(step, total_steps, start_val, end_val, strategy, args.scheduler_alpha)

def get_k_steps(step: int, total_steps: int, args) -> int:
    if args.k_scheduler == 'constant':
        return args.k_steps
    k_min = args.k_min
    k_max = args.k_max
    if 'increase' in args.k_scheduler:
        start_val = k_min
        end_val = k_max
    else:
        start_val = k_max
        end_val = k_min
    val = get_scheduled_value(step, total_steps, start_val, end_val, args.k_scheduler, args.k_scheduler_alpha)
    return max(1, int(val))

def get_gamma(step: int, total_steps: int, args) -> float:
    if args.gamma_scheduler == 'constant':
        return args.gamma
    limit_1 = args.gamma_max
    limit_2 = args.gamma_min
    start_val = limit_1
    end_val = limit_2
    if 'increase' in args.gamma_scheduler:
        start_val = limit_2 
        end_val = limit_1   
    elif 'decay' in args.gamma_scheduler or 'cosine' in args.gamma_scheduler:
         if args.gamma_scheduler == 'cosine_increase':
             start_val = limit_2
             end_val = limit_1
         else:
             start_val = limit_1
             end_val = limit_2
    return get_scheduled_value(step, total_steps, start_val, end_val, args.gamma_scheduler, args.gamma_scheduler_alpha)

def get_beta(step: int, total_steps: int, args) -> float:
    if args.beta_scheduler == 'constant':
        return args.beta
    limit_1 = args.beta_max
    limit_2 = args.beta_min
    start_val = limit_1
    end_val = limit_2
    if 'increase' in args.beta_scheduler:
        start_val = limit_2 
        end_val = limit_1   
    elif 'decay' in args.beta_scheduler or 'cosine' in args.beta_scheduler:
         if args.beta_scheduler == 'cosine_increase':
             start_val = limit_2
             end_val = limit_1
         else:
             start_val = limit_1
             end_val = limit_2
    return get_scheduled_value(step, total_steps, start_val, end_val, args.beta_scheduler, args.beta_scheduler_alpha)

def worker_process(
    rank: int,
    world_size: int,
    node_indices: List[int],
    node_shard_map: Dict[int, List[int]], # Map: node_idx -> list of shard indices
    args,
    barrier: Barrier,
    global_cpu_models: List[nn.Module],
    log_queue: queue.Queue,
    shared_train_losses=None,
    shared_val_losses=None,
    global_cpu_opt_states=None,
    global_cpu_models_new: List[nn.Module]=None,
    global_cpu_momentum: List[nn.Module]=None,
    global_cpu_local_momentums: List[nn.Module]=None
):
    """Worker process for Generic Sharded Multi-GPU training."""
    if torch.cuda.is_available():
        if args.hessian_distributed and not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(29500 + (args.data_seed % 100))
            dist.init_process_group(backend="nccl", rank=rank, world_size=args.num_gpus)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    worker_seed = args.data_seed + rank * 1000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
    torch.set_num_threads(1)

    print(f"[Worker {rank}] Loading Generic Sharded datasets from {args.data_dir}...")
    
    train_loaders = []
    
    for i, global_idx in enumerate(node_indices):
        assigned_shards = node_shard_map[global_idx]
        
        if args.iid:
            # IID Mode: Load ALL available shards (or a shared large subset)
            # GenericShardedDataset with shard_indices=None loads all compatible files
            if i == 0:
                print(f"[Worker {rank}] Node {global_idx} (and others) using IID mode: Loading ALL shards.")
            try:
                node_dataset = GenericShardedDataset(
                    data_dir=args.data_dir,
                    split='train',
                    shard_indices=None, # None = Load All
                    block_size=args.block_size,
                    max_samples=args.max_samples
                )
            except Exception as e:
                 print(f"[Worker {rank}] Error loading IID dataset: {e}")
                 raise e
        else:
            # Non-IID Mode: Load assigned shards only
            print(f"[Worker {rank}] Node {global_idx} assigned shards: {assigned_shards}")
            try:
                node_dataset = GenericShardedDataset(
                    data_dir=args.data_dir,
                    split='train',
                    shard_indices=assigned_shards,
                    block_size=args.block_size,
                    max_samples=args.max_samples
                )
            except Exception as e:
                print(f"[Worker {rank}] Error loading dataset for shards {assigned_shards}: {e}")
                raise e

        try:
            loader = DataLoader(
                node_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )
            train_loaders.append(loader)
        except Exception as e:
            print(f"[Worker {rank}] Error creating DataLoader for Node {global_idx}: {e}")
            raise e
        
    # Load Global Validation Data
    val_loader = None
    try:
        val_dataset = GenericShardedDataset(
            data_dir=args.data_dir,
            split='val',
            block_size=args.block_size,
            max_samples=args.max_samples
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        if rank == 0:
            print(f"[Worker {rank}] Global validation set loaded ({len(val_dataset)} samples).")
    except Exception as e:
        print(f"[Worker {rank}] Warning: Could not load global validation set (val.bin): {e}")
    
    print(f"[Worker {rank}] Data loaded. Initializing {len(node_indices)} models...")

    init_ckpt_state = None
    init_ckpt_obj = None
    init_optimizer_state = None
    init_scaler_state = None
    init_global_step = 0
    init_best_val_loss = float('inf')
    if args.init_ckpt:
        init_ckpt_obj = torch.load(args.init_ckpt, map_location='cpu')
        if isinstance(init_ckpt_obj, dict) and 'model_state' in init_ckpt_obj:
            init_ckpt_state = init_ckpt_obj.get('model_state', None)
            init_optimizer_state = init_ckpt_obj.get('optimizer_state', None)
            init_scaler_state = init_ckpt_obj.get('scaler_state', None)
            init_global_step = int(init_ckpt_obj.get('global_step', 0) or 0)
            init_best_val_loss = float(init_ckpt_obj.get('best_val_loss', float('inf')))
        else:
            init_ckpt_state = init_ckpt_obj
        if rank == 0:
            print(f"[Info] Loading init checkpoint from {args.init_ckpt}")

    # Initialize Models
    model_family = infer_model_family_from_name(args.model_config)
    
    local_models = []
    local_optimizers = []
    local_muon_optimizers = []
    
    for i, global_idx in enumerate(node_indices):
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
            
        torch.manual_seed(args.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.data_seed)
        
        model, _ = build_model_from_args(args, vocab_size=57728, device=device)
        if i == 0 and rank == 0:
            print(f"[Info] Using model family: {model_family} ({args.model_config})")
        
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        
        if model_family != 'llama':
            model = convert_to_asymmetric_model(model, args)
        elif args.use_asymmetry and i == 0 and rank == 0:
            print("[Info] Asymmetry conversion is currently skipped for llama models.")
        if init_ckpt_state is not None:
            model.load_state_dict(init_ckpt_state)
        local_models.append(model)
        
        opt_list, opt_handles = build_optimizers(args, model, model_family)
        optimizer = opt_handles['adamw'] if args.optimizer == 'muon' else opt_handles['single']
        muon_optimizer = opt_handles['muon']

        if args.optimizer == 'muon':
            loaded_any_state = False
            if isinstance(init_ckpt_obj, dict):
                adamw_state = init_ckpt_obj.get('optimizer_state_adamw', None)
                muon_state = init_ckpt_obj.get('optimizer_state_muon', None)
            else:
                adamw_state = None
                muon_state = None

            if adamw_state is not None:
                try:
                    optimizer.load_state_dict(adamw_state)
                    loaded_any_state = True
                except Exception as e:
                    if rank == 0:
                        print(f"[Warning] Failed to load AdamW optimizer state from init checkpoint on node {global_idx}: {e}")

            if muon_state is not None:
                try:
                    muon_optimizer.load_state_dict(muon_state)
                    loaded_any_state = True
                except Exception as e:
                    if rank == 0:
                        print(f"[Warning] Failed to load Muon optimizer state from init checkpoint on node {global_idx}: {e}")

            if (not loaded_any_state) and init_optimizer_state is not None:
                try:
                    optimizer.load_state_dict(init_optimizer_state)
                except Exception as e:
                    if rank == 0:
                        print(f"[Warning] Failed to load legacy optimizer state from init checkpoint on node {global_idx}: {e}")
        elif init_optimizer_state is not None:
            try:
                optimizer.load_state_dict(init_optimizer_state)
            except Exception as e:
                if rank == 0:
                    print(f"[Warning] Failed to load optimizer state from init checkpoint on node {global_idx}: {e}")
                
        local_optimizers.append(optimizer)
        local_muon_optimizers.append(muon_optimizer)
    
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    if init_scaler_state is not None:
        try:
            scaler.load_state_dict(init_scaler_state)
            if rank == 0:
                print("[Info] Loaded AMP scaler state from init checkpoint.")
        except Exception as e:
            if rank == 0:
                print(f"[Warning] Failed to load AMP scaler state from init checkpoint: {e}")

    data_iters = [iter(loader) for loader in train_loaders]
    cdat_manager = CDATManager(args, node_indices)
    update_step = max(0, init_global_step)
    iter_step = update_step * max(1, args.grad_accum_steps)
    effective_tokens_per_update = args.n_nodes * args.batch_size * args.block_size * max(1, args.grad_accum_steps)
    local_losses_accum = {idx: [] for idx in node_indices}
    
    hessian_v_blocks = None
    hessian_v_blocks_node0 = None 
    hessian_v_blocks_node0_pre = None 
    best_val_loss = init_best_val_loss

    if rank == 0 and args.save_checkpoints:
        os.makedirs(args.out_dir, exist_ok=True)
    
    for i, opt in enumerate(local_optimizers):
        opt.zero_grad()
        if local_muon_optimizers[i] is not None:
            local_muon_optimizers[i].zero_grad()

    def get_optimizer_group_lr(optimizer_obj, group_name: str):
        if optimizer_obj is None:
            return None
        for group in optimizer_obj.param_groups:
            if group.get('group_name') == group_name:
                return group.get('lr', None)
        return None

    def run_gossip(current_k_for_log: int):
        nonlocal hessian_v_blocks, hessian_v_blocks_node0, hessian_v_blocks_node0_pre

        for i, model in enumerate(local_models):
            global_idx = node_indices[i]
            if global_cpu_models_new is not None:
                global_cpu_models_new[global_idx].load_state_dict(model.state_dict())
            else:
                global_cpu_models[global_idx].load_state_dict(model.state_dict())

            if args.gossip_local_momentum and global_cpu_local_momentums is not None:
                optimizer = local_optimizers[i]
                target_mom_model = global_cpu_local_momentums[global_idx]
                for (name, param), (tgt_name, tgt_param) in zip(model.named_parameters(), target_mom_model.named_parameters()):
                    if param.requires_grad:
                        state = optimizer.state.get(param, None)
                        if state is not None and 'exp_avg' in state:
                            tgt_param.data.copy_(state['exp_avg'].detach().cpu())
                        else:
                            tgt_param.data.zero_()

        if (args.compute_hessian and args.hessian_step_interval is not None and
            update_step % args.hessian_step_interval == 0 and args.compute_hessian_precond):
            if global_cpu_opt_states is not None:
                for i, optimizer in enumerate(local_optimizers):
                    global_idx = node_indices[i]
                    local_model = local_models[i]
                    opt_state_list = []
                    for p in local_model.parameters():
                        if p.requires_grad:
                            st = optimizer.state.get(p, {})
                            exp_avg_sq = st.get('exp_avg_sq', None)
                            if exp_avg_sq is not None:
                                opt_state_list.append(exp_avg_sq.detach().cpu())
                            else:
                                opt_state_list.append(torch.zeros_like(p, device='cpu'))
                    global_cpu_opt_states[global_idx] = opt_state_list

        barrier.wait()

        if rank == 0:
            curr_connectivity = get_connectivity(update_step, args.max_iters, args)
            matrix = create_gossip_matrix(n_nodes=len(global_cpu_models), topology=args.topology, connectivity=curr_connectivity, bidirectional=True)
            curr_gamma = get_gamma(update_step, args.max_iters, args)
            curr_beta = get_beta(update_step, args.max_iters, args)
            log_queue.put({
                'type': 'schedule', 'step': update_step,
                'matrix_density': torch.count_nonzero(matrix).item() / matrix.numel(),
                'gamma': curr_gamma, 'beta': curr_beta, 'connectivity': curr_connectivity, 'k_steps': current_k_for_log
            })
            t0 = time.time()

            if global_cpu_models_new is not None:
                eye = torch.eye(len(global_cpu_models), device=matrix.device)
                W_old = (1.0 - curr_gamma) * eye + (curr_gamma - curr_beta) * matrix
                W_new = curr_beta * matrix

                if args.gossip_momentum > 0.0 and global_cpu_momentum is not None:
                    momentum_alpha = args.gossip_momentum
                    for param_group in global_cpu_momentum:
                        for p in param_group.parameters():
                             if p.requires_grad: p.data.mul_(momentum_alpha)
                    gossip_accumulate(global_cpu_models_new, W_new, global_cpu_momentum, exclude_embedding=args.no_gossip_embedding)
                    W_diff = W_old - eye
                    gossip_accumulate(global_cpu_models, W_diff, global_cpu_momentum, exclude_embedding=args.no_gossip_embedding)
                    for i in range(len(global_cpu_models)):
                        for (n_a, p_a), (n_b, p_b), (n_m, p_m) in zip(
                            global_cpu_models[i].named_parameters(),
                            global_cpu_models_new[i].named_parameters(),
                            global_cpu_momentum[i].named_parameters()
                        ):
                            if args.no_gossip_embedding and ('wte' in n_a or 'wpe' in n_a or 'embed_tokens' in n_a):
                                p_a.data.copy_(p_b.data)
                            else:
                                p_a.data.add_(p_m.data, alpha=args.outer_learning_rate)
                        for b_a, b_b in zip(global_cpu_models[i].buffers(), global_cpu_models_new[i].buffers()):
                             if not b_a.is_floating_point(): b_a.data.copy_(b_b.data)
                             else: b_a.data.copy_(b_b.data)
                else:
                    gossip_update(global_cpu_models, W_old, exclude_embedding=args.no_gossip_embedding)
                    gossip_update(global_cpu_models_new, W_new, exclude_embedding=args.no_gossip_embedding)
                    for i in range(len(global_cpu_models)):
                        for (n_a, p_a), (n_b, p_b) in zip(global_cpu_models[i].named_parameters(), global_cpu_models_new[i].named_parameters()):
                            if args.no_gossip_embedding and ('wte' in n_a or 'wpe' in n_a or 'embed_tokens' in n_a):
                                p_a.data.copy_(p_b.data)
                            else:
                                p_a.data.add_(p_b.data)
                        for b_a, b_b in zip(global_cpu_models[i].buffers(), global_cpu_models_new[i].buffers()):
                            if b_a.is_floating_point(): b_a.data.add_(b_b.data)
                            else: b_a.data.copy_(b_b.data)
                print(f"[Debug] Step {update_step}: Dual Gossip Aggregation Done (Time: {time.time()-t0:.4f}s) | K={current_k_for_log}")
            else:
                if curr_gamma < 1.0:
                    eye = torch.eye(len(global_cpu_models), device=matrix.device)
                    matrix = (1.0 - curr_gamma) * eye + curr_gamma * matrix

                gossip_update(global_cpu_models, matrix, exclude_embedding=args.no_gossip_embedding)

                if args.gossip_local_momentum and global_cpu_local_momentums is not None:
                    gossip_update(global_cpu_local_momentums, matrix, exclude_embedding=args.no_gossip_embedding)

                print(f"[Debug] Step {update_step}: Gossip Aggregation Done (Time: {time.time()-t0:.4f}s) | K={current_k_for_log}")

            if update_step % args.consensus_step_interval == 0:
                cons_err = compute_consensus_error(global_cpu_models)
                log_queue.put({'step': update_step, 'type': 'consensus_monitor_post', 'consensus_error': cons_err, 'k_steps': current_k_for_log})

            if (args.compute_hessian and args.hessian_step_interval is not None and
                update_step % args.hessian_step_interval == 0):
                print(f"[Debug] Step {update_step}: Computing Hessian (Distributed)")
                consensus_model = create_consensus_model(global_cpu_models).to(device)
                avg_opt_states = []
                cons_optimizer = None
                node0_optimizer = None
                if args.compute_hessian_precond and global_cpu_opt_states is not None:
                    n_params = len(global_cpu_opt_states[0])
                    for p_idx in range(n_params):
                        sum_tensor = sum(global_cpu_opt_states[n][p_idx] for n in range(args.n_nodes))
                        avg_tensor = sum_tensor / args.n_nodes
                        avg_opt_states.append(avg_tensor)
                    cons_optimizer = PseudoOptimizer(consensus_model.parameters(), avg_opt_states, step=update_step)
                    node0_opt_states = global_cpu_opt_states[0]

                if val_loader is not None:
                    try:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)

                    node0_model_gpu = deepcopy(global_cpu_models[0]).to(device)
                    if args.compute_hessian_precond and global_cpu_opt_states is not None:
                            node0_opt_states = global_cpu_opt_states[0]
                            node0_optimizer = PseudoOptimizer(node0_model_gpu.parameters(), node0_opt_states, step=update_step)

                    if args.hessian_distributed:
                        hessian_metrics = compute_hessian_distributed(
                            consensus_model, node0_model_gpu, cons_optimizer, node0_optimizer,
                            val_batch, None, device, args, hessian_v_blocks, hessian_v_blocks_node0, hessian_v_blocks_node0_pre
                        )
                    else:
                        if rank == 0:
                            hessian_metrics = compute_hessian_on_rank0(
                                consensus_model, node0_model_gpu, cons_optimizer, node0_optimizer,
                                val_batch, None, device, args, hessian_v_blocks, hessian_v_blocks_node0, hessian_v_blocks_node0_pre
                            )
                        else: hessian_metrics = None

                    if hessian_metrics:
                        hessian_v_blocks = hessian_metrics.get('v_blocks')
                        hessian_v_blocks_node0 = hessian_metrics.get('v_blocks_node0')
                        hessian_v_blocks_node0_pre = hessian_metrics.get('v_blocks_node0_pre')
                        if rank == 0:
                            log_metrics = {}
                            for k, v in hessian_metrics.items():
                                if k in ['v_blocks', 'v_blocks_node0', 'v_blocks_node0_pre']: continue
                                if isinstance(v, torch.Tensor): continue
                                log_metrics[k] = v
                            log_queue.put({'type': 'hessian', 'metrics': log_metrics, 'step': update_step})
                    del consensus_model
                    del node0_model_gpu
                    del cons_optimizer
                    del node0_optimizer
                    torch.cuda.empty_cache()

        barrier.wait()
        for i, model in enumerate(local_models):
            global_idx = node_indices[i]
            model.load_state_dict(global_cpu_models[global_idx].state_dict())

            if args.gossip_local_momentum and global_cpu_local_momentums is not None:
                optimizer = local_optimizers[i]
                target_mom_model = global_cpu_local_momentums[global_idx]

                with torch.no_grad():
                    for (name, param), (tgt_name, tgt_param) in zip(model.named_parameters(), target_mom_model.named_parameters()):
                        if param.requires_grad:
                             state = optimizer.state.get(param, None)
                             if state is not None and 'exp_avg' in state:
                                 state['exp_avg'].copy_(tgt_param.data.to(device))

        if not args.no_gossip_reset_optimizer:
            for i, opt in enumerate(local_optimizers):
                opt.state.clear()
                if local_muon_optimizers[i] is not None:
                    local_muon_optimizers[i].state.clear()

    while update_step < args.max_iters:
        cdat_manager.capture_params(local_models)
        current_k = max(1, get_k_steps(update_step, args.max_iters, args))

        for step_in_round in range(current_k):
            if update_step >= args.max_iters:
                break
            did_update_this_iter = False
            
            for i, model in enumerate(local_models):
                global_idx = node_indices[i]
                optimizer = local_optimizers[i]
                muon_optimizer = local_muon_optimizers[i]
                current_lr = cdat_manager.get_lr(update_step, args.learning_rate, global_idx)
                lr_scale = current_lr / args.learning_rate if args.learning_rate > 0 else 0.0
                if args.use_gossip_warmup and update_step < args.gossip_warmup_steps:
                    warmup_progress = update_step / max(1, args.gossip_warmup_steps)
                    warmup_factor = args.gossip_warmup_min_ratio + (1.0 - args.gossip_warmup_min_ratio) * warmup_progress
                    lr_scale = lr_scale * warmup_factor
                
                for param_group in optimizer.param_groups:
                    base_group_lr = param_group.get('initial_lr', args.learning_rate)
                    param_group['lr'] = base_group_lr * lr_scale
                if muon_optimizer is not None:
                    for param_group in muon_optimizer.param_groups:
                        base_group_lr = param_group.get('initial_lr', args.muon_learning_rate)
                        param_group['lr'] = base_group_lr * lr_scale
                
                try:
                    batch = next(data_iters[i])
                except StopIteration:
                    data_iters[i] = iter(train_loaders[i])
                    batch = next(data_iters[i])
                
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    logits, loss = model(input_ids, labels)
                    if args.grad_accum_steps > 1:
                        loss = loss / args.grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (iter_step + 1) % args.grad_accum_steps == 0:
                    if args.use_amp:
                        scaler.unscale_(optimizer)
                        if muon_optimizer is not None:
                            scaler.unscale_(muon_optimizer)
                    if args.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if args.use_amp:
                        scaler.step(optimizer)
                        if muon_optimizer is not None:
                            scaler.step(muon_optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        if muon_optimizer is not None:
                            muon_optimizer.step()
                    optimizer.zero_grad()
                    if muon_optimizer is not None:
                        muon_optimizer.zero_grad()
                    did_update_this_iter = True
                
                loss_val = loss.item()
                if args.grad_accum_steps > 1:
                    loss_val *= args.grad_accum_steps
                local_losses_accum[global_idx].append(loss_val)
            
            iter_step += 1
            if not did_update_this_iter:
                continue

            update_step += 1
            current_k = max(1, get_k_steps(update_step, args.max_iters, args))
            if args.use_cdat and update_step % 10 == 0:
                 cdat_manager.compute_update(update_step, local_models, data_iters, train_loaders)

            if update_step % args.log_interval == 0:
                for i, model in enumerate(local_models):
                    global_idx = node_indices[i]
                    if local_losses_accum[global_idx]:
                        avg = np.mean(local_losses_accum[global_idx])
                        if shared_train_losses is not None:
                            shared_train_losses[global_idx] = avg
                    else:
                        if shared_train_losses is not None:
                            shared_train_losses[global_idx] = 0.0
                barrier.wait()
                if rank == 0:
                    train_metrics = {}
                    if shared_train_losses is not None:
                        all_losses = [shared_train_losses[i] for i in range(args.n_nodes)]
                        avg_loss = np.mean(all_losses)
                        for i, l in enumerate(all_losses):
                            train_metrics[f'node_{i}_loss'] = l
                    else:
                        all_avgs = []
                        for idx in node_indices: 
                            if local_losses_accum[idx]: all_avgs.append(np.mean(local_losses_accum[idx]))
                        avg_loss = np.mean(all_avgs) if all_avgs else 0.0
                    curr_connectivity_log = get_connectivity(update_step, args.max_iters, args)
                    curr_gamma_log = get_gamma(update_step, args.max_iters, args)
                    curr_beta_log = get_beta(update_step, args.max_iters, args)
                    current_lr = 0.0
                    if local_optimizers:
                        current_lr = local_optimizers[0].param_groups[0]['lr']
                    current_wte_lr = None
                    if local_optimizers:
                        current_wte_lr = get_optimizer_group_lr(local_optimizers[0], 'wte')
                    current_muon_lr = 0.0
                    if local_muon_optimizers and local_muon_optimizers[0] is not None:
                        current_muon_lr = local_muon_optimizers[0].param_groups[0]['lr']
                    effective_tokens_total = update_step * effective_tokens_per_update
                    log_queue.put({
                        'step': update_step, 'loss': avg_loss, 'lr': current_lr,
                        'lr_wte': current_wte_lr,
                        'lr_muon': current_muon_lr,
                        'effective_tokens_total': effective_tokens_total,
                        'k_steps': current_k, 'connectivity': curr_connectivity_log,
                        'gamma': curr_gamma_log, 'beta': curr_beta_log, 'train_metrics': train_metrics
                    })
                for idx in node_indices: local_losses_accum[idx] = []

            if update_step % args.consensus_step_interval == 0:
                for i, model in enumerate(local_models):
                    global_idx = node_indices[i]
                    global_cpu_models[global_idx].load_state_dict(model.state_dict())
                barrier.wait()
                if rank == 0:
                    cons_err = compute_consensus_error(global_cpu_models)
                    log_queue.put({'step': update_step, 'type': 'consensus_monitor_pre', 'consensus_error': cons_err})
                barrier.wait()

            if update_step % args.eval_interval == 0 and val_loader is not None:
                local_val_losses = {}
                
                # Validation Batches Calculation
                # Target: zzp1012/LLM-optimizers (~10M tokens)
                batch_tokens = args.batch_size * args.block_size
                # Here we are running on a single GPU (rank 0 handles consensus val), 
                # so world_size=1 effectively for this sequential validation loop.
                # Or do we parallelize? 
                
                # In this script, 'consensus_model' validation is done sequentially on rank 0 GPU.
                # So step size is just based on 1 batch.
                target_val_steps = args.val_tokens // batch_tokens
                if target_val_steps < 1: target_val_steps = 1
                
                max_val_steps = max(1, target_val_steps)
                
                for i, model in enumerate(local_models):
                    global_idx = node_indices[i]
                    model.eval()
                    v_loss = 0.0
                    v_steps = 0
                    # Explicit new iterator (implicit via loop)
                    # This resets the loader state.
                    with torch.no_grad():
                        for val_batch in val_loader:
                            if v_steps >= max_val_steps: break
                            x, y = val_batch
                            x, y = x.to(device), y.to(device)
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                _, loss = model(x, y)
                            v_loss += loss.item()
                            v_steps += 1
                    model.train()
                    v_loss_avg = v_loss / max(1, v_steps)
                    local_val_losses[f"val_loss_node_{global_idx}"] = v_loss_avg
                    if shared_val_losses is not None:
                        shared_val_losses[global_idx] = v_loss_avg
                barrier.wait()
                log_queue.put({'step': update_step, 'type': 'local_val', 'metrics': local_val_losses})
                for i, model in enumerate(local_models):
                    global_idx = node_indices[i]
                    global_cpu_models[global_idx].load_state_dict(model.state_dict())
                barrier.wait()
                if rank == 0:
                    if shared_val_losses is not None:
                        all_val_losses = [shared_val_losses[i] for i in range(args.n_nodes)]
                        avg_local_val_loss = np.mean(all_val_losses)
                        log_queue.put({'step': update_step, 'type': 'val_avg', 'avg_local_val_loss': avg_local_val_loss})
                    consensus_model = create_consensus_model(global_cpu_models).to(device)
                    consensus_model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    # Explicit new iterator
                    with torch.no_grad():
                        for val_batch in val_loader:
                            if val_steps >= max_val_steps: break
                            x, y = val_batch
                            x, y = x.to(device), y.to(device)
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                _, loss = consensus_model(x, y)
                            val_loss += loss.item()
                            val_steps += 1
                    avg_val_loss = val_loss / max(1, val_steps)
                    cons_err = compute_consensus_error(global_cpu_models)
                    val_loss_diff = avg_val_loss - avg_local_val_loss

                    if args.save_checkpoints:
                        consensus_state = {
                            k: v.detach().cpu() for k, v in consensus_model.state_dict().items()
                        }
                        optimizer_state = local_optimizers[0].state_dict() if local_optimizers else None
                        optimizer_state_muon = local_muon_optimizers[0].state_dict() if (local_muon_optimizers and local_muon_optimizers[0] is not None) else None
                        ckpt_payload = {
                            'format_version': 2,
                            'checkpoint_type': 'full',
                            'model_state': consensus_state,
                            'optimizer_state': optimizer_state,
                            'optimizer_state_adamw': optimizer_state if args.optimizer == 'muon' else None,
                            'optimizer_state_muon': optimizer_state_muon,
                            'scaler_state': scaler.state_dict() if args.use_amp else None,
                            'global_step': int(update_step),
                            'best_val_loss': float(best_val_loss),
                            'val_loss': float(avg_val_loss),
                            'meta': {
                                'model_config': args.model_config,
                                'model_family': model_family,
                                'optimizer': args.optimizer,
                                'n_layer': args.n_layer,
                                'n_head': args.n_head,
                                'n_embd': args.n_embd,
                                'block_size': args.block_size,
                            },
                        }

                        if args.save_interval > 0 and update_step % args.save_interval == 0:
                            save_path = os.path.join(args.out_dir, f'ckpt_step_{update_step}.pt')
                            torch.save(ckpt_payload, save_path)
                            print(f"[Checkpoint] Saved step checkpoint: {save_path}")

                        if args.save_best_val and avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            ckpt_payload['best_val_loss'] = float(best_val_loss)
                            best_path = os.path.join(args.out_dir, 'ckpt_best.pt')
                            torch.save(ckpt_payload, best_path)
                            print(f"[Checkpoint] Saved best checkpoint: {best_path} (val={avg_val_loss:.6f})")

                    log_queue.put({
                        'step': update_step, 'val_loss': avg_val_loss,
                        'consensus_error': cons_err, 'val_loss_diff': val_loss_diff
                    })
                    print(f"Step {update_step}: Val Loss (Global Consensus) = {avg_val_loss:.4f} | Diff = {val_loss_diff:.4f}")
                    del consensus_model
                barrier.wait()

            if update_step % current_k == 0:
                run_gossip(current_k)

    if rank == 0:
        print("Training complete.")

# Hessian Utils copied
def compute_hessian_on_rank0(consensus_model, node0_model, cons_optimizer, node0_optimizer, batch, tokenizer, device, args, v_blocks, v_blocks_node0, v_blocks_node0_pre):
    if v_blocks is None: v_blocks = []
    if v_blocks_node0 is None: v_blocks_node0 = []
    if v_blocks_node0_pre is None: v_blocks_node0_pre = []
    input_ids, labels = batch
    hessian_bs = min(args.hessian_batch_size, input_ids.size(0))
    input_ids = input_ids[:hessian_bs].to(device)
    labels = labels[:hessian_bs].to(device)
    def loss_fn(pred, target): return nn.functional.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
    def get_blocks(model):
        return get_hessian_blocks(model)
    metrics = {}
    est_kwargs = { 'loss_fn': loss_fn, 'batch': input_ids, 'tokenizer': None, 'labels': labels, 'n_power_iter': args.hessian_power_iter, 'tol': 1e-3, 'use_eval_mode': True, 'sign_align': True }
    try:
        start = time.time()
        cons_results = estimate_precondhessian_lmax_blockdiag(model=consensus_model, blocks=get_blocks(consensus_model), optimizer=cons_optimizer, init_v_blocks=v_blocks, **est_kwargs)
        metric_key = 'consensus_precond' if cons_optimizer is not None else 'consensus_raw'
        metrics['global_lmax'] = cons_results[0]
        metrics[f'{metric_key}_lmax'] = cons_results[0]
        metrics[f'{metric_key}_per_block_lmax'] = cons_results[2]
        metrics['v_blocks'] = cons_results[3]
        metrics['elapsed'] = time.time() - start
    except Exception as e: print(f"Hessian computation failed: {e}")
    try:
        node0_results = estimate_precondhessian_lmax_blockdiag(model=node0_model, blocks=get_blocks(node0_model), optimizer=None, init_v_blocks=v_blocks_node0, **est_kwargs)
        metrics['node0_raw_lmax'] = node0_results[0]
        metrics['v_blocks_node0'] = node0_results[3]
        metrics['node0_raw_per_block_lmax'] = node0_results[2]
    except Exception as e: print(f"Node 0 Hessian computation failed: {e}")
    if node0_optimizer is not None:
        try:
            node0_pre_results = estimate_precondhessian_lmax_blockdiag(model=node0_model, blocks=get_blocks(node0_model), optimizer=node0_optimizer, init_v_blocks=v_blocks_node0_pre, **est_kwargs)
            metrics['node0_precond_lmax'] = node0_pre_results[0]
            metrics['v_blocks_node0_pre'] = node0_pre_results[3]
            metrics['node0_precond_per_block_lmax'] = node0_pre_results[2]
        except Exception as e: print(f"Node 0 Precond Hessian computation failed: {e}")
    return metrics

def compute_hessian_distributed(consensus_model, node0_model, cons_optimizer, node0_optimizer, local_val_batch, tokenizer, device, args, v_blocks, v_blocks_node0, v_blocks_node0_pre):
    if v_blocks is None: v_blocks = []
    if v_blocks_node0 is None: v_blocks_node0 = []
    if v_blocks_node0_pre is None: v_blocks_node0_pre = []
    input_ids, labels = local_val_batch 
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    target_local_bs = args.hessian_batch_size // world_size
    if rank < (args.hessian_batch_size % world_size): target_local_bs += 1
    real_bs = min(target_local_bs, input_ids.size(0))
    if real_bs > 0:
        batch_input = input_ids[:real_bs].to(device)
        batch_labels = labels[:real_bs].to(device)
    else:
        batch_input = input_ids[:1].to(device)
        batch_labels = labels[:1].to(device)
    def loss_fn(pred, target): return nn.functional.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1), reduction='sum')
    def get_blocks(model):
        return get_hessian_blocks(model)
    metrics = {}
    total_bs = args.hessian_batch_size
    def distributed_reduce(tensor_list):
        if real_bs == 0:
            for t in tensor_list: t.zero_()
        if dist.is_initialized():
             for t in tensor_list: dist.all_reduce(t, op=dist.ReduceOp.SUM)
        scale = 1.0 / max(1.0, float(total_bs))
        return [t * scale for t in tensor_list]
    est_kwargs = { 'loss_fn': loss_fn, 'batch': batch_input, 'tokenizer': None, 'labels': batch_labels, 'n_power_iter': args.hessian_power_iter, 'tol': 1e-3, 'use_eval_mode': True, 'sign_align': True, 'reduce_fn': distributed_reduce }
    try:
        start = time.time()
        cons_results = estimate_precondhessian_lmax_blockdiag(model=consensus_model, blocks=get_blocks(consensus_model), optimizer=cons_optimizer, init_v_blocks=v_blocks, **est_kwargs)
        metrics['v_blocks'] = cons_results[3] 
        if rank == 0:
            metric_key = 'consensus_precond' if cons_optimizer is not None else 'consensus_raw'
            metrics['global_lmax'] = cons_results[0]
            metrics[f'{metric_key}_lmax'] = cons_results[0]
            metrics[f'{metric_key}_per_block_lmax'] = cons_results[2]
            block_lmax_values = [res[1] for res in cons_results[2]]
            if block_lmax_values: metrics[f'{metric_key}_avg_block_lmax'] = sum(block_lmax_values) / len(block_lmax_values)
            metrics['elapsed'] = time.time() - start
    except Exception as e: print(f"[Rank {rank}] Consensus Hessian computation failed: {e}")
    try:
        node0_results = estimate_precondhessian_lmax_blockdiag(model=node0_model, blocks=get_blocks(node0_model), optimizer=None, init_v_blocks=v_blocks_node0, **est_kwargs)
        metrics['v_blocks_node0'] = node0_results[3]
        if rank == 0:
            metrics['node0_raw_lmax'] = node0_results[0]
            metrics['node0_raw_per_block_lmax'] = node0_results[2]
            node0_raw_block_lmax_values = [res[1] for res in node0_results[2]]
            if node0_raw_block_lmax_values: metrics['node0_raw_avg_block_lmax'] = sum(node0_raw_block_lmax_values) / len(node0_raw_block_lmax_values)
    except Exception as e: print(f"[Rank {rank}] Node 0 Hessian computation failed: {e}")
    if node0_optimizer is not None:
        try:
            node0_pre_results = estimate_precondhessian_lmax_blockdiag(model=node0_model, blocks=get_blocks(node0_model), optimizer=node0_optimizer, init_v_blocks=v_blocks_node0_pre, **est_kwargs)
            metrics['v_blocks_node0_pre'] = node0_pre_results[3]
            if rank == 0:
                metrics['node0_precond_lmax'] = node0_pre_results[0]
                metrics['node0_precond_per_block_lmax'] = node0_pre_results[2]
                node0_pre_block_lmax_values = [res[1] for res in node0_pre_results[2]]
                if node0_pre_block_lmax_values: metrics['node0_precond_avg_block_lmax'] = sum(node0_pre_block_lmax_values) / len(node0_pre_block_lmax_values)
        except Exception as e: print(f"[Rank {rank}] Node 0 Precond Hessian computation failed: {e}")
    return metrics if rank == 0 else {'v_blocks': metrics.get('v_blocks'), 'v_blocks_node0': metrics.get('v_blocks_node0'), 'v_blocks_node0_pre': metrics.get('v_blocks_node0_pre')}

class PseudoOptimizer:
    def __init__(self, model_params, state_list, step=1):
        self.param_groups = [{'params': list(model_params), 'betas': (0.9, 0.95), 'eps': 1e-8}]
        self.state = {}
        for p, s in zip(self.param_groups[0]['params'], state_list):
            if s is not None:
                s_device = s.to(p.device) if isinstance(s, torch.Tensor) else s
                self.state[p] = {'exp_avg_sq': s_device, 'step': step}
            else: self.state[p] = {'step': step}

def main():
    parser = argparse.ArgumentParser(description='Non-IID Decentralized Training (Generic Shards)')
    parser.add_argument('--n_nodes', type=int, default=16, help='Total number of nodes')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs available')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--wte_lr_ratio', type=float, default=1.0, help='LR ratio applied only to token embedding params (wte/embed_tokens) inside AdamW')
    parser.add_argument('--k_steps', type=int, default=10, help='Gossip interval in update steps (run gossip every k update steps)')
    parser.add_argument('--k_scheduler', type=str, default='constant', choices=['constant', 'linear_increase', 'linear_decay', 'cosine', 'cosine_increase', 'exp_increase', 'exp_decay'])
    parser.add_argument('--k_min', type=int, default=10)
    parser.add_argument('--k_max', type=int, default=100)
    parser.add_argument('--k_scheduler_alpha', type=float, default=1.0)
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--shards_per_node', type=int, default=None, help='Limit number of shards to use per node')
    parser.add_argument('--topology', type=str, default='random', choices=['ring', 'random', 'random_sym', 'complete'])
    parser.add_argument('--connectivity', type=float, default=2.0)
    parser.add_argument('--connectivity_max', type=float, default=16.0)
    parser.add_argument('--connectivity_min', type=float, default=2.0)
    parser.add_argument('--connectivity_scheduler', type=str, default='constant')
    parser.add_argument('--scheduler_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_cdat', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default='constant', choices=['constant', 'wsd', 'd2z'])
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--decay_steps', type=int, default=None)
    parser.add_argument('--muon_learning_rate', type=float, default=0.0036)
    parser.add_argument('--muon_momentum', type=float, default=0.9)
    parser.add_argument('--muon_ns_steps', type=int, default=5)
    parser.add_argument('--compute_hessian', action='store_true')
    parser.add_argument('--compute_hessian_precond', action='store_true')
    parser.add_argument('--hessian_step_interval', type=int, default=None)
    parser.add_argument('--hessian_power_iter', type=int, default=10)
    parser.add_argument('--hessian_batch_size', type=int, default=8)
    parser.add_argument('--hessian_distributed', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--consensus_step_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_tokens', type=int, default=10485760, help='Total tokens for validation')
    parser.add_argument('--wandb_project', type=str, default='DecentralizedLLM_Generic_MultiGPU')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs/decentralized_generic', help='Directory to save warm-start checkpoints')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save warm-start checkpoint every N steps')
    parser.add_argument('--save_checkpoints', action='store_true', help='Enable checkpoint saving')
    parser.add_argument('--save_best_val', action='store_true', help='Save best validation checkpoint')
    parser.add_argument('--init_ckpt', type=str, default='', help='Path to init checkpoint (full or model-only)')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/scratch/DEC_dataset/fineweb_sharded'), help='Directory containing train_*.bin and val.bin')
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--use_asymmetry', action='store_true')
    parser.add_argument('--asym_ratio_mlp', type=float, default=0.05)
    parser.add_argument('--asym_ratio_attn', type=float, default=0.05)
    parser.add_argument('--asym_version', type=str, default='v1')
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--no_gossip_embedding', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gamma_max', type=float, default=1.0)
    parser.add_argument('--gamma_min', type=float, default=0.0)
    parser.add_argument('--gamma_scheduler', type=str, default='constant')
    parser.add_argument('--gamma_scheduler_alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--beta_max', type=float, default=1.0)
    parser.add_argument('--beta_min', type=float, default=1.0)
    parser.add_argument('--beta_scheduler', type=str, default='constant')
    parser.add_argument('--beta_scheduler_alpha', type=float, default=1.0)
    parser.add_argument('--no_gossip_reset_optimizer', action='store_true')
    parser.add_argument('--use_gossip_warmup', action='store_true')
    parser.add_argument('--gossip_warmup_steps', type=int, default=5)
    parser.add_argument('--gossip_warmup_min_ratio', type=float, default=0.1)
    parser.add_argument('--gossip_momentum', type=float, default=0.0)
    parser.add_argument('--gossip_local_momentum', action='store_true', help='Enable gossiping of local optimizer momentum (exp_avg)')
    parser.add_argument('--outer_learning_rate', type=float, default=1.0, help='Outer learning rate for momentum application')
    parser.add_argument('--dataset_name', type=str, default='GenShards', help='Name of the dataset (e.g. fineweb, wikitext)')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    args = parser.parse_args()
    
    if args.model_config is not None:
        try:
            from model_configs import apply_model_config
            apply_model_config(args, args.model_config)
        except ImportError: print("Warning: model_configs.py not found.")
    
    # --- 1. Scan Data ---
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    shards = sorted(glob.glob(os.path.join(args.data_dir, 'train_*.bin')))
    if not shards and not os.path.exists(os.path.join(args.data_dir, 'train.bin')):
         raise FileNotFoundError(f"No train_*.bin shards found in {args.data_dir}")
    
    # If no numbered shards but 'train.bin' exists, treat as 1 shard
    if not shards and os.path.exists(os.path.join(args.data_dir, 'train.bin')):
         shards = [os.path.join(args.data_dir, 'train.bin')]
    
    print(f"Found {len(shards)} training shards.")
    
    # Assign shards to nodes
    # If shards > nodes, we distribute them round-robin so nodes get multiple shards
    # Current GenericShardedDataset can handle a list of shard indices? 
    #   -> No, it takes `shard_indices` list and loads them all into a ConcatDataset.
    
    node_shard_map = {i: [] for i in range(args.n_nodes)}
    
    # Check if shards are named with standard pattern to extract indices
    # We generally rely on the sorted order of glob for consistency
    
    for idx, shard_path in enumerate(shards):
        # Assign this shard to a node
        node_idx = idx % args.n_nodes
        
        # We need to determine the integer ID for this shard
        fname = os.path.basename(shard_path)
        if fname == 'train.bin':
            shard_id = 0 # Dummy ID for single file
        else:
            # Try to extract number from train_XXXX.bin
            # Regex or simple string manipulation
            try:
                # remove prefix 'train_' and suffix '.bin'
                shard_id = int(fname.replace('train_', '').replace('.bin', ''))
            except ValueError:
                # If naming is weird, fall back to enumeration index if relying on sorted order
                # But GenericShardedDataset constructs paths based on ID!
                # So we MUST have the correct ID that matches the filename.
                print(f"Warning: Could not extract specific ID from {fname}, using enumeration index {idx}")
                shard_id = idx
                
        node_shard_map[node_idx].append(shard_id)
    
    # Limit shards per node if requested
    if args.shards_per_node is not None and args.shards_per_node > 0:
        print(f"Applying shard limit: {args.shards_per_node} per node.")
        for node_idx in node_shard_map:
            if len(node_shard_map[node_idx]) > args.shards_per_node:
                node_shard_map[node_idx] = node_shard_map[node_idx][:args.shards_per_node]

    print("Node Shard Assignment:")
    for i in range(min(10, args.n_nodes)):
        print(f"  Node {i}: {node_shard_map[i]}")
    if args.n_nodes > 10: print("  ...")

    # --- 2. Setup Multiprocessing ---
    mp.set_start_method('spawn', force=True)
    
    # Initialize appropriate model for parameter counting/sizing
    temp_model, temp_model_family = build_model_from_args(args, vocab_size=57728, device=None)
    if temp_model_family != 'llama':
        temp_model = convert_to_asymmetric_model(temp_model, args)
    
    global_cpu_models = []
    global_cpu_models_new = [] 
    for _ in range(args.n_nodes):
        m = deepcopy(temp_model)
        m.share_memory()
        global_cpu_models.append(m)
        m_new = deepcopy(temp_model)
        m_new.share_memory()
        global_cpu_models_new.append(m_new)
    
    barrier = Barrier(args.num_gpus)
    log_queue = mp.Queue()
    global_cpu_momentum = []
    if args.gossip_momentum > 0.0:
        for _ in range(args.n_nodes):
            m_mom = deepcopy(temp_model)
            for p in m_mom.parameters(): p.data.zero_()
            for b in m_mom.buffers(): b.data.zero_()
            m_mom.share_memory()
            global_cpu_momentum.append(m_mom)
    else: global_cpu_momentum = None
        
    shared_train_losses = mp.Array('d', args.n_nodes)
    shared_val_losses = mp.Array('d', args.n_nodes)
    global_cpu_opt_states = Manager().list([None] * args.n_nodes)
    
    # Initialize shared memory for local momentum gossip (wrapped in models for convenience)
    global_cpu_local_momentums = []
    if args.gossip_local_momentum:
        for _ in range(args.n_nodes):
            m_mom = deepcopy(temp_model)
            for p in m_mom.parameters(): p.data.zero_()
            for b in m_mom.buffers(): b.data.zero_() # buffers valid but zeroed
            m_mom.share_memory()
            global_cpu_local_momentums.append(m_mom)
    else:
        global_cpu_local_momentums = None

    nodes_per_gpu = args.n_nodes // args.num_gpus
    processes = []
    
    if WANDB_AVAILABLE:
        model_tag = args.model_config if args.model_config else f"L{args.n_layer}H{args.n_head}E{args.n_embd}"
        run_name = f"{args.dataset_name}_N{args.n_nodes}_{args.topology}_K{args.k_steps}_ctx{args.block_size}_Step{args.max_iters}_{model_tag}_bs{args.batch_size}"
        
        if args.iid:
            run_name = run_name.replace(args.dataset_name, f"IID_{args.dataset_name}")
            
        if args.topology == 'random':
            if args.connectivity_scheduler != 'constant':
                if 'increase' in args.connectivity_scheduler:
                    c_start, c_end = args.connectivity_min, args.connectivity_max
                else:
                    c_start, c_end = args.connectivity_max, args.connectivity_min
                run_name += f"_Conn-{args.connectivity_scheduler}-{c_start:g}to{c_end:g}"
            else:
                run_name += f"_conn{args.connectivity:g}"
        
        # Add K-Scheduler to run name
        if args.k_scheduler != 'constant':
            if 'increase' in args.k_scheduler:
                k_start, k_end = args.k_min, args.k_max
            else:
                k_start, k_end = args.k_max, args.k_min
            run_name += f"_K-{args.k_scheduler}-{k_start}to{k_end}"
            
        # Add Gamma to run name
        if args.gamma_scheduler != 'constant':
            if 'increase' in args.gamma_scheduler:
                g_start, g_end = args.gamma_min, args.gamma
            else:
                g_start, g_end = args.gamma, args.gamma_min
            run_name += f"_Gamma-{args.gamma_scheduler}-{g_start:g}to{g_end:g}"
        else:
             run_name += f"_Gamma{args.gamma}"

        # Add Beta to run name
        if args.beta_scheduler != 'constant':
            if 'increase' in args.beta_scheduler:
                b_start, b_end = args.beta_min, args.beta
            else:
                b_start, b_end = args.beta, args.beta_min
            run_name += f"_Beta-{args.beta_scheduler}-{b_start:g}to{b_end:g}"
        elif args.beta != 0.0: 
             run_name += f"_Beta{args.beta}"
        
        run_name += f"_{args.optimizer}"
        if args.optimizer == 'muon':
             run_name += f"_MuonLR{args.muon_learning_rate:g}"
        else:
             run_name += f"_lr{args.learning_rate:g}"
             
        if args.use_amp:
            run_name += "_amp"
        
        if args.lr_scheduler != 'constant':
            run_name += f"_{args.lr_scheduler}"

        if args.use_cdat:
            run_name += "_CDAT"
        
        if args.gossip_momentum > 0.0:
            run_name += f"_Momentum{args.gossip_momentum:g}"
            
        if args.outer_learning_rate != 1.0:
            run_name += f"_OuterLR{args.outer_learning_rate:g}"

        if args.shards_per_node is not None and args.shards_per_node > 0:
            run_name += f"_{args.shards_per_node}shards"
        
        if args.wandb_run_name: run_name = args.wandb_run_name

        if "WANDB_API_KEY" in os.environ: wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            try: wandb.login()
            except: pass
            
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"✅ WandB Initialized: {wandb.run.name}")

        if args.topology == 'random' and args.connectivity_scheduler != 'constant':
            print("Generating Connectivity Schedule Plan...")
            data = []
            step_size = max(1, args.max_iters // 100)
            for s in range(0, args.max_iters + 1, step_size):
                c = get_connectivity(s, args.max_iters, args)
                data.append([s, c])
            
            table = wandb.Table(data=data, columns=["step", "planned_connectivity"])
            wandb.log({"connectivity_plan": wandb.plot.line(table, "step", "planned_connectivity", title="Connectivity Schedule Plan")})

        if args.k_scheduler != 'constant':
             print("Generating K-Step Schedule Plan...")
             data_k = []
             step_size_k = max(1, args.max_iters // 100)
             for s in range(0, args.max_iters + 1, step_size_k):
                 kval = get_k_steps(s, args.max_iters, args)
                 data_k.append([s, kval])
             
             table_k = wandb.Table(data=data_k, columns=["step", "planned_k_steps"])
             wandb.log({"k_steps_plan": wandb.plot.line(table_k, "step", "planned_k_steps", title="K-Steps Schedule Plan")})

        if args.gamma_scheduler != 'constant':
             print("Generating Gamma Schedule Plan...")
             data_g = []
             step_size_g = max(1, args.max_iters // 100)
             for s in range(0, args.max_iters + 1, step_size_g):
                 gval = get_gamma(s, args.max_iters, args)
                 data_g.append([s, gval])
             
             table_g = wandb.Table(data=data_g, columns=["step", "planned_gamma"])
             wandb.log({"gamma_plan": wandb.plot.line(table_g, "step", "planned_gamma", title="Gamma Schedule Plan")})

    def logger_thread():
        while True:
            try:
                data = log_queue.get(timeout=1)
                if data is None: break
                
                if 'loss' in data and 'val_loss' not in data: # Train loss, print
                     print(f"Step {data['step']}: Loss {data['loss']:.4f} | K {data.get('k_steps', args.k_steps)}")
                     
                if WANDB_AVAILABLE and wandb.run:
                    log_dict = {'step': data['step']}
                    
                    # 1. Hessian Metrics
                    if 'type' in data and data['type'] == 'hessian' and 'metrics' in data:
                        for k, v in data['metrics'].items():
                            if k.endswith('_per_block_lmax'):
                                base_name = k.replace('_per_block_lmax', '')
                                for block_idx, lmax_val in v: log_dict[f"hessian/{base_name}_block_{block_idx}_lmax"] = lmax_val
                            elif k == 'per_block_lmax':
                                for block_idx, lmax_val in v: log_dict[f"hessian/block_{block_idx}_lmax"] = lmax_val
                            elif v is not None: log_dict[f"hessian/{k}"] = v
                            
                    # 2. Local Validation
                    elif 'type' in data and data['type'] == 'local_val' and 'metrics' in data:
                        for k, v in data['metrics'].items():
                             if k.startswith("val_loss_node_"): log_dict[f"val/node_{k.split('_')[-1]}_loss"] = v
                             else: log_dict[f"val/{k}"] = v
                             
                    # 3. Validation Avg
                    elif 'type' in data and data['type'] == 'val_avg':
                        log_dict['val/avg_local_loss'] = data['avg_local_val_loss']
                        
                    # 4. Consensus Monitors
                    elif 'type' in data and (data['type'] == 'consensus_monitor' or data['type'] == 'consensus_monitor_post'):
                        log_dict['val/consensus_error'] = data['consensus_error']
                        current_k_for_data = data.get('k_steps', args.k_steps)
                        if args.consensus_step_interval > current_k_for_data: log_dict['val/consensus_error_post_gossip'] = data['consensus_error']
                    elif 'type' in data and data['type'] == 'consensus_monitor_pre':
                         log_dict['val/consensus_error_pre_gossip'] = data['consensus_error']
                         log_dict['val/consensus_error_trace'] = data['consensus_error']
                         
                    # 5. Consensus Validation
                    elif 'val_loss' in data:
                        log_dict['val/consensus_loss'] = data['val_loss']
                        if 'consensus_error' in data: log_dict['val/consensus_error'] = data['consensus_error']
                        if 'val_loss_diff' in data: log_dict['val/loss_difference'] = data['val_loss_diff']
                        
                    # 6. Training Metrics
                    elif 'loss' in data:
                        log_dict['train/loss'] = data['loss']
                        if 'effective_tokens_total' in data:
                            log_dict['train/effective_tokens_total'] = data['effective_tokens_total']
                        if 'connectivity' in data: log_dict['schedule/connectivity'] = data['connectivity']
                        if 'k_steps' in data: log_dict['schedule/k_steps'] = data['k_steps']
                        if 'gamma' in data: log_dict['schedule/gamma'] = data['gamma']
                        if 'beta' in data: log_dict['schedule/beta'] = data['beta']
                        if 'lr' in data: log_dict['train/lr'] = data['lr']
                        if 'lr_wte' in data and data['lr_wte'] is not None: log_dict['train/lr_wte'] = data['lr_wte']
                        if 'lr_muon' in data and data['lr_muon'] > 0: log_dict['train/lr_muon'] = data['lr_muon']
                        
                        # Fix: Expand per-node training metrics
                        if 'train_metrics' in data:
                            for k, v in data['train_metrics'].items():
                                log_dict[f"train/{k}"] = v
                                
                    # 7. Schedule Metrics
                    elif 'type' in data and data['type'] == 'schedule':
                        if 'matrix_density' in data: log_dict['schedule/matrix_density'] = data['matrix_density']
                        if 'gamma' in data: log_dict['schedule/gamma'] = data['gamma']
                        if 'connectivity' in data: log_dict['schedule/connectivity'] = data['connectivity']
                        if 'k_steps' in data: log_dict['schedule/k_steps'] = data['k_steps']
                        
                    wandb.log(log_dict)
            except queue.Empty: continue
            except Exception as e: 
                print(f"[Logger] Error processing log: {e}")
                continue

                
    logger = threading.Thread(target=logger_thread, daemon=True)
    logger.start()

    for rank in range(args.num_gpus):
        start_idx = rank * nodes_per_gpu
        end_idx = (rank + 1) * nodes_per_gpu if rank < args.num_gpus - 1 else args.n_nodes
        worker_node_indices = list(range(start_idx, end_idx))
        p = mp.Process(
            target=worker_process,
            args=(rank, args.num_gpus, worker_node_indices, node_shard_map, args, barrier, global_cpu_models, log_queue, shared_train_losses, shared_val_losses, global_cpu_opt_states, global_cpu_models_new, global_cpu_momentum, global_cpu_local_momentums)
        )
        p.start()
        processes.append(p)
    for p in processes: p.join()

from torch.multiprocessing import Queue
if __name__ == "__main__":
    main()
