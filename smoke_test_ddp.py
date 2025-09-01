#!/usr/bin/env python
# smoke_test_ddp.py
# Tiny DDP smoke test for streaming training pipeline (bf16-capable GPUs assumed)

import os
import time
import random
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.amp import autocast

# -------------------------
# DDP helpers
# -------------------------
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    """
    Initialize process group using env vars set by torchrun.
    Returns: (rank, local_rank, world_size, device)
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Single process fallback (no DDP)
    if world_size == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return rank, local_rank, world_size, device

    # choose backend: prefer nccl on CUDA (Linux); torchrun usually uses NCCL
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not ddp_is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, device

def is_main_process() -> bool:
    return (not ddp_is_initialized()) or (dist.get_rank() == 0)

def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM if DDP initialized; returns reduced tensor on all ranks."""
    if not ddp_is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

# -------------------------
# Tiny toy model (autoregressive-like)
# -------------------------
class TinyLM(nn.Module):
    """Tiny autoregressive-like model: token embedding + pos embedding + linear head."""
    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, block_size, n_embd) * 0.01)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        """
        x: (B, L) long
        y: (B, L) long (targets)
        returns (logits, loss_if_targets_provided)
        """
        B, L = x.shape
        emb = self.tok_emb(x) + self.pos_emb[:, :L, :]
        logits = self.head(emb)  # (B, L, vocab)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

# -------------------------
# Small deterministic IterableDataset
# -------------------------
class RandomTokenIterable(IterableDataset):
    """Streams deterministic pseudo-random tokens. Supports rank+worker sharding."""
    def __init__(self, total_sequences:int, block_size:int, vocab_size:int, seed:int=1337):
        self.total_sequences = int(total_sequences)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

    def _shard(self, seqs: List[int]) -> List[int]:
        # rank sharding
        rank, world = 0, 1
        if ddp_is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        # worker sharding
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        # pick sequences for this rank+worker
        chosen = seqs[rank::world] if world > 1 else seqs
        chosen = chosen[worker_id::num_workers] if num_workers > 1 else chosen
        return chosen

    def __iter__(self):
        # deterministically build a list of sequence ids
        all_ids = list(range(self.total_sequences))
        # shuffle deterministically per process/worker using global seed + rank
        base_seed = self.seed + (dist.get_rank() if ddp_is_initialized() else 0)
        rng = random.Random(base_seed)
        rng.shuffle(all_ids)
        my_ids = self._shard(all_ids)

        for sid in my_ids:
            # create deterministic tokens per sid
            rng2 = random.Random(self.seed + sid)
            seq = [rng2.randrange(0, self.vocab_size) for _ in range(self.block_size + 1)]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            yield x, y

# -------------------------
# Eval helper (small)
# -------------------------
@torch.no_grad()
def estimate_loss_from_loaders(model, train_loader, val_loader, device, device_type, autocast_dtype, eval_iters=4):
    model_was_training = model.training
    model.eval()
    totals = {"train":0.0, "val":0.0}
    counts = {"train":0, "val":0}

    for name, loader in (("train", train_loader), ("val", val_loader)):
        it = iter(loader)
        for _ in range(eval_iters):
            try:
                X, Y = next(it)
            except StopIteration:
                break
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            with autocast(device_type=device_type, dtype=autocast_dtype):
                _, loss = model(X, Y)
            totals[name] += float(loss.item())
            counts[name] += 1

    # pack and all-reduce across ranks
    packed = torch.tensor([totals["train"], counts["train"], totals["val"], counts["val"]], dtype=torch.float64, device=device)
    packed = all_reduce_sum(packed)
    t_train_sum, c_train_sum, t_val_sum, c_val_sum = float(packed[0].item()), int(packed[1].item()), float(packed[2].item()), int(packed[3].item())
    train_mean = t_train_sum / max(1, c_train_sum)
    val_mean = t_val_sum / max(1, c_val_sum)

    if model_was_training:
        model.train()
    return {"train": train_mean, "val": val_mean}

# -------------------------
# Main smoke test
# -------------------------
def main():
    # DDP init
    rank, local_rank, world_size, device = ddp_setup()
    print(f"[rank {rank}] device={device}, world_size={world_size}")

    # config
    vocab_size = 256
    block_size = 32
    n_embd = 64
    batch_size = 2
    grads_accum_steps = 2
    total_effective_steps = 8
    eval_interval = 4
    train_seqs = 64
    val_seqs = 16
    global_seed = 1234

    # deterministic seeding
    random.seed(global_seed + rank)
    np.random.seed(global_seed + rank)
    torch.manual_seed(global_seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(global_seed + rank)

    # model
    model = TinyLM(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size).to(device)

    # wrap DDP only when initialized and multi-process
    if ddp_is_initialized() and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # use bf16 autocast if on CUDA (per your environment)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    # dataset + loaders
    train_ds = RandomTokenIterable(total_sequences=train_seqs, block_size=block_size, vocab_size=vocab_size, seed=global_seed)
    val_ds = RandomTokenIterable(total_sequences=val_seqs, block_size=block_size, vocab_size=vocab_size, seed=global_seed+1)

    pin_memory = (device.type == "cuda")
    num_workers = 0  # keep simple for smoke test
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader( val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    # streaming loop (micro-batches -> effective steps)
    train_it = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    effective_step = 0

    print(f"[rank {rank}] starting training loop (effective steps={total_effective_steps})")
    t0 = time.time()
    try:
        while effective_step < total_effective_steps:
            # get micro-batch, wrap-around loader if needed
            try:
                xb, yb = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                xb, yb = next(train_it)

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # forward in autocast
            with autocast(device_type=device_type, dtype=autocast_dtype):
                _, loss = model(xb, yb)
                loss = loss / grads_accum_steps

            loss.backward()
            micro_step += 1

            if micro_step % grads_accum_steps == 0:
                # clip grads (optional)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                effective_step += 1

                if is_main_process():
                    print(f"[main] step {effective_step} loss={float(loss.item()*grads_accum_steps):.4f}")

                # eval
                if effective_step % eval_interval == 0:
                    metrics = estimate_loss_from_loaders(model, train_loader, val_loader, device, device_type, autocast_dtype, eval_iters=2)
                    if is_main_process():
                        print(f"[main] eval @ step {effective_step} -> train={metrics['train']:.4f} val={metrics['val']:.4f}")

    except Exception as e:
        print(f"[rank {rank}] Exception during training: {e}")
        raise
    finally:
        if is_main_process():
            print("Main: training finished. Total time:", time.time() - t0)

        # cleanup
        if ddp_is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
