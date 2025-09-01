#!/usr/bin/env python
# coding: utf-8


# ===================================================
# 1. Standard library imports
# ===================================================
import json
import os
import math
import re
import time
import random
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterable, Tuple


# ===================================================
# 2. Third-party imports
# ===================================================
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pyarrow.dataset as ds
import pyarrow as pa
import tiktoken
from tqdm import tqdm


# ===================================================
# 3. Local imports
# ===================================================
from src.gpt2_model import GPTLanguageModel


run_id = int(time.time())

# Optional: nvml for VRAM reporting (if available)
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gpt-train")




# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    """
    Initialize DDP using environment variables set by torchrun.
    Returns (rank, local_rank, world_size, device)
    """
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size == 1:
        # single-process fallback: don't init process group
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    # choose backend: prefer nccl on linux + cuda, fallback to gloo (required on Windows)
    if torch.cuda.is_available():
        backend = "nccl" if (torch.version.cuda is not None and os.name != "nt") else "gloo"
    else:
        backend = "gloo"

    if not ddp_is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, device


def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return (not ddp_is_initialized()) or dist.get_rank() == 0

def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM in-place and return tensor (no-op if DDP not initialized)."""
    if not ddp_is_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ---------------------------------------------------------------------------
# Utilities: VRAM, seed, safe checkpoint load/save
# ---------------------------------------------------------------------------

def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")


def show_vram_usage(local_index: int = 0) -> str:
    if not NVML_AVAILABLE:
        return "nvml not available"
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(local_index)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"Used: {info.used // (1024 ** 2)} MB, Free: {info.free // (1024 ** 2)} MB, Total: {info.total // (1024 ** 2)} MB"


def set_global_seed(seed: int, rank: int = 0, device: torch.device = torch.device("cpu")):
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(s)

# -------------------------
# DummyScaler (no-op)
# -------------------------
class DummyScaler:
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def unscale_(self, optimizer): return
    def update(self): return
    def state_dict(self): return {}
    def load_state_dict(self, _: dict): return



def _get_eos_token_id(enc: "tiktoken.Encoding", model_hint: str = "gpt2") -> Optional[int]:
    try:
        ids = enc.encode("", allowed_special={"<|endoftext|>"})
        if ids: return int(ids[0])
    except Exception: pass
    try:
        ids = enc.encode("")
        if ids: return int(ids[0])
    except Exception: pass
    if model_hint.lower().startswith("gpt2"):
        return 50256
    return None

class ParquetCausalIterable(IterableDataset):
    """
    IterableDataset for streaming text from Parquet files and producing causal LM
    training samples (x, y) where:
        - x is a sequence of length `block_size`
        - y is the same sequence shifted by one token

    Key Features
    ------------
    • Supports both single-GPU and multi-GPU training with DDP:
        - Files are sharded across ranks
        - Further sharded across DataLoader workers inside each rank
    • Deterministic shuffling with per-epoch seeds
    • Tokenization done on-the-fly using tiktoken
    • Efficient rolling buffer yields overlapping sequences with stride
    • EOS tokens appended per row, sequences can cross row/file boundaries
    """

    def __init__(
        self,
        files: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        pattern: str = "*.parquet",
        text_col: Optional[str] = None,
        batch_rows: int = 4096,
        target_chunk_bytes: int = 100_000_000,
        tokenizer_model: str = "gpt2",
        add_eos: bool = True,
        block_size: int = 1024,
        stride: Optional[int] = None,
        lang_mode: str = "skip",
        ascii_threshold: float = 0.5,
        shuffle_files: bool = True,
        seed: int = 1337,
    ):
        super().__init__()
        self._all_files = files
        self.input_dir = input_dir
        self.pattern = pattern
        self.text_col = text_col
        self.batch_rows = int(batch_rows)
        self.target_chunk_bytes = int(target_chunk_bytes)
        self.tokenizer_model = tokenizer_model
        self.add_eos = add_eos
        self.block_size = int(block_size)
        self.stride = int(stride if stride is not None else block_size)  # no overlap by default
        self.lang_mode = lang_mode
        self.ascii_threshold = ascii_threshold
        self.shuffle_files = shuffle_files
        self.seed = int(seed)

        # Tokenizer can be created in __iter__ to ensure one per worker process
        self._epoch = 0  # current epoch number for deterministic seeding

    # -------------------------
    # Public API
    # -------------------------

    def set_epoch(self, epoch: int):
        """Set current epoch for deterministic shuffling."""
        self._epoch = int(epoch)

    # -------------------------
    # Helpers
    # -------------------------

    def _files(self) -> List[str]:
        """Resolve file list either from explicit `files` or scanning `input_dir`."""
        if self._all_files is not None:
            return list(self._all_files)
        if not self.input_dir:
            raise ValueError("Provide either `files` or `input_dir`.")
        p = Path(self.input_dir)
        if not p.exists():
            raise FileNotFoundError(self.input_dir)
        return sorted(str(x) for x in p.glob(self.pattern))

    def _shard_for_worker(self, files: List[str]) -> List[str]:
        """
        Shard files across:
          1. DDP ranks (each GPU gets disjoint subset)
          2. DataLoader workers inside each rank
        Apply deterministic shuffling per epoch.
        """
        # --- Rank sharding ---
        rank, world = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            files = files[rank::world]

        # --- Worker sharding ---
        info = get_worker_info()
        worker_id, num_workers = 0, 1
        if info is not None:
            worker_id, num_workers = info.id, info.num_workers
            files = files[worker_id::num_workers]

        # --- Deterministic RNG seed ---
        seed = self.seed + self._epoch * 10_000 + rank * 100 + worker_id
        rng = random.Random(seed)

        if self.shuffle_files:
            rng.shuffle(files)
        return files

    def _yield_sequences(self, token_buffer: List[int]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given a token buffer, yield overlapping (x, y) sequences:
          • length block_size
          • stride self.stride
          • x = seq[:-1], y = seq[1:]
        """
        i = 0
        bs1 = self.block_size + 1
        COMPACT_EVERY = max(8 * self.block_size, 32768)

        while i + bs1 <= len(token_buffer):
            seq = token_buffer[i: i + bs1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            yield x, y
            i += self.stride

            # compact buffer occasionally to free memory
            if i > COMPACT_EVERY:
                del token_buffer[:i]
                i = 0

        # keep only remainder for continuity
        if i > 0:
            del token_buffer[:i]

    # -------------------------
    # Main iterator
    # -------------------------

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over tokenized text samples from sharded files,
        yielding (x, y) pairs of shape (block_size,).
        """
        # One tokenizer per worker process
        enc = tiktoken.get_encoding(self.tokenizer_model)
        eos_id = _get_eos_token_id(enc, self.tokenizer_model) if self.add_eos else None

        files = self._shard_for_worker(self._files())
        if len(files) == 0:
            # If this rank has no files, sleep forever or raise depending on design.
            # Better: return an empty iterator so DataLoader will eventually StopIteration.
            return

        token_buffer: List[int] = []

        for file in files:
            dataset = ds.dataset([file], format="parquet")
            col = self.text_col or self._choose_string_column(dataset)

            # Important: disable threads for safety with DataLoader
            scanner = dataset.scanner(batch_size=self.batch_rows, use_threads=False)

            for rb in scanner.to_batches():
                arr = rb[col]
                for raw in arr.to_pylist():
                    if raw is None:
                        continue
                    text = str(raw).strip()
                    if not text:
                        continue
                    if not self._is_english(text):
                        continue

                    try:
                        toks = enc.encode_ordinary(text)
                    except Exception:
                        toks = enc.encode(text)

                    if toks:
                        token_buffer.extend(toks)
                        if eos_id is not None:
                            token_buffer.append(eos_id)

                # Yield sequences as soon as buffer has enough
                for x, y in self._yield_sequences(token_buffer):
                    yield x, y

            # At file boundary: keep only block_size tokens
            if len(token_buffer) > self.block_size:
                token_buffer[:] = token_buffer[-self.block_size:]

    # -------------------------
    # Utils
    # -------------------------

    def _choose_string_column(self, dataset: ds.Dataset) -> str:
        """Pick a text column if none provided (robust w/ pyarrow schema)."""
        schema = dataset.schema
        # schema is a pyarrow.Schema containing Field objects
        for f in schema:
            if pa.types.is_string(f.type):
                return f.name
        # fallback: any field named 'text','content','body'
        for candidate in ("text", "content", "body"):
            if candidate in [f.name for f in schema]:
                return candidate
        raise ValueError("No string column found in parquet file")
    
    def _is_english(self, text: str) -> bool:
        """Basic language filter. Extend with proper langid if needed."""
        import langid
        if self.lang_mode == "skip": return True
        if not text: return False
        def ascii_fraction(s): 
            return 0.0 if not s else sum(1 for c in s if ord(c) < 128)/len(s)
        if self.lang_mode in ("ascii","fast"): return ascii_fraction(text) >= self.ascii_threshold
        try:
            lang, _ = langid.classify(text)
            if lang == "en": return True
        except Exception: pass
        return ascii_fraction(text) >= self.ascii_threshold
    


## -- Utility Functions ---------------------------------------------------
# -- Getting Size of Vocab Function ------------------------------------------
def get_vocab_size(encoding_name: str) -> int:
    """
    Return the vocabulary size for a given encoding in tiktoken.

    Args:
        encoding_name: Name of the encoding (e.g. "gpt2", "cl100k_base", etc.)

    Returns:
        The total number of tokens in that encoding’s vocabulary.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.n_vocab


# -- Estimate Loss -------------------------------------------------
@torch.no_grad()
def estimate_loss_from_loaders(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    device_type: str,
    autocast_dtype: torch.dtype,
    eval_iters: int,
    max_eval_iters: int = 50,
):
    """
    Evaluate model loss on a small number of batches from train/val loaders.

    - Respects caller-provided eval_iters (clamped by max_eval_iters).
    - Temporarily disables per-module checkpointing flags (if present).
    - Aggregates totals across DDP ranks using SUM then computes a global mean.
    """

    model_was_training = model.training
    model.eval()

    # Temporarily disable per-block checkpointing (walk modules)
    touched = []
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            touched.append((m, getattr(m, "use_checkpoint")))
            setattr(m, "use_checkpoint", False)

    # Clamp eval_iters reasonably
    eval_iters = int(max(1, min(max_eval_iters, eval_iters)))

    totals = {"train": 0.0, "val": 0.0}
    counts = {"train": 0, "val": 0}

    for name, loader in (("train", train_loader), ("val", val_loader)):
        it = iter(loader)
        for _ in range(eval_iters):
            try:
                X, Y = next(it)
            except StopIteration:
                break
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            # Use the same autocast dtype as training
            with autocast(device_type=device_type, dtype=autocast_dtype):
                _, loss = model(X, Y)
            totals[name] += float(loss.item())
            counts[name] += 1

    # Aggregate across ranks (SUM)
    # pack totals and counts into a single tensor: [t_train, c_train, t_val, c_val]
    packed = torch.tensor(
        [totals["train"], counts["train"], totals["val"], counts["val"]],
        dtype=torch.float64,
        device=device,
    )
    packed = all_reduce_sum(packed)

    # Unpack global sums
    t_train_sum = float(packed[0].item())
    c_train_sum = int(packed[1].item())
    t_val_sum = float(packed[2].item())
    c_val_sum = int(packed[3].item())

    # Compute means (guard divide-by-zero)
    train_mean = t_train_sum / max(1, c_train_sum)
    val_mean = t_val_sum / max(1, c_val_sum)

    # restore checkpoint flags
    for m, old in touched:
        setattr(m, "use_checkpoint", old)

    if model_was_training:
        model.train()

    return {"train": train_mean, "val": val_mean}


# ---------------------------------------------------------------------------
# worker seed function for DataLoader
# ---------------------------------------------------------------------------
def seed_worker(worker_id: int):
    # Per-worker deterministic seeding uses process-wide seed + worker_id
    rank = dist.get_rank() if ddp_is_initialized() else 0
    base = int(os.environ.get("GLOBAL_SEED", "1337"))
    seed = base + rank * 10_000 + worker_id
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: Optional[float],
    batches_seen,
    train_losses,
    val_losses,
    file_path: str,
    scaler = None,
    warmup_scheduler = None,
    plateau_scheduler = None,
):
    """
    Save checkpoint (main process only). Handles DDP model wrapping.
    """
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history": {
            "batches_seen": list(batches_seen),
            "train_losses": list(train_losses),
            "val_losses": list(val_losses),
        },
    }
    if scaler is not None:
        try:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            ckpt["scaler_state_dict"] = None
    if warmup_scheduler is not None:
        try:
            ckpt["warmup_state_dict"] = warmup_scheduler.state_dict()
        except Exception:
            ckpt["warmup_state_dict"] = None
    if plateau_scheduler is not None:
        try:
            ckpt["plateau_scheduler_state_dict"] = plateau_scheduler.state_dict()
        except Exception:
            ckpt["plateau_scheduler_state_dict"] = None

    if is_main_process():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(ckpt, file_path)
        logger.info("Checkpoint saved to %s", file_path)



def load_checkpoint(file_path: str, map_location="cpu"):
    """Return loaded checkpoint (caller will broadcast/load into model)."""
    return torch.load(file_path, map_location=map_location)


def ddp_load_and_broadcast_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, map_location='cpu'):
    """Rank 0 loads checkpoint, broadcasts state dicts to all other ranks, and all ranks load them."""
    ckpt = None
    if is_main_process():
        ckpt = torch.load(path, map_location=map_location)
    # broadcast None / ckpt presence
    ckpt_present = torch.tensor([0 if ckpt is None else 1], dtype=torch.int)
    if ddp_is_initialized():
        dist.broadcast(ckpt_present, src=0)
    if int(ckpt_present.item()) == 0:
        return None
    # now broadcast the dict keys & tensors: easiest is to let main broadcast state_dict bytes
    if is_main_process():
        sd = ckpt.get("model_state_dict")
        opt_sd = ckpt.get("optimizer_state_dict")
    else:
        sd = None
        opt_sd = None
    sd = dist.broadcast_object_list([sd], src=0)[0] if ddp_is_initialized() else sd
    opt_sd = dist.broadcast_object_list([opt_sd], src=0)[0] if ddp_is_initialized() else opt_sd
    # load into model/optimizer (map to device)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(sd)
    if opt_sd is not None:
        optimizer.load_state_dict(opt_sd)
    return ckpt



def get_total_params_gpt2(model: torch.nn.Module) -> int:
    """
    Compute the total number of *unique* trainable parameters in a GPT-2 model,
    correctly accounting for weight tying between the input embeddings and the
    output language modeling head.

    GPT-2 ties the weights of its input embedding matrix and the output projection
    (lm_head).  Although these weights appear twice in the model's parameter list,
    they should only be counted once when reporting the total number of parameters.

    Args:
        model (torch.nn.Module):
            A GPT-2 model instance (e.g. from Hugging Face's transformers library),
            which must have an attribute `lm_head` representing the output projection
            layer.

    Returns:
        int: The total number of unique parameters in the GPT-2 model, with the
             duplicated `lm_head` parameters subtracted out.
    """
    model_to_check = model.module if hasattr(model, "module") else model
    # 1) Count every parameter in the model (embeddings, transformer blocks, lm_head, etc.)
    total_params = sum(p.numel() for p in model_to_check.parameters())

    # 2) Count only the parameters in the output head (lm_head).
    #    These share weights with the input embedding matrix under weight-tying.
    if hasattr(model_to_check, "lm_head"):
        lm_head_params = sum(p.numel() for p in model_to_check.lm_head.parameters())
    else:
        lm_head_params = 0

    # 3) Subtract the lm_head params once so they're not double-counted,
    #    yielding the true GPT-2 parameter count.
    return total_params - lm_head_params

def get_total_model_size_mb(model: torch.nn.Module) -> float:
    model_to_check = model.module if hasattr(model, "module") else model
    total_params = sum(p.numel() for p in model_to_check.parameters())
    return total_params * 4 / (1024.0 * 1024.0)



# -- Plot training and validation loss vs batch number ------------------------------------------------------
def plot_train_val_loss(batches_seen, train_losses, val_losses):
    logger.info("Generating plot: training and validation loss vs batch number.")

    fig, ax1 = plt.subplots(figsize=(15,6))

    # Plot training and validation loss against epochs
    ax1.plot(batches_seen, train_losses, label="Train Loss")
    ax1.plot(batches_seen, val_losses, linestyle="-.", label="Val Loss")
    ax1.set_xlabel("Batch Number")
    #ax1.set_ylim(0)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss")
    ax1.legend(loc="upper right")
    # --- Add Subgrid (Minor Grid Lines) ---
    ax1.minorticks_on()  # Enable minor ticks
    # Major grid (main grid lines)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.8)
    # Minor grid (subgrid lines)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.grid(True)

    #fig.tight_layout()  # Adjust layout to make room
    plt.xticks(rotation=45)
    plt.savefig(f"loss-plot-{run_id}.pdf")
    plt.show()



# ---------------------------------------------------------------------------
# Main training that ties everything together
# ---------------------------------------------------------------------------
def main():
    """
    Streaming pre-training loop for a GPT-style autoregressive language model.

    High-level flow
    ---------------
    1. Data are streamed from Parquet shards → tokenized on-the-fly → overlapping
    sliding windows (stride < block_size) to maximize data efficiency.
    2. Mixed-precision (`bfloat16`) and gradient accumulation are used to fit large
    models and long sequences on modest GPUs.
    3. Training is measured in **effective steps** (optimizer updates) rather than
    micro-batches for intuitive scheduling of evals, checkpoints, LR warm-up, etc.
    4. Early stopping, LR-scheduling on plateau, NaN guards, and deterministic
    checkpointing are all built-in.

    Key symbols
    -----------
    effective_step       number of times the optimizer has actually stepped
    batches_processed    raw mini-batches consumed (micro-steps)
    grads_accum_steps    how many micro-batches are averaged into one effective step
    eval_interval        effective steps between validation runs
    save_interval        effective steps between checkpoints
    warmup_steps         effective steps over which LR rises from 0 → base_lr

    Implementation notes
    --------------------
    - optimizer.zero_grad is called once per effective step only.
    - Loss is **mean-reduced** across the accumulation window via
    `loss = loss / grads_accum_steps`.
    - tqdm bar updates on every effective step; logs and filenames use the same unit.
    - ReduceLROnPlateau is stepped after **every** validation (not only when the
    interval doubles) to guarantee LR decay when loss plateaus.
    - Warm-up uses effective_step, so `warmup_steps=2000` means 2000 optimizer updates.
    """

    # -- Logging Configuration ----------------------------------------------

    # Ensure logs directory exists
    LOG_DIR = os.path.join("logs")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Create file handler
    log_file = os.path.join(LOG_DIR, f"train_{run_id}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Optional: use same format as console
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # ---- DDP init ----
    rank, local_rank, world_size, device = ddp_setup()
    logger.info(f"[rank {rank}] Using device: {device}")

    # keep your autocast choice (bf16 on CUDA; fp32 on CPU)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32


    # Seeds
    # Deterministic per-process seeding (single place)
    global_seed = int(os.environ.get("GLOBAL_SEED", "1337"))
    set_global_seed(global_seed, rank=rank, device=device)
    # ensure deterministic DataLoader worker init uses GLOBAL_SEED
    os.environ["GLOBAL_SEED"] = str(global_seed)


    # -- Load Tokenizer -----------------------------------------------------
    encoding_name = "gpt2"
    tokenizer = tiktoken.get_encoding(encoding_name)
    vocab_size = get_vocab_size(encoding_name) 
    while(vocab_size%64 != 0): vocab_size+=1 # override with larger value that is divisable by 64 
    logger.info(f"Vocab size: {vocab_size}")

    # -- Model Configuration ------------------------------------------------
    block_size = 1024
    n_embd = 1024 # 768 1024 1280
    n_head = 16 # 12 16 20
    n_layer = 24 # 12 24 36
    dropout = 0.1
    batch_size = 3

    logger.info(f"Using device: {device}")



    # create an instance of GPTLanguageModel class
    model = GPTLanguageModel(
        vocab_size=vocab_size, 
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    ).to(device)

    # Wrap with DDP only when running distributed
    if ddp_is_initialized() and world_size > 1:
        # Wrap with DDP (Windows/gloo)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            static_graph=True  # safe if your graph never changes; speeds up comms
        )
    else:
        # single-process: leave model as-is (no DDP wrapper)
        if is_main_process():
            logger.info("Running single-process (no DDP wrapper).")

    #logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    logger.info(f"Model parameters: {get_total_params_gpt2(model)/1e6:.2f}M")
    logger.info(f"Total size of the model: {get_total_model_size_mb(model):.2f} MB")


    # -- Load Data ----------------------------------------------------------
    INPUT_DIR = "datasets/100BT"
    PARQUET_GLOB = "*.parquet"
    all_files = sorted(glob(os.path.join(INPUT_DIR, PARQUET_GLOB)))
    assert len(all_files) > 0, "No parquet files found."

    # -- Train/Val Split -----------------------------------------------------
    # allocating 90% for training and 10% for validation
    # simple file-level split (deterministic)
    split_at = max(1, int(0.9 * len(all_files)))
    train_files = all_files[:split_at]
    val_files   = all_files[split_at:] or all_files[-1:]

    # after computing train_files and val_files
    if len(train_files) < world_size:
        raise RuntimeError(f"Not enough training files ({len(train_files)}) for world_size={world_size}. Reduce world_size or provide more files.")
    if len(val_files) < world_size:
        logger.warning("Less validation files than ranks; some ranks will have no validation data.")
    # # or replication fallback (to be considered in the future)
    # if len(train_files) < world_size:
    #     # repeat files deterministically so every rank has something
    #     times = (world_size // len(train_files)) + 1
    #     train_files = (train_files * times)[:max(world_size, len(train_files))]

    # -- Prepare Overlapping Windows -----------------------------------
    # Desired fraction of overlap between successive windows
    overlap_frac = 0.0625   # 6.25% 
    #overlap_frac = 0.5     # 50% 
    # Compute the stride (how far the window moves each time)
    stride = int(block_size * (1 - overlap_frac))  


    # Per-process worker counts:
    cpus = max(1, os.cpu_count() or 1)
    workers_per_rank = min(max(1, cpus // max(1, world_size)), 4)  # tune; start small on Windows
    # On Windows default to 0 workers to avoid spawn/persistent worker edge cases
    if os.name == "nt":
        workers_per_rank = 0  
        logger.info(f"os.name == nt")
    prefetch = 2                                    # tune with GPU util
                  

    train_dataset = ParquetCausalIterable(
        files=train_files,
        pattern="*.parquet",
        tokenizer_model="gpt2",
        block_size=block_size,
        stride=stride,
        add_eos=True,
        shuffle_files=True,
        seed=global_seed,
    )

    val_dataset = ParquetCausalIterable(
        files=val_files,
        pattern="*.parquet",
        tokenizer_model="gpt2",
        block_size=block_size,
        stride=stride,
        add_eos=True,
        shuffle_files=False,   # validation must be deterministic
        seed=global_seed + 9999,
    )


    # ---------------------------------------------------
    # DDP-aware DataLoaders
    # ---------------------------------------------------
    # Note:
    # • No DistributedSampler needed because sharding is inside dataset
    # • Use pinned memory for async GPU transfer
    # • Persistent workers recommended for Parquet scanning
    # • Prefetch_factor > 1 improves throughput
    # ---------------------------------------------------

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers_per_rank,
        pin_memory=pin_memory,
        persistent_workers=(workers_per_rank > 0),
        prefetch_factor=prefetch if workers_per_rank > 0 else None,
        worker_init_fn=seed_worker,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=max(0, workers_per_rank // 2),
        pin_memory=pin_memory,
        persistent_workers=(max(0, workers_per_rank // 2) > 0),
        prefetch_factor=prefetch if workers_per_rank > 0 else None,
        worker_init_fn=seed_worker,
        drop_last=True
    )


    # -- Optimizer & Training Setup -----------------------------------------
    grads_accum_steps = 8
    initial_eval_interval = 32
    max_eval_interval = 1000
    save_interval = max_eval_interval * 10
    eval_interval = initial_eval_interval
    val_loss_threshold = 3.0  # stop if val loss exceeds threshold x train loss
    warmup_steps = 2000
    patience_limit = 10 # stop if no improvement in val loss after patience_limit checks


    learning_rate = 5e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01  # Regularization
    )
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda effective_step: min(1.0, effective_step / warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,   # new_LR = LR*factor
        patience=3,   # number of evals with no improvement to wait before appying new_LR
        min_lr = 1e-6
    )

    if is_main_process():
        try:
            model_to_gen = model.module if hasattr(model, "module") else model
            model_to_gen.eval()
            with torch.no_grad():
                input_tokens = tokenizer.encode("I like apple juice - I drink it")
                input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                output = model_to_gen.advanced_generation(input_tokens=input_tokens, max_new_tokens=40, temperature=1.3, top_k=20)
                decoded = tokenizer.decode([t for t in output[0].tolist() if t < tokenizer.n_vocab])
                logger.info(f"Model output: \n{decoded}")
        except Exception:
            logger.exception("Generation failed")

    model.train()




    # -------------------------
    # Prepare directories
    # -------------------------
    pre_training_dir = os.path.join('outputs', 'output_v17', 'pre_training', f'run_{run_id}')
    os.makedirs(pre_training_dir, exist_ok=True)



    scaler = DummyScaler()

    # -------------------------
    # Tracking metrics 
    # --------------------------
    batches_processed = 0        # raw mini-batches seen (micro-steps)
    effective_step = 0           # full optimizer steps
    last_grad_norm = 0.0
    train_losses, val_losses, batches_seen = [], [], []
    val_loss = None
    early_stop = False
    t0 = time.time()
    best_val_loss = float('inf')
    patience_counter = 0

    # -------------------------
    # Device / autocast / precision setup
    # -------------------------
    # Normalize device_type string for autocast
    # `device` may be a torch.device or string like 'cuda' or 'cpu' -- normalize:
    if isinstance(device, torch.device):
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    else:
        device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    autocast_dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32

    # Matmul precision setting (GPU)
    try:
        torch.set_float32_matmul_precision('high')
        print("torch.set_float32_matmul_precision('high')")
    except Exception:
        # ignore if not available
        pass

    # decide how many *effective* steps you want (or math.inf for endless streaming)
    max_effective_steps = math.inf   # change to taste

    # -------------------------
    # Prepare tqdm bar
    # -------------------------
    # create pbar only on main process
    pbar = tqdm(total=None if math.isinf(max_effective_steps) else int(max_effective_steps),
                desc="Training", unit="step", leave=True) if is_main_process() else None
    # debug sync flag (set env DEBUG_SYNC=1 if you want synchronous ops for debugging)
    DEBUG_SYNC = os.environ.get("DEBUG_SYNC", "0") == "1"

    # initialize a consistent postfix dictionary
    postfix = {
        "loss": "-",
        "grad": "-",
        "train": "-",
        "val": "-",
        "lr": "-",
        "t/step": "-",
        "t/eval": "-"
    }


    # -------------------------
    # Start training loop
    # -------------------------
    # we now zero once *before* the very first micro-batch
    optimizer.zero_grad(set_to_none=True)
    # Streaming training loop (no epochs) 
    train_iter = iter(train_loader)  # fresh iterator


    try:
        while not early_stop and effective_step < max_effective_steps:
            micro_start = time.time()          # <── start timer

            # fetch next micro-batch (streaming)
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                # reset iterator if streaming dataset exhausted
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)
                
            # move to device (non_blocking requires pinned memory)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # forward with autocast — handle dtype/device gracefully
            # Precompute a display_loss before backward (to avoid issues with scaled tensors)
            with autocast(device_type=device_type, dtype=autocast_dtype):
                _, loss = model(xb, yb)
                # scale down the loss for accumulation (important)
                loss = loss / grads_accum_steps
                display_loss = float((loss * grads_accum_steps).detach().cpu())

            if DEBUG_SYNC and device.type == "cuda":
                torch.cuda.synchronize()

            if torch.isnan(loss).any():
                if is_main_process():
                    logger.error("NaN loss at micro-batch %d (batches_processed=%d)", batches_processed, batches_processed)
                    # Save checkpoint for debugging (optional)
                    try:
                        ckpt_path = os.path.join(pre_training_dir, f"checkpoint_nan_at_micro_{batches_processed}.pth")
                        save_checkpoint(
                            model, optimizer, effective_step,
                            float(loss.item()) if hasattr(loss, "item") else None,
                            batches_seen, train_losses, val_losses,
                            ckpt_path, scaler=scaler, warmup_scheduler=warmup, plateau_scheduler=scheduler
                        )
                        logger.info("Saved NaN checkpoint to %s", ckpt_path)
                    except Exception as e:
                        logger.exception("Failed to save NaN checkpoint: %s", e)

                early_stop = True
                break

            # backward (no GradScaler needed)
            loss.backward()
            batches_processed += 1

            # When we reached accumulation boundary -> optimizer step + scheduler + logging + checkpoints
            if batches_processed % grads_accum_steps == 0:

                # Clip grads (returns total_norm)
                last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # optional warmup apply AFTER step
                if effective_step < warmup_steps:
                    try:
                        warmup.step()
                        logger.debug(f"Warmup step {effective_step}/{warmup_steps} - LR: {optimizer.param_groups[0]['lr']:.2e}")
                    except Exception:
                        # fail gracefully if warmup not configured properly
                        logger.debug("Warmup step failed or warmup scheduler missing")
                # zero grads (set_to_none for slight perf improvement)
                optimizer.zero_grad(set_to_none=True)

                effective_step += 1

                # Update progress bar & logging only on main process
                if is_main_process():
                    postfix.update({
                        "loss": f"{display_loss:.4f}",
                        "grad": f"{last_grad_norm:.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "t/step": f"{time.time() - micro_start:.3f}s"
                    })
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(postfix)


                # Periodic checkpoint (main only): save at multiples of save_interval including the first
                if is_main_process() and save_interval and effective_step % save_interval == 0 and effective_step > 1:
                    ckpt_path = os.path.join(pre_training_dir, f"checkpoint_step_{effective_step}.pth")
                    save_checkpoint(model, optimizer, effective_step, float(loss.item()) if hasattr(loss, "item") else None,
                                    batches_seen, train_losses, val_losses, ckpt_path,
                                    scaler=scaler, warmup_scheduler=warmup, plateau_scheduler=scheduler)
                    logger.info(f"Saved periodic checkpoint @ step {effective_step}")


                # Evaluation (all ranks compute partial sums, main aggregates/prints)
                if effective_step % eval_interval == 0 and effective_step > 1:
                    dt = time.time() - t0
                    t0 = time.time()
                    model.eval()
                    with torch.no_grad():
                        eval_iters = max(1, min(8, eval_interval // 16))
                        metrics = estimate_loss_from_loaders(
                            model, train_loader, val_loader, device,
                            device_type, autocast_dtype, eval_iters
                        )
                    model.train()

                    train_loss = metrics.get("train", float("nan"))
                    val_loss = metrics.get("val", float("nan"))

                    # step plateau scheduler only on main process (it uses global val)
                    if is_main_process():

                        try:
                            scheduler.step(val_loss)
                        except Exception:
                            logger.debug("Scheduler.step(val_loss) failed")

                        # update postfix and logging
                        postfix.update({
                            "train": f"{train_loss:.4f}",
                            "val": f"{val_loss:.4f}",
                            "t/eval": f"{dt:.1f}s"
                        })
                        if pbar is not None:
                            pbar.set_postfix(postfix)

                        logger.info(f"[Step {effective_step}] train={train_loss:.4f} | val={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | time={dt:.1f}s")

                        # store metrics history on main only
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        batches_seen.append(batches_processed)

                        # Divergence guard / patience checks and saves (main only)
                        if train_loss and not math.isnan(train_loss) and val_loss > val_loss_threshold * train_loss:
                            logger.warning("Validation diverged")
                            ckpt_path = os.path.join(pre_training_dir, f"checkpoint_divergence_step_{effective_step}.pth")
                            save_checkpoint(model, optimizer, effective_step, float(loss.item()) if hasattr(loss, "item") else None,
                                            batches_seen, train_losses, val_losses, ckpt_path, scaler=scaler,
                                            warmup_scheduler=warmup, plateau_scheduler=scheduler)
                            early_stop = True
                            break

                        # Patience counter guard    
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            logger.info(f"No val improvement {patience_counter}/{patience_limit}")
                            if patience_counter >= patience_limit:
                                logger.warning("Early stopping triggered")
                                ckpt_path = os.path.join(pre_training_dir, f"checkpoint_patience_step_{effective_step}.pth")
                                save_checkpoint(model, optimizer, effective_step, float(loss.item()) if hasattr(loss, "item") else None,
                                                batches_seen, train_losses, val_losses, ckpt_path, scaler=scaler,
                                                warmup_scheduler=warmup, plateau_scheduler=scheduler)
                                early_stop = True
                                break

                    # sample generation on main only
                    if is_main_process():                    
                        with torch.no_grad():
                            input_tokens = tokenizer.encode("I like apple juice, I drink it")
                            input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                            # call module if wrapped
                            gen_model = model.module if hasattr(model, "module") else model
                            output = gen_model.generate(input_tokens=input_tokens, max_new_tokens=30)
                            decoded = tokenizer.decode([t for t in output[0].tolist() if t < tokenizer.n_vocab])
                            logger.info(f"Sample: {decoded}")

        # final log on main only
        if is_main_process():                            
            # -- Final log ------------------------------------------------------
            if early_stop:
                logger.info("Training stopped by early stopping criteria.")
            else:
                logger.info("Training stopped manually or completed streaming run.")

    except KeyboardInterrupt:
        if is_main_process():
            logger.warning("KeyboardInterrupt — saving checkpoint before exit")
            safe_loss = None
            try:
                safe_loss = float(locals().get("loss").item()) if isinstance(locals().get("loss"), torch.Tensor) else None
            except Exception:
                safe_loss = None
            save_checkpoint(model, optimizer, effective_step, safe_loss, batches_seen, train_losses, val_losses,
                            os.path.join(pre_training_dir, "checkpoint_interrupt.pth"),
                            scaler=scaler, warmup_scheduler=warmup, plateau_scheduler=scheduler)
        raise
    except Exception:
        if is_main_process():
            logger.exception("Unexpected exception. Saving checkpoint.")
            safe_loss = None
            try:
                safe_loss = float(locals().get("loss").item()) if isinstance(locals().get("loss"), torch.Tensor) else None
            except Exception:
                safe_loss = None
            save_checkpoint(model, optimizer, effective_step, safe_loss, batches_seen, train_losses, val_losses,
                            os.path.join(pre_training_dir, "checkpoint_exception.pth"),
                            scaler=scaler, warmup_scheduler=warmup, plateau_scheduler=scheduler)
        raise
    finally:
        if pbar is not None:
            pbar.close()
        ddp_cleanup()

        # final generation & plot only on main
        if is_main_process():
            try:
                # optional plotting helper if available
                plot_train_val_loss(batches_seen, train_losses, val_losses)

                model_to_gen = model.module if hasattr(model, "module") else model
                model_to_gen.eval()
                with torch.no_grad():
                    input_tokens = tokenizer.encode("I like apple juice - I drink it")
                    input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                    output = model_to_gen.advanced_generation(input_tokens=input_tokens, max_new_tokens=40, temperature=1.3, top_k=20)
                    decoded = tokenizer.decode([t for t in output[0].tolist() if t < tokenizer.n_vocab])
                    logger.info(f"Model output: \n{decoded}")
            except Exception:
                logger.exception("Final generation failed")


if __name__ == "__main__":
    main()