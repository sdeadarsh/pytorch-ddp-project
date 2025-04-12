import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset
import time
import psutil
from datetime import timedelta
import argparse
import traceback
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime

# --- DDP Configuration ---
RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 3))
MASTER_ADDR = "localhost"
MASTER_PORT = "8080"
BACKEND = 'gloo' # Default, can be overridden by args


# --- Language Model Hyperparameters ---
batch_size = 64          # Global batch size (will be divided among ranks)
block_size = 256         # Context length
max_iters = 5000         # Total training iterations
eval_interval = 500      # Evaluation frequency
learning_rate = 3e-4
# Device will be set in setup_distributed
eval_iters = 200         # Batches for evaluation
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337 + RANK) # Add rank for different initializations if desired

# --- Data Loading (Executed on all ranks) ---
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Rank {RANK}: Successfully read input.txt", flush=True)
except FileNotFoundError:
    print(f"Rank {RANK}: ERROR - input.txt not found in the current directory. Please create it.", flush=True)
    exit(1)

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data_full = data[:n]
val_data_full = data[n:]

# --- DDP Helper Functions ---
def log_memory(stage=""):
    # (Keep this function as is)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"Rank {RANK} Memory ({stage}): RSS={rss_mb:.2f} MB", flush=True)

def setup_distributed(backend):
    # (Keep this function as is, it returns is_distributed, device_string)
    if RANK == -1 or WORLD_SIZE <= 0:
        print("DDP environment variables not set. Running in non-distributed mode.", flush=True)
        return False, 'cpu'

    if backend == 'nccl' and torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE:
        device_id = RANK % torch.cuda.device_count()
        device = f'cuda:{device_id}'
        torch.cuda.set_device(device)
        print(f"Rank {RANK}: Using GPU {device}", flush=True)
    else:
        if backend == 'nccl':
            print(f"Rank {RANK}: WARNING - NCCL backend requested but CUDA not available or not enough GPUs. Falling back to Gloo backend on CPU.", flush=True)
            backend = 'gloo'
        device = 'cpu'
        print(f"Rank {RANK}: Using CPU", flush=True)
        if backend == 'gloo' and 'GLOO_SOCKET_IFNAME' not in os.environ:
            default_iface = "eth0"
            print(f"Rank {RANK}: Using default GLOO_SOCKET_IFNAME='{default_iface}'. Set env var if needed.", flush=True)
            os.environ['GLOO_SOCKET_IFNAME'] = default_iface

    init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
    print(f"Rank {RANK}: Attempting DDP init... Backend: {backend}, Init: {init_method}, World: {WORLD_SIZE}", flush=True)
    log_memory("Before Init")
    try:
        dist.init_process_group(
            backend=backend, init_method=init_method, rank=RANK,
            world_size=WORLD_SIZE, timeout=timedelta(seconds=120)
        )
        print(f"Rank {RANK}: DDP Init SUCCESS.", flush=True)
        log_memory("After Init")
        return True, device # Return True and the device string
    except Exception as e:
        print(f"Rank {RANK}: DDP Init FAILED: {e}\n{traceback.format_exc()}", flush=True)
        exit(1)

def cleanup_distributed():
    # (Keep this function as is)
    if dist.is_initialized():
        print(f"Rank {RANK}: Cleaning up DDP...", flush=True)
        try:
            dist.barrier()
        except Exception as e:
            print(f"Rank {RANK}: Warning - Barrier during cleanup failed: {e}", flush=True)
        dist.destroy_process_group()
        print(f"Rank {RANK}: DDP Cleanup SUCCESS.", flush=True)
        log_memory("After Cleanup")
    else:
        print(f"Rank {RANK}: DDP not initialized, skipping cleanup.", flush=True)


# --- Language Model Components ---
# (Keep Head, MultiHeadAttention, FeedForward, Block, BigramLanguageModel classes as they are)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.size(-1)**-0.5 # Use k.size(-1) for head_dim
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd) # Input size adjusted
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.position_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights) # Apply custom weight initialization

    def _init_weights(self, module): # Standard practice for Transformers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_embd = self.token_embd(idx) # (B, T, n_embd)
        # Need device for arange
        pos_embd = self.position_embd(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_embd + pos_embd # (B, T, n_embd)
        x = self.blocks(x)      # (B, T, n_embd)
        x = self.ln_f(x)        # (B, T, n_embd)
        logits = self.lm_head(x)# (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad() # Generation doesn't need gradients
    def generate(self, idx, max_new_tokens):
        self.eval() # Set to eval mode for generation
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        self.train() # Set back to train mode
        return idx

# --- Dataset for Language Model ---
class CharDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size
        # Ensure data is long enough
        if len(self.data) <= self.block_size:
            raise ValueError(f"Data length ({len(self.data)}) must be greater than block size ({self.block_size})")

    def __len__(self):
        # Number of possible starting points for a block
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab chunk of text
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- Evaluation Function (Adapted for DDP - Gloo Compatible) ---
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, is_distributed):
    out = {}
    model.eval()
    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        if loader is None: continue

        if is_distributed and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(0)

        local_total_loss = 0.0
        local_total_samples = 0
        max_eval_batches = eval_iters # Limit number of batches for speed

        for k, (X, Y) in enumerate(loader):
            if k >= max_eval_batches: break
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            if loss is not None:
                # Accumulate the total loss for the batches processed by this rank
                local_total_loss += loss.item() * X.size(0) # loss * batch_size
                local_total_samples += X.size(0)

        # --- Aggregation across ranks ---
        if is_distributed:
            # Create tensors for the local sum of losses and count
            # Ensure they are float for reduction operations that might involve averaging later
            # (though here we only use SUM)
            local_stats = torch.tensor([local_total_loss, local_total_samples], dtype=torch.float64, device=device)

            # Perform all_reduce SUM to get global totals
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

            # Extract global totals from the reduced tensor
            global_total_loss = local_stats[0].item()
            global_total_samples = local_stats[1].item()

            # Calculate the final average loss
            if global_total_samples > 0:
                final_avg_loss = global_total_loss / global_total_samples
            else:
                final_avg_loss = float('nan') # No samples processed across all ranks

        else: # Non-distributed case
            if local_total_samples > 0:
                final_avg_loss = local_total_loss / local_total_samples
            else:
                final_avg_loss = float('nan') # No samples processed locally

        out[split_name] = final_avg_loss
        # Optional: Log if NaN occurred
        if final_avg_loss != final_avg_loss: # Check for NaN
            print(f"Rank {RANK}: WARNING - Calculated NaN for loss in split '{split_name}'. "
                f"Local Samples: {local_total_samples}, Global Samples: {global_total_samples if is_distributed else local_total_samples}", flush=True)


    model.train()
    return out


# --- Main Training Function ---
def run_training(args):
    """Main DDP training function for BigramLanguageModel."""

    is_distributed, device_str = setup_distributed(backend=args.backend)
    device = torch.device(device_str) # Convert string to torch.device

    # Calculate batch size per rank
    if batch_size % WORLD_SIZE != 0:
        print(f"Rank {RANK}: WARNING - Global batch size {batch_size} not divisible by world size {WORLD_SIZE}. Effective batch size may vary.")
    # Effective batch size per GPU
    per_rank_batch_size = batch_size // WORLD_SIZE

    # Create Datasets and Samplers
    train_dataset = CharDataset(train_data_full, block_size)
    val_dataset = CharDataset(val_data_full, block_size)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
        # Don't shuffle validation data, use DistributedSampler to give each rank a slice
        val_sampler = DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Shuffle only if not using DistributedSampler
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_rank_batch_size,
        sampler=val_sampler,
        shuffle=False, # Never shuffle validation
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    log_memory("After DataLoaders")

    # Model Initialization
    model = BigramLanguageModel().to(device)
    if is_distributed:
        # Specify device_ids only for CUDA backend
        device_ids = [device.index] if device.type == 'cuda' else None
        ddp_model = DDP(model, device_ids=device_ids, find_unused_parameters=False) # Set find_unused if needed
        print(f"Rank {RANK}: Wrapped model with DDP.", flush=True)
    else:
        ddp_model = model # Use the plain model if not distributed

    # Print parameter count from Rank 0
    if RANK == 0:
        param_count = sum(p.numel() for p in ddp_model.parameters()) / 1e6
        print(f"{param_count:.2f} M parameters", flush=True)

    # Optimizer (use DDP model parameters)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate) # Use AdamW

    # Training Loop
    print(f"Rank {RANK}: Starting training for {max_iters} iterations...", flush=True)
    log_memory("Start Training Loop")
    total_start_time = time.time()
    current_iter = 0
    epochs = 0 # Track epochs for sampler
    training_successful = False

    ddp_model.train() # Set model to training mode

    while current_iter < max_iters:
        if is_distributed:
            train_sampler.set_epoch(epochs) # Set epoch for sampler shuffling

        for batch_idx, (xb, yb) in enumerate(train_loader):
            if current_iter >= max_iters: break # Exit if max_iters reached within epoch

            # Evaluate loss periodically (only Rank 0 prints)
            if current_iter % eval_interval == 0:
                log_memory(f"Eval at iter {current_iter}")
                # Use estimate_loss with DDP handling
                losses = estimate_loss(ddp_model, train_loader, val_loader, device, is_distributed)
                if RANK == 0:
                    print(f"step {current_iter}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}", flush=True)
                ddp_model.train() # Ensure model is back in train mode
                log_memory(f"End Eval at iter {current_iter}")

            # Perform training step
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = ddp_model(xb, yb) # Forward pass using DDP model

            # Check if loss is valid (might be None if targets were None, though unlikely with CharDataset)
            if loss is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward() # DDP handles gradient synchronization
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                print(f"Rank {RANK}: Warning - Loss was None at iteration {current_iter}. Skipping backward/step.", flush=True)


            current_iter += 1 # Increment iteration counter

        epochs += 1 # Increment epoch after iterating through loader

    # --- End of Training Loop ---
    training_successful = True # Mark as successful if loop completes
    total_time = time.time() - total_start_time
    print(f"Rank {RANK}: Training finished. Total time: {total_time:.2f}s for {max_iters} iterations.", flush=True)
    log_memory("End Training Loop")

    # --- Generate Text (Rank 0 Only) ---
    if RANK == 0:
        print("\n--- Generating Text (Rank 0) ---", flush=True)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        # Use the underlying model for generation if DDP wrapped
        model_to_generate = ddp_model.module if is_distributed else ddp_model
        generated_indices = model_to_generate.generate(context, max_new_tokens=500)[0].tolist()
        generated_text = decode(generated_indices)
        print(generated_text)
        print("--- End Generation ---", flush=True)

    # --- Save Model (Rank 0 Only) ---
    if RANK == 0 and training_successful and args.save_path:
        print(f"Rank 0: Saving model state_dict to {args.save_path}...", flush=True)
        try:
            model_to_save = ddp_model.module if is_distributed else ddp_model
            state_dict_cpu = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
            torch.save(state_dict_cpu, args.save_path)
            print(f"Rank 0: Model saved successfully to {args.save_path}", flush=True)
        except Exception as e:
            print(f"Rank 0: ERROR saving model: {e}\n{traceback.format_exc()}", flush=True)
            
    elif RANK == 0 and not training_successful:
        print(f"Rank 0: Skipping model save as training did not complete successfully.", flush=True)


# --- Main Execution Guard ---
if __name__ == "__main__":
    
    start_time = datetime.now()
    print(f"++++++++++++  Starting time of the script : {start_time}  ++++++++++++")
    
    parser = argparse.ArgumentParser(description='PyTorch DDP Training for BigramLanguageModel')
    # Removed args related to tabular data (num_samples)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='DDP backend')
    parser.add_argument('--save_path', type=str, default="bigram_language_model.pth", help='Path to save the final model state_dict (Rank 0 only)')
    # Add other relevant args if needed (e.g., learning_rate, max_iters)
    args = parser.parse_args()

    # Update global BACKEND
    BACKEND = args.backend

    print("--- Starting DDP Script for BigramLanguageModel ---", flush=True)
    print(f"Args: {args}", flush=True) # Print relevant args

    exit_code = 0
    try:
        run_training(args)
        if dist.is_initialized():
            print(f"Rank {RANK}: Reached final barrier before cleanup...", flush=True)
            dist.barrier()
            print(f"Rank {RANK}: Passed final barrier.", flush=True)
        print(f"--- Process {RANK} finished successfully ---", flush=True)

    except KeyboardInterrupt:
        print(f"--- Process {RANK} received KeyboardInterrupt ---", flush=True)
        exit_code = 130
    except SystemExit as e:
        print(f"--- Process {RANK} exited with code {e.code} ---", flush=True)
        exit_code = e.code
    except Exception as e:
        print(f"--- Process {RANK} encountered an unhandled exception in main block ---", flush=True)
        print(f"Error: {e}\n{traceback.format_exc()}", flush=True)
        exit_code = 1
    finally:
        cleanup_distributed()
        print(f"--- Process {RANK} exiting with code {exit_code} ---", flush=True)
        exit(exit_code)
        
    end_time = datetime.now()
    print(f"++++++++++++  Ending time of the script : {end_time}  ++++++++++++")
    