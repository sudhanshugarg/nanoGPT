"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import tiktoken

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from model import GPTConfig, GPT
from model_params import compute_model_params

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    print("RUNNING ON SINGLE DEVICE WITH 1 PROCESS")
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# Pre-load all examples into memory
print("Pre-loading dataset...")
_data_cache = {}
for split in ['train', 'val']:
    examples_path = os.path.join(data_dir, f'{split}_examples.pkl')
    if os.path.exists(examples_path):
        with open(examples_path, 'rb') as f:
            _data_cache[split] = pickle.load(f)
        print(f"  Loaded {len(_data_cache[split])} {split} examples")

class CopyTaskDataset(Dataset):
    """Efficient dataset for copy task with pre-filtered valid examples"""
    def __init__(self, examples, stoi, block_size, load_meta=False):
        self.examples = examples
        self.stoi = stoi
        self.block_size = block_size
        self.load_meta = load_meta
        self.padding_token_id = stoi.get('<pad>', 4) if load_meta else 0
        self.out_token_id = stoi.get('<out>', None) if load_meta else None

        # Pre-filter valid examples (with <out> token not at end)
        self.valid_indices = []
        for i, ex in enumerate(examples):
            if self.out_token_id is not None:
                try:
                    out_idx = ex.index(self.out_token_id)
                    if out_idx < len(ex) - 1:  # <out> not at end
                        self.valid_indices.append(i)
                except ValueError:
                    pass
            else:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_example = self.examples[self.valid_indices[idx]]

        example = original_example if len(original_example) <= self.block_size else original_example[:self.block_size]
        out_idx = example.index(self.out_token_id) if self.out_token_id is not None else 0

        remaining_output = example[out_idx + 1:]
        num_output_tokens = torch.randint(1, len(remaining_output) + 1, (1,)).item() if remaining_output else 1
        x_seq = example[:out_idx + 1] + remaining_output[:num_output_tokens]

        y_content = original_example[1:len(x_seq) + 1]
        y_seq = y_content + [self.padding_token_id] * (self.block_size - len(y_content))
        x_seq = x_seq + [self.padding_token_id] * (self.block_size - len(x_seq))

        x_tensor = torch.tensor(x_seq[:self.block_size], dtype=torch.long)
        y_tensor = torch.tensor(y_seq[:self.block_size], dtype=torch.long)

        return x_tensor, y_tensor, out_idx

def collate_fn(batch):
    """Collate function to create masks for batch"""
    x_list, y_list, out_indices = zip(*batch)
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Create mask (vectorized)
    mask = torch.zeros_like(y, dtype=torch.bool)
    out_token_id = stoi.get('<out>', None) if load_meta else None
    padding_token_id = stoi.get('<pad>', 4) if load_meta else 0

    if load_meta and out_token_id is not None:
        # Vectorized mask creation
        for batch_idx in range(y.shape[0]):
            out_positions = (x[batch_idx] == out_token_id).nonzero(as_tuple=True)[0]
            if len(out_positions) > 0:
                last_out_pos = out_positions[-1].item()
                mask[batch_idx, last_out_pos:] = (y[batch_idx, last_out_pos:] != padding_token_id)
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    return x, y, mask

def get_batch(split):
    """Get next batch from DataLoader iterator (initialized after stoi is defined)"""
    # Initialize dataloaders on first call (after stoi is available)
    if not hasattr(get_batch, '_iterators'):
        get_batch._dataloaders = {}
        for split_name in ['train', 'val']:
            if split_name in _data_cache:
                dataset = CopyTaskDataset(_data_cache[split_name], stoi, block_size, load_meta)
                get_batch._dataloaders[split_name] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split_name == 'train'),
                    num_workers=8,
                    pin_memory=True,
                    prefetch_factor=16,  # Queue up to 16*8=128 batches ahead
                    collate_fn=collate_fn,
                )
        get_batch._iterators = {}

    # Create new iterator if needed
    if split not in get_batch._iterators or get_batch._iterators[split] is None:
        get_batch._iterators[split] = iter(get_batch._dataloaders[split])

    try:
        x, y, mask = next(get_batch._iterators[split])
    except StopIteration:
        # Restart iterator when epoch ends
        get_batch._iterators[split] = iter(get_batch._dataloaders[split])
        x, y, mask = next(get_batch._iterators[split])

    # Move to device (DataLoader already pinned, so just transfer)
    if device_type == 'cuda':
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        mask = mask.to(device)

    return x, y, mask

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 1
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
load_meta = False
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    load_meta = True
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# for sampling during training
if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    # Identify special tokens from stoi
    special_tokens = [token for token in stoi.keys() if token.startswith('<') and token.endswith('>')]

    def _encode_func(s):
        """Encode a string to token ids, treating special tokens like <bos> as single tokens."""
        tokens = []
        i = 0
        while i < len(s):
            found_special = False
            for token in special_tokens:
                if s[i:i+len(token)] == token:
                    tokens.append(stoi[token])
                    i += len(token)
                    found_special = True
                    break
            if not found_special:
                if s[i] in stoi:
                    tokens.append(stoi[s[i]])
                i += 1
        return tokens

    encode = _encode_func
    decode = lambda l: ''.join([itos.get(i, '<pad>') if isinstance(i, int) else itos.get(int(i), '<pad>') for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# Print and save model parameter breakdown
if master_process:
    compute_model_params(
        model.config.vocab_size,
        model.config.block_size,
        model.config.n_embd,
        model.config.n_layer,
        batch_size=batch_size,
        bias=model.config.bias,
        print_breakdown=True
    )

    # Also save parameters to hyperparams.txt
    vocab_size = model.config.vocab_size
    sequence_length = model.config.block_size
    n_embd = model.config.n_embd
    n_layer = model.config.n_layer

    token_emb_params = vocab_size * n_embd
    pos_emb_params = sequence_length * n_embd
    attention_params = sum([n_embd * (3 * n_embd) + (3 * n_embd if model.config.bias else 0) +
                           n_embd * n_embd + (n_embd if model.config.bias else 0) for _ in range(n_layer)])
    mlp_params = sum([n_embd * (4 * n_embd) + (4 * n_embd if model.config.bias else 0) +
                     (4 * n_embd) * n_embd + (n_embd if model.config.bias else 0) for _ in range(n_layer)])
    layernorm_params = sum([n_embd * 2 + n_embd * 2 for _ in range(n_layer)]) + n_embd * 2
    total_params = token_emb_params + pos_emb_params + attention_params + mlp_params + layernorm_params
    bytes_float32 = total_params * 4

    total_params_val, _, memory_breakdown = compute_model_params(
        vocab_size, sequence_length, n_embd, n_layer,
        batch_size=batch_size, bias=model.config.bias, print_breakdown=False
    )

    hyperparams_str = f"""Model Architecture
{'='*60}
Vocabulary Size:     {vocab_size:,}
Embedding Dimension: {n_embd:,}
Number of Layers:    {n_layer:,}
Block Size:          {sequence_length:,}
Bias:                {model.config.bias}

Model Parameters
{'='*60}
Token Embedding:     {token_emb_params:15,} params ({token_emb_params * 4 / (1024**2):8.2f} MB)
Position Embedding:  {pos_emb_params:15,} params ({pos_emb_params * 4 / (1024**2):8.2f} MB)
Layer Norms:         {layernorm_params:15,} params ({layernorm_params * 4 / (1024**2):8.2f} MB)
Attention (x{n_layer}):      {attention_params:15,} params ({attention_params * 4 / (1024**2):8.2f} MB)
MLP (x{n_layer}):           {mlp_params:15,} params ({mlp_params * 4 / (1024**2):8.2f} MB)
{'-'*60}
TOTAL:               {total_params:15,} params ({bytes_float32 / (1024**3):8.2f} GB)

Memory Breakdown (batch_size={batch_size})
{'='*60}
Model Weights:       {memory_breakdown['weights'] / (1024**3):8.2f} GB
Gradients:           {memory_breakdown['gradients'] / (1024**3):8.2f} GB
Optimizer State:     {memory_breakdown['optimizer_state'] / (1024**3):8.2f} GB
Activations:         {memory_breakdown['activations'] / (1024**3):8.2f} GB
KV Cache:            {memory_breakdown['kv_cache'] / (1024**3):8.2f} GB
{'-'*60}
TOTAL MEMORY:        {memory_breakdown['total'] / (1024**3):8.2f} GB

Training Configuration
{'='*60}
Learning Rate:       {learning_rate}
Weight Decay:        {weight_decay}
Max Iterations:      {max_iters:,}
Batch Size:          {batch_size}
Block Size:          {block_size}
Gradient Accumulation: {gradient_accumulation_steps}
Data Type:           {dtype}
Compile:             {compile}
"""

    hyperparams_path = os.path.join(out_dir, 'hyperparams.txt')
    with open(hyperparams_path, 'w') as f:
        f.write(hyperparams_str)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def sample(model: nn.Module):
    start_ids = encode("<bos><in>hello<out>")
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

def print_token_ids(token_ids: np.ndarray, suffix: str = ""):
    msg1 = f"got {len(token_ids)}"
    token_msg = ''.join(decode(token_ids))
    ans = f"{msg1}, token_ids = {token_ids}, tokens = #{token_msg}#, suffix: #{suffix}#"
    print(ans)

def print_logits(logits: torch.Tensor, X: torch.Tensor, Y: torch.Tensor):
        batch_sz, context_sz, vocab_sz = logits.shape
        print(f"logits.shape = {logits.shape}")
        for i in range(batch_sz):
            row_logits = logits[i] # [context_sz, vocab_sz]
            row_token_ids = X[i]
            for j in range(context_sz):
                # we are predicitng a next token for each substring
                # i need the substring, and i need the next letter being predicted, and i need the actual letter.
                input_token_ids = row_token_ids[:j+1].numpy()

                row_logits_sequence_so_far = row_logits[j]
                row_logits_sequence_so_far_softmax = F.softmax(row_logits_sequence_so_far, dim=-1).detach().numpy()
                sorted_logit_token_ids = np.argsort(row_logits_sequence_so_far_softmax)[::-1]
                top_k_token_ids = sorted_logit_token_ids[:5]
                #print out each of the letters
                next_letters = decode(top_k_token_ids)
                print_token_ids(input_token_ids, ''.join(next_letters))


        # token_probs = F.softmax(row_logits, dim=-1).detach().numpy()
        # top_token_probs = np.argsort(token_probs)[::-1]
        # top_5_tokens_ids = top_token_probs[:5]
        # top_5_tokens = decode(top_5_tokens_ids)
        # print(f"top_token_probs = {top_token_probs}")
        # print(f"softmax(logits[0][0]) = {F.softmax(logits[0][0], dim=-1)}, top_5_tokens_ids = {top_5_tokens_ids}, top_5_tokens = {top_5_tokens}")

# logging
if __name__ == '__main__':
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode='offline')

    # training loop
    X, Y, mask = get_batch('train') # fetch the very first batch
    #X.shape = [2, 10], i.e. [batch_size, sequence_length]. X[0] is [1, 10]
    X2 = X.cpu()
    Y2 = Y.cpu()
    print_token_ids(X2[0].numpy())
    print_token_ids(Y2[0].numpy())
    print_token_ids(X2[1].numpy())
    print_token_ids(Y2[1].numpy())

    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            # lets print a small sample also
            sample(raw_model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    # Create filename with metrics
                    ckpt_filename = f"ckpt_iter{iter_num}_train{losses['train']:.4f}_val{losses['val']:.4f}.pt"
                    ckpt_path = os.path.join(out_dir, ckpt_filename)
                    print(f"saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y, mask)

                # print_logits(logits, X, Y)
                # print(f"reached line 418"); exit(0)

                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # load next batch (now fast since examples are in memory)
            X, Y, mask = get_batch('train')



            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

if ddp:
    destroy_process_group()
