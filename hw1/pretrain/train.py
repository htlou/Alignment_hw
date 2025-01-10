from modeling_gpt2 import GPT, GPTConfig
import torch
import tiktoken
import time
import math
import os
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DataLoader:
    def __init__(self, B, T, rank, num_workers):
        self.B = B
        self.T = T

        self.rank = rank
        self.num_workers = num_workers

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        f.close()
        self.tokens = torch.tensor(enc.encode(text))
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current = self.B * self.T * self.rank

    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current : self.current + B*T+1]
        x = (buffer[:-1]).view(B, T)
        y = (buffer[1:]).view(B, T)
        self.current += B * T * self.num_workers
        if self.current + (B*T*self.num_workers+1) > len(self.tokens):
            self.current = self.B * self.T * self.rank
        
        return x, y

# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
# print(f"Using model at {device}")

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend='nccl')
    device = f'cuda:{torch.distributed.get_rank()}'
    ddp_rank = torch.distributed.get_rank()
    ddp_world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    if torch.cuda.is_available():
        device = f'cuda:{torch.distributed.get_rank()}'
    else:
        device = 'cpu'

print(f"Using device: {device}")

enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
f.close()
text = text[:1000]
tokens = enc.encode(text)

batch_size = 524288
B, T = 16, 1024
assert batch_size % (B * T * ddp_world_size) == 0
grad_accum = batch_size // (B * T * ddp_world_size)

if master_process:
    print("At Master Process")
    print(f"Batch size: {batch_size}, Gradient accumulation steps: {grad_accum}")

print(f"Batch size: {batch_size}, Gradient accumulation steps: {grad_accum}")

buffer = torch.tensor(tokens[:B*T + 1]).to(device)
x = buffer[:-1].view(B, T)
y = buffer[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[device])

max_lr = 3e-4
min_lr = 1e-4
warmup_steps = 10
steps = 1000

def get_lr(it):
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    elif it > steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

optimizer = torch.optim.AdamW(model.parameters(), lr = max_lr, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.1, fused=True)

data_loader = DataLoader(B, T, ddp_rank, ddp_world_size)
torch.set_float32_matmul_precision('high')

eval_interval = 100

for i in range(steps):
    start_time = time.time()
    x, y = data_loader.next_batch()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    for j in range(grad_accum):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        accumulated_loss += loss.item()
        if ddp:
            model.require_backward_grad_sync = (j == grad_accum - 1)
    loss = accumulated_loss / grad_accum

    if ddp:
        dist.all_reduce(loss, op = dist.ReduceOp.AVG)

    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param in optimizer.param_groups:
        param['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    if master_process:
        print(f"Step {i}, Loss: {loss.item()}, Norm: {norm.item()}, Time: {end_time - start_time}")

    if i % eval_interval == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
        model.train()

if ddp:
    destroy_process_group()