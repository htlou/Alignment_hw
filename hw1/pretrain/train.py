from modeling_gpt2 import GPT, GPTConfig
import torch
import tiktoken
import time
import math

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        f.close()
        self.tokens = torch.tensor(enc.encode(text))
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current : self.current + B*T+1]
        x = (buffer[:-1]).view(B, T)
        y = (buffer[1:]).view(B, T)
        self.current += B * T
        if self.current + (B*T+1) > len(self.tokens):
            self.current = 0
        
        return x, y

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using model at {device}")

enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
f.close()
text = text[:1000]
tokens = enc.encode(text)

batch_size = 524288
B, T = 16, 1024
assert batch_size % (B * T) == 0
grad_accum = batch_size // (B * T)

print(f"Batch size: {batch_size}, Gradient accumulation steps: {grad_accum}")

buffer = torch.tensor(tokens[:B*T + 1]).to(device)
x = buffer[:-1].view(B, T)
y = buffer[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

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

data_loader = DataLoader(B, T)
torch.set_float32_matmul_precision('high')

for i in range(steps):
    start_time = time.time()
    x, y = data_loader.next_batch()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    for j in range(grad_accum):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        accumulated_loss += loss.item()
    loss = accumulated_loss / grad_accum
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param in optimizer.param_groups:
        param['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Step {i}, Loss: {loss.item()}, Norm: {norm.item()}, Time: {end_time - start_time}")
