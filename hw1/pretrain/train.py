from modeling_gpt2 import GPT, GPTConfig
import torch
import tiktoken

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
B, T = 4, 32
buffer = torch.tensor(tokens[:B*T + 1]).to(device)
x = buffer[:-1].view(B, T)
y = buffer[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

for i in range(100):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item()}")
