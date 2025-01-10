from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
import math

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    bias: bool = False

class CausalSelfAttention(nn.Moudule):
    def __init__(self, config):
        super().__init__()
        
        # Key Query Value prjections in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))
    def forward(self, x):
        # Batch size, Seq length, dimension
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k , v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)
        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Skeleton
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Hidden layer
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Final classifier
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            return logits

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), 
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # init hf module
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy paramether
        sd_keys_hf = sd_hf.keys()
        
        # buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Treatment for Conv1D
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla for other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

# logits, loss = model(x, y)

# print(logits.shape)
# print(loss)

# model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig())

# if model is None:
#     print('model is None')
# else:
#     print('model is not None')

# max_length = 30
# num_return_seq = 5

# model.eval()
# model.to('cuda')
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I am a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
# x = tokens.to('cuda')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)
#         # only care last col logits
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim = -1)
#         topk_probs , topk_indices = torch.topk(probs, 50, dim = -1)
#         # (B, 1) select tokens and coresponding idx
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim= -1)

# for i in range(num_return_seq):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(decoded)