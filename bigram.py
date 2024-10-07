import torch
from torch import nn
import torch.nn.functional as F


with open("input.txt", "r") as f:
  text = f.read()
print(f"length of text : {len(text)}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab_size={vocab_size}")

# encode decode
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {v:k for k,v in stoi.items()}
encode = lambda s : [stoi[i] for i in s]
decode = lambda l : "".join(itos[i] for i in l)

# !wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(train_data.shape , val_data.shape)

block_size = 8
batch_size = 32
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  xb = torch.stack([data[i:i+block_size] for i in ix])
  yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return xb, yb

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)
        if targets is not None:
            loss = F.cross_entropy( logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]), targets.view(-1))
        else:
            loss = None
        return logits, loss

    def train(self, optimizer, epochs=100):
        for i in range(epochs):
            xb, yb = get_batch("train")
            logits, loss = self.forward(xb, yb) 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if(i%5000 == 0):
                print(f"epochs : {i}, loss : {loss.item()}")

        print("training completed")

    def generate(self, idx, max_new_tokens= 1):
        for i in range(max_new_tokens):
            logits, loss = self(torch.tensor([idx], dtype=torch.long))
            logc = logits[:,-1,:]
            probs = F.softmax(logc, dim=-1)
            pred = torch.multinomial(probs, num_samples=1, replacement=True)
            # print(pred)
            idx = idx[1:] + [pred.item()]
            # print(idx)
            print(decode(idx), end="")

class Head(nn.Module):
  def __init__(self, block_size, emb_size, head_size):
    super().__init__()
    self.query = nn.Linear(emb_size, head_size)
    self.key = nn.Linear(emb_size, head_size)
    self.value = nn.Linear(emb_size, head_size)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape # C-head_size
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    wei =  q @ k.transpose(-2,-1) * C**-0.5  # (B,T,C ) @ (B,C,T) =  (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    out = wei @ v  # (B,T,T) @ (B,T,C) = B,T,C
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, block_size, emb_size, head_size, num_heads):
    super().__init__()
    self.heads = [Head(block_size, emb_size, head_size) for i in range(num_heads)]

  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)



m = BigramModel(vocab_size)
optim = torch.optim.Adam(m.parameters(), lr=1e-3)

m.train(optimizer=optim, epochs=3000)

m.generate(encode("Person  "),1000)
