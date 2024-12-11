import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from torchviz import make_dot


batch_size = 64
block_size = 256
max_iters = 30000
eval_interval = 30
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


def scr(name, tensor, tt='w'):
    timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()) * 100)
    fig_width, fig_height = (20, 20)
    if tt == 'w':
        tensor = tensor.cpu().weight.data
        height, width = tensor.shape
        fig_width = 20
        fig_height = fig_width * (height / width)
        plt.figure(figsize=(fig_width, fig_height))
        im = plt.imshow(tensor, aspect='equal')
        plt.colorbar(im, shrink=0.5)
    elif tt == 't':
        plt.imshow(tensor.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
    else:
        return

    plt.savefig(f'img/{name}_{timestamp}.png')
    tensor = tensor.to(device)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # scr('Head_qXk', wei[0], 't')
        # scr('Head_q', self.query, 'w')
        # scr('Head_k', self.key, 'w')
        # scr('Head_v', self.value, 'w')
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # self.token_embedding_table таблица встраивания (embedding table) для преобразования токенов (слов или символов) в векторные
        # представления фиксированного размера. Здесь vocab_size - размер словаря (количество уникальных токенов),
        # а n_embd - размерность вектора встраивания для каждого токена. При нициализации - случайные веса
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # создаем блоки трансформара те что Nx из статьи AIAYN
        # nn.Sequential - создает слои данные по которым проходят последовательно, например
        # например жтой штукой можно создать сетку из например 3х полносвязных слоев
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm


        self.lm_head = nn.Linear(n_embd, vocab_size)
        # распределение весов до нормализации всех инициализированых весов всех слоев
        # методом self.apply(self._init_weights)
        # weights = self.lm_head.weight.data.numpy()
        # plt.hist(weights.flatten(), bins=100)
        # plt.title("Распределение весов линейного слоя до нормализации")
        # plt.xlabel("Значения весов")
        # plt.ylabel("Частота")
        # plt.savefig(f'before_apply.png')


        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
        # распределение весов до нормализации всех инициализированых весов всех слоев
        # методом self.apply(self._init_weights)
        # weights = self.lm_head.weight.data.numpy()
        # plt.hist(weights.flatten(), bins=100)
        # plt.title("Распределение весов линейного слоя после нормализации")
        # plt.xlabel("Значения весов")
        # plt.ylabel("Частота")
        # plt.savefig(f'after_apply.png')
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        # B - строк, колво по вертикали, батчей = 64
        # T - кол-во токенов в контексте, столбцов, предложений = 256
        # C - вектор встраивания = 384
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        pass
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



if __name__ == '__main__':
    model = GPTLanguageModel()
    # слои модели
    # for name, module in model.named_modules():
    #     print(name, module)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in tqdm(range(max_iters)):
        # every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval == 0 or iter == max_iters - 1):
            # losses = estimate_loss()
            # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(model.state_dict(), 'model')
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(f'\nSaved after {iter} iters')
            print('Predict example:' + decode(m.generate(context, max_new_tokens=500)[0].tolist()))
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        # визуализация модели
        # y = model(xb, yb)
        # dot = make_dot(y, params=dict(list(model.named_parameters())))
        # dot.render('model_graph', format='png')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model')
    model = GPTLanguageModel()  # Создание экземпляра модели
    model.load_state_dict(torch.load('model'))  # Загрузка весов модели
    model.to(device)  # Перемещение модели на устройство (CPU или GPU

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
