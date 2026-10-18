"""Project implementations of CoOp and original training-free Tip-Adapter.

See docs/UPSTREAM.md for algorithm references and implementation differences.
"""
import torch
from torch import nn
from torch.nn import functional as F


def accuracy(logits, labels):
    return 100.0 * (logits.argmax(-1) == labels).float().mean().item()


@torch.no_grad()
def text_weights(model, classes, template):
    import clip
    device = next(model.parameters()).device
    tokens = clip.tokenize([template.format(c) for c in classes]).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def tip_logits(features, text, keys, values, alpha, beta, scale=100.0):
    return scale * features @ text.T + alpha * torch.exp(beta * (features @ keys.T - 1)) @ values


@torch.no_grad()
def tune_tip(features, labels, text, keys, values, alphas, betas):
    """Only validation features/labels may be passed here."""
    if not alphas or not betas or min(alphas) < 0 or min(betas) <= 0:
        raise ValueError("Tip grid needs nonnegative alpha and positive beta")
    best = (-1.0, None, None)
    for alpha in alphas:
        for beta in betas:
            score = accuracy(tip_logits(features, text, keys, values, alpha, beta), labels)
            if score > best[0]:
                best = (score, alpha, beta)
    return {"val_accuracy": best[0], "alpha": best[1], "beta": best[2]}


class CoOp(nn.Module):
    """Shared learned context; class tokens at end; frozen CLIP text backbone."""
    def __init__(self, model, classes, n_ctx=16, ctx_init=""):
        import clip
        from clip.simple_tokenizer import SimpleTokenizer
        super().__init__()
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        device = next(model.parameters()).device
        tokenizer = SimpleTokenizer()
        if ctx_init:
            init_ids = tokenizer.encode(ctx_init)
            if len(init_ids) != n_ctx:
                raise ValueError(f"Initialization has {len(init_ids)} BPE tokens; n_ctx={n_ctx}")
            prefix = ctx_init
        else:
            prefix = " ".join(["X"] * n_ctx)
        if n_ctx < 1 or any(n_ctx + len(tokenizer.encode(c + ".")) + 2 > 77 for c in classes):
            raise ValueError("Context and class tokens exceed CLIP's token budget")
        tokens = clip.tokenize([f"{prefix} {c}." for c in classes]).to(device)
        with torch.no_grad():
            embedded = model.token_embedding(tokens)
        self.context = nn.Parameter(embedded[0, 1:1+n_ctx].clone())
        if not ctx_init:
            nn.init.normal_(self.context, std=.02)
        self.register_buffer("prefix", embedded[:, :1].clone())
        self.register_buffer("suffix", embedded[:, 1+n_ctx:].clone())
        self.register_buffer("eot", tokens.argmax(-1))

    def text_features(self):
        ctx = self.context.unsqueeze(0).expand(len(self.prefix), -1, -1)
        x = torch.cat([self.prefix, ctx, self.suffix], dim=1)
        x = x + self.model.positional_embedding
        x = self.model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.model.ln_final(x)
        x = x[torch.arange(len(x), device=x.device), self.eot] @ self.model.text_projection
        return F.normalize(x, dim=-1)

    def forward(self, image_features):
        return self.model.logit_scale.exp() * image_features @ self.text_features().T


def train_coop(prompt, train, validation, epochs, batch_size, lr, seed):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    features, labels = train
    optimizer = torch.optim.SGD([prompt.context], lr=lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator().manual_seed(seed)
    best_score, history, selected_epoch = -1.0, [], 0
    best_context = prompt.context.detach().clone()
    # Cached, deterministic image features are fixed. Text encoder autograd stays enabled.
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(labels), generator=generator)
        loss_sum = 0.0
        for indices in order.split(batch_size):
            indices = indices.to(features.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(prompt(features[indices]), labels[indices])
            loss.backward()
            if prompt.context.grad is None or not torch.isfinite(prompt.context.grad).all():
                raise RuntimeError("Invalid CoOp context gradient")
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        scheduler.step()
        with torch.no_grad():
            score = accuracy(prompt(validation[0]), validation[1])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(labels), "val_accuracy": score})
        print(f"CoOp epoch {epoch}/{epochs}: val={score:.2f}%", flush=True)
        if score > best_score:
            best_score, selected_epoch = score, epoch
            best_context = prompt.context.detach().clone()
    with torch.no_grad():
        prompt.context.copy_(best_context)
    return {"val_accuracy": best_score, "selected_epoch": selected_epoch, "history": history}
