"""
PeptideBERT: a small single-sequence, alignment-free masked-language
transformer for plant signalling-peptide sequences -- same recipe as
NucleicBERT (Upadhyay et al. 2026, Nat. Mach. Intell.), scaled down by
~2500x in parameter count and re-tokenized over amino acids instead of
nucleotides. See ../REPORT.md for the full architecture rationale.
"""
import numpy as np
from tensor import Tensor, embedding_lookup
from peptide_data import VOCAB_SIZE, PAD_ID

D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
D_FF = 256
MAX_LEN = 180  # headroom above the indel-augmented corpus's observed max (~155 aa + 2 special tokens)
assert D_MODEL % N_HEADS == 0
D_HEAD = D_MODEL // N_HEADS


def _init(rng, *shape, scale=None):
    fan_in = shape[0] if len(shape) > 1 else 1
    scale = scale or 1.0 / np.sqrt(fan_in)
    return Tensor(rng.randn(*shape) * scale)


class PeptideBERT:
    """All parameters live in self.p (name -> Tensor) so that zero_grad /
    optimizer steps / freezing (linear-probe) can iterate them generically.
    forward() returns MLM logits; encode() exposes hidden states and
    attention weights for downstream heads and explainability."""

    def __init__(self, n_classes=5, seed=0, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_ff=D_FF, max_len=MAX_LEN):
        rng = np.random.RandomState(seed)
        self.d_model, self.n_layers, self.n_heads, self.d_ff = d_model, n_layers, n_heads, d_ff
        self.d_head = d_model // n_heads
        self.max_len = max_len
        p = {}
        p["tok_emb"] = _init(rng, VOCAB_SIZE, d_model, scale=0.02)
        p["pos_emb"] = _init(rng, max_len, d_model, scale=0.02)
        for l in range(n_layers):
            pre = f"L{l}."
            p[pre + "ln1.g"] = Tensor(np.ones((1, 1, d_model)))
            p[pre + "ln1.b"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "Wq"] = _init(rng, d_model, d_model)
            p[pre + "bq"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "Wk"] = _init(rng, d_model, d_model)
            p[pre + "bk"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "Wv"] = _init(rng, d_model, d_model)
            p[pre + "bv"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "Wo"] = _init(rng, d_model, d_model)
            p[pre + "bo"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "ln2.g"] = Tensor(np.ones((1, 1, d_model)))
            p[pre + "ln2.b"] = Tensor(np.zeros((1, 1, d_model)))
            p[pre + "W1"] = _init(rng, d_model, d_ff)
            p[pre + "b1"] = Tensor(np.zeros((1, 1, d_ff)))
            p[pre + "W2"] = _init(rng, d_ff, d_model)
            p[pre + "b2"] = Tensor(np.zeros((1, 1, d_model)))
        p["ln_f.g"] = Tensor(np.ones((1, 1, d_model)))
        p["ln_f.b"] = Tensor(np.zeros((1, 1, d_model)))
        # MLM head: transform + decoder, as in BERT
        p["mlm.W1"] = _init(rng, d_model, d_model)
        p["mlm.b1"] = Tensor(np.zeros((1, d_model)))
        p["mlm.ln.g"] = Tensor(np.ones((1, d_model)))
        p["mlm.ln.b"] = Tensor(np.zeros((1, d_model)))
        p["mlm.W2"] = _init(rng, d_model, VOCAB_SIZE)
        p["mlm.b2"] = Tensor(np.zeros((1, VOCAB_SIZE)))
        # classification head (family prediction from pooled [CLS])
        p["cls.W"] = _init(rng, d_model, n_classes)
        p["cls.b"] = Tensor(np.zeros((1, n_classes)))
        self.p = p

    def all_params(self):
        return list(self.p.values())

    def backbone_params(self):
        return [t for k, t in self.p.items() if not k.startswith("cls.")]

    def zero_grad(self):
        for t in self.p.values():
            t.zero_grad()

    # ---------------- forward pass ----------------
    def _attn_block(self, x, mask_add, l, want_attn=False):
        p, pre = self.p, f"L{l}."
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def split_heads(t):
            return t.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # (B,H,T,Dh)

        h = x.layer_norm(p[pre + "ln1.g"], p[pre + "ln1.b"])
        Q = split_heads(h @ p[pre + "Wq"] + p[pre + "bq"])
        K = split_heads(h @ p[pre + "Wk"] + p[pre + "bk"])
        V = split_heads(h @ p[pre + "Wv"] + p[pre + "bv"])
        scores = (Q @ K.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(Dh))
        if mask_add is not None:
            scores = scores + mask_add  # (B,1,1,T) broadcasts
        attn = scores.softmax(axis=-1)  # (B,H,T,T)
        context = attn @ V  # (B,H,T,Dh)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = context @ p[pre + "Wo"] + p[pre + "bo"]
        result = x + out  # residual
        return (result, attn) if want_attn else (result, None)

    def _ff_block(self, x, l):
        p, pre = self.p, f"L{l}."
        h = x.layer_norm(p[pre + "ln2.g"], p[pre + "ln2.b"])
        h = (h @ p[pre + "W1"] + p[pre + "b1"]).gelu()
        h = h @ p[pre + "W2"] + p[pre + "b2"]
        return x + h

    def encode(self, input_ids, attn_mask, want_attn=False):
        """input_ids: (B,T) int array. attn_mask: (B,T) 1/0 float array.
        Returns hidden states (B,T,D) Tensor and, if want_attn, a list of
        per-layer attention Tensors (B,H,T,T)."""
        B, T = input_ids.shape
        tok = embedding_lookup(self.p["tok_emb"], input_ids)  # (B,T,D)
        pos = self.p["pos_emb"][np.arange(T)]  # (T,D) via Tensor.__getitem__
        x = tok + pos.reshape(1, T, self.d_model)
        mask_add = None
        if attn_mask is not None:
            add = (1.0 - attn_mask) * -1e9  # (B,T)
            mask_add = Tensor(add.reshape(B, 1, 1, T), requires_grad=False)
        attns = []
        for l in range(self.n_layers):
            x, a = self._attn_block(x, mask_add, l, want_attn=want_attn)
            if want_attn:
                attns.append(a)
            x = self._ff_block(x, l)
        x = x.layer_norm(self.p["ln_f.g"], self.p["ln_f.b"])
        return (x, attns) if want_attn else (x, None)

    def mlm_logits(self, hidden, positions=None):
        """hidden: (B,T,D). If positions given (list of (b,t) tuples or a
        boolean/int index), gather those rows first (saves compute since
        we usually only care about masked positions)."""
        B, T, D = hidden.shape
        flat = hidden.reshape(B * T, D)
        if positions is not None:
            flat = flat[positions]
        h = (flat @ self.p["mlm.W1"] + self.p["mlm.b1"]).gelu()
        h = h.layer_norm(self.p["mlm.ln.g"], self.p["mlm.ln.b"])
        logits = h @ self.p["mlm.W2"] + self.p["mlm.b2"]
        return logits

    def classify_logits(self, hidden):
        """Pool via the [CLS] token (position 0) and apply the linear head."""
        cls = hidden[:, 0, :]  # (B,D)
        return cls @ self.p["cls.W"] + self.p["cls.b"]


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad
            i = id(p)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def count_params(model):
    return sum(p.data.size for p in model.all_params())


if __name__ == "__main__":
    from peptide_data import encode
    m = PeptideBERT()
    print(f"PeptideBERT parameter count: {count_params(m):,}")
    ids, attn = encode("MDKSFTLFLTLTILVVFIISSPPVQAGFAN", MAX_LEN)
    ids_b = np.stack([ids, ids])
    attn_b = np.stack([attn, attn])
    hidden, _ = m.encode(ids_b, attn_b)
    print("hidden shape:", hidden.shape)
    logits = m.mlm_logits(hidden)
    print("mlm logits shape:", logits.shape)
    cls_logits = m.classify_logits(hidden)
    print("classify logits shape:", cls_logits.shape)
