"""
Minimal reverse-mode autograd engine over NumPy arrays.

Why this exists: the sandbox this was built in has no PyTorch/JAX (a torch
install blew the disk quota pulling in ~2GB of bundled CUDA wheels for a
CPU-only job). For a genuinely *small* peptide-BERT, a hand-rolled autograd
is actually the more honest choice anyway: zero heavy dependencies, every
gradient is inspectable, and it ports trivially to torch later (swap Tensor
for torch.Tensor and the model code barely changes).

Design follows the standard "micrograd/tinygrad" pattern: each Tensor
remembers the op that created it and a closure that pushes gradients to its
parents. `backward()` does a reverse topological traversal, calling each
node's `_backward(grad_output)` with the incoming gradient passed in as an
argument.

IMPORTANT implementation note (learned the hard way -- see history): each
node's `_backward` closure must NOT capture the node itself (`out`) to read
`out.grad`; that creates a self-referential cycle at every single graph
node, which CPython's refcounter can't clean up immediately and which
floods the cyclic GC faster than it can keep up (observed: >500MB RSS
growth per training batch, then an OOM kill). Instead, `backward()` reads
`.grad` off each node itself and *passes it in* to that node's
`_backward(grad_out)`, so closures only ever reference their *parents*
(who never reference their children) -- a genuine DAG, not a graph with
per-node cycles.

Every primitive here is checked against finite differences in
test_tensor.py -- run that file before trusting anything built on top of it.
"""
import numpy as np

_EPS = 1e-8


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _noop(grad_out):
    pass


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "requires_grad", "aux", "__weakref__")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = _noop
        self._prev = _children
        self._op = _op
        self.requires_grad = requires_grad
        self.aux = None

    @property
    def shape(self):
        return self.data.shape

    # ---------- elementwise / broadcasting ops ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a_shape, b_shape = self.data.shape, other.data.shape
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward(grad_out):
            self.grad += _unbroadcast(grad_out, a_shape)
            other.grad += _unbroadcast(grad_out, b_shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward(grad_out):
            self.grad += -grad_out
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a_data, b_data = self.data, other.data
        a_shape, b_shape = a_data.shape, b_data.shape
        out = Tensor(a_data * b_data, (self, other), "*")

        def _backward(grad_out):
            self.grad += _unbroadcast(grad_out * b_data, a_shape)
            other.grad += _unbroadcast(grad_out * a_data, b_shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other._pow(-1.0)

    def _pow(self, p):
        base = self.data
        out = Tensor(base ** p, (self,), f"pow{p}")

        def _backward(grad_out):
            self.grad += (p * base ** (p - 1)) * grad_out
        out._backward = _backward
        return out

    __radd__ = __add__

    def __rmul__(self, other):
        return self * other

    # ---------- matmul (supports batched leading dims) ----------
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self.data, other.data
        a_shape, b_shape = a.shape, b.shape
        out = Tensor(a @ b, (self, other), "matmul")

        def _backward(grad_out):
            if a.ndim == 1 and b.ndim == 1:
                self.grad += grad_out * b
                other.grad += grad_out * a
                return
            bT = np.swapaxes(b, -1, -2)
            aT = np.swapaxes(a, -1, -2)
            ga = grad_out @ bT
            gb = aT @ grad_out
            self.grad += _unbroadcast(ga, a_shape)
            other.grad += _unbroadcast(gb, b_shape)
        out._backward = _backward
        return out

    # ---------- shape ops ----------
    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        out = Tensor(np.transpose(self.data, axes), (self,), "transpose")
        inv = tuple(np.argsort(axes))

        def _backward(grad_out):
            self.grad += np.transpose(grad_out, inv)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward(grad_out):
            self.grad += grad_out.reshape(orig_shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        orig_shape = self.data.shape
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward(grad_out):
            g = grad_out
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis if isinstance(axis, int) else tuple(axis))
            self.grad += np.ones(orig_shape) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.data.shape[a] for a in (axis if isinstance(axis, (tuple, list)) else [axis])])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---------- nonlinearities ----------
    def exp(self):
        y = np.exp(self.data)
        out = Tensor(y, (self,), "exp")

        def _backward(grad_out):
            self.grad += y * grad_out
        out._backward = _backward
        return out

    def log(self):
        x = self.data
        out = Tensor(np.log(x + _EPS), (self,), "log")

        def _backward(grad_out):
            self.grad += grad_out / (x + _EPS)
        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation, as used in BERT/GPT
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        y = 0.5 * x * (1.0 + t)
        out = Tensor(y, (self,), "gelu")

        def _backward(grad_out):
            sech2 = 1.0 - t ** 2
            dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
            dy_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner
            self.grad += dy_dx * grad_out
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        out = Tensor(np.maximum(x, 0), (self,), "relu")

        def _backward(grad_out):
            self.grad += (x > 0) * grad_out
        out._backward = _backward
        return out

    # ---------- fused ops with analytically-known backward ----------
    def softmax(self, axis=-1):
        x = self.data
        z = x - x.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _backward(grad_out):
            dot = (grad_out * s).sum(axis=axis, keepdims=True)
            self.grad += s * (grad_out - dot)
        out._backward = _backward
        return out

    def layer_norm(self, gamma, beta, axis=-1, eps=1e-5):
        x = self.data
        mu = x.mean(axis=axis, keepdims=True)
        xc = x - mu
        var = (xc ** 2).mean(axis=axis, keepdims=True)
        std = np.sqrt(var + eps)
        xhat = xc / std
        gamma_shape, beta_shape = gamma.data.shape, beta.data.shape
        y = xhat * gamma.data + beta.data
        out = Tensor(y, (self, gamma, beta), "layer_norm")
        n = x.shape[axis]

        def _backward(grad_out):
            gamma.grad += _unbroadcast(grad_out * xhat, gamma_shape)
            beta.grad += _unbroadcast(grad_out, beta_shape)
            dxhat = grad_out * gamma.data
            dvar_term = (dxhat * xc).sum(axis=axis, keepdims=True) * (-0.5) * std ** -3
            dmu_term = -(dxhat / std).sum(axis=axis, keepdims=True) - 2.0 * xc.mean(axis=axis, keepdims=True) * dvar_term
            dx = dxhat / std + dvar_term * 2 * xc / n + dmu_term / n
            self.grad += dx
        out._backward = _backward
        return out

    def cross_entropy(self, targets, mask=None):
        """self: logits of shape (N, V). targets: int array (N,).
        mask: optional 0/1 float array (N,) selecting which rows count
        (e.g. only masked-language-model positions). Returns a scalar
        Tensor = mean NLL over the selected rows; per-row correctness for
        accuracy bookkeeping is stashed in `.aux` (plain numpy, not part
        of the graph)."""
        x = self.data
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=-1, keepdims=True)
        n = x.shape[0]
        row = np.arange(n)
        logp = np.log(probs[row, targets] + _EPS)
        if mask is None:
            mask = np.ones(n)
        denom = max(mask.sum(), 1.0)
        loss_val = -(logp * mask).sum() / denom
        out = Tensor(loss_val, (self,), "cross_entropy")
        preds = probs.argmax(axis=-1)
        correct = (preds == targets).astype(np.float64) * mask

        def _backward(grad_out):
            d = probs.copy()
            d[row, targets] -= 1.0
            d = d * mask[:, None] / denom
            self.grad += d * grad_out
        out._backward = _backward
        out.aux = {"probs": probs, "correct": correct, "mask": mask, "preds": preds}
        return out

    def __getitem__(self, idx):
        orig_shape = self.data.shape
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward(grad_out):
            g = np.zeros(orig_shape)
            np.add.at(g, idx, grad_out)
            self.grad += g
        out._backward = _backward
        return out

    # ---------- graph traversal ----------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward(v.grad)
            v._backward = _noop   # drop closure references to parents ASAP
            v._prev = ()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"


def embedding_lookup(table: Tensor, indices: np.ndarray):
    """table: (V, D) Tensor of embedding rows. indices: int array of any
    shape. Returns a Tensor of shape indices.shape + (D,)."""
    out_data = table.data[indices]
    table_shape = table.data.shape
    out = Tensor(out_data, (table,), "embedding_lookup")

    def _backward(grad_out):
        g = np.zeros(table_shape)
        np.add.at(g, indices, grad_out)
        table.grad += g
    out._backward = _backward
    return out
