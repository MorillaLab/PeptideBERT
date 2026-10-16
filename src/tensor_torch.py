"""
PyTorch port of tensor.py.

Drop-in replacement: `from tensor_torch import Tensor, embedding_lookup`
instead of `from tensor import Tensor, embedding_lookup`. The model code
does not change.

Key differences from the NumPy autograd:
  * .data is a torch.Tensor (leaf, requires_grad=True) instead of np.ndarray.
  * .grad is read through the leaf, not stored on the wrapper.
  * .backward() delegates to torch.autograd (no manual topological sort,
    no closure leaks by construction — torch keeps the graph alive only
    while it's needed).
  * .aux is still a plain dict, populated on the wrapper, not on the graph.
"""
import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-8


class Tensor:
    """Thin wrapper over a torch.Tensor that mimics the tinygrad-style API
    of the original tensor.py. Holds a reference to the underlying leaf
    (or graph node) so .grad and .data behave as expected."""

    __slots__ = ("_t", "aux", "_is_leaf")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        if isinstance(data, Tensor):
            self._t = data._t
            self._is_leaf = data._is_leaf
        else:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float64)
            if data.dtype != torch.float64 and data.dtype != torch.float32:
                data = data.to(torch.float64)
            # keep float64 by default to match the original autograd's
            # numerical behaviour; switch to float32 for speed.
            self._t = data.clone().detach().requires_grad_(requires_grad) \
                if requires_grad else data.clone().detach()
            self._is_leaf = requires_grad
        self.aux = None

    # ---------- duck-typed accessors ----------
    @property
    def data(self):
        return self._t

    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self._t.dtype)
        self._t.data = value

    @property
    def grad(self):
        return self._t.grad

    @grad.setter
    def grad(self, value):
        if value is None:
            self._t.grad = None
        else:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=self._t.dtype)
            self._t.grad = value

    @property
    def shape(self):
        return tuple(self._t.shape)

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def requires_grad(self):
        return self._t.requires_grad

    # ---------- arithmetic ----------
    def __add__(self, other):
        return Tensor._wrap(self._t + Tensor._unwrap(other))

    __radd__ = __add__

    def __neg__(self):
        return Tensor._wrap(-self._t)

    def __sub__(self, other):
        return Tensor._wrap(self._t - Tensor._unwrap(other))

    def __rsub__(self, other):
        return Tensor._wrap(Tensor._unwrap(other) - self._t)

    def __mul__(self, other):
        return Tensor._wrap(self._t * Tensor._unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Tensor._wrap(self._t / Tensor._unwrap(other))

    def __rtruediv__(self, other):
        return Tensor._wrap(Tensor._unwrap(other) / self._t)

    def _pow(self, p):
        return Tensor._wrap(self._t ** p)

    def __matmul__(self, other):
        return Tensor._wrap(self._t @ Tensor._unwrap(other))

    def __rmatmul__(self, other):
        return Tensor._wrap(Tensor._unwrap(other) @ self._t)

    # ---------- shape ops ----------
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes) if axes else None
        if axes is None:
            return Tensor._wrap(self._t.T)
        return Tensor._wrap(self._t.permute(*axes))

    @property
    def T(self):
        return Tensor._wrap(self._t.T)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor._wrap(self._t.reshape(*shape))

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return Tensor._wrap(self._t.sum())
        return Tensor._wrap(self._t.sum(dim=axis, keepdim=keepdims))

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return Tensor._wrap(self._t.mean())
        return Tensor._wrap(self._t.mean(dim=axis, keepdim=keepdims))

    # ---------- nonlinearities ----------
    def exp(self):
        return Tensor._wrap(self._t.exp())

    def log(self):
        return Tensor._wrap(torch.log(self._t + _EPS))

    def gelu(self):
        # match the tanh approximation in the NumPy version exactly
        x = self._t
        c = (2.0 / np.pi) ** 0.5
        inner = c * (x + 0.044715 * x ** 3)
        y = 0.5 * x * (1.0 + torch.tanh(inner))
        return Tensor._wrap(y)

    def relu(self):
        return Tensor._wrap(F.relu(self._t))

    # ---------- fused ops ----------
    def softmax(self, axis=-1):
        return Tensor._wrap(F.softmax(self._t, dim=axis))

    def layer_norm(self, gamma, beta, axis=-1, eps=1e-5):
        # keep gamma/beta as Tensors in the graph so their .grad is filled
        g = gamma._t if isinstance(gamma, Tensor) else gamma
        b = beta._t if isinstance(beta, Tensor) else beta
        y = F.layer_norm(self._t, (self._t.shape[axis],), g, b, eps)
        return Tensor._wrap(y)

    def cross_entropy(self, targets, mask=None):
        # targets: int array-like; mask: 0/1 float array-like or None
        t = torch.as_tensor(targets, dtype=torch.long, device=self._t.device)
        logits = self._t
        n = logits.shape[0]
        logp = F.log_softmax(logits, dim=-1)
        row = torch.arange(n, device=logits.device)
        per_row = -logp[row, t]
        if mask is None:
            loss = per_row.mean()
            mask_t = torch.ones(n, dtype=logits.dtype, device=logits.device)
        else:
            mask_t = torch.as_tensor(mask, dtype=logits.dtype, device=logits.device)
            loss = (per_row * mask_t).sum() / mask_t.sum().clamp(min=1.0)
        out = Tensor._wrap(loss)
        with torch.no_grad():
            probs = logp.exp()
            preds = probs.argmax(dim=-1)
            correct = (preds == t).to(logits.dtype) * mask_t
        out.aux = {
            "probs": probs.detach().cpu().numpy(),
            "correct": correct.detach().cpu().numpy(),
            "mask": mask_t.detach().cpu().numpy(),
            "preds": preds.detach().cpu().numpy(),
        }
        return out

    def __getitem__(self, idx):
        return Tensor._wrap(self._t[idx])

    def __setitem__(self, idx, value):
        self._t[idx] = Tensor._unwrap(value)

    # ---------- graph ----------
    def backward(self, grad=None):
        if not self._t.requires_grad:
            return
        if grad is None:
            self._t.backward()
        else:
            g = grad._t if isinstance(grad, Tensor) else torch.as_tensor(grad)
            self._t.backward(g)

    def zero_grad(self):
        self._t.grad = None

    def detach(self):
        out = Tensor.__new__(Tensor)
        out._t = self._t.detach()
        out._is_leaf = False
        out.aux = None
        return out

    def numpy(self):
        return self._t.detach().cpu().numpy()

    # ---------- internals ----------
    @staticmethod
    def _unwrap(x):
        return x._t if isinstance(x, Tensor) else x

    @staticmethod
    def _wrap(t):
        out = Tensor.__new__(Tensor)
        out._t = t
        out._is_leaf = t.requires_grad and t.grad_fn is None
        out.aux = None
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._t.grad_fn.__class__.__name__ if self._t.grad_fn else 'leaf'})"


def embedding_lookup(table: Tensor, indices: np.ndarray):
    """table: (V, D) Tensor. indices: int array of any shape.
    Returns Tensor of shape indices.shape + (D,)."""
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=table._t.device)
    return Tensor._wrap(table._t[idx])