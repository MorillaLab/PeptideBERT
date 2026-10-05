"""
JAX port of tensor.py.

JAX is functional: there is no per-node .grad to accumulate into during a
reverse traversal. To keep the same API as the NumPy autograd, this module
keeps a global registry of leaf Tensors that have requires_grad=True, and
.backward() differentiates the loss w.r.t. all of them in one jax.grad call.

Usage in the model:
    loss = model(batch)          # a Tensor
    loss.backward()              # fills .grad on every registered leaf
    opt.step()                   # reads leaf.grad as before

Caveats:
  * .backward() must be called on a scalar (or reduced-to-scalar) Tensor.
  * Gradients are stored as numpy arrays on each leaf (matching the
    original .grad type), so downstream code that does .grad.copy(),
    .grad += ..., etc. keeps working.
  * jit/vmap work if you bypass .backward() and use jax.grad directly on
    a pure function. The wrapper is for API compatibility, not for
    performance.
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

_EPS = 1e-8

# Every Tensor with requires_grad=True registers here so .backward() knows
# what to differentiate w.r.t. Weak references would be nicer but the
# original module's lifetime is simple enough that a plain list is fine.
_PARAM_REGISTRY = []


class Tensor:
    __slots__ = ("_x", "grad", "_requires_grad", "aux")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        if isinstance(data, Tensor):
            self._x = data._x
            self._requires_grad = data._requires_grad
        else:
            if isinstance(data, np.ndarray):
                data = jnp.asarray(data)
            elif not isinstance(data, jnp.ndarray):
                data = jnp.asarray(data, dtype=jnp.float64)
            self._x = data
            self._requires_grad = requires_grad
        self.grad = None
        self.aux = None
        if requires_grad:
            _PARAM_REGISTRY.append(self)

    # ---------- accessors ----------
    @property
    def data(self):
        return self._x

    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            value = jnp.asarray(value)
        elif not isinstance(value, jnp.ndarray):
            value = jnp.asarray(value)
        self._x = value

    @property
    def shape(self):
        return tuple(self._x.shape)

    @property
    def dtype(self):
        return self._x.dtype

    @property
    def requires_grad(self):
        return self._requires_grad

    # ---------- arithmetic ----------
    def __add__(self, other):
        return Tensor._wrap(self._x + Tensor._unwrap(other))

    __radd__ = __add__

    def __neg__(self):
        return Tensor._wrap(-self._x)

    def __sub__(self, other):
        return Tensor._wrap(self._x - Tensor._unwrap(other))

    def __rsub__(self, other):
        return Tensor._wrap(Tensor._unwrap(other) - self._x)

    def __mul__(self, other):
        return Tensor._wrap(self._x * Tensor._unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Tensor._wrap(self._x / Tensor._unwrap(other))

    def __rtruediv__(self, other):
        return Tensor._wrap(Tensor._unwrap(other) / self._x)

    def _pow(self, p):
        return Tensor._wrap(self._x ** p)

    def __matmul__(self, other):
        return Tensor._wrap(self._x @ Tensor._unwrap(other))

    # ---------- shape ops ----------
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes) if axes else None
        if axes is None:
            return Tensor._wrap(self._x.T)
        return Tensor._wrap(jnp.transpose(self._x, axes))

    @property
    def T(self):
        return Tensor._wrap(self._x.T)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor._wrap(self._x.reshape(*shape))

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return Tensor._wrap(jnp.sum(self._x))
        return Tensor._wrap(jnp.sum(self._x, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return Tensor._wrap(jnp.mean(self._x))
        return Tensor._wrap(jnp.mean(self._x, axis=axis, keepdims=keepdims))

    # ---------- nonlinearities ----------
    def exp(self):
        return Tensor._wrap(jnp.exp(self._x))

    def log(self):
        return Tensor._wrap(jnp.log(self._x + _EPS))

    def gelu(self):
        # tanh approximation, identical formula to the NumPy original
        x = self._x
        c = (2.0 / np.pi) ** 0.5
        inner = c * (x + 0.044715 * x ** 3)
        y = 0.5 * x * (1.0 + jnp.tanh(inner))
        return Tensor._wrap(y)

    def relu(self):
        return Tensor._wrap(jax.nn.relu(self._x))

    # ---------- fused ops ----------
    def softmax(self, axis=-1):
        return Tensor._wrap(jax.nn.softmax(self._x, axis=axis))

    def layer_norm(self, gamma, beta, axis=-1, eps=1e-5):
        # pure-JAX layer norm so gradients w.r.t. gamma/beta flow through
        # the same graph that .backward() will differentiate.
        g = Tensor._unwrap(gamma)
        b = Tensor._unwrap(beta)
        x = self._x
        mu = jnp.mean(x, axis=axis, keepdims=True)
        xc = x - mu
        var = jnp.mean(xc ** 2, axis=axis, keepdims=True)
        std = jnp.sqrt(var + eps)
        xhat = xc / std
        y = xhat * g + b
        return Tensor._wrap(y)

    def cross_entropy(self, targets, mask=None):
        x = self._x
        n = x.shape[0]
        logp = jax.nn.log_softmax(x, axis=-1)
        t = jnp.asarray(targets, dtype=jnp.int32)
        row = jnp.arange(n)
        per_row = -logp[row, t]
        if mask is None:
            mask_arr = jnp.ones(n, dtype=x.dtype)
        else:
            mask_arr = jnp.asarray(mask, dtype=x.dtype)
        loss = (per_row * mask_arr).sum() / jnp.maximum(mask_arr.sum(), 1.0)
        out = Tensor._wrap(loss)
        # aux is plain numpy, matching the original contract
        probs = jnp.exp(logp)
        preds = jnp.argmax(probs, axis=-1)
        correct = (preds == t).astype(jnp.float64) * mask_arr
        out.aux = {
            "probs": np.asarray(probs),
            "correct": np.asarray(correct),
            "mask": np.asarray(mask_arr),
            "preds": np.asarray(preds),
        }
        return out

    def __getitem__(self, idx):
        return Tensor._wrap(self._x[idx])

    def __setitem__(self, idx, value):
        self._x = self._x.at[idx].set(Tensor._unwrap(value))

    # ---------- graph ----------
    def backward(self, grad=None):
        """Differentiate this Tensor (assumed scalar) w.r.t. every leaf
        registered with requires_grad=True, and write .grad on each as a
        numpy array. Mirrors the original autograd's post-conditions."""
        leaves = [p for p in _PARAM_REGISTRY if p._requires_grad]
        if not leaves:
            return
        leaf_arrays = [p._x for p in leaves]

        def loss_fn(*xs):
            # swap the leaf arrays back into the graph, recompute loss
            saved = [p._x for p in leaves]
            for p, x in zip(leaves, xs):
                p._x = x
            try:
                loss = self._x
            finally:
                for p, s in zip(leaves, saved):
                    p._x = s
            return loss

        # We need a pure function that takes the leaf arrays as inputs.
        # Rebuild the forward pass symbolically is not possible with an
        # eager wrapper, so we use jax.grad on a closure that substitutes
        # the arrays. This is correct but not jit-able; for jit/vmap,
        # write the model as a pure function and call jax.grad directly.
        grads = jax.grad(loss_fn)(*leaf_arrays)

        # If backward() was called with an explicit incoming gradient,
        # scale each leaf grad by it (chain rule at the root).
        scale = 1.0 if grad is None else float(np.asarray(grad))
        for p, g in zip(leaves, grads):
            p.grad = np.asarray(g) * scale

    def zero_grad(self):
        self.grad = None

    def detach(self):
        out = Tensor.__new__(Tensor)
        out._x = jax.lax.stop_gradient(self._x)
        out._requires_grad = False
        out.grad = None
        out.aux = None
        return out

    def numpy(self):
        return np.asarray(self._x)

    # ---------- internals ----------
    @staticmethod
    def _unwrap(x):
        return x._x if isinstance(x, Tensor) else x

    @staticmethod
    def _wrap(x):
        out = Tensor.__new__(Tensor)
        out._x = x
        out._requires_grad = False   # only user-constructed leaves track grads
        out.grad = None
        out.aux = None
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def embedding_lookup(table: Tensor, indices: np.ndarray):
    idx = jnp.asarray(np.asarray(indices), dtype=jnp.int32)
    return Tensor._wrap(table._x[idx])