"""Gradient checks for tensor.py against numerical finite differences.
Run: python3 test_tensor.py
Every op used anywhere in the model must pass here before it's trusted.
"""
import numpy as np
from tensor import Tensor, embedding_lookup

np.random.seed(0)
TOL = 3e-4


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f(x)
        x[idx] = orig - eps
        f2 = f(x)
        x[idx] = orig
        grad[idx] = (f1 - f2) / (2 * eps)
        it.iternext()
    return grad


def check(name, build_fn, x0):
    """build_fn(x_data) -> scalar loss (python float), used for numeric grad.
    Separately builds the Tensor graph and checks .grad against numeric."""
    x_np = x0.copy()

    def f(xd):
        return build_fn(xd)[0]  # scalar

    num_g = numerical_grad(lambda xd: f(xd), x_np)

    _, grad_analytic = build_fn(x_np, want_grad=True)
    err = np.max(np.abs(num_g - grad_analytic))
    status = "OK" if err < TOL else "FAIL"
    print(f"[{status}] {name}: max abs error = {err:.2e}")
    assert err < TOL, f"{name} gradient check failed (err={err:.2e})"


def t_add():
    def build(xd, want_grad=False):
        x = Tensor(xd)
        y = Tensor(np.array([[1.0, 2.0, 3.0]]))
        out = (x + y).sum()
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("add(broadcast)", build, np.random.randn(4, 3))


def t_mul_matmul():
    W = np.random.randn(5, 4)

    def build(xd, want_grad=False):
        x = Tensor(xd)
        w = Tensor(W)
        out = (x @ w).sum()
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("matmul", build, np.random.randn(3, 5))


def t_softmax():
    def build(xd, want_grad=False):
        x = Tensor(xd)
        s = x.softmax(axis=-1)
        target = Tensor(np.array([[0.1, 0.2, 0.3, 0.4]] * xd.shape[0]))
        out = (s * target).sum()
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("softmax", build, np.random.randn(2, 4))


def t_layernorm():
    gamma = Tensor(np.random.randn(1, 6) * 0.1 + 1.0)
    beta = Tensor(np.random.randn(1, 6) * 0.1)

    def build(xd, want_grad=False):
        gamma.zero_grad(); beta.zero_grad()
        x = Tensor(xd)
        y = x.layer_norm(gamma, beta)
        target = Tensor(np.random.RandomState(1).randn(*xd.shape))
        out = (y * target).sum()
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("layer_norm", build, np.random.randn(3, 6) * 2 + 1)


def t_gelu():
    def build(xd, want_grad=False):
        x = Tensor(xd)
        out = x.gelu().sum()
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("gelu", build, np.random.randn(4, 5))


def t_cross_entropy():
    targets = np.array([0, 2, 1])
    mask = np.array([1.0, 1.0, 0.0])

    def build(xd, want_grad=False):
        x = Tensor(xd)
        out = x.cross_entropy(targets, mask=mask)
        out.backward()
        return out.data.item(), (x.grad if want_grad else None)
    check("cross_entropy(masked)", build, np.random.randn(3, 4))


def t_embedding():
    idx = np.array([0, 2, 2, 1])

    def build(xd, want_grad=False):
        table = Tensor(xd)
        out = embedding_lookup(table, idx)
        target = Tensor(np.random.RandomState(2).randn(*out.shape))
        loss = (out * target).sum()
        loss.backward()
        return loss.data.item(), (table.grad if want_grad else None)
    check("embedding_lookup", build, np.random.randn(5, 3))


def t_batched_matmul_attention_shapes():
    # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T), mimicking attention scores
    B, H, T, D = 2, 2, 3, 4
    K = np.random.randn(B, H, D, T)

    def build(xd, want_grad=False):
        Q = Tensor(xd)
        k = Tensor(K)
        scores = Q @ k
        out = scores.sum()
        out.backward()
        return out.data.item(), (Q.grad if want_grad else None)
    check("batched_matmul(attn-shape)", build, np.random.randn(B, H, T, D))


if __name__ == "__main__":
    t_add()
    t_mul_matmul()
    t_softmax()
    t_layernorm()
    t_gelu()
    t_cross_entropy()
    t_embedding()
    t_batched_matmul_attention_shapes()
    print("\nAll gradient checks passed.")
