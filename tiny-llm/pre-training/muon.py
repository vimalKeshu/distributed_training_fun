"""Muon optimizer (Momentum Orthogonalized by Newton-Schulz).

Muon (Keller Jordan et al.) updates 2D weight matrices by orthogonalizing the
momentum buffer via a few Newton-Schulz iterations before applying it. In the
nanoGPT speedruns this reaches a target loss in noticeably fewer steps than
AdamW, which is attractive here because low-precision tricks (bf16/TF32) are not
available on a Maxwell/Pascal card.

Muon is only appropriate for 2D hidden weights. Embeddings, the (tied) LM head,
norm gains, and biases must be optimized with AdamW instead. The Newton-Schulz
iteration runs in float32 (not bf16) so it works on older GPUs.

torch.optim.Muon exists only in newer torch (>=2.8); this is a self-contained
equivalent for the 2.5.x line. Swapping to the upstream class later is a
one-line change.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2D matrix via a quintic Newton-Schulz iteration."""
    assert grad.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        update = b * gram + c * gram @ gram
        x = a * x + update @ x
    if transposed:
        x = x.T
    return x.to(grad.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters; route others to AdamW")

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale so the update's spectral norm is comparable across shapes.
                scale = max(1.0, parameter.size(0) / parameter.size(1)) ** 0.5

                if weight_decay != 0.0:
                    parameter.mul_(1 - lr * weight_decay)
                parameter.add_(update, alpha=-lr * scale)
