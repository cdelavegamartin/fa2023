import copy

import torch

from platefno.nn.training import clip_groups, train_step


class Scale(torch.nn.Module):
    """Model whose prediction is a learnt scale of the input, repeated over time."""

    def __init__(self, value):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))

    def forward(self, x, num_time_steps):
        return (self.scale * x).unsqueeze(1).repeat(1, num_time_steps, 1)


def step(models, clip_jointly):
    optimizers = {n: torch.optim.SGD(m.parameters(), lr=1.0) for n, m in models.items()}
    x = torch.ones(4, 3)
    y = torch.zeros(4, 2, 3)
    train_step(models, optimizers, clip_groups(models, clip_jointly), x, y, max_norm=1e-3)
    return {n: m.scale.item() for n, m in models.items()}


def models():
    # d log10(MSE) / d scale = 2 / (scale ln 10): b's gradient dominates the joint norm
    return {"a": Scale(1.0), "b": Scale(1e-3)}


def test_separate_clipping_ignores_other_models():
    alone = step({"a": Scale(1.0)}, clip_jointly=False)["a"]
    together = step(models(), clip_jointly=False)["a"]
    assert together == alone


def test_joint_clipping_couples_models():
    separate = step(models(), clip_jointly=False)["a"]
    joint = step(models(), clip_jointly=True)["a"]
    # with joint clipping, model a's step shrinks because of model b's gradient
    assert abs(1.0 - joint) < abs(1.0 - separate)


def test_train_step_returns_mse_per_model():
    ms = models()
    optimizers = {n: torch.optim.SGD(m.parameters(), lr=0.0) for n, m in ms.items()}
    mse = train_step(ms, optimizers, clip_groups(ms, False), torch.ones(2, 3), torch.zeros(2, 1, 3))
    assert abs(mse["a"] - 1.0) < 1e-6 and abs(mse["b"] - 1e-6) < 1e-12
