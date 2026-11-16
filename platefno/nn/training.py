import torch


def clip_groups(models, clip_jointly):
    """Parameter groups whose gradient norm is clipped together."""
    if clip_jointly:
        return [[p for model in models.values() for p in model.parameters()]]
    return [list(model.parameters()) for model in models.values()]


def train_step(models, optimizers, groups, model_input, target, max_norm=1.0):
    """One optimisation step of every model on a batch; returns {name: MSE}.

    Each model minimises log10 of its MSE; gradients are clipped per group.
    """
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    mse = {}
    for name, model in models.items():
        pred = model(model_input, num_time_steps=target.shape[1])
        loss = torch.log10(torch.nn.functional.mse_loss(pred, target))
        loss.backward()
        mse[name] = 10 ** loss.item()
    for group in groups:
        torch.nn.utils.clip_grad_norm_(group, max_norm)
    for optimizer in optimizers.values():
        optimizer.step()
    return mse
