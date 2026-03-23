from __future__ import annotations

from pathlib import Path

import torch

from .losses import WeightedL1Loss


DEFAULT_PATIENCE = 50


def build_optimizer(model, lr: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 500], gamma=0.5)


def save_battery_checkpoint(model, optimizer=None, epoch: int = 0, save_path: str | Path = ""):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, save_path)


def load_battery_checkpoint(model, save_path: str | Path, optimizer=None, map_location=None):
    checkpoint = torch.load(save_path, map_location=map_location)
    model.load_state_dict(checkpoint['net'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def _train_battery_model(model, train_x, train_y, config, save_path=None, kl_beta=0.0, patience=DEFAULT_PATIENCE, print_per=50):
    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer)
    loss_fn = WeightedL1Loss()

    inputs = torch.from_numpy(train_x.astype('float32')).to(model.device)
    labels = torch.from_numpy(train_y.astype('float32')).to(model.device)
    model.set_batch_size(inputs.shape[0])

    min_loss = float('inf')
    stop = 0
    best_checkpoint = None
    losses = []

    for epoch in range(config.epoch):
        optimizer.zero_grad()
        pred, _ = model.predict(inputs)
        data_loss = loss_fn(pred, labels)
        boundary_loss = model.boundary_loss(pred)
        kl = model.kl_loss()
        total_loss = data_loss + boundary_loss + kl_beta * kl
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(total_loss.item())

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if epoch == 0 or (epoch + 1) % print_per == 0:
            print(
                f"Epoch {epoch + 1:4d} loss={total_loss.item():.5e} "
                f"L1={data_loss.item():.7f} boundary={boundary_loss.item():.7f} "
                f"KL={kl.item():.7f} lr={lr:.4f}"
            )

        stop += 1
        if total_loss.item() < min_loss:
            min_loss = total_loss.item()
            stop = 0
            best_checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

        if patience is not None and stop > patience:
            print('\nearly stop')
            print('=' * 100)
            break

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['net'])
        if save_path is not None:
            save_battery_checkpoint(model, optimizer, best_checkpoint['epoch'], save_path)
    elif save_path is not None:
        save_battery_checkpoint(model, optimizer, len(losses) - 1, save_path)

    return model, losses


def train_battery_deterministic(model, train_x, train_y, config, save_path=None):
    return _train_battery_model(model, train_x, train_y, config, save_path=save_path, kl_beta=0.0, patience=DEFAULT_PATIENCE)


def train_battery_variational(model, train_x, train_y, config, save_path=None):
    kl_beta = getattr(config, 'kl_beta', 1e-5)
    return _train_battery_model(model, train_x, train_y, config, save_path=save_path, kl_beta=kl_beta, patience=None)


def fine_tune_battery_model(model, train_x, train_y, config, save_path=None):
    if getattr(model, 'is_variational', False):
        return train_battery_variational(model, train_x, train_y, config, save_path=save_path)
    return train_battery_deterministic(model, train_x, train_y, config, save_path=save_path)
