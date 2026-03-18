import numpy as np
import torch


def _extract_predictions(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def _compute_kl_weight(epoch, num_epochs, kl_schedule):
    if kl_schedule == "linear":
        return epoch / num_epochs
    if kl_schedule == "sigmoid_growth":
        return 0.1 / (1 + np.exp(-2 * (epoch - 0.7 * num_epochs))) + 0.001
    if kl_schedule == "sigmoid_decay":
        return 0.1 / (1 + np.exp(2 * (epoch - 0.15 * num_epochs))) + 0.001
    return 1e-4


def train_deterministic(
    model,
    train_loader,
    optimizer,
    loss_fn,
    num_epochs,
    device=torch.device("cpu"),
    val_loader=None,
    log_every=1,
    return_history=True,
):
    model.to(device)
    history = {"train_losses": [], "val_losses": []}

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = _extract_predictions(model(inputs))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = _extract_predictions(model(val_inputs))
                    running_val_loss += loss_fn(val_outputs, val_targets).item()
            avg_val_loss = running_val_loss / len(val_loader)
            history["val_losses"].append(avg_val_loss)

        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            if avg_val_loss is None:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
            else:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

    if not return_history:
        return None
    return history


def train_deterministic_rolling(
    model,
    train_loader,
    optimizer,
    loss_fn,
    num_epochs,
    device=torch.device("cpu"),
    log_every=10,
    return_history=False,
):
    history = train_deterministic(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device=device,
        val_loader=None,
        log_every=log_every,
        return_history=True,
    )
    if not return_history:
        return None
    return history


def train_variational(
    model,
    train_loader,
    optimizer,
    reconstruction_loss_fn,
    num_epochs,
    device=torch.device("cpu"),
    val_loader=None,
    kl_schedule=None,
    log_every=10,
    return_history=True,
):
    model.to(device)
    history = {
        "train_losses": [],
        "val_losses": [],
        "recon_losses": [],
        "kl_losses": [],
        "val_recon_losses": [],
        "val_kl_losses": [],
        "kl_weights": [],
    }

    for epoch in range(num_epochs):
        model.train()
        kl_weight = _compute_kl_weight(epoch, num_epochs, kl_schedule)
        running_train_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()
            running_train_loss += total_loss.item()
            running_recon_loss += reconstruction_loss.item()
            running_kl_loss += kl_loss.item()

        n_train_batches = len(train_loader)
        avg_train_loss = running_train_loss / n_train_batches
        avg_recon_loss = running_recon_loss / n_train_batches
        avg_kl_loss = running_kl_loss / n_train_batches
        history["train_losses"].append(avg_train_loss)
        history["recon_losses"].append(avg_recon_loss)
        history["kl_losses"].append(avg_kl_loss)
        history["kl_weights"].append(kl_weight)

        avg_val_loss = None
        avg_val_recon = None
        avg_val_kl = None
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            running_val_recon = 0.0
            running_val_kl = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs, val_kl_loss = model(val_inputs)
                    val_reconstruction_loss = reconstruction_loss_fn(val_outputs, val_targets)
                    running_val_loss += (val_reconstruction_loss + kl_weight * val_kl_loss).item()
                    running_val_recon += val_reconstruction_loss.item()
                    running_val_kl += val_kl_loss.item()
            n_val_batches = len(val_loader)
            avg_val_loss = running_val_loss / n_val_batches
            avg_val_recon = running_val_recon / n_val_batches
            avg_val_kl = running_val_kl / n_val_batches
            history["val_losses"].append(avg_val_loss)
            history["val_recon_losses"].append(avg_val_recon)
            history["val_kl_losses"].append(avg_val_kl)

        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            if avg_val_loss is None:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, "
                    f"MSE Loss: {avg_recon_loss:.6f},KL Loss: {avg_kl_loss:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, KL Weight: {kl_weight}"
                )

    if not return_history:
        return None
    return history


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    reconstruction_loss_fn,
    optimizer,
    device=torch.device("cpu"),
    kl_schedule=None,
):
    history = train_variational(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        reconstruction_loss_fn=reconstruction_loss_fn,
        num_epochs=num_epochs,
        device=device,
        val_loader=val_loader,
        kl_schedule=kl_schedule,
        log_every=1,
        return_history=True,
    )
    return history["train_losses"], history["val_losses"]
