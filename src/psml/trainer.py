import numpy as np
import torch


# Training process (with KL divergence handling)
def train_model(model, train_loader, val_loader, num_epochs, reconstruction_loss_fn, optimizer, device=torch.device('cpu'), kl_schedule=None):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()

        if kl_schedule == 'linear':
            kl_weight = epoch / num_epochs
        elif kl_schedule == 'sigmoid_growth':
            kl_weight = 0.1 / (1 + np.exp(-2 * (epoch - 0.7 * num_epochs))) + 0.001
        elif kl_schedule == 'sigmoid_decay':
            kl_weight = 0.1 / (1 + np.exp(2 * (epoch - 0.15 * num_epochs))) + 0.001
        else:
            kl_weight = 1e-4

        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss

            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs, val_kl_loss = model(val_inputs)

                val_reconstruction_loss = reconstruction_loss_fn(val_outputs, val_targets)
                val_total_loss = val_reconstruction_loss + kl_weight * val_kl_loss
                running_val_loss += val_total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, KL Weight: {kl_weight}')

    return train_losses, val_losses
