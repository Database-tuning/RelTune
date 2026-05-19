import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def prepare_dataloaders(graph_dataset, batch_size=64, test_size=0.3, val_ratio=0.3):
    train_data, temp_data = train_test_split(graph_dataset, test_size=test_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=val_ratio, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device,
                lr=0.0005, weight_decay=1e-5, metric_weight=0.3,
                max_epochs=3000, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_recon_loss, total_metric_loss, total_combined_loss = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            recon_config, metric_pred, _ = model(batch)

            target_config = batch.x[:, 0].view(batch.num_graphs, -1)
            metric_target = batch.y.view(batch.num_graphs, -1)

            recon_loss = loss_fn(recon_config, target_config)
            metric_loss = loss_fn(metric_pred, metric_target)
            loss = recon_loss + metric_weight * metric_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_metric_loss += metric_loss.item()
            total_combined_loss += loss.item()

        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_metric_loss = total_metric_loss / len(train_loader)
        avg_train_loss = total_combined_loss / len(train_loader)

        model.eval()
        val_recon_loss, val_metric_loss, val_combined_loss = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_config, metric_pred, _ = model(batch)

                target_config = batch.x[:, 0].view(batch.num_graphs, -1)
                metric_target = batch.y.view(batch.num_graphs, -1)

                recon_loss = loss_fn(recon_config, target_config)
                metric_loss = loss_fn(metric_pred, metric_target)
                loss = recon_loss + metric_weight * metric_loss

                val_recon_loss += recon_loss.item()
                val_metric_loss += metric_loss.item()
                val_combined_loss += loss.item()

        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_metric = val_metric_loss / len(val_loader)
        avg_val_loss = val_combined_loss / len(val_loader)

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}]")
            print(f"  Train  Recon: {avg_recon_loss:.4f} | Metric: {avg_metric_loss:.4f} | Total: {avg_train_loss:.4f}")
            print(f"  Val    Recon: {avg_val_recon:.4f} | Metric: {avg_val_metric:.4f} | Total: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} | Best Val Loss: {best_val_loss:.4f}")
                break

    return model


def extract_latent_vectors(model, test_loader, device):
    model.eval()
    z_list, y_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, _, z = model(batch)
            z_list.append(z.cpu())
            y_list.append(batch.y.cpu())

    Z_all = torch.cat(z_list, dim=0).numpy()
    Y_all = torch.cat(y_list, dim=0).numpy()
    return Z_all, Y_all
