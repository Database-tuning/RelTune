import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATEncoder(nn.Module):
    def __init__(self, in_dim=385, hidden_dim=128, hidden_sec_dim=64, z_dim=32, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
        self.gat2 = GATConv(hidden_dim, hidden_sec_dim // heads, heads=heads)
        self.gat3 = GATConv(hidden_sec_dim, z_dim, heads=1, concat=False)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.gat3(x, edge_index)
        z = global_mean_pool(x, batch)
        return z


class ConfigDecoder(nn.Module):
    def __init__(self, z_dim=32, output_dim=138):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.fc(z)


class Surrogate(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        return self.fc(z)


class GNN_Autoencoder_With_Surrogate(nn.Module):
    def __init__(self, in_dim=385, hidden_dim=128, hidden_sec_dim=64,
                 z_dim=32, out_config_dim=138, heads=4):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, hidden_sec_dim, z_dim, heads=heads)
        self.decoder = ConfigDecoder(z_dim, out_config_dim)
        self.surrogate = Surrogate(z_dim)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        recon_config = self.decoder(z)
        metric_pred = self.surrogate(z)
        return recon_config, metric_pred, z
