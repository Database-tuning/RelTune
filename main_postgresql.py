import torch

from graph import build_parameter_embeddings, build_relational_graph, build_graph_dataset, visualize_graph
from data import load_postgresql_data
from models import GNN_Autoencoder_With_Surrogate
from train import prepare_dataloaders, train_model, extract_latent_vectors
from optimize import run_hybrid_bo, run_vanilla_bo, recover_config

Z_DIM         = 32
METRIC_WEIGHT = 0.3
ALPHA         = 0.5
GAMMA         = 0.3
SIGMA         = 1.0
TPS_THRESHOLD = 0.7
LAT_THRESHOLD = 0.3
MARGIN        = 0.1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_names, embeddings, desc_embeddings, _ = build_parameter_embeddings(
        "Postgresql_Description_fixed.json"
    )

    edge_index = build_relational_graph(embeddings, similarity_threshold=0.75)
    visualize_graph(embeddings, edge_index, param_names, title="PostgreSQL Parameter Graph")

    scaled_config, scaled_metrics, scaler, metric_scaler, param_columns = load_postgresql_data(
        "post_data_TPCC.csv"
    )

    graph_dataset = build_graph_dataset(
        scaled_config, scaled_metrics, param_names, desc_embeddings, edge_index
    )

    train_loader, val_loader, test_loader = prepare_dataloaders(graph_dataset)

    model = GNN_Autoencoder_With_Surrogate(out_config_dim=111).to(device)
    model = train_model(model, train_loader, val_loader, device, metric_weight=METRIC_WEIGHT)

    Z_all, Y_all = extract_latent_vectors(model, test_loader, device)

    _, _, _, best_config_h = run_hybrid_bo(
        model, Z_all, Y_all, device,
        z_dim=Z_DIM, alpha=ALPHA, gamma=GAMMA, sigma=SIGMA,
        tps_threshold=TPS_THRESHOLD, latency_threshold=LAT_THRESHOLD,
        margin=MARGIN
    )
    recover_config(best_config_h, scaler, param_names)

    _, _, _, best_config_v = run_vanilla_bo(
        model, Z_all, device,
        z_dim=Z_DIM, alpha=ALPHA, margin=MARGIN
    )
    recover_config(best_config_v, scaler, param_names)
