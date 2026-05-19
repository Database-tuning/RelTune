import json
import itertools

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def build_parameter_embeddings(desc_json_path: str):
    with open(desc_json_path, "r") as f:
        data = json.load(f)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    param_names = list(data.keys())
    descriptions = list(data.values())

    embeddings = model.encode(descriptions, convert_to_tensor=True)
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)

    results = []
    for i, j in itertools.combinations(range(len(param_names)), 2):
        sim = F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0)
        ).item()
        results.append((param_names[i], param_names[j], sim))

    results.sort(key=lambda x: -x[2])
    return param_names, embeddings, desc_embeddings, results


def build_relational_graph(embeddings: torch.Tensor, similarity_threshold: float = 0.75):
    edge_list = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
            if sim >= similarity_threshold:
                edge_list.append((i, j))
                edge_list.append((j, i))

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index


def build_graph_dataset(scaled_config, scaled_metrics, param_names,
                        desc_embeddings, edge_index):
    import torch
    from torch_geometric.data import Data

    graph_dataset = []
    for i in range(len(scaled_config)):
        feature_list = []
        for j in range(len(param_names)):
            value_tensor = torch.tensor([scaled_config[i][j]], dtype=torch.float)
            desc_tensor = desc_embeddings[j]
            node_feature = torch.cat([value_tensor, desc_tensor], dim=0)
            feature_list.append(node_feature)

        x = torch.stack(feature_list, dim=0)
        y = torch.tensor(scaled_metrics[i], dtype=torch.float).view(1, -1)
        graph_dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return graph_dataset


def visualize_graph(embeddings, edge_index, param_names, title="Parameter Graph"):
    graph_data = Data(x=embeddings, edge_index=edge_index)
    G = to_networkx(graph_data, to_undirected=True)
    G = nx.relabel_nodes(G, {i: name for i, name in enumerate(param_names)})

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.show()
