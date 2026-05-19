import torch
import numpy as np
from skopt import gp_minimize
from skopt.space import Real


def affinity_score(z: np.ndarray, Z_good: np.ndarray, sigma: float = 1.0) -> float:
    if len(Z_good) == 0:
        return 0.0
    dists = np.linalg.norm(Z_good - z, axis=1)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
    return float(np.mean(weights))


def run_hybrid_bo(model, Z_all, Y_all, device,
                  z_dim, alpha, gamma, sigma,
                  tps_threshold, latency_threshold,
                  margin=0.1, n_calls=300,
                  save_path="z_trajectory_hybrid.npy"):
    def is_good(metric):
        return metric[0] >= tps_threshold and metric[1] <= latency_threshold

    Z_good = np.array([z for z, m in zip(Z_all, Y_all) if is_good(m)])

    z_min = Z_all.min(axis=0)
    z_max = Z_all.max(axis=0)
    z_range = z_max - z_min
    z_min -= margin * z_range
    z_max += margin * z_range

    space = [Real(float(z_min[i]), float(z_max[i]), name=f'z_{i}') for i in range(z_dim)]
    aff_score_list = []

    def objective(z_flat):
        z_tensor = torch.tensor(z_flat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            metric = model.surrogate(z_tensor)
            tps = metric[:, 0].item()
            latency = metric[:, 1].item()
            score = tps - alpha * latency

        aff = affinity_score(np.array(z_flat), Z_good, sigma=sigma)
        hybrid_score = score + gamma * aff
        aff_score_list.append(hybrid_score)
        return -hybrid_score

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=10,
        acq_func="EI",
        random_state=42
    )

    np.save(save_path, np.array(res.x_iters))

    best_z = torch.tensor(res.x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        best_metric = model.surrogate(best_z)
        best_config = model.decoder(best_z)

    print(f"\n[Hybrid BO]")
    print(f"Best Score (TPS - alpha*Latency + gamma*Affinity): {-res.fun:.4f}")
    print(f"TPS: {best_metric[0,0].item():.4f} | Latency: {best_metric[0,1].item():.4f}")
    print(f"Recovered Config (normalized): {best_config}")

    return res, best_z, best_metric, best_config


def run_vanilla_bo(model, Z_all, device,
                   z_dim, alpha,
                   margin=0.1, n_calls=300):
    z_min = Z_all.min(axis=0)
    z_max = Z_all.max(axis=0)
    z_min -= margin * (z_max - z_min)
    z_max += margin * (z_max - z_min)

    space = [Real(float(z_min[i]), float(z_max[i]), name=f'z_{i}') for i in range(z_dim)]
    vbo_score_list = []

    def vanilla_objective(z_flat):
        z_tensor = torch.tensor(z_flat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            metric = model.surrogate(z_tensor)
            tps = metric[:, 0].item()
            latency = metric[:, 1].item()
            score = tps - alpha * latency
        vbo_score_list.append(score)
        return -score

    res = gp_minimize(
        func=vanilla_objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=10,
        acq_func="EI",
        random_state=42
    )

    best_z = torch.tensor(res.x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        best_metric = model.surrogate(best_z)
        best_config = model.decoder(best_z)

    print(f"\n[Vanilla BO]")
    print(f"Best Score (TPS - alpha*Latency): {-res.fun:.4f}")
    print(f"TPS: {best_metric[0,0].item():.4f} | Latency: {best_metric[0,1].item():.4f}")
    print(f"Recovered Config (normalized): {best_config}")

    return res, best_z, best_metric, best_config


def recover_config(best_config, scaler, param_names):
    recovered = best_config.cpu().detach().numpy()
    original_config = scaler.inverse_transform(recovered.reshape(1, -1)).flatten()
    for name, value in zip(param_names, original_config):
        print(f"{name} = {int(round(value))}")
