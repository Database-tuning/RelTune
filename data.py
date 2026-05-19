import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_mysql_data(config_path: str, metrics_path: str):
    config = pd.read_csv(config_path)
    scaler = MinMaxScaler()
    scaled_config = scaler.fit_transform(config)

    metrics_df = pd.read_csv(metrics_path)
    metric_scaler = MinMaxScaler()
    scaled_metrics = metric_scaler.fit_transform(metrics_df[['tps', 'latency']])

    return scaled_config, scaled_metrics, scaler, metric_scaler


def load_postgresql_data(data_path: str):
    data = pd.read_csv(data_path)

    metric = data[["throughput", "latency"]]
    metric_scaler = MinMaxScaler()
    scaled_metrics = metric_scaler.fit_transform(metric)

    config = data.drop(columns=["throughput", "latency"]).copy()
    param_columns = list(config.columns)
    scaler = MinMaxScaler()
    scaled_config = scaler.fit_transform(config)

    return scaled_config, scaled_metrics, scaler, metric_scaler, param_columns


def load_cnf_configs(config_dir: str = "configs"):
    knob_list = glob.glob(os.path.join(config_dir, "my_*.cnf"))
    configs = None

    for xx in range(len(knob_list)):
        path = os.path.join(config_dir, f"my_{xx}.cnf")
        a_all = pd.read_csv(path, sep="=", names=['Sample', 'value'], header=2)
        a_all = a_all.set_index("Sample")
        cur_all_df = a_all.T

        if configs is None:
            configs = cur_all_df
        else:
            configs = pd.concat([configs, cur_all_df], axis=0)

    configs = configs.reset_index().drop(["index"], axis=1)
    configs = configs.drop(configs.columns[[0, 1]], axis=1)
    return configs
