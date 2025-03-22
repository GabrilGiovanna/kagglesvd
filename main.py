import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import copy
import random
import numpy as np
import torch

import model
from parse import parse_args
from utils import log_end_epoch, get_item_propensity, get_common_path, set_seed, preprocess_svd, preprocess_ease, convert_sp_mat_to_sp_tensor
from grouping import grouping_factory
from eval import evaluate
from aggregation.aggregation import Average, BordaCount
import torch
import torch_xla
import torch_xla.core.xla_model as xm

args = parse_args()

def train(hyper_params, data):

    device = xm.xla_device()


    if hyper_params['model'] == 'svd-ae':
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        PATH = os.getcwd()
        adj_mat, norm_adj, ut, s, vt = preprocess_svd(hyper_params['load'], hyper_params['dataset'], adj_mat, hyper_params['k'], os.path.join(PATH, 'checkpoints'), device)
        train_model = model.SVD_AE(adj_mat, norm_adj, ut, vt, device)
    else:
        print('This model is not supported!')
        exit()

    return train_model, s


def evaluate_model(hyper_params, data, train_model, s):
    from eval import evaluate
    from aggregation.aggregation import Average, BordaCount
    import torch

    device = xm.xla_device()

    item_propensity = get_item_propensity(hyper_params, data)

    # Iterate through different cluster and group sizes
    #clusters = [5,10,20]
    group_sizes = [5,10]
    SIMILARITY_THRESHOLDS = [0.8, 0.9]
    #group_sizes = [5, 10, 20, 50]
    #clusters = [10,20]
    #group_sizes = [50,100,200,500]
    # Convert model output tensor
    #s = s.to(device='cpu')
    s = s.to(device)
    rating = train_model(s)

    for similarity in SIMILARITY_THRESHOLDS:
        for group_size in group_sizes:
            print(f"\nEvaluating with similarity={similarity}, group_size={group_size}")

            # Update clustering/grouping hyperparameters
            hyper_params['similarity_threshold'] = similarity
            hyper_params['group_size'] = group_size

            # Define base log filename
            base_log_path = f"./results/logs/{get_common_path(hyper_params)}"

            # **1. Individual Recommendation (individual=True)**
            hyper_params['individual'] = True
            hyper_params['log_file'] = f"./results/logs/{get_common_path(hyper_params)}.txt"
            print(f"Running group evaluation (log: {hyper_params['log_file']})")
            print(hyper_params)
            test_metrics, preds = evaluate(rating, hyper_params, data, item_propensity, None, test_set_eval=True)
            log_end_epoch(hyper_params, test_metrics, 0, 0)

            # **2. Group Recommendation with Average Aggregation**
            hyper_params['individual'] = False
            hyper_params['aggregation'] = 'Average'
            hyper_params['log_file'] = f"./results/logs/{get_common_path(hyper_params)}.txt"
            print(f"Running individual evaluation (Average) (log: {hyper_params['log_file']})")
            print(hyper_params)
            test_metrics, preds = evaluate(rating, hyper_params, data, item_propensity, None, test_set_eval=True)
            log_end_epoch(hyper_params, test_metrics, 0, 0)

            # **3. Group Recommendation with BordaCount Aggregation**
            hyper_params['aggregation'] = 'BordaCount'
            hyper_params['log_file'] = f"./results/logs/{get_common_path(hyper_params)}.txt"
            print(f"Running individual evaluation (BordaCount) (log: {hyper_params['log_file']})")
            print(hyper_params)
            test_metrics, preds = evaluate(rating, hyper_params, data, item_propensity, None, test_set_eval=True)
            log_end_epoch(hyper_params, test_metrics, 0, 0)

            print(f"Finished evaluations for similarity={similarity}, group_size={group_size}\n")


def top_pop(hyper_params, data, topk=10):
    item_count = np.zeros(hyper_params['num_items'])
    
    # Get the top popular items
    for u, i, r in data.data['train']:
        item_count[i] += 1

    top_items = np.argsort(item_count)[::-1][:topk]
    return top_items

def evaluate_top_pop(hyper_params, data, topk=10):
    top_items = top_pop(hyper_params, data, topk)

    



def main(hyper_params, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax import config
    if 'float64' in hyper_params and hyper_params['float64']:
        config.update('jax_enable_x64', True)

    from data import Dataset

    os.makedirs("./results/logs/", exist_ok=True)
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params)  # Updated w/ data-stats

    # Train model once
    train_model, s = train(hyper_params, data)


    # Evaluate multiple times with different settings
    evaluate_model(hyper_params, data, train_model, s)


def test_eval(hyper_params):
    gpu_id = None
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax import config
    if 'float64' in hyper_params and hyper_params['float64']:
        config.update('jax_enable_x64', True)

    from data import Dataset

    os.makedirs("./results/logs/", exist_ok=True)
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params)  # Updated w/ data-stats

    # Train model once
    train_model, s = train(hyper_params, data)

    device = xm.xla_device()

    #s = s.to(device='cpu')
    s = s.to(device)
    rating = train_model(s)

    item_propensity = get_item_propensity(hyper_params, data)

    hyper_params['log_file'] = f"./results/logs/{get_common_path(hyper_params)}.txt"

    test_metrics, preds = evaluate(rating, hyper_params, data, item_propensity, None, test_set_eval=True)
    log_end_epoch(hyper_params, test_metrics, 0, 0)

if __name__ == "__main__":
    from grouping import FCMWithPCCGrouping
    from dataset import SteamRSDataset, MovieLensRSDataset, dataset_factory, ML1m, MINDRSDataset
    from aggregation.aggregation import Average, BordaCount
    from aggregation import aggregation_factory
    from hyper_params import hyper_params
    from data import Dataset
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    set_seed(hyper_params['seed'])
    #GPU = torch.cuda.is_available()
    #device = torch.device('cuda:0' if GPU else 'cpu')

    # TPU
    device = xm.xla_device()


    # Ml-latest-small dataset
    #hyper_params['dataset'] = 'ml-latest-small'
    #hyper_params['dataset'] = 'ml-1m'
    #hyper_params['dataset'] = 'steam'
    #hyper_params['dataset'] = 'MIND'
    #hyper_params['grouping_method'] = 'ContentBasedPCC'
    #hyper_params['grouping_method'] = 'FCMWithPCC'
    #hyper_params['aggregation'] = 'Average'
    #hyper_params['individual'] = False
    #hyper_params['similarity_threshold'] = 0.9
    #hyper_params['group_size'] = 10

    #hyper_params['k'] = 148
    print(hyper_params)

    #test_eval(hyper_params)
    #train_ds, val_ds, test_ds = dataset_factory(SteamRSDataset.code())

    #SteamRSDataset.datasetconversion(train_ds, val_ds, test_ds)
    main(hyper_params)