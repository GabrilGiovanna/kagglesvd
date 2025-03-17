import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import copy
import random
import numpy as np
import torch
import jax
from jax import numpy as jnp
import jax.experimental.sparse as jax_sparse

import model
from parse import parse_args
from utils import log_end_epoch, get_item_propensity, get_common_path, set_seed, preprocess_svd, preprocess_ease, convert_sp_mat_to_sp_tensor
from grouping import grouping_factory

args = parse_args()

def train(hyper_params, data):
    from model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(hyper_params)
    #sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    if hyper_params['model'] == 'svd-ae':
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        PATH = os.getcwd()
        adj_mat, norm_adj, ut, s, vt = preprocess_svd(hyper_params['load'], hyper_params['dataset'], adj_mat, hyper_params['k'], os.path.join(PATH, 'checkpoints'), device)
        train_model = model.SVD_AE(adj_mat, norm_adj, ut, vt, device = 'cpu')
        
    elif hyper_params['model'] == 'ease':
        adj_mat = data.data['train_matrix']
        adj_mat, item_adj = preprocess_ease(adj_mat, device)
        train_model = model.EASE(adj_mat, item_adj, device)

    elif hyper_params['model'] == 'inf-ae':
        rating = None

    else:
        print('This model is not supported!')
        exit()

    #sampled_matrix = jnp.array(sampled_matrix.todense())

    #sampled_matrix = jax_sparse.BCOO.from_scipy_sparse(sampled_matrix)

    '''
    NOTE: No training required! We will compute dual-variables \alpha on the fly in `kernelized_rr_forward`
          However, if we needed to perform evaluation multiple times, we could pre-compute \alpha like so:
    
    import jax, jax.numpy as jnp, jax.scipy as sp
    @jax.jit
    def precompute_alpha(X, lamda=0.1):
        K = kernel_fn(X, X)
        K_reg = (K + jnp.abs(lamda) * jnp.trace(K) * jnp.eye(K.shape[0]) / K.shape[0])
        return sp.linalg.solve(K_reg, X, sym_pos=True)
    alpha = precompute_alpha(sampled_matrix, lamda=0.1) # Change for the desired value of lamda
    '''

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)
    
    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric, best_lamda = None, None

    if hyper_params['model'] == 'svd-ae':
        print(len(s))
        s = s.to(device = 'cpu')
        rating = train_model(s)
        test_metrics, preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, None, test_set_eval = True)
        

        # MSE
        print("Computing MSE")
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        #adj_mat = jnp.array(convert_sp_mat_to_sp_tensor(adj_mat).to_dense(), dtype=jnp.bfloat16)
        #print("Converting to dense")
        #adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()
        #adj_mat_cpu = jax.device_put(adj_mat, jax.devices("cpu")[0])
        #preds_cpu = jax.device_put(preds, jax.devices("cpu")[0])

        #err = (preds - adj_mat) ** 2
        #mse = sum(sum(err)) / (adj_mat.shape[0] * adj_mat.shape[1])
        import torch
        import numpy as np
        from tqdm import tqdm

        # Convert to a PyTorch sparse tensor (keeping everything on CPU)
        adj_mat = torch.tensor(adj_mat.toarray(), dtype=torch.float16)  # No CUDA
        preds = torch.tensor(preds, dtype=torch.float16)  # Ensure preds is a tensor

        groups = grouping_factory(grouping_method = hyper_params['grouping_method'], 
                                dataset_name= hyper_params['dataset'], 
                                group_size= hyper_params['group_size'], 
                                n_clusters= hyper_params['n_clusters'], 
                                similarity_threshold = hyper_params['similarity_threshold'])
        

        unique_users = groups.get_unique_users()

        u_map = data.data['user_map']
        group = [u_map[user_group] for user_group in unique_users]

        #Adj mat with only unique users
        adj_mat = adj_mat[group]
        # Compute MSE in batches
        print("Computing MSE")
        mse_values = []
        for i, user in tqdm(enumerate(group)):
            #if user is float, convert it to int
            if isinstance(user, float):
                user = int(user)
            test_indices = torch.tensor(list(data.data['test_positive_set'][user]), dtype=torch.long)
            neg_indices = torch.tensor(list(data.data['negatives'][user]), dtype=torch.long)
            preds_batch = preds[i * 101 : i * 101 + 101]
            mask = torch.zeros(adj_mat.shape[1], dtype=torch.bool)
            mask[test_indices.tolist()+neg_indices.tolist()] = True
            adj_batch = adj_mat[i,mask]

            err = (preds_batch - adj_batch) ** 2
            mse_values.append(err.sum().item())  # Convert to Python number to save memory

        mse = sum(mse_values) / (adj_mat.shape[0] * 101)
        print("\nMSE value: {}".format(mse))


    elif hyper_params['model'] == 'ease':
        # Validate on the validation-set
        for lamda in [ 1.0, 10.0, 100.0, 1000.0, 10000.0 ] if hyper_params['grid_search_lamda'] else [ hyper_params['lamda'] ]:
            hyper_params['lamda'] = lamda
            rating = train_model(lamda)
            val_metrics,  preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix)
            log_end_epoch(hyper_params, val_metrics, 0, time.time() - start_time)
            if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric): best_metric, best_lamda = val_metrics[VAL_METRIC], lamda
        print("\nBest lambda value: {}".format(best_lamda))
        hyper_params['lamda'] = best_lamda

        # Test on the train + validation set
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        adj_mat, item_adj = preprocess_ease(adj_mat, device)
        train_model = model.EASE(adj_mat, item_adj, device)
        rating = train_model(best_lamda)
        test_metrics, preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
       
        # MSE
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        adj_mat = jnp.array(convert_sp_mat_to_sp_tensor(adj_mat).to_dense(), dtype=jnp.bfloat16)
        #adj_mat_cpu = jax.device_put(adj_mat, jax.devices("cpu")[0])
        #preds_cpu = jax.device_put(preds, jax.devices("cpu")[0])

        err = (preds - adj_mat) ** 2
        mse = sum(sum(err)) / (adj_mat.shape[0] * adj_mat.shape[1])
        print("\nMSE value: {}".format(mse))


    else:
        # Validate on the validation-set
        for lamda in [ 0.0, 1.0, 5.0, 20.0, 50.0, 100.0 ] if hyper_params['grid_search_lamda'] else [ hyper_params['lamda'] ]:
            hyper_params['lamda'] = lamda
            val_metrics, preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix)
            log_end_epoch(hyper_params, val_metrics, 0, time.time() - start_time)
            if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric): best_metric, best_lamda = val_metrics[VAL_METRIC], lamda
        print("Best lambda value: {}".format(best_lamda))
        hyper_params['lamda'] = best_lamda
        test_metrics, preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
        
        # MSE
        adj_mat = data.data['train_matrix'] + data.data['val_matrix']
        adj_mat = jnp.array(convert_sp_mat_to_sp_tensor(adj_mat).to_dense(), dtype=jnp.bfloat16)
        #adj_mat_cpu = jax.device_put(adj_mat, jax.devices("cpu")[0])
        #preds_cpu = jax.device_put(preds, jax.devices("cpu")[0])

        err = (preds - adj_mat) ** 2
        mse = sum(sum(err)) / (adj_mat.shape[0] * adj_mat.shape[1])
        print("\nMSE value: {}".format(mse))


    # Return metrics with the best lamda on the test-set
    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics

def main(hyper_params, gpu_id = None):
    if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax import config
    if 'float64' in hyper_params and hyper_params['float64'] == True: config.update('jax_enable_x64', True)

    from data import Dataset

    os.makedirs("./results/logs/", exist_ok=True)
    hyper_params['log_file'] = "./results/logs/" + get_common_path(hyper_params) + ".txt"
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params) # Updated w/ data-stats

    return train(hyper_params, data)

if __name__ == "__main__":
    from grouping import FCMWithPCCGrouping
    from dataset import SteamRSDataset, MovieLensRSDataset, dataset_factory, ML1m, MINDRSDataset
    from aggregation.aggregation import Average, BordaCount
    from aggregation import aggregation_factory
    from hyper_params import hyper_params
    from data import Dataset
    set_seed(hyper_params['seed'])
    GPU = torch.cuda.is_available()
    device = torch.device('cuda:0' if GPU else 'cpu')

    # Ml-latest-small dataset
    #hyper_params['dataset'] = 'ml-latest-small'
    #hyper_params['dataset'] = 'ml-1m'
    #hyper_params['dataset'] = 'steam'
    #hyper_params['dataset'] = 'MIND'

    #hyper_params['k'] = 65

    #clusters
    #hyper_params['n_clusters'] = 10

    #group size
    #hyper_params['group_size'] = 10

    #hyper_params['individual'] = False
    #print(hyper_params)

    #train_ds, val_ds, test_ds = dataset_factory(SteamRSDataset.code())

    #SteamRSDataset.datasetconversion(train_ds, val_ds, test_ds)

    main(hyper_params)