import jax
import numpy as np
import jax.numpy as jnp
from numba import jit, float64
import time
from tqdm import tqdm
import pandas as pd
from dataset import SteamRSDataset, MovieLensRSDataset, dataset_factory
from grouping import FCMWithPCCGrouping , grouping_factory
from aggregation.aggregation import Average, BordaCount
from aggregation import aggregation_factory

INF = float(1e6)

def evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, train_x, topk = [1, 5, 10, 20, 50, 100 ], test_set_eval = False):
    preds, y_binary, metrics = [], [], {}
    for kind in [ 'HR', 'NDCG', 'PSP', 'RECALL', 'PRECISION', 'MRR' ]: # [ 'HR', 'NDCG', 'PSP' ]:
        for k in topk: 
            metrics['{}@{}'.format(kind, k)] = 0.0
    # Train positive set -- these items will be set to -infinity while prediction on the val/test set

    train_positive_list = list(map(list, data.data['train_positive_set']))
    if test_set_eval:
        for u in range(len(train_positive_list)): train_positive_list[u] += list(data.data['val_positive_set'][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data['train_matrix']
    if test_set_eval: eval_context += data.data['val_matrix']

    # What needs to be predicted
    to_predict = data.data['val_positive_set']
    if test_set_eval: to_predict = data.data['test_positive_set']

    bsz = hyper_params['num_users']
    bsz = 10000
    # bsz = 20_000 # These many users
    
    train_time = 0
    u_map = data.data['user_map']

    u_map = data.data['user_map']


    train, val, test = dataset_factory(hyper_params['dataset'])
    #print("Dataset statistics of train set:")
    #print(train.get_statistics())

    train.ratings = pd.concat([train.ratings, val.ratings], ignore_index=True)

    #groups = FCMWithPCCGrouping(train,group_size = hyper_params['group_size'],n_clusters = hyper_params['n_clusters'])

    groups = grouping_factory(grouping_method= hyper_params['grouping_method'], 
                            dataset_name= hyper_params['dataset'],
                            group_size= hyper_params['group_size'], 
                            n_clusters= hyper_params['n_clusters'],similarity_threshold = hyper_params['similarity_threshold'])

    #aggregation = Average()
    aggregation = aggregation_factory(hyper_params['aggregation'])

    temp_preds = torch.zeros(hyper_params['num_users'], hyper_params['num_items'])

    unique_users = groups.get_unique_users()

    user_mapping = data.data['user_mapping']

    
    unique_users_map = [u_map[user_group] for user_group in unique_users]
    
    if hyper_params['individual']== False:
        for user in tqdm(unique_users_map):
            user_id = list(u_map.keys())[list(u_map.values()).index(user)]
            group = groups.get_user_group(user_id)
            group = [u_map[user_group] for user_group in unique_users]
            if hyper_params['aggregation'] == 'BordaCount':
                test_indices = list(data.data['test_positive_set'][user])
                neg_indices = data.data['negatives'][user].tolist()
                rating_group = rating[group]
                rating_group = rating_group[:,test_indices+neg_indices]
                temp_preds[user,test_indices+neg_indices] = aggregation.aggregate_pytorch(rating_group).to(torch.float)
            else:
                rating_group = rating[group]
                temp_preds[user] = aggregation.aggregate_pytorch(rating_group)
    
    #get unique users in list_of_group_users and get temp_preds only for those users
   
    temp_preds = temp_preds[unique_users_map]

    #if unique_users_map is list of float, convert it to list of int
    if isinstance(unique_users_map[0], float):
        unique_users_map = [int(u) for u in unique_users_map]

    #get train_positive_list and to_predict for list_of_group_users
    train_positive_list = [train_positive_list[u] for u in unique_users_map]
    to_predict = [to_predict[u] for u in unique_users_map]

    
    metrics, temp_preds, temp_y = evaluate_batch(
        data.data['negatives'][unique_users_map], np.array(temp_preds), 
        train_positive_list, to_predict, item_propensity, 
        topk, metrics
    )

    preds += temp_preds
    y_binary += temp_y


    """ for i in tqdm(range(0, hyper_params['num_users'], bsz)):
        if hyper_params['model'] == 'ease' or hyper_params['model'] == 'svd-ae':
            end = min(i+bsz, hyper_params['num_users'])
            #import jax.experimental.sparse as jax_sparse
            #temp_preds = jax_sparse.BCOO.from_scipy_sparse(rating.to_sparse().cpu().coalesce().to_scipy())
            #temp_preds=jax_sparse.BCOO.from_scipy_sparse(rating)
            #temp_preds = jnp.array(rating)
            #temp_preds = jnp.array(rating.to_dense().cpu())
            #temp_preds_copy = temp_preds.copy()
            temp_preds = rating
            predicted_rating = temp_preds
        else:
            train_start_time = time.time()
            temp_preds = kernelized_rr_forward(train_x, eval_context[i:end].todense(), reg = hyper_params['lamda'])
            #temp_preds_copy = temp_preds.copy()
            temp_train_time = time.time() - train_start_time
            train_time += temp_train_time
            predicted_rating = temp_preds # predicted_rating_score
        #if i == 0:
            #print('Train_positive_list:', train_positive_list[0])
            #print('To_predict:', to_predict[0])
            #print('temp_preds:', temp_preds[0])
        metrics, temp_preds, temp_y = evaluate_batch(
            data.data['negatives'][i:end], np.array(temp_preds[i:end]), 
            train_positive_list[i:end], to_predict[i:end], item_propensity, 
            topk, metrics
        )
        #print(to_predict[i:end])
        preds += temp_preds
        y_binary += temp_y """

    if hyper_params['model'] == 'inf-ae':
        print('Training time: {}'.format(train_time))

    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        metrics['AUC'] = round(fast_auc(y_binary, preds), 4)
    
    for kind in [ 'HR', 'NDCG', 'PSP', 'RECALL', 'PRECISION', 'MRR' ]: # [ 'HR', 'NDCG', 'PSP' ]:
        for k in topk: 
            metrics['{}@{}'.format(kind, k)] = round(
                float(100.0 * metrics['{}@{}'.format(kind, k)]) / len(unique_users), 4
            )

    # metrics['num_users'] = int(train_x.shape[0])
    # metrics['num_interactions'] = int(jnp.count_nonzero(train_x.astype(np.int8)))

    return metrics, temp_preds

import torch

INF = float(1e6)

def evaluate_batch(auc_negatives, logits, train_positive, test_positive_set, item_propensity, topk, metrics, train_metrics=False):
    """
    logits: predicted rating tensor (batch_size, num_items)
    train_positive: list of train positive items
    test_positive_set: list of test positive items
    """
    temp_preds, temp_y = [], []
    logits = torch.tensor(logits)  # Ensure logits is a tensor
    
    for b in range(len(logits)):
         
        test_indices = torch.tensor(list(test_positive_set[b]), dtype=torch.long)
        temp_preds.append(logits[b, test_indices])
        temp_y.extend([1.0] * len(test_positive_set[b]))

        neg_indices = torch.tensor(auc_negatives[b], dtype=torch.long)
        temp_preds.append(logits[b, neg_indices])
        temp_y.extend([0.0] * len(auc_negatives[b]))
        mask = torch.ones(logits.shape[1], dtype=torch.bool)
        mask[test_indices.tolist()+neg_indices.tolist()] = False
        logits[b, mask] = -INF
    temp_preds = torch.cat(temp_preds).tolist()
    
    
    # Use torch.topk instead of sorting manually
    _, indices = torch.topk(logits, max(topk), dim=1, largest=True, sorted=True)
    
    for k in topk:
        for b in range(len(logits)):
            num_pos = float(len(test_positive_set[b]))
            if num_pos == 0:
                continue
            
            top_k_set = set(indices[b, :k].tolist())
            test_set = set(test_positive_set[b])

            first_relevant = next((i for i, x in enumerate(indices[b, :k]) if x.item() in test_set), None)
            
            metrics[f'HR@{k}'] += len(top_k_set & test_set) / float(min(num_pos, k))
            metrics[f'RECALL@{k}'] += len(top_k_set & test_set) / float(num_pos)
            metrics[f'PRECISION@{k}'] += len(top_k_set & test_set) / float(k)

            if first_relevant is not None:
                metrics[f'MRR@{k}'] += 1.0 / (first_relevant + 1.0)
            
            else:
                metrics[f'MRR@{k}'] += 0.0
            
            test_positive_sorted_psp = sorted([item_propensity[x] for x in test_positive_set[b]], reverse=True)
            
            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, pred in enumerate(indices[b, :k]):
                if pred.item() in test_set:
                    dcg += 1.0 / torch.log2(torch.tensor(at + 2.0))
                    psp += float(item_propensity[pred.item()]) / float(min(num_pos, k))
                if at < num_pos:
                    idcg += 1.0 / torch.log2(torch.tensor(at + 2.0))
                    max_psp += test_positive_sorted_psp[at]
            
            metrics[f'NDCG@{k}'] += dcg / idcg if idcg > 0 else 0
            metrics[f'PSP@{k}'] += psp / max_psp if max_psp > 0 else 0
    
    return metrics, temp_preds, temp_y
    
@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += (1 - y_true[i])
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))