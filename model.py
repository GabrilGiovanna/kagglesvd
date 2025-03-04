import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax
import torch
from torch import nn
import numpy as np
import neural_tangents as nt

from tqdm import tqdm

def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params['depth'],
        num_classes=hyper_params['num_items']
    )
    
    kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get='ntk')

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        num_users = X_train.shape[0]
        
        print("Computing K_train (this may take time)...")
        with tqdm(total=num_users) as pbar:
            K_train = kernel_fn(X_train, X_train).astype(jnp.float16)
            pbar.update(num_users)  # Since it's one big operation
        
        print("Computing K_predict...")
        with tqdm(total=num_users) as pbar:
            K_predict = kernel_fn(X_predict, X_train).astype(jnp.float16)
            pbar.update(num_users)

        K_reg = (K_train + jnp.abs(reg) * jnp.trace(K_train) * jnp.eye(K_train.shape[0]) / K_train.shape[0])

        return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))

    return kernelized_rr_forward, kernel_fn


def FullyConnectedNetwork( 
    depth,
    W_std = 2 ** 0.5, 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk'
):
    activation_fn = stax.Relu()
    dense = functools.partial(stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth): layers += [dense(1024), activation_fn] 
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    return stax.serial(*layers)

class EASE(nn.Module):
    def __init__(self, adj_mat, item_adj, device='cuda:0'):
        super(EASE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.item_adj = item_adj.to(device)

    def forward(self, lambda_):
        G = self.item_adj
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_

        print("Inverting the matrix (this may take time)...")
        with tqdm(total=1) as pbar:
            P = torch.linalg.solve(G, torch.eye(G.shape[0], device=G.device))  # Avoid inverse
            pbar.update(1)

        B = P / (-torch.diag(P))
        B[diagIndices] = 0

        print("Computing final rating matrix...")
        rating = torch.mm(self.adj_mat, B)

        return rating

class SVD_AE(nn.Module):
    def __init__(self, adj_mat, norm_adj, user_sv, item_sv, device='cuda:0'):
        super(SVD_AE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.norm_adj = norm_adj.to(device)
        self.user_sv = user_sv.to(device)  # (K, M)
        self.item_sv = item_sv.to(device)  # (K, N)

    def forward(self, lambda_mat):
        print("Computing A matrix...")
        with tqdm(total=1) as pbar:
            A = self.item_sv @ (torch.diag(1/lambda_mat)) @ self.user_sv.T
            pbar.update(1)

        print("Computing sparse matrix multiplication...")
        with tqdm(total=1) as pbar:
            import torch.profiler

            with torch.profiler.profile(record_shapes=True) as prof:
                rating = torch.sparse.mm(self.norm_adj, A @ self.adj_mat)
            print(prof.key_averages().table(sort_by="cpu_time_total"))


            
            #rating = torch.mm(self.norm_adj, A @ self.adj_mat.to_dense())
            pbar.update(1)

        return rating