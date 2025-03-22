import functools
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class EASE(nn.Module):
    def __init__(self, adj_mat, item_adj, device='cuda:0'):
        super(EASE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.item_adj = item_adj.to(device)

    def forward(self, lambda_):
        G = self.item_adj
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = torch.inverse(G)
        B = P / (-torch.diag(P))
        B[diagIndices] = 0
        rating = torch.mm(self.adj_mat, B)

        return rating


class SVD_AE(nn.Module):
    def __init__(self, adj_mat, norm_adj, user_sv, item_sv, device='cuda:0', batch_size=512):
        super(SVD_AE, self).__init__()
        self.device = device
        self.adj_mat = adj_mat
        self.norm_adj = norm_adj
        self.user_sv = user_sv  # (M, K)
        self.item_sv = item_sv  # (N, K)
        
        self.batch_size = batch_size

    @staticmethod
    def __slice_sparse_columns(sparse_mat, col_indices):
        sparse_mat = sparse_mat.coalesce()
        i = sparse_mat.indices()
        v = sparse_mat.values()

        # Get mask for the selected columns
        mask = torch.isin(i[1], torch.tensor(col_indices))

        # Filter indices and values
        new_i = i[:, mask]
        new_v = v[mask]

        # Adjust column indices (to match new column positions)
        col_map = {old: new for new, old in enumerate(col_indices)}
        new_i[1] = torch.tensor([col_map[c.item()] for c in new_i[1]])

        new_size = (sparse_mat.shape[0], len(col_indices))
        return torch.sparse_coo_tensor(new_i, new_v, new_size)

    @staticmethod
    def __slice_sparse_rows(sparse_mat, row_indices):
        sparse_mat = sparse_mat.coalesce()
        i = sparse_mat.indices()  # Shape (2, num_nonzero_elements)
        v = sparse_mat.values()

        # Get mask for the selected rows
        mask = torch.isin(i[0], torch.tensor(row_indices))

        # Filter indices and values
        new_i = i[:, mask]
        new_v = v[mask]

        # Adjust row indices (to match new row positions)
        row_map = {old: new for new, old in enumerate(row_indices)}
        new_i[0] = torch.tensor([row_map[r.item()] for r in new_i[0]])

        new_size = (len(row_indices), sparse_mat.shape[1])  # Number of selected rows, all columns
        return torch.sparse_coo_tensor(new_i, new_v, new_size)

    def forward(self, lambda_mat):
        # Avoid creating a large diagonal matrix
        inv_lambda = 1 / lambda_mat

        # Element wise multiplication instead of matrix mult with diagonal matrix
        scaled_user_sv = (inv_lambda[:, None] * self.user_sv.T).to(self.device)
        num_users = self.user_sv.shape[0]
        num_items = self.item_sv.shape[0]
        rating = torch.zeros((num_users, num_items), device='cpu')
        for start_item_sv in tqdm(range(0, self.item_sv.shape[0], self.batch_size), desc='Computing ratings by batches'):
            end_item_sv = min(start_item_sv + self.batch_size, num_items)

            # Compute the batch slice for items
            batch_item_sv = self.item_sv[start_item_sv:end_item_sv, :].to(self.device)  # (batch_size, K)

            # Compute batch-wise interaction
            batch_ratings = torch.mm(batch_item_sv, scaled_user_sv) # (batch_size, user_size)

            for start_adj_mat in range(0, self.adj_mat.shape[1], self.batch_size):
                #print(f"Step 1: start_adj_mat{start_adj_mat}\n")
                end_adj_mat = min(start_adj_mat + self.batch_size, num_items)
                # Slice adj_mat and norm_adj
                adj_mat_batch = self.__slice_sparse_columns(self.adj_mat, range(start_adj_mat, end_adj_mat))
                adj_mat_batch = adj_mat_batch.to_dense().to(self.device)
                # Apply adjacency matrices
                batch_ratings_adj = torch.mm(batch_ratings, adj_mat_batch)
                for start_norm_adj in tqdm(range(0, self.norm_adj.shape[0], self.batch_size), desc='Computing ratings by batches, start: {}'.format(start_adj_mat)):
                    #print(f"Step 2: start_norm_adj{start_norm_adj}\n")
                    end_norm_adj = min(start_norm_adj + self.batch_size, num_users)
                    norm_adj_batch = self.__slice_sparse_rows(self.norm_adj, range(start_norm_adj, end_norm_adj))
                    norm_adj_batch = norm_adj_batch.to_dense()[:, start_item_sv:end_item_sv].to(self.device)
                    rating[start_norm_adj:end_norm_adj, start_adj_mat:end_adj_mat] += (norm_adj_batch @ batch_ratings_adj).cpu()

        return rating

    def forward_2(self, lambda_mat):
        A = self.item_sv @ (torch.diag(1/lambda_mat)) @ self.user_sv.T
        rating = torch.mm(self.norm_adj, A @ self.adj_mat.to_dense())

        return rating