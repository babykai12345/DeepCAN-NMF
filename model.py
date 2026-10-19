"""DeepCAN-NMF encoders, decoder and objective terms."""
from __future__ import annotations

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import (
    SEED,
)


def build_laplacian(latent_dim: int) -> torch.Tensor:
    rng = np.random.default_rng(config.SEED)
    adj = rng.random((latent_dim, latent_dim))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    deg = np.diag(adj.sum(axis=1))
    return torch.tensor(deg - adj, dtype=torch.float32)


class DeepCANNMF_Dynamic(nn.Module):
    def __init__(self, dims, hidden_dim, latent_dim):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.Softplus()
            ) for dim in dims
        ])
        self.W_shared = nn.Parameter(torch.rand(latent_dim, hidden_dim) * 0.015)
        self.W_decs   = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_dim, dim) * 0.015) for dim in dims
        ])

    def forward(self, x_list):
        z_list    = [enc(x) for enc, x in zip(self.encoders, x_list)]
        z_shared  = sum(z_list) / len(z_list)
        W_sh_nn   = torch.clamp(self.W_shared, min=1e-6)
        W_decs_nn = [torch.clamp(W, min=1e-6) for W in self.W_decs]
        h_layer   = torch.matmul(z_shared, W_sh_nn)
        recons    = [torch.matmul(h_layer, W_dec) for W_dec in W_decs_nn]
        return z_list, z_shared, recons, W_sh_nn


def deep_cca_loss(H1, H2, eps=1e-2):
    m       = H1.size(0)
    H1_bar  = H1 - H1.mean(dim=0, keepdim=True)
    H2_bar  = H2 - H2.mean(dim=0, keepdim=True)
    S11 = torch.matmul(H1_bar.t(), H1_bar) / (m-1) + eps * torch.eye(H1.size(1), device=H1.device)
    S22 = torch.matmul(H2_bar.t(), H2_bar) / (m-1) + eps * torch.eye(H2.size(1), device=H2.device)
    S12 = torch.matmul(H1_bar.t(), H2_bar) / (m-1)
    D1, V1 = torch.linalg.eigh(S11)
    D2, V2 = torch.linalg.eigh(S22)
    D1 = torch.clamp(D1, min=eps);  D2 = torch.clamp(D2, min=eps)
    S11inv = torch.matmul(V1, torch.matmul(torch.diag(D1**-0.5), V1.t()))
    S22inv = torch.matmul(V2, torch.matmul(torch.diag(D2**-0.5), V2.t()))
    Tval   = torch.matmul(torch.matmul(S11inv, S12), S22inv)
    return -torch.sqrt(torch.sum(Tval**2) + 1e-8)


def graph_reg_loss(H, L):
    return torch.trace(torch.matmul(torch.matmul(H.t(), L), H))


def orthogonality_loss(W):
    W_norm   = F.normalize(W, p=2, dim=1)
    corr_mat = torch.matmul(W_norm, W_norm.t())
    identity = torch.eye(corr_mat.size(0), device=corr_mat.device)
    return torch.sum((corr_mat - identity)**2)
