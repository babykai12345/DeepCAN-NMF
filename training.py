"""Training, Optuna tuning and final refitting."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import optuna
from itertools import combinations as _cv_combinations
import random as _cv_random
from config import (
    OPTUNA_EPOCHS,
    TRIAL_LOG_PATH,
    _CV_PARAM_KEYS,
)
from model import (
    DeepCANNMF_Dynamic,
    build_laplacian,
    deep_cca_loss,
    graph_reg_loss,
    orthogonality_loss,
)


def run_training(train_tensors, val_tensors, dims,
                 hidden_dim, latent_dim, lr, weight_decay,
                 lambda_cca, lambda_graph, lambda_ortho, lambda_l1,
                 epochs, early_stop_patience=15, trial=None, verbose=False):
    L_matrix  = build_laplacian(latent_dim)
    num_mods  = len(dims)
    mse_fn    = nn.MSELoss()
    model     = DeepCANNMF_Dynamic(dims, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    history          = {k: [] for k in
                        ["epoch","train_total","train_recon","val_total","val_recon"]}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z_list, z_shared, recons, W_sh = model(train_tensors)
        loss_recon = sum(mse_fn(recons[i], train_tensors[i]) for i in range(num_mods))
        loss_cca, pairs = 0.0, 0
        for i in range(num_mods):
            for j in range(i+1, num_mods):
                loss_cca += deep_cca_loss(z_list[i], z_list[j]); pairs += 1
        if pairs > 0: loss_cca /= pairs
        loss_graph = graph_reg_loss(W_sh, L_matrix)
        loss_ortho = orthogonality_loss(W_sh)
        loss_l1    = sum(torch.norm(W, p=1) for W in model.W_decs)
        loss_train = (loss_recon + lambda_cca*loss_cca + lambda_graph*loss_graph
                      + lambda_ortho*loss_ortho + lambda_l1*loss_l1)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vz_list, _, vrecons, vW_sh = model(val_tensors)
            val_recon = sum(mse_fn(vrecons[i], val_tensors[i]) for i in range(num_mods))
            val_cca, vpairs = 0.0, 0
            for i in range(num_mods):
                for j in range(i+1, num_mods):
                    val_cca += deep_cca_loss(vz_list[i], vz_list[j]); vpairs += 1
            if vpairs > 0: val_cca /= vpairs
            val_graph = graph_reg_loss(vW_sh, L_matrix)
            val_ortho = orthogonality_loss(vW_sh)
            val_l1    = sum(torch.norm(W, p=1) for W in model.W_decs)
            loss_val  = (val_recon + lambda_cca*val_cca + lambda_graph*val_graph
                         + lambda_ortho*val_ortho + lambda_l1*val_l1)

        scheduler.step(loss_val)
        cur_val = loss_val.item()
        history["epoch"].append(epoch+1)
        history["train_total"].append(loss_train.item())
        history["train_recon"].append(loss_recon.item())
        history["val_total"].append(cur_val)
        history["val_recon"].append(val_recon.item())

        if cur_val < best_val_loss:
            best_val_loss = cur_val; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch+1) % 10 == 0:
            print(f"  Epoch [{epoch+1:03d}] Train: {loss_train.item():.4f} | "
                  f"Val: {cur_val:.4f} | Recon(val): {val_recon.item():.4f}")

        if trial is not None:
            trial.report(cur_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= early_stop_patience:
            if verbose: print(f"  ⚠️  Early stopping at epoch {epoch+1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss, history


class TrialLogCallback:
    def __init__(self, path=TRIAL_LOG_PATH):
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("trial_number,state,value,hidden_dim,latent_dim,"
                    "lr,weight_decay,lambda_cca,lambda_graph,"
                    "lambda_ortho,lambda_l1\n")

    def __call__(self, study, trial):
        p = trial.params
        row = (f"{trial.number},{trial.state.name},"
               f"{trial.value if trial.value is not None else ''},"
               f"{p.get('hidden_dim','')},"
               f"{p.get('latent_dim','')},"
               f"{p.get('lr','')},"
               f"{p.get('weight_decay','')},"
               f"{p.get('lambda_cca','')},"
               f"{p.get('lambda_graph','')},"
               f"{p.get('lambda_ortho','')},"
               f"{p.get('lambda_l1','')}\n")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(row)


def make_objective(train_tensors, val_tensors, dims, epochs=OPTUNA_EPOCHS):
    def objective(trial):
        hidden_dim   = trial.suggest_categorical("hidden_dim",   [128, 256, 512])
        latent_dim   = trial.suggest_int(        "latent_dim",   20, 80, step=10)
        lr           = trial.suggest_float(      "lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float(      "weight_decay", 1e-6, 1e-3, log=True)
        lambda_cca   = trial.suggest_float(      "lambda_cca",   0.05, 1.0)
        lambda_graph = trial.suggest_float(      "lambda_graph", 0.01, 0.5)
        lambda_ortho = trial.suggest_float(      "lambda_ortho", 0.1,  2.0)
        lambda_l1    = trial.suggest_float(      "lambda_l1",    1e-5, 1e-3, log=True)
        try:
            _, best_val_loss, _ = run_training(
                train_tensors=train_tensors, val_tensors=val_tensors, dims=dims,
                hidden_dim=hidden_dim, latent_dim=latent_dim,
                lr=lr, weight_decay=weight_decay,
                lambda_cca=lambda_cca, lambda_graph=lambda_graph,
                lambda_ortho=lambda_ortho, lambda_l1=lambda_l1,
                epochs=epochs, early_stop_patience=10,
                trial=trial, verbose=False,
            )
        except optuna.TrialPruned:
            raise
        return best_val_loss
    return objective


def _cv_seed(seed):
    _cv_random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _cv_tune_params(args,train_tensors,val_tensors,dims,seed):
    if args.optuna_trials==0:
        return dict(args._base_params), None
    _cv_seed(seed)
    sampler=optuna.samplers.TPESampler(seed=seed)
    study=optuna.create_study(
        direction="minimize",sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(args.optuna_trials,5)))
    study.optimize(make_objective(train_tensors,val_tensors,dims,
                                  epochs=args.hpo_epochs),
                   n_trials=args.optuna_trials,show_progress_bar=False)
    if not study.best_trials:
        raise ValueError("该fold的Optuna搜索全部失败或被剪枝")
    return {k:study.best_params[k] for k in _CV_PARAM_KEYS},float(study.best_value)


def _cv_train_full_best_epoch(tensors,params,best_epoch,args):
    """Refit all available labeled subjects AFTER CV with fixed chosen epochs."""
    dims=[t.shape[1] for t in tensors]
    _cv_seed(args.seed+99999)
    model=DeepCANNMF_Dynamic(
        dims,params["hidden_dim"],params["latent_dim"])
    lap=build_laplacian(params["latent_dim"])
    opt=optim.Adam(model.parameters(),lr=params["lr"],
                   weight_decay=params["weight_decay"])
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="min",factor=.5,patience=5)
    hist=[]
    for epoch in range(best_epoch):
        model.train()
        opt.zero_grad()
        zlist,_,recons,shared=model(tensors)
        loss_recon=sum(F.mse_loss(pred,truth)
                       for pred,truth in zip(recons,tensors))
        cca_terms=[
            deep_cca_loss(zlist[i],zlist[j])
            for i,j in _cv_combinations(range(len(zlist)),2)]
        loss_cca=sum(cca_terms)/max(len(cca_terms),1)
        loss_graph=graph_reg_loss(shared,lap)
        loss_ortho=orthogonality_loss(shared)
        loss_l1=sum(torch.norm(dec,p=1) for dec in model.W_decs)
        loss=(loss_recon+params["lambda_cca"]*loss_cca+
              params["lambda_graph"]*loss_graph+
              params["lambda_ortho"]*loss_ortho+
              params["lambda_l1"]*loss_l1)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"全样本训练第 {epoch+1} epoch 出现非有限损失")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
        opt.step()
        scheduler.step(float(loss))
        hist.append({"epoch":epoch+1,"total_loss":float(loss.detach()),
                     "reconstruction_loss":float(loss_recon.detach())})
    model.eval()
    return model,pd.DataFrame(hist)
