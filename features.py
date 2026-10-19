"""Module ranking and cumulative-weight hub selection."""
from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd
from config import (
    CUMSUM_THRESH,
    ELBOW_MIN_N,
    FEATURE_MAX_N,
    FEATURE_MIN_N,
    FEATURE_SELECT_STRATEGY,
    MODALITY_NAMES,
    MOD_COLORS,
    TOP_N_MODULES_SHOW,
    ZSCORE_THRESH,
)


def _select_zscore(weights, z_thresh):
    mean = weights.mean(); std = weights.std() + 1e-10
    z    = (weights - mean) / std
    idx  = np.where(z > z_thresh)[0]
    return idx[np.argsort(weights[idx])[::-1]] if len(idx) > 0 else np.array([], dtype=int)


def _select_cumsum(weights, cum_thresh):
    sorted_idx = np.argsort(weights)[::-1]
    sorted_w   = weights[sorted_idx]
    cumsum     = np.cumsum(sorted_w) / (sorted_w.sum() + 1e-10)
    n_select   = int(np.searchsorted(cumsum, cum_thresh)) + 1
    return sorted_idx[:n_select]


def _select_elbow(weights, min_n):
    sorted_idx = np.argsort(weights)[::-1]
    n          = min(len(weights), FEATURE_MAX_N + 5)
    sorted_w   = weights[sorted_idx[:n]]
    if len(sorted_w) < 4:
        return sorted_idx[:len(sorted_w)]
    diff2    = np.diff(np.diff(sorted_w))
    elbow    = int(np.argmax(diff2)) + 2
    return sorted_idx[:max(min_n, elbow)]


def select_hub_features(weights, strategy=FEATURE_SELECT_STRATEGY,
                         z_thresh=ZSCORE_THRESH, cum_thresh=CUMSUM_THRESH,
                         elbow_min_n=ELBOW_MIN_N,
                         min_n=FEATURE_MIN_N, max_n=FEATURE_MAX_N):
    w = weights.copy().astype(float)
    if strategy == "zscore":
        idx  = _select_zscore(w, z_thresh)
        desc = f"Z-score(thresh={z_thresh})"
    elif strategy == "cumsum":
        idx  = _select_cumsum(w, cum_thresh)
        desc = f"CumSum(thresh={cum_thresh:.0%})"
    elif strategy == "elbow":
        idx  = _select_elbow(w, elbow_min_n)
        desc = f"Elbow(min={elbow_min_n})"
    elif strategy == "all":
        idx_z = _select_zscore(w, z_thresh)
        idx_c = _select_cumsum(w, cum_thresh)
        idx_e = _select_elbow(w, elbow_min_n)
        union = np.union1d(np.union1d(idx_z, idx_c), idx_e)
        idx   = union[np.argsort(w[union])[::-1]] if len(union) > 0 else np.array([], dtype=int)
        desc  = f"Union(Z={len(idx_z)}, CumSum={len(idx_c)}, Elbow={len(idx_e)})"
    else:
        raise ValueError(f"未知策略: {strategy}")
    if len(idx) < min_n:
        fallback = np.argsort(w)[::-1][:min_n]
        idx      = np.union1d(idx, fallback)
        idx      = idx[np.argsort(w[idx])[::-1]]
        desc    += f" → fallback to Top-{min_n}"
    return idx[:max_n].astype(int), desc


def extract_and_score_modules(model, z_shared, names_list,
                               sample_ids=None,
                               strategy=FEATURE_SELECT_STRATEGY):
    with torch.no_grad():
        W_sh_nn = torch.clamp(model.W_shared, min=1e-6)
        bases   = [torch.matmul(W_sh_nn, torch.clamp(W, min=1e-6)).cpu().numpy()
                   for W in model.W_decs]
        sample_scores = z_shared.cpu().numpy()

    num_mods          = len(bases)
    latent_dim        = sample_scores.shape[1]
    module_importance = sum(np.linalg.norm(b, axis=1) for b in bases)
    ranked_idx        = np.argsort(module_importance)[::-1]
    mod_labels = (MODALITY_NAMES[:num_mods]
                  if len(MODALITY_NAMES) >= num_mods
                  else [f"Mod{i+1}" for i in range(num_mods)])

    print("\n" + "=" * 65)
    print(f"  🏆  {num_mods}-MODALITY HUB FEATURES  [策略: {strategy.upper()}]")
    print("=" * 65)
    for rank_i, mod_idx in enumerate(ranked_idx[:3]):
        score = module_importance[mod_idx]
        print(f"\n⭐ [Rank {rank_i+1}] Latent Module {mod_idx+1} (Score: {score:.4f})")
        for m_idx, basis in enumerate(bases):
            w   = basis[mod_idx, :]
            idx, desc = select_hub_features(w, strategy=strategy)
            feats = [names_list[m_idx][j] for j in idx]
            print(f"   ➤ {mod_labels[m_idx]} [{len(feats)} features, {desc}]: "
                  f"{', '.join(feats)}")

    all_rows = []
    for rank_i, mod_idx in enumerate(ranked_idx):
        score = module_importance[mod_idx]
        for m_idx, basis in enumerate(bases):
            w   = basis[mod_idx, :]
            idx, desc = select_hub_features(w, strategy=strategy)
            for feat_rank, fidx in enumerate(idx):
                all_rows.append({
                    "global_rank": rank_i+1, "module_index": mod_idx+1,
                    "module_score": round(float(score), 6),
                    "modality": mod_labels[m_idx], "select_strategy": desc,
                    "n_features_selected": len(idx),
                    "feature_rank": feat_rank+1,
                    "feature_name": names_list[m_idx][fidx],
                    "weight": round(float(w[fidx]), 6),
                })
    pd.DataFrame(all_rows).to_csv(
        "hub_features_all_modules.csv", index=False, encoding="utf-8-sig")
    print("\n  ✅  已导出: hub_features_all_modules.csv")

    os.makedirs("module_features", exist_ok=True)
    for rank_i, mod_idx in enumerate(ranked_idx[:TOP_N_MODULES_SHOW]):
        score = module_importance[mod_idx]
        rows  = []
        for m_idx, basis in enumerate(bases):
            w = basis[mod_idx, :]
            for strat in ["zscore", "cumsum", "elbow"]:
                idx_s, desc_s = select_hub_features(w, strategy=strat)
                for feat_rank, fidx in enumerate(idx_s):
                    rows.append({
                        "modality": mod_labels[m_idx], "strategy": strat,
                        "strategy_detail": desc_s, "feature_rank": feat_rank+1,
                        "feature_name": names_list[m_idx][fidx],
                        "weight": round(float(w[fidx]), 6),
                        "module_score": round(float(score), 6),
                    })
        pd.DataFrame(rows).to_csv(
            f"module_features/hub_module_{mod_idx+1}_rank{rank_i+1}.csv",
            index=False, encoding="utf-8-sig")
    print(f"  ✅  已导出 Top {TOP_N_MODULES_SHOW} 模块特征至 module_features/")

    col_names = [f"Module_{k+1}" for k in range(latent_dim)]
    df_scores = pd.DataFrame(sample_scores, columns=col_names)
    if sample_ids is not None:
        df_scores.insert(0, "sample_id", sample_ids)
    df_scores.to_csv("sample_module_scores.csv", index=False, encoding="utf-8-sig")
    print("  ✅  已导出: sample_module_scores.csv")

    count_rows = []
    for rank_i, mod_idx in enumerate(ranked_idx[:TOP_N_MODULES_SHOW]):
        for m_idx, basis in enumerate(bases):
            w = basis[mod_idx, :]
            for strat in ["zscore", "cumsum", "elbow"]:
                idx_s, desc_s = select_hub_features(w, strategy=strat)
                count_rows.append({
                    "module_index": mod_idx+1, "global_rank": rank_i+1,
                    "modality": mod_labels[m_idx], "strategy": strat,
                    "n_selected": len(idx_s), "strategy_detail": desc_s,
                })
    pd.DataFrame(count_rows).to_csv(
        "feature_selection_counts.csv", index=False, encoding="utf-8-sig")
    print("  ✅  已导出: feature_selection_counts.csv")

    return bases, module_importance, sample_scores, ranked_idx


def save_best_module_expression(bases, ranked_idx, names_list, raw_dfs,
                                 sample_ids,
                                 strategy=FEATURE_SELECT_STRATEGY):
    """
    将最佳模块（ranked_idx[0]）在各模态选中的特征，
    拼接为一个 [n_samples × n_features] 的原始表达矩阵并导出为 CSV。
    """
    num_mods     = len(bases)
    best_mod_idx = int(ranked_idx[0])
    mod_labels   = (MODALITY_NAMES[:num_mods]
                    if len(MODALITY_NAMES) >= num_mods
                    else [f"Mod{i+1}" for i in range(num_mods)])

    expr_dfs  = []
    meta_rows = []

    for m_idx in range(num_mods):
        df_raw = raw_dfs[m_idx]
        w      = bases[m_idx][best_mod_idx, :]
        feat_idx, desc = select_hub_features(w, strategy=strategy)
        feat_names     = [names_list[m_idx][j] for j in feat_idx]
        weights_sel    = w[feat_idx]

        available   = [f for f in feat_names if f in df_raw.columns]
        unavailable = [f for f in feat_names if f not in df_raw.columns]
        if unavailable:
            print(f"  ⚠️  {mod_labels[m_idx]}: "
                  f"{len(unavailable)} 个特征在原始数据中未找到，已跳过")

        if not available:
            continue

        df_sub = df_raw.loc[sample_ids, available].copy()
        df_sub.columns = [f"{mod_labels[m_idx]}__{c}" for c in available]
        expr_dfs.append(df_sub)

        for rank_i, fname in enumerate(feat_names):
            if fname in available:
                meta_rows.append({
                    "modality":          mod_labels[m_idx],
                    "feature_name":      fname,
                    "column_name":       f"{mod_labels[m_idx]}__{fname}",
                    "feature_rank":      rank_i + 1,
                    "weight":            round(float(weights_sel[rank_i]), 6),
                    "module_index":      best_mod_idx + 1,
                    "select_strategy":   desc,
                })

    if not expr_dfs:
        print("  ⚠️  无有效特征，表达矩阵未导出。")
        return None

    expr_matrix = pd.concat(expr_dfs, axis=1)
    expr_matrix.index.name = "sample_id"
    expr_matrix.to_csv("best_module_expression_matrix.csv", encoding="utf-8-sig")
    print(f"\n  ✅  已导出: best_module_expression_matrix.csv  "
          f"（{expr_matrix.shape[0]} 样本 × {expr_matrix.shape[1]} 特征）")

    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv("best_module_feature_meta.csv", index=False, encoding="utf-8-sig")
    print(f"  ✅  已导出: best_module_feature_meta.csv  "
          f"（{len(df_meta)} 条特征元信息）")

    return expr_matrix


def _get_best_module_feature_matrix(bases, ranked_idx, names_list, raw_dfs,
                                    sample_ids, strategy=FEATURE_SELECT_STRATEGY):
    """
    提取最佳模块各模态选中特征，返回
      expr_matrix : DataFrame [n_samples × n_selected_features]，列名格式 MOD__feat
      feat_meta_df: DataFrame，记录每列所属模态、原始特征名、模块权重及对应颜色
    """
    num_mods     = len(bases)
    best_mod_idx = int(ranked_idx[0])
    mod_labels   = (MODALITY_NAMES[:num_mods]
                    if len(MODALITY_NAMES) >= num_mods
                    else [f"Mod{i+1}" for i in range(num_mods)])

    expr_dfs  = []
    meta_rows = []

    for m_idx in range(num_mods):
        df_raw = raw_dfs[m_idx]
        w      = bases[m_idx][best_mod_idx, :]
        feat_idx, desc = select_hub_features(w, strategy=strategy)
        feat_names     = [names_list[m_idx][j] for j in feat_idx]
        weights_sel    = w[feat_idx]

        available = [f for f in feat_names if f in df_raw.columns]
        if not available:
            continue

        df_sub = df_raw.loc[sample_ids, available].copy()
        col_map = {f: f"{mod_labels[m_idx]}__{f}" for f in available}
        df_sub.rename(columns=col_map, inplace=True)
        expr_dfs.append(df_sub)

        for rank_i, fname in enumerate(available):
            orig_rank = feat_names.index(fname)
            meta_rows.append({
                "col_name":  f"{mod_labels[m_idx]}__{fname}",
                "modality":  mod_labels[m_idx],
                "feature":   fname,
                "weight":    float(weights_sel[orig_rank]),
                "mod_color": MOD_COLORS[m_idx % len(MOD_COLORS)],
            })

    if not expr_dfs:
        return None, None

    expr_matrix  = pd.concat(expr_dfs, axis=1)
    feat_meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    return expr_matrix, feat_meta_df


def _cv_model_bases(model):
    with torch.no_grad():
        shared=torch.clamp(model.W_shared,min=1e-6)
        return [(shared@torch.clamp(dec,min=1e-6)).detach().cpu().numpy()
                for dec in model.W_decs]


def _cv_module_ranking(bases):
    importance=sum(np.linalg.norm(b,axis=1) for b in bases)
    order=np.argsort(importance)[::-1]
    return importance,order


def _cv_select_fold_module(bases,names,args,fold):
    """Select ONE top module; 80% hub rule and Top features are fold-local."""
    importance,ranking=_cv_module_ranking(bases)
    module=int(ranking[0])
    hub_rows, top_rows=[],[]
    hub_sets,top_sets={},{}
    for m,modality in enumerate(MODALITY_NAMES):
        weights=np.asarray(bases[m][module],dtype=float)
        cap=len(weights) if args.max_hub_features==0 else args.max_hub_features
        chosen,description=select_hub_features(
            weights,strategy="cumsum",cum_thresh=args.cumsum_threshold,
            min_n=FEATURE_MIN_N,max_n=cap)
        top=chosen[:args.top_n]
        total=float(np.sum(weights))
        cumulative=np.cumsum(weights[chosen])/max(total,1e-30)
        hub_sets[modality]=set(str(names[m][int(i)]) for i in chosen)
        top_sets[modality]=set(str(names[m][int(i)]) for i in top)
        for rank,j in enumerate(chosen,1):
            row=dict(fold=fold,leading_module_1based=module+1,
                     modality=modality,feature=str(names[m][int(j)]),
                     feature_column=int(j),weight=float(weights[int(j)]),
                     rank=rank,cumulative_weight_fraction=float(cumulative[rank-1]),
                     in_top_n=bool(rank<=len(top)),
                     requested_cumsum=args.cumsum_threshold,
                     reached_threshold=bool(cumulative[-1]>=args.cumsum_threshold-1e-9),
                     selection_limit=cap)
            hub_rows.append(row)
            if rank<=len(top): top_rows.append(dict(row))
        if cumulative.size and cumulative[-1]<args.cumsum_threshold-1e-9:
            print(f"  [NOTE] Fold {fold}, {modality}: "
                  f"{len(chosen)} features selected (cap={cap}), "
                  f"cumulative weight only {cumulative[-1]:.1%} < "
                  f"{args.cumsum_threshold:.0%}. Use --max-hub-features 0 "
                  "to lift the legacy feature cap.")
    ranking_rows=[
        {"fold":fold,"rank":i+1,"module_1based":int(k)+1,
         "importance":float(importance[k])}
        for i,k in enumerate(ranking)]
    return dict(module=module,hub_rows=hub_rows,top_rows=top_rows,
                ranking_rows=ranking_rows,hub_sets=hub_sets,
                top_sets=top_sets,importance=importance,ranking=ranking)
