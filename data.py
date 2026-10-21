"""Input loading, sample alignment and training-only normalization."""
from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path as _CVPath
from config import (
    SEED,
    SPLIT_DATA_DIR,
    VAL_RATIO,
)


def load_omics_data_dynamic(file_paths):
    num_mods = len(file_paths)
    if num_mods < 2:
        raise ValueError("至少需要 2 个模态的输入文件。")
    for f in file_paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"数据文件不存在: {f}")

    dfs = [pd.read_csv(f, index_col=0) for f in file_paths]
    common_samples = dfs[0].index
    for i in range(1, num_mods):
        common_samples = common_samples.intersection(dfs[i].index)
    if len(common_samples) == 0:
        raise ValueError("所有模态之间没有共同的样本 ID。")

    dfs     = [df.loc[common_samples] for df in dfs]
    dims    = [df.shape[1] for df in dfs]
    raw_dfs = [df.copy() for df in dfs]
    tensors = [torch.tensor(df.values, dtype=torch.float32) for df in dfs]

    for i in range(num_mods):
        if i == 0:
            tensors[i] = tensors[i] / 2.0
        else:
            X_min = tensors[i].min(dim=0)[0]
            X_max = tensors[i].max(dim=0)[0]
            tensors[i] = (tensors[i] - X_min) / (X_max - X_min + 1e-8)

    proc_dfs = [
        pd.DataFrame(tensors[i].numpy(), index=common_samples, columns=dfs[i].columns)
        for i in range(num_mods)
    ]
    names      = [df.columns.tolist() for df in dfs]
    sample_ids = list(common_samples)
    return tensors, proc_dfs, raw_dfs, names, sample_ids, len(common_samples), dims


def train_val_split(tensors, val_ratio=0.2, seed=SEED):
    n     = tensors[0].shape[0]
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx   = torch.tensor(idx[:n_val],  dtype=torch.long)
    train_idx = torch.tensor(idx[n_val:],  dtype=torch.long)
    return ([t[train_idx] for t in tensors],
            [t[val_idx]   for t in tensors],
            idx[n_val:].tolist(),
            idx[:n_val].tolist())


def save_split_data(proc_dfs, raw_dfs, sample_ids,
                    train_idx_list, val_idx_list,
                    mod_names, out_dir=SPLIT_DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    num_mods = len(proc_dfs)
    n_train  = len(train_idx_list)
    n_val    = len(val_idx_list)
    n_total  = len(sample_ids)
    train_ids = [sample_ids[i] for i in train_idx_list]
    val_ids   = [sample_ids[i] for i in val_idx_list]

    pd.DataFrame({"sample_id": train_ids, "split": ["train"] * n_train,
                  "global_index": train_idx_list}
                 ).to_csv(f"{out_dir}/train_sample_ids.csv",
                          index=False, encoding="utf-8-sig")
    pd.DataFrame({"sample_id": val_ids, "split": ["val"] * n_val,
                  "global_index": val_idx_list}
                 ).to_csv(f"{out_dir}/val_sample_ids.csv",
                          index=False, encoding="utf-8-sig")

    split_labels = [""] * n_total
    for i in train_idx_list: split_labels[i] = "train"
    for i in val_idx_list:   split_labels[i] = "val"
    pd.DataFrame({"sample_id": sample_ids, "split": split_labels,
                  "global_index": list(range(n_total))}
                 ).to_csv(f"{out_dir}/all_sample_split_labels.csv",
                          index=False, encoding="utf-8-sig")

    for m_idx, (proc_df, raw_df) in enumerate(zip(proc_dfs, raw_dfs)):
        mname = mod_names[m_idx] if m_idx < len(mod_names) else f"Mod{m_idx+1}"
        proc_df.iloc[train_idx_list].to_csv(
            f"{out_dir}/train_{mname}_preprocessed.csv", encoding="utf-8-sig")
        proc_df.iloc[val_idx_list].to_csv(
            f"{out_dir}/val_{mname}_preprocessed.csv", encoding="utf-8-sig")
        raw_df.iloc[train_idx_list].to_csv(
            f"{out_dir}/train_{mname}_raw.csv", encoding="utf-8-sig")
        raw_df.iloc[val_idx_list].to_csv(
            f"{out_dir}/val_{mname}_raw.csv", encoding="utf-8-sig")

    summary_lines = [
        "=" * 55, "  DeepCAN-NMF  训练/验证集分割摘要", "=" * 55,
        f"  随机种子 (SEED)       : {SEED}",
        f"  验证集比例 (VAL_RATIO): {VAL_RATIO}",
        f"  全量样本数            : {n_total}",
        f"  训练集样本数          : {n_train}",
        f"  验证集样本数          : {n_val}", "",
        "  模态信息：",
    ]
    for m_idx, df in enumerate(proc_dfs):
        mname = mod_names[m_idx] if m_idx < len(mod_names) else f"Mod{m_idx+1}"
        summary_lines.append(f"    {mname:<8s}: {df.shape[1]} 个特征")
    summary_lines += ["", "=" * 55]
    with open(f"{out_dir}/split_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n  ✅  训练/验证集数据已导出至 {out_dir}/")
    print(f"      训练集: {n_train} 样本  验证集: {n_val} 样本")


def save_latent_and_recon(model, tensors, sample_ids,
                          train_idx_list, val_idx_list,
                          mod_names, latent_dim,
                          out_dir=SPLIT_DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    num_mods  = len(tensors)
    model.eval()
    with torch.no_grad():
        z_list, z_shared, recons, _ = model(tensors)

    z_shared_np = z_shared.cpu().numpy()
    z_list_np   = [z.cpu().numpy() for z in z_list]
    recons_np   = [r.cpu().numpy() for r in recons]
    train_ids   = [sample_ids[i] for i in train_idx_list]
    val_ids     = [sample_ids[i] for i in val_idx_list]
    z_cols      = [f"z_{k+1}" for k in range(latent_dim)]

    for split_name, split_idx, ids in [
        ("train", train_idx_list, train_ids),
        ("val",   val_idx_list,   val_ids),
    ]:
        df_z = pd.DataFrame(z_shared_np[split_idx, :], columns=z_cols)
        df_z.insert(0, "sample_id", ids)
        df_z.to_csv(f"{out_dir}/{split_name}_latent_z_shared.csv",
                    index=False, encoding="utf-8-sig")

    for m_idx in range(num_mods):
        mname    = mod_names[m_idx] if m_idx < len(mod_names) else f"Mod{m_idx+1}"
        zm_cols  = [f"z_{k+1}" for k in range(z_list_np[m_idx].shape[1])]
        n_feats  = recons_np[m_idx].shape[1]
        rec_cols = [f"feat_{j+1}" for j in range(n_feats)]
        for split_name, split_idx, ids in [
            ("train", train_idx_list, train_ids),
            ("val",   val_idx_list,   val_ids),
        ]:
            df_zm = pd.DataFrame(z_list_np[m_idx][split_idx, :], columns=zm_cols)
            df_zm.insert(0, "sample_id", ids)
            df_zm.to_csv(f"{out_dir}/{split_name}_latent_z_{mname}.csv",
                         index=False, encoding="utf-8-sig")
            df_rec = pd.DataFrame(recons_np[m_idx][split_idx, :], columns=rec_cols)
            df_rec.insert(0, "sample_id", ids)
            df_rec.to_csv(f"{out_dir}/{split_name}_reconstruction_{mname}.csv",
                          index=False, encoding="utf-8-sig")

    print(f"\n  ✅  潜在表示与重建值已导出至 {out_dir}/")


def _cv_resolve(data_dir, filename, gene_fallback=False):
    p = _CVPath(filename)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    folder = _CVPath(data_dir)
    candidates = [folder / filename, folder / "输入文件" / filename, p]
    if gene_fallback and filename == "d.csv":
        candidates.extend([folder / "d_select_degs.csv",
                           folder / "输入文件" / "d_select_degs.csv"])
    for attempt in candidates:
        if attempt.is_file():
            return attempt.resolve()
    raise FileNotFoundError(
        f"找不到输入文件 {filename!r}。搜索位置: " + ", ".join(map(str,candidates)))


def _cv_canonical_label(value):
    """Treat integer-looking 0/1 labels equally whether CSV reads 0, 0.0 or '0'."""
    if pd.isna(value):
        return None
    string=str(value).strip()
    try:
        number=float(string)
    except (TypeError,ValueError):
        return string
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return string


def _cv_load_labeled(args):
    """Align all omics AND labels first. Only labeled subjects enter CV/final."""
    paths = [_cv_resolve(args.data_dir, name, gene_fallback=(j == 3))
             for j, name in enumerate(args.files)]
    label_file = _cv_resolve(args.data_dir, args.labels)
    dfs = [pd.read_csv(p, index_col=0) for p in paths]
    for source, df in zip(paths, dfs):
        df.index = pd.Index(df.index.map(lambda t: str(t).strip()))
        if not df.index.is_unique:
            raise ValueError(f"重复样本ID: {source}")
        if not df.columns.is_unique:
            raise ValueError(f"重复特征名: {source}")
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index, sort=False)
    label_df = pd.read_csv(label_file, index_col=0)
    label_df.index = pd.Index(label_df.index.map(lambda t: str(t).strip()))
    if not label_df.index.is_unique:
        raise ValueError("label.csv 存在重复样本ID")
    if args.label_col in label_df.columns:
        label_col = args.label_col
    elif len(label_df.columns) == 1:
        label_col = label_df.columns[0]
        print(f"  [NOTE] 标签列 {args.label_col} 不存在，使用唯一列 {label_col}")
    else:
        raise KeyError(f"未找到标签列 {args.label_col!r}: {label_df.columns.tolist()}")
    raw_labels = label_df[label_col].reindex(common)
    norm_labels = raw_labels.map(_cv_canonical_label)
    positive=_cv_canonical_label(args.positive)
    negative=_cv_canonical_label(args.negative)
    keep = norm_labels.isin([positive,negative])
    ids = common[keep]
    y = (norm_labels.loc[ids].to_numpy() == positive).astype(np.int64)
    if args.expected_n and len(ids) != args.expected_n:
        raise ValueError(f"预计 {args.expected_n} 位有标签受试者，但匹配到 {len(ids)} 位。"
                         "请核对ID、输入文件或用 --expected-n 0 关闭限制。")
    if len(ids)<10 or min(np.bincount(y,minlength=2))<3:
        raise ValueError("每类样本过少，无法进行分层留出验证。")
    matrices = []
    for df in dfs:
        z = df.loc[ids].apply(pd.to_numeric,errors="coerce").to_numpy(
            dtype=np.float32)
        z[~np.isfinite(z)] = np.nan
        matrices.append(z)
    names = [[str(col) for col in df.columns] for df in dfs]
    print(f"仅使用有标签四模态共同样本: {len(ids)} 位；"
          f"标签 {args.negative} = {int((y==0).sum())}, "
          f"标签 {args.positive} = {int((y==1).sum())}")
    print("实际输入:", *[str(p.name) for p in paths], "标签:",label_file.name)
    return dict(ids=np.array(ids,dtype=str), y=y, X=matrices, names=names,
                input_paths=[str(p) for p in paths], label_path=str(label_file),
                common_unlabeled_count=int(len(common)-len(ids)))


class _CVNormalizer:
    """Per-feature imputation and min-max fit STRICTLY on fitting records."""
    def fit(self, X_list, fit_indices):
        self.stats=[]
        for m, raw in enumerate(X_list):
            x=raw[fit_indices].astype(np.float64,copy=True)
            with np.errstate(all="ignore"):
                median=np.nanmedian(x,axis=0)
            median=np.where(np.isfinite(median),median,0.0)
            x=np.where(np.isfinite(x),x,median)
            if m == 0:  # SNP dosage: same /2 scaling as uploaded aa.py
                self.stats.append(("SNP",median,None,None))
            else:
                minimum=np.min(x,axis=0)
                maximum=np.max(x,axis=0)
                self.stats.append(("MINMAX",median,minimum,
                                   np.maximum(maximum-minimum,1e-8)))
        return self

    def transform(self,X_list,indices):
        out=[]
        for raw,(kind,median,vmin,span) in zip(X_list,self.stats):
            x=raw[indices].astype(np.float64,copy=True)
            x=np.where(np.isfinite(x),x,median)
            if kind=="SNP":
                x=x/2.0
            else:
                x=(x-vmin)/span
                x=np.clip(x,0.0,1.0)
            if not np.isfinite(x).all():
                raise ValueError("归一化后仍有非有限值")
            out.append(torch.tensor(x,dtype=torch.float32))
        return out
