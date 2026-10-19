"""Descriptive figures and ROC visualization."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interpolate as scipy_interp
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice
import plotly.io as pio
from sklearn.metrics import roc_auc_score as _cv_roc_auc_score
from sklearn.metrics import roc_curve as _cv_roc_curve
from config import (
    COLOR_RECON_TRAIN,
    COLOR_RECON_VAL,
    COLOR_TRAIN,
    COLOR_VAL,
    CROSSMODAL_CORR_THRESH,
    CROSSMODAL_PVAL_THRESH,
    CUMSUM_THRESH,
    ELBOW_MIN_N,
    FEATURE_SELECT_STRATEGY,
    LABEL_COL,
    LABEL_FILE,
    MODALITY_NAMES,
    MOD_COLORS,
    POSITIVE_CLASS,
    ROC_MAX_LEGEND,
    SCI_RC,
    SEED,
    TOP_N_MODULES_SHOW,
    ZSCORE_THRESH,
    _ROC_PALETTE,
)
from features import (
    _get_best_module_feature_matrix,
    select_hub_features,
)


import warnings
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _save_pdf(fig, stem: str, dpi: int = 300) -> None:
    """统一保存为 PDF（dpi 控制栅格化元素精度）并关闭 figure。"""
    path = f"{stem}.pdf"
    fig.savefig(path, format="pdf", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    print(f"  ✅  已保存: {path}")
    plt.close(fig)


def load_sample_labels(label_file, sample_ids,
                        label_col=LABEL_COL,
                        positive_class=POSITIVE_CLASS):
    """
    从 CSV 文件加载二分类标签，与 sample_ids 对齐后返回。
    """
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"标签文件不存在: {label_file}")

    df_lb = pd.read_csv(label_file, index_col=0)

    if label_col not in df_lb.columns:
        label_col_used = df_lb.columns[0]
        print(f"  ⚠️  标签列 '{label_col}' 不存在，自动使用第一列: '{label_col_used}'")
    else:
        label_col_used = label_col

    common_ids = [s for s in sample_ids if s in df_lb.index]
    missing    = len(sample_ids) - len(common_ids)
    if missing > 0:
        print(f"  ⚠️  有 {missing} 个样本在标签文件中未找到，已跳过")
    if len(common_ids) == 0:
        raise ValueError("标签文件与数据集无共同样本，请检查 sample ID 格式。")

    labels_raw = df_lb.loc[common_ids, label_col_used]
    unique_vals = sorted(labels_raw.unique().tolist(), key=str)

    if len(unique_vals) < 2:
        raise ValueError(f"标签列仅含 1 个类别，无法进行 ROC 分析。")
    if len(unique_vals) > 2:
        print(f"  ⚠️  检测到 {len(unique_vals)} 个类别 {unique_vals}，"
              "将进行 one-vs-rest（正类由 POSITIVE_CLASS 指定或取字典序最后一个）。")

    if positive_class is not None and positive_class in unique_vals:
        pos_name = positive_class
        neg_name = [v for v in unique_vals if str(v) != str(positive_class)][0]
    else:
        pos_name = unique_vals[-1]
        neg_name = unique_vals[0]

    label_map = {v: (1 if str(v) == str(pos_name) else 0) for v in unique_vals}
    y_aligned = labels_raw.map(label_map).values.astype(int)

    print(f"  📌  标签编码: {label_map}  "
          f"（正类={pos_name}, 负类={neg_name}）  "
          f"共 {len(common_ids)} 个对齐样本")

    return y_aligned, common_ids, label_map, str(pos_name), str(neg_name)


def _compute_mean_roc(fprs, tprs):
    """将多条 ROC 曲线在公共 FPR 网格上插值，计算均值 ± 标准差。"""
    mean_fpr = np.linspace(0, 1, 300)
    interp_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        sort_idx = np.argsort(fpr)
        fpr_s, tpr_s = fpr[sort_idx], tpr[sort_idx]
        fpr_s, unique_idx = np.unique(fpr_s, return_index=True)
        tpr_s = tpr_s[unique_idx]
        f_interp = scipy_interp.interp1d(fpr_s, tpr_s,
                                          kind="linear",
                                          bounds_error=False,
                                          fill_value=(tpr_s[0], tpr_s[-1]))
        interp_tprs.append(f_interp(mean_fpr))
    interp_tprs = np.array(interp_tprs)
    return mean_fpr, interp_tprs.mean(axis=0), interp_tprs.std(axis=0)


def plot_roc_best_module(bases, ranked_idx, names_list, raw_dfs,
                          sample_ids, label_file=LABEL_FILE,
                          strategy=FEATURE_SELECT_STRATEGY):
    """
    为最重要模块（ranked_idx[0]）各模态的选中特征绘制 SCI 级 ROC 曲线。
    每个模态单独保存为独立 PDF：panel_ROC_best_module_{MOD}.pdf
    """
    try:
        y, common_ids, label_map, pos_name, neg_name = \
            load_sample_labels(label_file, sample_ids)
    except Exception as e:
        print(f"\n  ⚠️  ROC 曲线跳过（标签加载失败）: {e}")
        return

    num_mods     = len(bases)
    best_mod_idx = int(ranked_idx[0])
    mod_labels   = (MODALITY_NAMES[:num_mods]
                    if len(MODALITY_NAMES) >= num_mods
                    else [f"Mod{i+1}" for i in range(num_mods)])

    for m_idx in range(num_mods):
        _apply_sci_theme()
        fig, ax = plt.subplots(figsize=(9.5, 8.2))

        df_raw = raw_dfs[m_idx]

        w = bases[m_idx][best_mod_idx, :]
        feat_idx, desc = select_hub_features(w, strategy=strategy)
        feat_names     = [names_list[m_idx][j] for j in feat_idx]

        roc_results = []
        for fname in feat_names:
            if fname not in df_raw.columns:
                continue
            vals = df_raw.loc[common_ids, fname].values.astype(float)
            nan_mask = np.isnan(vals)
            if nan_mask.all():
                continue
            vals[nan_mask] = np.nanmedian(vals)

            fpr, tpr, _ = roc_curve(y, vals, drop_intermediate=True)
            roc_auc     = auc(fpr, tpr)
            if roc_auc < 0.5:
                fpr, tpr   = 1.0 - fpr[::-1], 1.0 - tpr[::-1]
                roc_auc    = 1.0 - roc_auc
            roc_results.append((fname, fpr, tpr, roc_auc))

        if not roc_results:
            ax.text(0.5, 0.5, "No valid features",
                    ha="center", va="center", transform=ax.transAxes, fontsize=16)
        else:
            roc_results.sort(key=lambda x: -x[3])
            n_feat = len(roc_results)

            show_in_legend = min(n_feat, ROC_MAX_LEGEND)
            for fi, (fname, fpr, tpr, rauc) in enumerate(roc_results):
                if fi < show_in_legend:
                    color = _ROC_PALETTE[fi % len(_ROC_PALETTE)]
                    lw, alpha, zorder = 1.9, 0.88, 3
                    label = f"{fname}  (AUC = {rauc:.3f})"
                else:
                    color = "#bbbbbb"
                    lw, alpha, zorder = 1.0, 0.45, 2
                    label = None
                ax.plot(fpr, tpr, color=color, lw=lw, alpha=alpha,
                        zorder=zorder, label=label)

            all_fprs = [r[1] for r in roc_results]
            all_tprs = [r[2] for r in roc_results]
            mean_fpr, mean_tpr, std_tpr = _compute_mean_roc(all_fprs, all_tprs)
            mean_auc = np.mean([r[3] for r in roc_results])
            std_auc  = np.std( [r[3] for r in roc_results])

            ax.plot(mean_fpr, mean_tpr,
                    color="#111111", lw=3.0, linestyle="-", zorder=5,
                    label=f"Mean ROC  (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
            ax.fill_between(mean_fpr,
                             np.clip(mean_tpr - std_tpr, 0, 1),
                             np.clip(mean_tpr + std_tpr, 0, 1),
                             color="#444444", alpha=0.12, zorder=4,
                             label="± 1 SD")
            ax.plot([0, 1], [0, 1], linestyle="--", lw=1.4,
                    color="#999999", alpha=0.7, zorder=1, label="Chance (AUC = 0.500)")

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.05])
            ax.set_xlabel("1 – Specificity  (False Positive Rate)", fontsize=20)
            ax.set_ylabel("Sensitivity  (True Positive Rate)", fontsize=20)
            ax.set_title(
                f"{mod_labels[m_idx]}  —  Best Module M{best_mod_idx + 1}\n"
                f"({n_feat} features, {desc})\n"
                f"Positive: {pos_name}   vs   Negative: {neg_name}",
                fontsize=21, pad=10,
            )
            ax.tick_params(axis="both", labelsize=17)

            handles, labels_leg = ax.get_legend_handles_labels()
            priority   = ["Mean ROC", "± 1 SD", "Chance"]
            front_h, front_l, rest_h, rest_l = [], [], [], []
            for h, lb in zip(handles, labels_leg):
                if any(lb.startswith(p) for p in priority):
                    front_h.append(h); front_l.append(lb)
                else:
                    rest_h.append(h); rest_l.append(lb)
            final_h = front_h + rest_h
            final_l = front_l + rest_l

            leg = ax.legend(
                final_h, final_l,
                loc="lower right",
                fontsize=13,
                frameon=True, framealpha=0.92, edgecolor="#cccccc",
                title=(f"Features (Top {show_in_legend} shown)"
                       if n_feat > show_in_legend else "Features"),
                title_fontsize=14,
                handlelength=2.0, handletextpad=0.6,
                borderpad=0.8, labelspacing=0.4,
            )
            leg.get_frame().set_linewidth(0.8)

            if n_feat > show_in_legend:
                ax.text(0.02, 0.02,
                        f"Remaining {n_feat - show_in_legend} features shown as grey lines",
                        transform=ax.transAxes, fontsize=12, color="#777777",
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.7, edgecolor="none"))

            sns.despine(ax=ax)

        plt.tight_layout()
        stem = f"panel_ROC_best_module_{mod_labels[m_idx]}"
        _save_pdf(fig, stem)


def _plot_panel_I_block_corr(corr_df, pval_df, feat_meta_df,
                              mod_labels, best_mod_idx,
                              pval_thresh=CROSSMODAL_PVAL_THRESH,
                              corr_thresh=CROSSMODAL_CORR_THRESH):
    """分块相关热图：特征按模态分组排列，块边框颜色区分模态，• 标注显著互作。"""
    _apply_sci_theme()

    # 按模态顺序排列特征列
    ordered_cols = []
    mod_blocks   = []   # (start, end, mod_name, color)
    for m_idx, mod in enumerate(mod_labels):
        mod_cols = (feat_meta_df[feat_meta_df["modality"] == mod]["col_name"]
                    .tolist())
        mod_cols = [c for c in mod_cols if c in corr_df.columns]
        start = len(ordered_cols)
        ordered_cols.extend(mod_cols)
        end = len(ordered_cols)
        if end > start:
            mod_blocks.append((start, end, mod, MOD_COLORS[m_idx % len(MOD_COLORS)]))

    if len(ordered_cols) < 2:
        print("  ⚠️  Panel I 跳过：有效特征数不足")
        return

    corr_ord = corr_df.loc[ordered_cols, ordered_cols]
    pval_ord = pval_df.loc[ordered_cols, ordered_cols]
    n        = len(ordered_cols)

    # 显著性掩码（跨模态 + |r| ≥ thresh + p ≤ thresh）
    sig_mask = (pval_ord.values < pval_thresh) & (np.abs(corr_ord.values) >= corr_thresh)
    np.fill_diagonal(sig_mask, False)

    # 动态图幅
    cell   = np.clip(18.0 / n, 0.32, 0.68)
    fig_w  = max(10, n * cell + 3.5)
    fig_h  = max(9,  n * cell + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    short_labels = [c.split("__", 1)[-1] if "__" in c else c for c in ordered_cols]

    # 主热图
    sns.heatmap(
        corr_ord, ax=ax,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.0,
        cbar_kws={"label": "Pearson r", "shrink": 0.72, "pad": 0.02},
        annot=False,
    )

    # 显著性标记（白色小点）
    for i in range(n):
        for j in range(n):
            if sig_mask[i, j]:
                ax.text(j + 0.5, i + 0.5, "•",
                        ha="center", va="center",
                        fontsize=max(6, int(10 - n / 12)),
                        color="white", fontweight="bold")

    # 模态分块边框
    for start, end, mod_name, color in mod_blocks:
        rect = plt.Rectangle(
            (start, start), end - start, end - start,
            fill=False, edgecolor=color, lw=3.2, zorder=10, clip_on=False,
        )
        ax.add_patch(rect)

    # 顶部彩色模态标签条
    bar_h = max(0.5, n * 0.04)
    for m_idx, (start, end, mod_name, color) in enumerate(mod_blocks):
        ax.add_patch(plt.Rectangle(
            (start, -bar_h * 1.5), end - start, bar_h * 1.2,
            color=color, zorder=11, clip_on=False,
        ))
        ax.text(
            (start + end) / 2, -bar_h * 0.9, mod_name,
            ha="center", va="center",
            fontsize=max(9, 15 - n // 6),
            color="white", fontweight="bold", zorder=12, clip_on=False,
        )

    tick_fs = max(7, 15 - n // 6)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=14)
    ax.collections[0].colorbar.set_label("Pearson r", fontsize=16)
    ax.set_title(
        f"Cross-modal Feature Correlation Heatmap  —  Best Module M{best_mod_idx + 1}\n"
        f"(colored borders = modality blocks;  • = |r| ≥ {corr_thresh}, p < {pval_thresh})",
        fontsize=20, pad=20,
    )
    plt.tight_layout()
    _save_pdf(fig, "panel_I_crossmodal_block_corr")


def _plot_panel_J_scatter_matrix(expr_z, feat_meta_df, mod_labels, best_mod_idx):
    """
    各模态选中特征 z-score 均值向量的两两散点矩阵：
    对角线为分布直方图，非对角线为散点+OLS 回归线+95% CI，标注 Pearson r 与 p 值。
    """
    from scipy import stats as scipy_stats
    _apply_sci_theme()

    num_mods  = len(mod_labels)
    n_samples = expr_z.shape[0]

    # 各模态均值向量
    mod_means = {}
    for m_idx, mod in enumerate(mod_labels):
        cols = feat_meta_df[feat_meta_df["modality"] == mod]["col_name"].tolist()
        cols = [c for c in cols if c in expr_z.columns]
        mod_means[mod] = expr_z[cols].mean(axis=1).values if cols else np.zeros(n_samples)

    fig, axes = plt.subplots(
        num_mods, num_mods,
        figsize=(4.8 * num_mods, 4.5 * num_mods),
        squeeze=False,
    )

    for i, mod_i in enumerate(mod_labels):
        for j, mod_j in enumerate(mod_labels):
            ax   = axes[i][j]
            x    = mod_means[mod_j]
            y    = mod_means[mod_i]
            ci   = MOD_COLORS[i % len(MOD_COLORS)]

            if i == j:
                # 对角：密度直方图
                ax.hist(x, bins=16, color=ci, alpha=0.72, edgecolor="none")
                ax.set_title(mod_i, fontsize=17, fontweight="bold", color=ci, pad=6)
            else:
                # 非对角：散点 + 回归线 + CI
                r_val, p_val = scipy_stats.pearsonr(x, y)
                ax.scatter(x, y, s=26, alpha=0.62, color=ci,
                           edgecolors="none", linewidths=0, zorder=3)

                # OLS 回归
                slope, intercept = np.polyfit(x, y, 1)
                x_lin = np.linspace(x.min(), x.max(), 200)
                y_lin = slope * x_lin + intercept
                ax.plot(x_lin, y_lin, color="#222222", lw=2.0, alpha=0.88, zorder=4)

                # 95% CI（解析式）
                ss_x  = np.sum((x - x.mean()) ** 2) + 1e-10
                y_hat = slope * x + intercept
                mse   = np.sum((y - y_hat) ** 2) / max(n_samples - 2, 1)
                se_lin = np.sqrt(mse * (1 / n_samples +
                                        (x_lin - x.mean()) ** 2 / ss_x))
                t_crit = scipy_stats.t.ppf(0.975, df=max(n_samples - 2, 1))
                ax.fill_between(x_lin,
                                y_lin - t_crit * se_lin,
                                y_lin + t_crit * se_lin,
                                color="#333333", alpha=0.12, zorder=2)

                # 标注
                p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
                ax.text(
                    0.05, 0.95,
                    f"r = {r_val:+.3f}\n{p_str}",
                    transform=ax.transAxes, fontsize=12,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85,
                              edgecolor="#cccccc", linewidth=0.8),
                )

            sns.despine(ax=ax)
            if i == num_mods - 1:
                ax.set_xlabel(mod_j, fontsize=15)
            if j == 0:
                ax.set_ylabel(mod_i, fontsize=15)
            ax.tick_params(axis="both", labelsize=12)

    fig.suptitle(
        f"Cross-modal Pairwise Scatter Matrix  —  Best Module M{best_mod_idx + 1}\n"
        "(modality-averaged z-scored expression per sample)",
        fontsize=22, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save_pdf(fig, "panel_J_crossmodal_pairwise_scatter")


def _plot_panel_K_network(corr_df, pval_df, feat_meta_df, mod_labels, best_mod_idx,
                           corr_thresh=CROSSMODAL_CORR_THRESH,
                           pval_thresh=CROSSMODAL_PVAL_THRESH):
    """
    圆形布局跨模态互作网络：
    - 节点按模态分弧排列，大小正比于模块权重
    - 红/蓝边分别表示正/负显著跨模态相关，粗细正比于 |r|
    - 每模态仅标注权重最高的 Top-N 特征名
    """
    _apply_sci_theme()

    num_mods = len(mod_labels)

    # 有效节点（在 corr_df 中）
    all_nodes = feat_meta_df["col_name"].tolist()
    all_nodes = [n for n in all_nodes if n in corr_df.columns]
    if len(all_nodes) < 2:
        print("  ⚠️  Panel K 跳过：节点数不足")
        return

    node_mod    = dict(zip(feat_meta_df["col_name"], feat_meta_df["modality"]))
    node_color  = dict(zip(feat_meta_df["col_name"], feat_meta_df["mod_color"]))
    node_weight = dict(zip(feat_meta_df["col_name"],
                           feat_meta_df["weight"].abs()))

    # 按模态分组并计算圆弧坐标
    grouped = []
    n_per_mod = {}
    for mod in mod_labels:
        grp = [n for n in all_nodes if node_mod.get(n) == mod]
        grouped.extend(grp)
        n_per_mod[mod] = len(grp)

    total     = len(grouped)
    gap       = 0.05           # 模态间角度间隙（占比）
    n_mods_nz = sum(1 for m in mod_labels if n_per_mod.get(m, 0) > 0)
    total_gap = gap * n_mods_nz

    pos = {}
    cursor = 0.0
    for mod in mod_labels:
        grp = [n for n in grouped if node_mod.get(n) == mod]
        if not grp:
            continue
        arc     = (1.0 - total_gap) * len(grp) / total
        angles  = np.linspace(cursor * 2 * np.pi,
                              (cursor + arc) * 2 * np.pi,
                              len(grp), endpoint=False)
        for node, ang in zip(grp, angles):
            pos[node] = (np.cos(ang), np.sin(ang))
        cursor += arc + gap

    # 收集显著跨模态边
    edges = []
    for ii, na in enumerate(all_nodes):
        for jj, nb in enumerate(all_nodes):
            if jj <= ii:
                continue
            if node_mod.get(na) == node_mod.get(nb):
                continue
            r_val = corr_df.loc[na, nb]
            p_val = pval_df.loc[na, nb]
            if abs(r_val) >= corr_thresh and p_val <= pval_thresh:
                edges.append((na, nb, float(r_val)))

    # 绘图
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    # 画边（按 |r| 从小到大，使重要边在上层）
    if edges:
        max_r = max(abs(e[2]) for e in edges)
        for na, nb, r_val in sorted(edges, key=lambda e: abs(e[2])):
            x0, y0 = pos[na]
            x1, y1 = pos[nb]
            edge_col = "#C63A2F" if r_val > 0 else "#2E5FA3"
            lw       = 0.6 + 3.2 * abs(r_val) / max(max_r, 1e-6)
            alpha    = 0.22 + 0.58 * abs(r_val) / max(max_r, 1e-6)
            ax.plot([x0, x1], [y0, y1],
                    color=edge_col, lw=lw, alpha=alpha, zorder=1,
                    solid_capstyle="round")

    # 画节点
    max_w = max((node_weight.get(n, 0) for n in grouped), default=1.0)
    for node in grouped:
        if node not in pos:
            continue
        x, y  = pos[node]
        w     = node_weight.get(node, 0)
        size  = 70 + 260 * w / max(max_w, 1e-6)
        color = node_color.get(node, "#888888")
        ax.scatter(x, y, s=size, color=color,
                   edgecolors="white", linewidths=1.4,
                   zorder=3, alpha=0.93)

    # 节点标签（每模态显示权重 Top-N，避免重叠）
    top_n = max(3, min(7, 40 // max(total, 1) + 2))
    labeled = set()
    for mod in mod_labels:
        grp = sorted(
            [(n, node_weight.get(n, 0)) for n in grouped if node_mod.get(n) == mod],
            key=lambda x: -x[1],
        )
        for node, _ in grp[:top_n]:
            if node in labeled or node not in pos:
                continue
            labeled.add(node)
            x, y  = pos[node]
            short = node.split("__", 1)[-1] if "__" in node else node
            angle = np.degrees(np.arctan2(y, x))
            ha    = "left" if -90 <= angle <= 90 else "right"
            r_off = 1.14
            ax.text(x * r_off, y * r_off, short,
                    ha=ha, va="center", fontsize=10,
                    color="#1a1a1a", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.55,
                              edgecolor="none"))

    # 模态弧形色块（外圈装饰）
    for m_idx, mod in enumerate(mod_labels):
        grp = [n for n in grouped if node_mod.get(n) == mod and n in pos]
        if not grp:
            continue
        angs = [np.arctan2(pos[n][1], pos[n][0]) for n in grp]
        a_min, a_max = min(angs), max(angs)
        theta = np.linspace(a_min - 0.05, a_max + 0.05, 60)
        r_arc = 1.22
        ax.plot(r_arc * np.cos(theta), r_arc * np.sin(theta),
                color=MOD_COLORS[m_idx % len(MOD_COLORS)],
                lw=6.0, alpha=0.72, solid_capstyle="round", zorder=2)
        mid_a = (a_min + a_max) / 2
        ax.text(1.31 * np.cos(mid_a), 1.31 * np.sin(mid_a), mod,
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=MOD_COLORS[m_idx % len(MOD_COLORS)], zorder=6)

    # 图例
    for m_idx, mod in enumerate(mod_labels):
        if n_per_mod.get(mod, 0) == 0:
            continue
        ax.scatter([], [], s=110,
                   color=MOD_COLORS[m_idx % len(MOD_COLORS)],
                   edgecolors="white", linewidths=1.2,
                   label=mod)
    ax.plot([], [], color="#C63A2F", lw=2.2, label="Positive r")
    ax.plot([], [], color="#2E5FA3", lw=2.2, label="Negative r")

    ax.legend(
        loc="lower left", fontsize=14,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        title="Modality / Edge sign", title_fontsize=14,
        markerscale=1.2, handlelength=1.8, borderpad=0.8,
    )

    n_edges = len(edges)
    ax.set_title(
        f"Cross-modal Feature Interaction Network  —  Best Module M{best_mod_idx + 1}\n"
        f"({n_edges} significant cross-modal edges,  |r| ≥ {corr_thresh},  p < {pval_thresh})\n"
        f"Node size ∝ module weight;  top-weighted feature labels shown per modality",
        fontsize=18, fontweight="bold", pad=14,
    )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.58, 1.58)
    plt.tight_layout()
    _save_pdf(fig, "panel_K_crossmodal_network")


def _plot_panel_L_corr_distribution(corr_df, feat_meta_df, mod_labels, best_mod_idx,
                                     corr_thresh=CROSSMODAL_CORR_THRESH):
    """
    每个模态对的跨模态 Pearson r 分布：violin + box + strip，
    标注均值、显著比例与阈值参考线。
    """
    from scipy import stats as scipy_stats
    _apply_sci_theme()

    pair_data   = []
    pair_labels = []
    pair_colors = []

    _PAIR_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
    ]
    p_idx = 0

    for i_m, mod_a in enumerate(mod_labels):
        for j_m, mod_b in enumerate(mod_labels):
            if i_m >= j_m:
                continue
            cols_a = feat_meta_df[feat_meta_df["modality"] == mod_a]["col_name"].tolist()
            cols_b = feat_meta_df[feat_meta_df["modality"] == mod_b]["col_name"].tolist()
            cols_a = [c for c in cols_a if c in corr_df.columns]
            cols_b = [c for c in cols_b if c in corr_df.columns]
            if not cols_a or not cols_b:
                continue
            pair_r = [corr_df.loc[ca, cb]
                      for ca in cols_a for cb in cols_b]
            pair_data.append(pair_r)
            pair_labels.append(f"{mod_a}\n× {mod_b}")
            pair_colors.append(_PAIR_PALETTE[p_idx % len(_PAIR_PALETTE)])
            p_idx += 1

    if not pair_data:
        print("  ⚠️  Panel L 跳过：无有效模态对")
        return

    n_pairs = len(pair_data)
    fig, ax  = plt.subplots(figsize=(max(10, 3.8 * n_pairs), 9))

    # Violin
    vp = ax.violinplot(pair_data, positions=range(n_pairs),
                       showmedians=False, showextrema=False, widths=0.65)
    for body, color in zip(vp["bodies"], pair_colors):
        body.set_facecolor(color)
        body.set_alpha(0.40)
        body.set_edgecolor("#555555")
        body.set_linewidth(1.2)

    # Boxplot
    bp = ax.boxplot(
        pair_data, positions=range(n_pairs), widths=0.18,
        patch_artist=True,
        medianprops=dict(color="#111111", lw=2.8),
        whiskerprops=dict(color="#555555", lw=1.6),
        capprops=dict(color="#555555", lw=1.6),
        flierprops=dict(marker="o", ms=3.5, alpha=0.38,
                        markerfacecolor="#888888",
                        markeredgecolor="none"),
    )
    for patch, color in zip(bp["boxes"], pair_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)

    # Strip jitter
    rng = np.random.default_rng(SEED)
    for i, (data, color) in enumerate(zip(pair_data, pair_colors)):
        jitter = rng.uniform(-0.13, 0.13, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   s=20, color=color, alpha=0.52,
                   edgecolors="none", zorder=4)

    # 参考线
    ax.axhline(0,            color="#888888", lw=1.5,
               linestyle="--", alpha=0.65, zorder=0)
    ax.axhline( corr_thresh, color="#C63A2F", lw=1.4,
               linestyle=":",  alpha=0.70, zorder=0,
               label=f"|r| = {corr_thresh} (threshold)")
    ax.axhline(-corr_thresh, color="#2E5FA3", lw=1.4,
               linestyle=":",  alpha=0.70, zorder=0)

    # 统计标注（均值 + 显著对占比）
    for i, (data, color) in enumerate(zip(pair_data, pair_colors)):
        arr  = np.array(data)
        mn   = arr.mean()
        pct_sig = 100 * np.mean(np.abs(arr) >= corr_thresh)
        ax.text(i, max(arr) + 0.055,
                f"μ={mn:+.3f}\n{pct_sig:.0f}% sig.",
                ha="center", va="bottom", fontsize=12, color="#222222")

    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, fontsize=17)
    ax.set_ylabel("Pearson r", fontsize=20)
    ax.set_ylim(-1.18, 1.35)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_title(
        f"Cross-modal Correlation Distributions  —  Best Module M{best_mod_idx + 1}\n"
        f"(all feature pairs between each modality pair;  μ = mean r;  sig. = |r| ≥ {corr_thresh})",
        fontsize=21, pad=14,
    )
    ax.legend(loc="upper right", fontsize=15,
              frameon=True, framealpha=0.88, edgecolor="#cccccc")
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_L_crossmodal_corr_distribution")


def plot_crossmodal_interaction_best_module(bases, ranked_idx, names_list, raw_dfs,
                                            sample_ids,
                                            strategy=FEATURE_SELECT_STRATEGY,
                                            corr_thresh=CROSSMODAL_CORR_THRESH,
                                            pval_thresh=CROSSMODAL_PVAL_THRESH):
    """
    最佳模块四模态互作可视化主函数，依次输出：
      panel_I_crossmodal_block_corr.pdf          — 跨模态分块相关热图
      panel_J_crossmodal_pairwise_scatter.pdf    — 模态均值散点矩阵
      panel_K_crossmodal_network.pdf             — 跨模态特征互作网络
      panel_L_crossmodal_corr_distribution.pdf  — 模态对相关分布
      crossmodal_feature_corr_matrix.csv         — 完整特征×特征相关矩阵
      crossmodal_sig_interactions.csv            — 显著跨模态特征对
      crossmodal_modality_summary.csv            — 模态对摘要统计
    """
    from scipy import stats as scipy_stats

    print("\n  🔗  计算跨模态特征互作...")

    expr_matrix, feat_meta_df = _get_best_module_feature_matrix(
        bases, ranked_idx, names_list, raw_dfs, sample_ids, strategy)

    if expr_matrix is None or expr_matrix.shape[1] < 2:
        print("  ⚠️  跨模态互作可视化跳过（有效特征数不足）")
        return

    num_mods     = len(bases)
    best_mod_idx = int(ranked_idx[0])
    mod_labels   = (MODALITY_NAMES[:num_mods]
                    if len(MODALITY_NAMES) >= num_mods
                    else [f"Mod{i+1}" for i in range(num_mods)])

    # z-score 标准化
    expr_z = expr_matrix.apply(
        lambda col: (col - col.mean()) / (col.std() + 1e-10))

    cols_all = expr_z.columns.tolist()
    n_feat   = len(cols_all)
    n_samp   = expr_z.shape[0]

    # ── 完整相关矩阵 ─────────────────────────────────────────
    corr_df = expr_z.corr(method="pearson")

    # ── p 值矩阵（双侧 t 检验） ───────────────────────────────
    pval_mat = np.ones((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r = corr_df.iloc[i, j]
            t = r * np.sqrt(max(n_samp - 2, 1)) / np.sqrt(max(1 - r ** 2, 1e-10))
            p = 2 * scipy_stats.t.sf(abs(t), df=max(n_samp - 2, 1))
            pval_mat[i, j] = pval_mat[j, i] = p
    pval_df = pd.DataFrame(pval_mat, index=cols_all, columns=cols_all)

    # ── 导出完整相关矩阵 ─────────────────────────────────────
    corr_df.to_csv("crossmodal_feature_corr_matrix.csv", encoding="utf-8-sig")
    print("  ✅  已导出: crossmodal_feature_corr_matrix.csv")

    # ── 导出显著跨模态特征对 ──────────────────────────────────
    sig_rows = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            ca, cb = cols_all[i], cols_all[j]
            ma = feat_meta_df.loc[feat_meta_df["col_name"] == ca, "modality"]
            mb = feat_meta_df.loc[feat_meta_df["col_name"] == cb, "modality"]
            if ma.empty or mb.empty or ma.values[0] == mb.values[0]:
                continue
            r_val = corr_df.iloc[i, j]
            p_val = pval_df.iloc[i, j]
            if abs(r_val) >= corr_thresh and p_val <= pval_thresh:
                sig_rows.append({
                    "feature_A":  ca,
                    "modality_A": ma.values[0],
                    "feature_B":  cb,
                    "modality_B": mb.values[0],
                    "pearson_r":  round(float(r_val), 4),
                    "p_value":    round(float(p_val), 6),
                    "abs_r":      round(abs(r_val), 4),
                })
    sig_df = (pd.DataFrame(sig_rows)
              .sort_values("abs_r", ascending=False)
              .reset_index(drop=True))
    sig_df.to_csv("crossmodal_sig_interactions.csv",
                  index=False, encoding="utf-8-sig")
    print(f"  ✅  已导出: crossmodal_sig_interactions.csv  "
          f"（{len(sig_df)} 对显著跨模态互作）")

    # ── 导出模态对摘要统计 ────────────────────────────────────
    summ_rows = []
    for i_m, mod_a in enumerate(mod_labels):
        for j_m, mod_b in enumerate(mod_labels):
            if i_m >= j_m:
                continue
            cols_a = feat_meta_df[feat_meta_df["modality"] == mod_a]["col_name"].tolist()
            cols_b = feat_meta_df[feat_meta_df["modality"] == mod_b]["col_name"].tolist()
            cols_a = [c for c in cols_a if c in corr_df.columns]
            cols_b = [c for c in cols_b if c in corr_df.columns]
            if not cols_a or not cols_b:
                continue
            pair_r = [corr_df.loc[ca, cb] for ca in cols_a for cb in cols_b]
            arr    = np.array(pair_r)
            summ_rows.append({
                "modality_A":    mod_a,
                "modality_B":    mod_b,
                "n_features_A":  len(cols_a),
                "n_features_B":  len(cols_b),
                "n_pairs":       len(pair_r),
                "mean_r":        round(float(arr.mean()), 4),
                "median_r":      round(float(np.median(arr)), 4),
                "mean_abs_r":    round(float(np.abs(arr).mean()), 4),
                "max_abs_r":     round(float(np.abs(arr).max()), 4),
                "n_sig_pos":     int((arr >=  corr_thresh).sum()),
                "n_sig_neg":     int((arr <= -corr_thresh).sum()),
                "pct_sig":       round(100 * (np.abs(arr) >= corr_thresh).mean(), 2),
            })
    pd.DataFrame(summ_rows).to_csv(
        "crossmodal_modality_summary.csv", index=False, encoding="utf-8-sig")
    print("  ✅  已导出: crossmodal_modality_summary.csv")

    # ── 绘制 4 张 PDF ─────────────────────────────────────────
    print("\n  📊  绘制跨模态互作图（共 4 张 PDF）")
    _plot_panel_I_block_corr(corr_df, pval_df, feat_meta_df,
                              mod_labels, best_mod_idx,
                              pval_thresh, corr_thresh)
    _plot_panel_J_scatter_matrix(expr_z, feat_meta_df, mod_labels, best_mod_idx)
    _plot_panel_K_network(corr_df, pval_df, feat_meta_df,
                           mod_labels, best_mod_idx,
                           corr_thresh, pval_thresh)
    _plot_panel_L_corr_distribution(corr_df, feat_meta_df,
                                     mod_labels, best_mod_idx, corr_thresh)
    print("  ✅  跨模态互作图已全部保存（dpi=300）")


def _apply_sci_theme():
    """每次绘图前重置主题，避免 seaborn 残留覆盖字体配置。"""
    matplotlib.rcParams.update(SCI_RC)
    sns.set_theme(style="ticks", rc=SCI_RC)


def _plot_panel_A(history):
    _apply_sci_theme()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(history["epoch"], history["train_total"],
            label="Train", color=COLOR_TRAIN, lw=2.5)
    ax.plot(history["epoch"], history["val_total"],
            label="Validation", color=COLOR_VAL, lw=2.5, linestyle="--")
    ax.set_title("Total loss", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(fontsize=17)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_A_total_loss")


def _plot_panel_B(history):
    _apply_sci_theme()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(history["epoch"], history["train_recon"],
            label="Train", color=COLOR_RECON_TRAIN, lw=2.5)
    ax.plot(history["epoch"], history["val_recon"],
            label="Validation", color=COLOR_RECON_VAL, lw=2.5, linestyle="--")
    ax.set_title("Reconstruction loss (MSE)", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(fontsize=17)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_B_recon_loss")


def _plot_panel_C(module_importance, ranked_idx):
    _apply_sci_theme()
    fig, ax = plt.subplots(figsize=(12, 7))
    top15     = min(15, len(module_importance))
    top15_idx = ranked_idx[:top15]
    bar_vals  = module_importance[top15_idx]
    cmap_c    = plt.cm.Blues(np.linspace(0.35, 0.85, top15))[::-1]
    ax.bar(range(top15), bar_vals, color=cmap_c, width=0.7, edgecolor="none")
    ax.set_xticks(range(top15))
    ax.set_xticklabels([f"M{i+1}" for i in top15_idx], rotation=45, ha="right",
                       fontsize=16)
    ax.set_title("Module importance (top 15)", pad=12)
    ax.set_xlabel("Latent module")
    ax.set_ylabel("Cross-modal L2 score")
    ax.tick_params(axis="y", labelsize=17)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_C_module_importance")


def _plot_panel_D(model, tensors, mod_labels):
    _apply_sci_theme()
    MOD_PALETTE = [
        "#0072B2","#E69F00","#009E73","#CC79A7",
        "#56B4E9","#D55E00","#F0E442","#000000",
    ]
    MOD_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

    num_mods  = len(tensors)
    n_samples = tensors[0].shape[0]

    model.eval()
    with torch.no_grad():
        z_list_np = [enc(x).cpu().numpy()
                     for enc, x in zip(model.encoders, tensors)]

    z_all    = np.concatenate(z_list_np, axis=0)
    mod_ids  = np.repeat(np.arange(num_mods), n_samples)

    perp = min(30, max(5, n_samples - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED)
    z_2d = tsne.fit_transform(z_all)

    fig, ax = plt.subplots(figsize=(11, 8))
    for m_idx, mname in enumerate(mod_labels):
        mask = (mod_ids == m_idx)
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            s=50, alpha=0.78,
            color=MOD_PALETTE[m_idx % len(MOD_PALETTE)],
            marker=MOD_MARKERS[m_idx % len(MOD_MARKERS)],
            edgecolors="none", linewidths=0,
            label=mname,
            # rasterized=True  ← [Fix 2] 已删除，避免 MemoryError
        )
    ax.set_title("t-SNE of per-modality latent representations\n"
                 "(each sample appears once per modality)", pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(title="Modality", title_fontsize=18, fontsize=17,
              markerscale=1.8, loc="best", frameon=True,
              framealpha=0.88, edgecolor="#cccccc")
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_D_tsne_by_modality")


def _plot_panel_E(sample_scores, ranked_idx):
    _apply_sci_theme()
    top_k     = min(TOP_N_MODULES_SHOW, 10)
    hmap_data = sample_scores[:, ranked_idx[:top_k]].T
    sort_ord  = np.argsort(sample_scores[:, ranked_idx[0]])

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        hmap_data[:, sort_ord], ax=ax,
        cmap="RdBu_r", center=0,
        xticklabels=False,
        yticklabels=[f"M{i+1}" for i in ranked_idx[:top_k]],
        cbar_kws={"label": "Activation", "shrink": 0.85},
        linewidths=0,
        # rasterized=True  ← [Fix 3] 已删除，避免 MemoryError
    )
    ax.set_title(f"Sample × module heatmap (top {top_k})", pad=12)
    ax.set_xlabel("Samples (sorted by top module)", fontsize=20)
    ax.tick_params(axis="y", labelsize=17)
    ax.collections[0].colorbar.ax.tick_params(labelsize=16)
    ax.collections[0].colorbar.set_label("Activation", fontsize=19)
    plt.tight_layout()
    _save_pdf(fig, "panel_E_sample_module_heatmap")


def _plot_panel_F(model, tensors, mod_labels):
    _apply_sci_theme()
    num_mods = len(tensors)
    model.eval()
    with torch.no_grad():
        z_np = [enc(x).cpu().numpy() for enc, x in zip(model.encoders, tensors)]
    corr_mat = np.zeros((num_mods, num_mods))
    for i in range(num_mods):
        for j in range(num_mods):
            corrs = [np.corrcoef(z_np[i][:, k], z_np[j][:, k])[0, 1]
                     for k in range(z_np[i].shape[1])]
            corr_mat[i, j] = float(np.nanmean(corrs))

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        corr_mat, ax=ax,
        xticklabels=mod_labels, yticklabels=mod_labels,
        annot=True, fmt=".2f", annot_kws={"size": 18},
        cmap="RdBu_r", vmin=-1, vmax=1,
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Mean Pearson r", "shrink": 0.85},
        # rasterized=True  ← [Fix 4] 已删除，避免 MemoryError
    )
    ax.set_title("Cross-modality latent correlation", pad=12)
    ax.tick_params(axis="both", labelsize=17)
    ax.collections[0].colorbar.ax.tick_params(labelsize=16)
    ax.collections[0].colorbar.set_label("Mean Pearson r", fontsize=19)
    plt.tight_layout()
    _save_pdf(fig, "panel_F_crossmodal_corr")


def _plot_panel_G(bases, ranked_idx, mod_labels):
    _apply_sci_theme()
    num_mods  = len(bases)
    top_k     = min(TOP_N_MODULES_SHOW, len(ranked_idx))
    top_k_idx = ranked_idx[:top_k]
    x     = np.arange(top_k)
    width = 0.75 / num_mods

    fig, ax = plt.subplots(figsize=(12, 7))
    for m_idx in range(num_mods):
        l2_sc = [np.linalg.norm(bases[m_idx][mi, :]) for mi in top_k_idx]
        ax.bar(x + m_idx*width, l2_sc, width,
               label=mod_labels[m_idx],
               color=MOD_COLORS[m_idx % len(MOD_COLORS)],
               edgecolor="none", alpha=0.9)
    ax.set_xticks(x + width*(num_mods-1)/2)
    ax.set_xticklabels([f"M{i+1}" for i in top_k_idx], fontsize=17)
    ax.set_title(f"Per-modality contribution (top {top_k} modules)", pad=12)
    ax.set_xlabel("Latent module")
    ax.set_ylabel("L2 norm of basis row")
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(ncol=2, fontsize=17, handlelength=1.4,
              handletextpad=0.6, columnspacing=1.2)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_G_modality_contribution")


def _plot_panel_H(bases, ranked_idx):
    _apply_sci_theme()
    top_k     = min(TOP_N_MODULES_SHOW, len(ranked_idx))
    top_k_idx = ranked_idx[:top_k]
    num_mods  = len(bases)
    strats       = ["zscore", "cumsum", "elbow"]
    strat_names  = [
        f"Z-score (z>{ZSCORE_THRESH})",
        f"CumSum ({CUMSUM_THRESH:.0%})",
        f"Elbow (min={ELBOW_MIN_N})",
    ]
    strat_colors = ["#2166AC", "#4DAC26", "#D6604D"]
    bar_x = np.arange(top_k)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    for s_i, (strat, sname, scol) in enumerate(zip(strats, strat_names, strat_colors)):
        avg_counts = []
        for mod_idx in top_k_idx:
            counts = [len(select_hub_features(bases[m_idx][mod_idx, :],
                                              strategy=strat)[0])
                      for m_idx in range(num_mods)]
            avg_counts.append(np.mean(counts))
        ax.bar(bar_x + s_i*bar_w, avg_counts, bar_w,
               label=sname, color=scol, edgecolor="none", alpha=0.9)
    ax.set_xticks(bar_x + bar_w)
    ax.set_xticklabels([f"M{i+1}" for i in top_k_idx], fontsize=17)
    ax.set_title("Feature count by selection strategy\n(mean across modalities)", pad=12)
    ax.set_xlabel("Latent module")
    ax.set_ylabel("No. features selected")
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(loc="upper right", fontsize=16, handlelength=1.4)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save_pdf(fig, "panel_H_feature_selection_count")


def plot_comprehensive_results(history, sample_scores, bases, module_importance,
                               ranked_idx, names_list, tensors, model, strategy):
    """将原 8-panel 综合图拆分为独立 PDF，逐张输出（dpi=300）。"""
    num_mods   = len(bases)
    mod_labels = (MODALITY_NAMES[:num_mods]
                  if len(MODALITY_NAMES) >= num_mods
                  else [f"Mod{i+1}" for i in range(num_mods)])

    print("\n  📊  逐张输出综合分析图（共 8 张 PDF）")
    _plot_panel_A(history)
    _plot_panel_B(history)
    _plot_panel_C(module_importance, ranked_idx)
    _plot_panel_D(model, tensors, mod_labels)
    _plot_panel_E(sample_scores, ranked_idx)
    _plot_panel_F(model, tensors, mod_labels)
    _plot_panel_G(bases, ranked_idx, mod_labels)
    _plot_panel_H(bases, ranked_idx)
    print("  ✅  8 张图已全部保存为独立 PDF（dpi=300）")


def plot_optuna_results(study):
    print("\n正在生成 Optuna 分析图表...")
    figs = {
        "optuna_history.html":    plot_optimization_history(study),
        "optuna_importance.html": plot_param_importances(study),
        "optuna_contour.html":    plot_contour(
            study, params=["lr","latent_dim","lambda_cca","lambda_ortho"]),
        "optuna_slice.html":      plot_slice(study),
    }
    for fname, fig in figs.items():
        pio.write_html(fig, fname)
        print(f"  ✅  Saved: {fname}")
    study.trials_dataframe().to_csv(
        "optuna_all_trials.csv", index=False, encoding="utf-8-sig")
    print("  ✅  Saved: optuna_all_trials.csv")


def _cv_write_plot_roc(y,prob,out,title):
    if len(np.unique(y))!=2:
        return
    fpr,tpr,_=_cv_roc_curve(y,prob)
    score=float(_cv_roc_auc_score(y,prob))
    fig,ax=plt.subplots(figsize=(6.5,5.5))
    ax.plot(fpr,tpr,lw=2,label=f"Out-of-fold AUC = {score:.3f}")
    ax.plot([0,1],[0,1],"--",lw=1,color="grey",label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out)+".pdf",bbox_inches="tight")
    fig.savefig(str(out)+".png",dpi=160,bbox_inches="tight")
    plt.close(fig)


def _cv_plot_outer_fold_rocs(pred_rows,output):
    """Display separate outer-test ROC curves; scores are NEVER refitted."""
    df=pd.DataFrame(pred_rows)
    fig,ax=plt.subplots(figsize=(6.5,5.6))
    for fold,group in df.groupby("fold",sort=True):
        y=group["label"].to_numpy()
        prob=group["prob_positive"].to_numpy()
        fpr,tpr,_=_cv_roc_curve(y,prob)
        score=float(_cv_roc_auc_score(y,prob))
        ax.plot(fpr,tpr,lw=1.7,label=f"Fold {int(fold)} (AUC={score:.3f})")
    ax.plot([0,1],[0,1],"--",lw=1,color="grey",label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Selection-repeated fivefold outer-test ROC")
    ax.legend(loc="lower right",fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output)+".pdf",bbox_inches="tight")
    fig.savefig(str(output)+".png",dpi=160,bbox_inches="tight")
    plt.close(fig)
