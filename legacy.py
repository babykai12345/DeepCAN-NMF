"""Original exploratory plotting and Optuna workflow."""
from __future__ import annotations

import json
import torch
import numpy as np
import pandas as pd
import optuna
from config import (
    FEATURE_SELECT_STRATEGY,
    FINAL_EPOCHS,
    LABEL_FILE,
    MODALITY_NAMES,
    MY_MODALITY_FILES,
    N_TRIALS,
    OPTUNA_EPOCHS,
    SEED,
    SPLIT_DATA_DIR,
    STORAGE,
    STUDY_NAME,
    TOP_N_MODULES_SHOW,
    TRIAL_LOG_PATH,
    VAL_RATIO,
)
from data import (
    load_omics_data_dynamic,
    save_latent_and_recon,
    save_split_data,
    train_val_split,
)
from features import (
    extract_and_score_modules,
    save_best_module_expression,
)
from training import (
    TrialLogCallback,
    make_objective,
    run_training,
)
from visualization import (
    plot_comprehensive_results,
    plot_crossmodal_interaction_best_module,
    plot_optuna_results,
    plot_roc_best_module,
)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Step 1：加载数据 ─────────────────────────────────────
    print("=" * 65)
    print("  Step 1 / 4 : 加载数据")
    print("=" * 65)
    tensors, proc_dfs, raw_dfs, names_list, sample_ids, n_samples, dims = \
        load_omics_data_dynamic(MY_MODALITY_FILES)
    print(f"样本数: {n_samples}  |  各模态特征数: {dims}")

    train_tensors, val_tensors, train_idx, val_idx = \
        train_val_split(tensors, val_ratio=VAL_RATIO)
    print(f"训练集: {len(train_idx)} 样本  |  验证集: {len(val_idx)} 样本")
    print(f"特征选择策略: {FEATURE_SELECT_STRATEGY.upper()}")

    save_split_data(
        proc_dfs=proc_dfs, raw_dfs=raw_dfs, sample_ids=sample_ids,
        train_idx_list=train_idx, val_idx_list=val_idx,
        mod_names=MODALITY_NAMES, out_dir=SPLIT_DATA_DIR,
    )

    # ── Step 2：Optuna 搜索 ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 2 / 4 : Optuna 超参数搜索")
    print(f"  共 {N_TRIALS} 次 Trial，每次最多训练 {OPTUNA_EPOCHS} Epochs")
    print(f"  Trial 日志实时写入: {TRIAL_LOG_PATH}")
    print("=" * 65)

    trial_logger = TrialLogCallback(TRIAL_LOG_PATH)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(
        study_name=STUDY_NAME, direction="minimize",
        sampler=sampler, pruner=pruner, storage=STORAGE, load_if_exists=True,
    )
    study.optimize(make_objective(train_tensors, val_tensors, dims),
                   n_trials=N_TRIALS, callbacks=[trial_logger],
                   show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 65)
    print(f"  🎯  搜索完成！最佳验证损失: {best.value:.6f}")
    print("  最佳超参数：")
    for k, v in best.params.items():
        print(f"    {k:>15s} = {v}")
    print("=" * 65)

    with open("best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best.value, "params": best.params},
                  f, indent=2, ensure_ascii=False)
    print("  ✅  最佳参数已保存至 best_hyperparams.json")

    try:
        plot_optuna_results(study)
    except Exception as e:
        print(f"  ⚠️  Optuna 可视化跳过（{e}）")

    # ── Step 3：最终完整训练 ─────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 3 / 4 : 用最佳超参数进行最终完整训练")
    print(f"  最大 Epochs: {FINAL_EPOCHS}")
    print("=" * 65)

    p = best.params
    final_model, final_val_loss, final_history = run_training(
        train_tensors=train_tensors, val_tensors=val_tensors, dims=dims,
        hidden_dim=p["hidden_dim"],     latent_dim=p["latent_dim"],
        lr=p["lr"],                     weight_decay=p["weight_decay"],
        lambda_cca=p["lambda_cca"],     lambda_graph=p["lambda_graph"],
        lambda_ortho=p["lambda_ortho"], lambda_l1=p["lambda_l1"],
        epochs=FINAL_EPOCHS, early_stop_patience=20, trial=None, verbose=True,
    )
    print(f"\n  最终最优验证损失: {final_val_loss:.6f}")

    torch.save(final_model.state_dict(), "best_model_weights.pth")
    print("  ✅  模型权重已保存至 best_model_weights.pth")
    pd.DataFrame(final_history).to_csv(
        "final_training_history.csv", index=False, encoding="utf-8-sig")
    print("  ✅  最终训练历史已保存至 final_training_history.csv")

    save_latent_and_recon(
        model=final_model, tensors=tensors, sample_ids=sample_ids,
        train_idx_list=train_idx, val_idx_list=val_idx,
        mod_names=MODALITY_NAMES, latent_dim=p["latent_dim"],
        out_dir=SPLIT_DATA_DIR,
    )

    # ── Step 4：特征提取、综合可视化、★ ROC & 表达矩阵 ────────
    print("\n" + "=" * 65)
    print("  Step 4 / 4 : 特征提取与综合可视化")
    print("=" * 65)

    final_model.eval()
    with torch.no_grad():
        _, z_shared_full, _, _ = final_model(tensors)

    bases, mod_scores, sample_scores, ranked_idx = extract_and_score_modules(
        final_model, z_shared_full, names_list,
        sample_ids=sample_ids, strategy=FEATURE_SELECT_STRATEGY,
    )

    plot_comprehensive_results(
        history=final_history, sample_scores=sample_scores,
        bases=bases, module_importance=mod_scores,
        ranked_idx=ranked_idx, names_list=names_list,
        tensors=tensors, model=final_model,
        strategy=FEATURE_SELECT_STRATEGY,
    )

    # ★ ROC 曲线（最佳模块各模态特征）
    print("\n  📈  绘制最佳模块 ROC 曲线...")
    plot_roc_best_module(
        bases=bases, ranked_idx=ranked_idx,
        names_list=names_list, raw_dfs=raw_dfs,
        sample_ids=sample_ids,
        label_file=LABEL_FILE,
        strategy=FEATURE_SELECT_STRATEGY,
    )

    # ★ 导出最佳模块特征表达矩阵
    print("\n  📤  导出最佳模块特征表达矩阵...")
    save_best_module_expression(
        bases=bases, ranked_idx=ranked_idx,
        names_list=names_list, raw_dfs=raw_dfs,
        sample_ids=sample_ids,
        strategy=FEATURE_SELECT_STRATEGY,
    )

    # ★ 跨模态互作可视化
    print("\n  🔗  绘制最佳模块跨模态互作图...")
    plot_crossmodal_interaction_best_module(
        bases=bases, ranked_idx=ranked_idx,
        names_list=names_list, raw_dfs=raw_dfs,
        sample_ids=sample_ids,
        strategy=FEATURE_SELECT_STRATEGY,
    )

    # ── 输出文件清单 ──────────────────────────────────────────
    n_mods = len(dims)
    print("\n✅  全流程完成！输出文件清单：")
    outputs = [
        ("best_hyperparams.json",                        "最优超参数"),
        ("best_model_weights.pth",                       "模型权重"),
        (TRIAL_LOG_PATH,                                 "Optuna 实时 Trial 日志"),
        ("optuna_all_trials.csv",                        "所有 Trial 汇总表"),
        ("optuna_history.html",                          "搜索收敛曲线（交互）"),
        ("optuna_importance.html",                       "超参重要性排名（交互）"),
        ("optuna_contour.html",                          "参数交互热图（交互）"),
        ("optuna_slice.html",                            "参数切片图（交互）"),
        ("final_training_history.csv",                   "最终训练 Loss 数据"),
        ("hub_features_all_modules.csv",                 "所有模块 Hub 特征汇总"),
        ("module_features/hub_module_*.csv",             f"Top {TOP_N_MODULES_SHOW} 模块详细特征"),
        ("sample_module_scores.csv",                     "样本激活分数矩阵"),
        ("feature_selection_counts.csv",                 "各策略选中特征数量统计"),
        # ── 8 张独立 PDF ──────────────────────────────────────
        ("panel_A_total_loss.pdf",                       "Panel A — 总损失曲线"),
        ("panel_B_recon_loss.pdf",                       "Panel B — 重建损失曲线"),
        ("panel_C_module_importance.pdf",                "Panel C — 模块重要性柱状图"),
        ("panel_D_tsne_by_modality.pdf",                 "Panel D — t-SNE 散点图（按模态着色）"),
        ("panel_E_sample_module_heatmap.pdf",            "Panel E — 样本×模块热图"),
        ("panel_F_crossmodal_corr.pdf",                  "Panel F — 跨模态潜在相关热图"),
        ("panel_G_modality_contribution.pdf",            "Panel G — 各模态 L2 贡献柱状图"),
        ("panel_H_feature_selection_count.pdf",          "Panel H — 特征选择数量对比图"),
        # ★ 跨模态互作
        ("panel_I_crossmodal_block_corr.pdf",
         "★ Panel I  — 跨模态分块相关热图"),
        ("panel_J_crossmodal_pairwise_scatter.pdf",
         "★ Panel J  — 模态均值散点矩阵（含回归线+CI）"),
        ("panel_K_crossmodal_network.pdf",
         "★ Panel K  — 跨模态特征互作网络"),
        ("panel_L_crossmodal_corr_distribution.pdf",
         "★ Panel L  — 模态对相关系数分布（violin+box+strip）"),
        ("crossmodal_feature_corr_matrix.csv",
         "★ 完整特征×特征 Pearson r 矩阵"),
        ("crossmodal_sig_interactions.csv",
         "★ 显著跨模态特征对（r、p 值）"),
        ("crossmodal_modality_summary.csv",
         "★ 模态对摘要统计（均值/显著对数/占比等）"),
        # ★ NEW
        ("panel_ROC_best_module.pdf",                    "★ Panel ROC — 最佳模块各特征 ROC 曲线"),
        ("best_module_expression_matrix.csv",            "★ 最佳模块特征原始表达矩阵（样本×特征）"),
        ("best_module_feature_meta.csv",                 "★ 最佳模块特征元信息（模态/权重/排名）"),
        # ── split_data 目录 ───────────────────────────────────
        (f"{SPLIT_DATA_DIR}/train_sample_ids.csv",       "训练集样本 ID"),
        (f"{SPLIT_DATA_DIR}/val_sample_ids.csv",         "验证集样本 ID"),
        (f"{SPLIT_DATA_DIR}/all_sample_split_labels.csv","全量 split 标签"),
        (f"{SPLIT_DATA_DIR}/train_{{MOD}}_preprocessed.csv", "训练集预处理特征"),
        (f"{SPLIT_DATA_DIR}/val_{{MOD}}_preprocessed.csv",   "验证集预处理特征"),
        (f"{SPLIT_DATA_DIR}/train_{{MOD}}_raw.csv",           "训练集原始特征"),
        (f"{SPLIT_DATA_DIR}/val_{{MOD}}_raw.csv",             "验证集原始特征"),
        (f"{SPLIT_DATA_DIR}/train_latent_z_shared.csv",       "训练集共享潜在表示"),
        (f"{SPLIT_DATA_DIR}/val_latent_z_shared.csv",         "验证集共享潜在表示"),
        (f"{SPLIT_DATA_DIR}/train_latent_z_{{MOD}}.csv",      "训练集各模态潜在表示"),
        (f"{SPLIT_DATA_DIR}/val_latent_z_{{MOD}}.csv",        "验证集各模态潜在表示"),
        (f"{SPLIT_DATA_DIR}/train_reconstruction_{{MOD}}.csv","训练集各模态重建值"),
        (f"{SPLIT_DATA_DIR}/val_reconstruction_{{MOD}}.csv",  "验证集各模态重建值"),
        (f"{SPLIT_DATA_DIR}/split_summary.txt",               "分割参数摘要"),
    ]
    for fname, desc in outputs:
        prefix = "🆕" if "★" in desc else "📄"
        print(f"  {prefix}  {fname:<55s} ← {desc}")

    return study, final_model, final_history
