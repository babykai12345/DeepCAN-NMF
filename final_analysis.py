"""Independent final biological module and FDR analysis."""
from __future__ import annotations

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations as _cv_combinations
from sklearn.model_selection import train_test_split as _cv_train_test_split
from scipy.stats import pearsonr as _cv_pearsonr
from config import (
    MODALITY_NAMES,
)
from data import (
    _CVNormalizer,
)
from evaluation import (
    _cv_flat_normed_loadings,
)
from features import (
    _cv_model_bases,
    _cv_select_fold_module,
)
from training import (
    _cv_seed,
    _cv_train_full_best_epoch,
    _cv_tune_params,
    run_training,
)
from visualization import (
    _plot_panel_C,
    _plot_panel_D,
    _plot_panel_E,
    _plot_panel_F,
    _plot_panel_G,
)


def _cv_bh_adjust(p):
    """Benjamini–Hochberg adjusted q values for ALL cross-modal tests."""
    arr=np.asarray(p,dtype=np.float64)
    if arr.size==0:return arr.copy()
    index=np.argsort(arr)
    sorted_p=arr[index]
    estimates=np.minimum.accumulate(
        (sorted_p*len(arr)/np.arange(1,len(arr)+1))[::-1])[::-1]
    out=np.empty_like(arr)
    out[index]=np.clip(estimates,0.0,1.0)
    return out


def _cv_final_associations(data,selected,output,max_per_modality=50):
    """Exploratory final links with BH q<.05 AND |r|>=.30.

    Full 80% cumulative-weight lists can contain thousands of features.
    Restrict pairwise testing to top-ranked candidates per modality unless
    max_per_modality=0; report the testing family explicitly in the outputs.
    """
    mod_to_i={mod:i for i,mod in enumerate(MODALITY_NAMES)}
    pairs=[]
    rows=[r for r in selected["hub_rows"]
          if max_per_modality==0 or r["rank"]<=max_per_modality]
    pd.DataFrame(rows).to_csv(
        output/"association_tested_feature_scope.csv",
        index=False,encoding="utf-8-sig")
    print(f"  Testing cross-modal links among {len(rows)} final-model hub "
          f"features (max_per_modality={max_per_modality}, "
          f"total_hubs={len(selected['hub_rows'])}).")
    for a,b in _cv_combinations(rows,2):
        if a["modality"]==b["modality"]:continue
        i,j=mod_to_i[a["modality"]],mod_to_i[b["modality"]]
        va=data["X"][i][:,int(a["feature_column"])]
        vb=data["X"][j][:,int(b["feature_column"])]
        good=np.isfinite(va)&np.isfinite(vb)
        if good.sum()<4 or np.nanstd(va[good])==0 or np.nanstd(vb[good])==0:
            rr,pp=np.nan,1.0
        else:
            rr,pp=_cv_pearsonr(va[good],vb[good])
        pairs.append({
            "modality_A":a["modality"],"feature_A":a["feature"],
            "modality_B":b["modality"],"feature_B":b["feature"],
            "pearson_r":float(rr),"p_value":float(pp),
            "subjects_used":int(good.sum())})
    if not pairs:
        df=pd.DataFrame(columns=["modality_A","feature_A","modality_B",
                                 "feature_B","pearson_r","p_value",
                                 "subjects_used","bh_q","significant"])
        df.to_csv(output/"crossmodal_all_links_BH_FDR.csv",
                  index=False,encoding="utf-8-sig")
        return 0
    df=pd.DataFrame(pairs)
    df["bh_q"]=_cv_bh_adjust(df["p_value"].fillna(1))
    df["significant"]=(df["bh_q"]<.05)&(df["pearson_r"].abs()>=.30)
    df=df.sort_values(["bh_q","p_value"])
    df.to_csv(output/"crossmodal_all_links_BH_FDR.csv",
              index=False,encoding="utf-8-sig")
    df[df.significant].to_csv(
        output/"crossmodal_FDR_significant_links.csv",
        index=False,encoding="utf-8-sig")
    print(f"  探索性跨模态关联：BH q<0.05 且 |r|>=0.30 的特征对数量"
          f" = {int(df.significant.sum())}")
    return int(df.significant.sum())


def _cv_final_run(data,args,root_out,cv_result=None):
    print("\n" + "="*72)
    print("阶段 B：最终模型独立拟合 -> 模块排名一次 -> Hub/生物学分析")
    print("="*72)
    out=root_out/"final_model_biological_exploration"
    out.mkdir(parents=True,exist_ok=True)
    y=data["y"]
    # Choose hyperparameters / best epoch from this FINAL model's own
    # development validation; any CV has already been fully completed.
    train_idx,val_idx=_cv_train_test_split(
        np.arange(len(y)),test_size=args.inner_val_fraction,
        stratify=y,random_state=args.seed+90009)
    normalizer=_CVNormalizer().fit(data["X"],train_idx)
    inner_train=normalizer.transform(data["X"],train_idx)
    inner_val=normalizer.transform(data["X"],val_idx)
    dims=[x.shape[1] for x in data["X"]]
    _cv_seed(args.seed+70707)
    params,hpo_loss=_cv_tune_params(
        args,inner_train,inner_val,dims,args.seed+80808)
    _cv_seed(args.seed+90909)
    temp_model,val_loss,history=run_training(
        inner_train,inner_val,dims,
        epochs=args.epochs,early_stop_patience=args.patience,
        trial=None,verbose=False,**params)
    best_epoch=int(np.argmin(history["val_total"])+1)
    print(f"  最终模型内层验证选择 epoch={best_epoch}, "
          f"latent_dim={params['latent_dim']}")
    # Fit the FINAL preprocessing on full labeled cohort: legitimate for
    # post-hoc biological interpretation; NEVER re-used in CV predictions.
    full_scaler=_CVNormalizer().fit(
        data["X"],np.arange(len(y),dtype=int))
    tensors=full_scaler.transform(
        data["X"],np.arange(len(y),dtype=int))
    final_model,refit_history=_cv_train_full_best_epoch(
        tensors,params,best_epoch,args)
    bases=_cv_model_bases(final_model)
    selected=_cv_select_fold_module(
        bases,data["names"],args,fold="FINAL")
    module=selected["module"]+1
    print(f"  最终模型独立自动选出的模块: M{module}")
    if args.expect_final_module:
        if module!=args.expect_final_module:
            print(f"  [WARNING] 与期望M{args.expect_final_module}不同。"
                  f"真实结果为 M{module}，不会强制改名或重新指定。")
    torch.save(final_model.state_dict(),out/"final_model_weights.pth")
    pd.DataFrame(history).to_csv(out/"final_inner_training_history.csv",
                                 index=False,encoding="utf-8-sig")
    refit_history.to_csv(out/"final_refit_on_all_subjects_history.csv",
                         index=False,encoding="utf-8-sig")
    pd.DataFrame(selected["ranking_rows"]).to_csv(
        out/"final_module_rankings.csv",index=False,encoding="utf-8-sig")
    final_hubs=pd.DataFrame(selected["hub_rows"])
    final_hubs.to_csv(out/"final_hub_features_80pct.csv",
                      index=False,encoding="utf-8-sig")
    pd.DataFrame(selected["top_rows"]).to_csv(
        out/"final_top_features.csv",index=False,encoding="utf-8-sig")
    for modality in MODALITY_NAMES:
        gene_list=[row["feature"] for row in selected["hub_rows"]
                   if row["modality"]==modality]
        (out/f"final_{modality}_hub_list.txt").write_text(
            "\n".join(gene_list)+"\n",encoding="utf-8")
    (out/"GENE_for_GO_KEGG_PPI.txt").write_text(
        (out/"final_GENE_hub_list.txt").read_text(encoding="utf-8"),
        encoding="utf-8")
    n_links=_cv_final_associations(
        data,selected,out,max_per_modality=args.association_top_n)

    # Show the final list's recurrence in folds, but do not define the final
    # list by intersection or let final-model features leak into CV.
    recurrence_map={}
    if cv_result is not None:
        recurrence_map={
            (x["selection_type"],x["modality"],x["feature"]):
            int(x["frequency"]) for x in cv_result["recurrence"]}
    # A final-model feature absent from every fold has frequency ZERO,
    # not missing. The frequency is unknown only in --mode final without CV.
    absent_default=0 if cv_result is not None else None
    final_hubs["fold_frequency_of_hub"]=[
        recurrence_map.get(("cumsum_hubs",row["modality"],row["feature"]),
                           absent_default)
        for _,row in final_hubs.iterrows()]
    final_hubs["fold_frequency_of_top"]=[
        recurrence_map.get(("top_features",row["modality"],row["feature"]),
                           absent_default)
        for _,row in final_hubs.iterrows()]
    final_hubs.to_csv(
        out/"final_hubs_with_outer_fold_recurrence.csv",
        index=False,encoding="utf-8-sig")

    # Compare final dominant module's loading to each fold's dominant
    # module; matching by numeric index is not proof of same content.
    if cv_result is not None:
        final_vec=_cv_flat_normed_loadings(bases)[selected["module"]]
        rows=[]
        for f in cv_result["folds"]:
            v=_cv_flat_normed_loadings(f["bases"])
            similarities=v@final_vec
            rows.append({
                "fold":f["fold"],"final_module_1based":module,
                "fold_top_module_1based":f["module"]+1,
                "final_vs_fold_top_cosine":
                    float(similarities[f["module"]]),
                "best_matched_fold_module_1based":
                    int(np.argmax(similarities))+1,
                "best_match_cosine":float(np.max(similarities)),
                "fold_top_is_best_matching":
                    bool(f["module"]==np.argmax(similarities)),
            })
        pd.DataFrame(rows).to_csv(
            out/"final_module_loading_similarity_to_CV.csv",
            index=False,encoding="utf-8-sig")

    # Optional export of sample-by-hub original values; SNP/gene inputs can
    # be sensitive. Default OFF; subject ID export additionally opt-in.
    if args.export_final_matrix:
        blocks=[]
        for m,modality in enumerate(MODALITY_NAMES):
            part=final_hubs[final_hubs.modality==modality]
            cols=part.feature_column.astype(int).tolist()
            values=data["X"][m][:,cols]
            columns=[f"{modality}__{name}" for name in part.feature.tolist()]
            blocks.append(pd.DataFrame(values,columns=columns))
        matrix=pd.concat(blocks,axis=1)
        if args.export_ids: matrix.insert(0,"sample_id",data["ids"])
        else:matrix.insert(0,"row_index",np.arange(len(y)))
        matrix.to_csv(out/"final_hub_raw_expression_matrix_PRIVATE.csv",
                      index=False,encoding="utf-8-sig")
        print("  [PRIVACY] 导出了基因型/表达矩阵；请勿直接作为期刊补充材料上传。")

    fig,ax=plt.subplots(figsize=(7,4.7))
    scores=selected["importance"]
    order=selected["ranking"][:min(len(scores),15)]
    ax.bar([f"M{int(i)+1}" for i in order],scores[order])
    ax.set_title("Final-model module importance")
    ax.set_ylabel("Cross-modal decoder row-L2 importance")
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout()
    fig.savefig(out/"final_module_importance.pdf",bbox_inches="tight")
    plt.close(fig)

    if args.final_plots:
        # Only show descriptive figures computed from the actual FINAL fit.
        # No fictitious validation curve is drawn for the all-subject refit,
        # and the legacy label-flipped fixed-feature ROC is never invoked.
        print("  输出基于最终模型的描述性模块/模态图件 C-G")
        cwd=os.getcwd()
        try:
            os.chdir(out)
            with torch.no_grad():
                _,zshared,_,_=final_model(tensors)
            _plot_panel_C(selected["importance"],selected["ranking"])
            _plot_panel_D(final_model,tensors,MODALITY_NAMES)
            _plot_panel_E(zshared.detach().cpu().numpy(),
                          selected["ranking"])
            _plot_panel_F(final_model,tensors,MODALITY_NAMES)
            _plot_panel_G(bases,selected["ranking"],MODALITY_NAMES)
            fig,ax=plt.subplots(figsize=(7.4,4.5))
            ax.plot(refit_history["epoch"],
                    refit_history["total_loss"],label="Training objective")
            ax.plot(refit_history["epoch"],
                    refit_history["reconstruction_loss"],
                    label="Reconstruction loss")
            ax.set_xlabel("Refit epoch (all labeled subjects)")
            ax.set_ylabel("Training loss")
            ax.set_title("Final-model full-cohort refit")
            ax.legend()
            fig.tight_layout()
            fig.savefig("final_full_cohort_refit_loss.pdf",
                        bbox_inches="tight")
            plt.close(fig)
        except Exception as error:
            print(f"  [NOTE] 描述性 Panel 生成未完成: {error}")
        finally:
            os.chdir(cwd)

    summary={
        "n_labeled":len(y),
        "module_1based":module,
        "module_is_M44":module==44,
        "latent_dim":params["latent_dim"],
        "final_params":params,
        "inner_selected_best_epoch":best_epoch,
        "best_inner_validation_loss":float(val_loss),
        "final_hub_count":len(selected["hub_rows"]),
        "final_top_count":len(selected["top_rows"]),
        "exploratory_FDR_significant_crossmodal_links":n_links,
        "hub_definition":"One final independently trained model's 80% cumulative "
          "weights with configured cap; NOT necessarily the intersection of folds.",
        "hpo_validation_loss":hpo_loss,
        "avoid_diagnostic_leakage":
          "No final-model hub features were used in outer-test predictions.",
        "source_data_caveat":"Gene features already preselected upstream; "
          "fold-local DEG filtering has NOT been established.",
    }
    (out/"final_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"  最终模型生物学探索数据输出: {out}")
    return summary
