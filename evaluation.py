"""Fold-wise model fitting, feature re-selection and diagnosis."""
from __future__ import annotations

import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from itertools import combinations as _cv_combinations
from sklearn.model_selection import StratifiedKFold as _CVStratifiedKFold
from sklearn.model_selection import train_test_split as _cv_train_test_split
from sklearn.preprocessing import StandardScaler as _CVStandardScaler
from sklearn.impute import SimpleImputer as _CVSimpleImputer
from sklearn.pipeline import make_pipeline as _cv_make_pipeline
from sklearn.linear_model import LogisticRegression as _CVLogisticRegression
from sklearn.metrics import roc_auc_score as _cv_roc_auc_score
from scipy.optimize import linear_sum_assignment as _cv_linear_sum_assignment
from config import (
    MODALITY_NAMES,
)
from data import (
    _CVNormalizer,
)
from features import (
    _cv_model_bases,
    _cv_select_fold_module,
)
from training import (
    _cv_seed,
    _cv_tune_params,
    run_training,
)
from visualization import (
    _cv_plot_outer_fold_rocs,
    _cv_write_plot_roc,
)


def _cv_fit_predict_features(data,train_idx,test_idx,selected,args):
    """Preprocess / fit logistic regression ONLY on outer-development data."""
    raw_train=[]
    raw_test=[]
    for m,modality in enumerate(MODALITY_NAMES):
        feature_cols=[int(x["feature_column"]) for x in
                      selected["top_rows"] if x["modality"]==modality]
        raw_train.append(data["X"][m][np.ix_(train_idx,feature_cols)])
        raw_test.append(data["X"][m][np.ix_(test_idx,feature_cols)])
    stacked_train=np.concatenate(raw_train,axis=1)
    stacked_test=np.concatenate(raw_test,axis=1)
    estimator=_cv_make_pipeline(
        _CVSimpleImputer(strategy="median"),
        _CVStandardScaler(),
        _CVLogisticRegression(class_weight="balanced",solver="liblinear",
                             random_state=args.seed,max_iter=3000))
    estimator.fit(stacked_train,data["y"][train_idx])
    prob=estimator.predict_proba(stacked_test)[:,list(estimator.classes_).index(1)]
    return np.asarray(prob,dtype=float)


def _cv_flat_normed_loadings(bases):
    """All modalities concatenated in fixed feature order: one vector/module."""
    flat=np.concatenate(bases,axis=1)
    norm=np.linalg.norm(flat,axis=1,keepdims=True)
    return flat/np.maximum(norm,1e-12)


def _cv_compute_stability(fold_results,cv_dir):
    """Compare loadings, not only M44 labels; account for factor permutations."""
    rows=[]
    for a,b in _cv_combinations(fold_results,2):
        wa=_cv_flat_normed_loadings(a["bases"])
        wb=_cv_flat_normed_loadings(b["bases"])
        matrix=wa@wb.T
        ra,cb=_cv_linear_sum_assignment(-matrix)
        align={int(i):int(j) for i,j in zip(ra,cb)}
        topa,topb=int(a["module"]),int(b["module"])
        rows.append({
            "fold_A":a["fold"],"fold_B":b["fold"],
            "top_module_A_1based":topa+1,"top_module_B_1based":topb+1,
            "top_loading_cosine":float(matrix[topa,topb]),
            "optimal_match_of_A_top_in_B_1based":align.get(topa,-1)+1,
            "optimal_match_is_B_top":bool(align.get(topa,-1)==topb),
            "matched_mean_loading_cosine":float(np.mean(matrix[ra,cb])),
            "same_numeric_module_id":bool(topa==topb),
        })
    pd.DataFrame(rows).to_csv(cv_dir/"module_similarity_across_folds.csv",
                              index=False,encoding="utf-8-sig")
    selected=[]; consensus=[]
    for modality in MODALITY_NAMES:
        for family in ("hub_sets","top_sets"):
            universe=sorted(set().union(
                *(x[family][modality] for x in fold_results)))
            for feat in universe:
                flags=[feat in x[family][modality] for x in fold_results]
                selected.append({
                    "selection_type":"cumsum_hubs" if family=="hub_sets" else "top_features",
                    "modality":modality,"feature":feat,"frequency":sum(flags),
                    "fraction":sum(flags)/len(flags),
                    **{f"fold_{x['fold']}":int(flag)
                       for x,flag in zip(fold_results,flags)},
                    "strict_5of5_consensus":bool(all(flags)),
                })
                if all(flags):
                    consensus.append({
                        "selection_type":"cumsum_hubs" if family=="hub_sets" else "top_features",
                        "modality":modality,"feature":feat,
                        "frequency":len(fold_results)
                    })
    pd.DataFrame(selected).to_csv(cv_dir/"feature_selection_frequency.csv",
                                  index=False,encoding="utf-8-sig")
    pd.DataFrame(consensus,columns=[
        "selection_type","modality","feature","frequency"
    ]).to_csv(cv_dir/"strict_five_fold_feature_intersection.csv",
               index=False,encoding="utf-8-sig")
    # Quantify reproducibility of selected FEATURE NAMES from the leading
    # module in each fold, independently of the factor's numeric index.
    jac_rows=[]
    for family,selection_type in (
        ("hub_sets","cumsum_hubs"),("top_sets","top_features")):
        for modality in MODALITY_NAMES:
            for a,b in _cv_combinations(fold_results,2):
                sa=a[family][modality];sb=b[family][modality]
                union=sa|sb;inter=sa&sb
                jac_rows.append({
                    "selection_type":selection_type,"modality":modality,
                    "fold_A":a["fold"],"fold_B":b["fold"],
                    "intersection_n":len(inter),"union_n":len(union),
                    "jaccard":len(inter)/len(union) if union else 1.0
                })
    pd.DataFrame(jac_rows).to_csv(
        cv_dir/"fold_pair_feature_jaccard.csv",
        index=False,encoding="utf-8-sig")
    return selected


def _cv_run(data,args,root_out):
    print("\n" + "="*72)
    print("阶段 A：5-fold selection-nested classification + module stability")
    print("="*72)
    out=root_out/"cross_validation"
    out.mkdir(parents=True,exist_ok=True)
    splitter=_CVStratifiedKFold(n_splits=args.folds,shuffle=True,
                               random_state=args.seed)
    all_folds=[]
    all_hubs=[]
    all_top=[]
    all_rankings=[]
    all_predictions=[]
    split_manifest=[]
    oof=np.full(len(data["y"]),np.nan,dtype=float)
    dims=[x.shape[1] for x in data["X"]]

    for fold,(dev_idx,test_idx) in enumerate(
        splitter.split(np.zeros(len(data["y"])),data["y"]),1):
        print(f"\nFold {fold}/{args.folds}: dev={len(dev_idx)}, "
              f"test={len(test_idx)}; test positives={int(data['y'][test_idx].sum())}")
        # Strictly no outer test information in preprocessing/early stopping.
        fit_idx,val_idx=_cv_train_test_split(
            dev_idx,test_size=args.inner_val_fraction,
            random_state=args.seed+fold,stratify=data["y"][dev_idx])
        normalizer=_CVNormalizer().fit(data["X"],fit_idx)
        train_t=normalizer.transform(data["X"],fit_idx)
        val_t=normalizer.transform(data["X"],val_idx)
        _cv_seed(args.seed+101*fold)
        params,hpo_loss=_cv_tune_params(
            args,train_t,val_t,dims,args.seed+1001*fold)
        _cv_seed(args.seed+2001*fold)
        model,best_val,history=run_training(
            train_tensors=train_t,val_tensors=val_t,dims=dims,
            epochs=args.epochs,early_stop_patience=args.patience,
            trial=None,verbose=False,**params)
        model.eval()
        bases=_cv_model_bases(model)
        selected=_cv_select_fold_module(
            bases,data["names"],args,fold)
        predictions=_cv_fit_predict_features(
            data,dev_idx,test_idx,selected,args)
        if len(predictions)!=len(test_idx):
            raise AssertionError("测试人数与预测长度不一致")
        if not np.isfinite(predictions).all():
            raise ValueError(f"Fold {fold} 产生非有限预测概率")
        if np.any(np.isfinite(oof[test_idx])):
            raise AssertionError("交叉验证测试折出现受试者重叠")
        oof[test_idx]=predictions
        auc=float(_cv_roc_auc_score(data["y"][test_idx],predictions))
        selected_mod=selected["module"]+1
        print(f"  Selected module M{selected_mod} "
              f"(automatically ranked, NOT pre-specified); "
              f"held-out combined AUC={auc:.4f}")
        folder=out/f"fold_{fold:02d}"
        folder.mkdir(exist_ok=True)
        pd.DataFrame(selected["ranking_rows"]).to_csv(
            folder/"module_rankings.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(selected["hub_rows"]).to_csv(
            folder/"selected_80pct_hubs.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(selected["top_rows"]).to_csv(
            folder/"selected_top_features.csv",index=False,encoding="utf-8-sig")
        for m,modality in enumerate(MODALITY_NAMES):
            np.save(folder/f"top_module_loadings_{modality}.npy",
                    bases[m][selected["module"]])
        torch.save(model.state_dict(),folder/"checkpoint_weights.pth")
        pd.DataFrame(history).to_csv(
            folder/"training_history.csv",index=False,encoding="utf-8-sig")
        (folder/"fold_settings.json").write_text(
            json.dumps({
                "fold":fold,"seed":args.seed+2001*fold,
                "selected_module_1based":selected_mod,
                "hyperparams":params,
                "hpo_best_val_loss":hpo_loss,
                "best_val_loss":float(best_val),
                "epochs_run":len(history["epoch"]),
                "test_auc":auc,
                "outer_development_n":len(dev_idx),
                "inner_fit_n":len(fit_idx),"inner_checkpoint_val_n":len(val_idx),
                "outer_test_n":len(test_idx),
                "outer_test_positive":int(data["y"][test_idx].sum()),
            },indent=2,ensure_ascii=False),encoding="utf-8")
        for split,indices in (
            ("development",dev_idx),("inner_fit",fit_idx),
            ("inner_val",val_idx),("outer_test",test_idx)):
            for i in indices:
                row={"fold":fold,"role":split,"row_index":int(i)}
                if args.export_ids:row["sample_id"]=data["ids"][int(i)]
                split_manifest.append(row)
        for i,pred in zip(test_idx,predictions):
            row=dict(fold=fold,row_index=int(i),
                     label=int(data["y"][i]),prob_positive=float(pred))
            if args.export_ids:row["sample_id"]=data["ids"][int(i)]
            all_predictions.append(row)
        all_hubs.extend(selected["hub_rows"])
        all_top.extend(selected["top_rows"])
        all_rankings.extend(selected["ranking_rows"])
        all_folds.append(dict(
            fold=fold,module=int(selected["module"]),bases=bases,
            hub_sets=selected["hub_sets"],top_sets=selected["top_sets"],
            auc=auc,train_size=len(dev_idx),test_size=len(test_idx),
            test_positive=int(data["y"][test_idx].sum()),
            epochs_run=len(history["epoch"]),params=params))

    if not np.isfinite(oof).all():
        raise AssertionError("存在未被任何测试折预测的样本")
    if (pd.DataFrame(all_predictions).groupby("row_index")
            .size().ne(1).any()):
        raise AssertionError("每位受试者必须在一次且仅一次测试折出现")

    fold_df=pd.DataFrame([dict(fold=x["fold"],
          selected_module_1based=x["module"]+1,
          held_out_auc=x["auc"],outer_test_n=x["test_size"],
          outer_test_positive=x["test_positive"],
          best_epochs_run=x["epochs_run"],
          latent_dim=x["params"]["latent_dim"])
          for x in all_folds])
    fold_df.to_csv(out/"fivefold_performance_and_module_rank.csv",
                   index=False,encoding="utf-8-sig")
    pd.DataFrame(all_predictions).sort_values("row_index").to_csv(
        out/"out_of_fold_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(split_manifest).to_csv(
        out/"split_roles.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_rankings).to_csv(
        out/"all_fold_module_rankings.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_hubs).to_csv(
        out/"all_fold_80pct_hubs.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_top).to_csv(
        out/"all_fold_top_features.csv",index=False,encoding="utf-8-sig")
    recurrence=_cv_compute_stability(all_folds,out)
    pooled=float(_cv_roc_auc_score(data["y"],oof))
    a=np.array([r["auc"] for r in all_folds])
    mean=float(a.mean())
    std=float(a.std(ddof=1)) if len(a)>1 else 0.
    _cv_write_plot_roc(
        data["y"],oof,out/"pooled_out_of_fold_ROC",
        "DeepCAN-NMF: selection-repeated outer-fold ROC")
    _cv_plot_outer_fold_rocs(
        all_predictions,out/"individual_outer_fold_ROC_curves")
    results={
        "sample_n":len(data["y"]),
        "fold_n":args.folds,
        "positive_label":args.positive,"negative_label":args.negative,
        "fold_auc_mean":mean,"fold_auc_sample_sd":std,
        "pooled_oof_auc":pooled,
        "module_indices_1based":[x["module"]+1 for x in all_folds],
        "all_module_indices_same":len({x["module"] for x in all_folds})==1,
        "note":"Module number identity does not imply biological loading identity; "
               "see module_similarity_across_folds.csv",
        "different_final_biology_panel":
            "Final-model selected hubs must never be used to score CV test folds.",
        "deg_preselection_note":
            "Gene CSV already preselected: upstream limma fold-local selection "
            "is NOT independently verified by this script.",
        "averaging_note":
            "Foldwise mean AUC is not the pooled OOF ROC AUC.",
    }
    (out/"cv_summary.json").write_text(
        json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\n  Fivefold AUC mean±sampleSD = "
          f"{mean:.4f} ± {std:.4f}; "
          f"pooled out-of-fold AUC = {pooled:.4f}")
    print("  实际各折所选模块:",results["module_indices_1based"])
    if not results["all_module_indices_same"]:
        print("  [NOTE] 各折模块编号不同；不允许把所有fold写成 M44。")
    print(f"  Export: {out}")
    return dict(summary=results,folds=all_folds,recurrence=recurrence,
                oof=oof)
