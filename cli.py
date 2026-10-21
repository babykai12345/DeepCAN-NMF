"""Command-line interface for DeepCAN-NMF."""
from __future__ import annotations

import config
import json
import torch
import argparse as _cv_argparse
from pathlib import Path as _CVPath
from config import (
    _CV_MODES,
    _CV_PARAM_KEYS,
)
from data import (
    _cv_load_labeled,
)
from evaluation import (
    _cv_run,
    _cv_holdout_run,
)
from final_analysis import (
    _cv_final_run,
)
from legacy import (
    main,
)
from training import (
    _cv_seed,
)


def _cv_cli():
    parser=_cv_argparse.ArgumentParser(
        formatter_class=_cv_argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "DeepCAN-NMF reviewer-ready workflow: foldwise model/module/"
            "Top feature re-selection -> held-out ROC -> ONE final exploratory "
            "M44-or-other-leading-module analysis. The foldwise stage never "
            "forces M44 or leaks final-model features into test folds."
        ))
    parser.add_argument("--mode",choices=_CV_MODES,default="both",
                        help="holdout / both / cv / final all use single 4:1 holdout; "
                             "legacy is exploratory original")
    parser.add_argument("--data-dir",default=".")
    parser.add_argument("--files",nargs=4,default=[
        "c.csv","a.csv","b.csv","d.csv"],
        metavar=("SNP","MRI","PET","GENE"),
        help="4 omics CSVs, first column is subject ID; for GENE "
             "d.csv is tried first, then d_select_degs.csv")
    parser.add_argument("--labels",default="label.csv")
    parser.add_argument("--label-col",default="Label")
    parser.add_argument("--positive",default="1",
                        help="Numeric label for ROC positive class; verify source codebook")
    parser.add_argument("--negative",default="0",
                        help="Numeric label for ROC negative class; verify source codebook")
    parser.add_argument("--expected-n",type=int,default=0,
                        help="Require exactly this many labeled overlapping subjects; "
                             "0 disables this check")
    parser.add_argument("--folds",type=int,default=5,
                        help="Compatibility only: not used for primary holdout")
    parser.add_argument("--test-fraction",type=float,default=0.20)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--inner-val-fraction",type=float,default=0.20)
    parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--patience",type=int,default=20)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--latent-dim",type=int,default=60,
                        help="default >=44 makes M44 possible but never forces it")
    parser.add_argument("--hidden-dim",type=int,default=128)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--weight-decay",type=float,default=1e-5)
    parser.add_argument("--lambda-cca",type=float,default=0.2)
    parser.add_argument("--lambda-graph",type=float,default=0.03)
    parser.add_argument("--lambda-ortho",type=float,default=0.2)
    parser.add_argument("--lambda-l1",type=float,default=1e-5)
    parser.add_argument("--optuna-trials",type=int,default=0,
                        help="if >0, separate inner-fold Optuna per outer fold and "
                             "independent final-model tuning; potentially costly")
    parser.add_argument("--hpo-epochs",type=int,default=50)
    parser.add_argument("--params-json",default=None,
                        help="Optional externally pre-specified params, not tuned "
                             "on current evaluation subjects")
    parser.add_argument("--cumsum-threshold",type=float,default=.80)
    parser.add_argument("--max-hub-features",type=int,default=50,
                        help="legacy cumsum cap; 0 means uncapped true 80%%")
    parser.add_argument("--top-n",type=int,default=12)
    parser.add_argument("--association-top-n",type=int,default=50,
                        help="for final exploratory FDR correlation analysis "
                             "only, test at most this many top hub features "
                             "per modality (0 = all; may be very expensive). "
                             "Does not change final hub selection or CV.")
    parser.add_argument("--expect-final-module",type=int,default=44,
                        help="Only warn if final module differs; NEVER forcibly select")
    parser.add_argument("--final-plots",action="store_true",
                        help="make final-fit descriptive panels C-G and refit-loss plot; "
                             "never make fixed-feature diagnostic ROC")
    parser.add_argument("--export-final-matrix",action="store_true",
                        help="export sensitive subject-level selected-omics values")
    parser.add_argument("--export-ids",action="store_true",
                        help="include raw subject identifiers in private outputs")
    parser.add_argument("--out",default="DeepCAN_NMF_holdout_output")
    args=parser.parse_args()

    if args.mode=="legacy":
        print("[WARNING] legacy ROC uses final features on evaluation subjects "
              "and reverses AUC<0.5. It is EXPLORATORY, not leakage-free ROC.")
        return main()

    if args.positive==args.negative or args.folds<2:
        parser.error("positive/negative label must differ, and folds >=2")
    if not 0<args.test_fraction<.5:
        parser.error("test-fraction must be in (0,0.5)")
    if not 0<args.inner_val_fraction<.5 or args.expected_n<0:
        parser.error("invalid inner-val-fraction or expected-n")
    if args.epochs<1 or args.hpo_epochs<1 or args.patience<1:
        parser.error("epochs, hpo-epochs and patience must be >=1")
    if (args.threads<1 or args.top_n<1 or args.max_hub_features<0
            or args.association_top_n<0):
        parser.error("threads/top-n invalid; hub and association limits "
                     "must be >=0")
    if args.hidden_dim<1 or args.latent_dim<1:
        parser.error("hidden-dim and latent-dim must be positive")
    if not (0<args.cumsum_threshold<=1):
        parser.error("cumsum-threshold must be in (0,1]")
    if args.optuna_trials<0 or (args.optuna_trials and args.params_json):
        parser.error("optuna-trials cannot be negative, and cannot combine "
                     "inner-fold Optuna with external params-json")
    if args.optuna_trials>0:
        print("[NOTE] Fold-local Optuna may select latent_dim <44; "
              "M44 is not guaranteed.")

    params={
        "hidden_dim":args.hidden_dim,"latent_dim":args.latent_dim,
        "lr":args.lr,"weight_decay":args.weight_decay,
        "lambda_cca":args.lambda_cca,"lambda_graph":args.lambda_graph,
        "lambda_ortho":args.lambda_ortho,"lambda_l1":args.lambda_l1,
    }
    if args.params_json:
        incoming=json.loads(_CVPath(args.params_json).read_text(encoding="utf-8"))
        if "params" in incoming:
            incoming=incoming["params"]
        if not all(k in incoming for k in _CV_PARAM_KEYS):
            parser.error(f"params-json must contain these keys: {_CV_PARAM_KEYS}")
        params={k:incoming[k] for k in _CV_PARAM_KEYS}
        print("[CAUTION] Externally supplied hyperparameters are valid for "
              "unbiased CV ONLY when chosen without using this cohort's "
              "future outer-test outcomes.")
    args._base_params=params
    torch.set_num_threads(args.threads)
    config.SEED=args.seed
    _cv_seed(args.seed)
    data=_cv_load_labeled(args)
    out=_CVPath(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    meta={
        "mode":"holdout","seed":args.seed,
        "dataset_n":len(data["y"]),
        "negative_label":args.negative,
        "positive_label":args.positive,
        "n_negative":int((data["y"]==0).sum()),
        "n_positive":int((data["y"]==1).sum()),
        "number_of_unlabeled_four_modality_subjects":
            data["common_unlabeled_count"],
        "modality_input_paths":data["input_paths"],
        "label_input_path":data["label_path"],
        "modality_feature_dimensions":[x.shape[1] for x in data["X"]],
        "feature_selection":
            f"Original aa.py's cumsum strategy at {args.cumsum_threshold:.0%}; "
            f"max_hub_features={args.max_hub_features} "
            f"(0 means uncapped, true requested weight); "
            f"Top {args.top_n} from each fold's OWN leading module.",
        "final_association_test_max_features_per_modality":
            args.association_top_n,
        "hyperparams":params if args.optuna_trials==0 else
            {"inner_fold_Optuna_trials":args.optuna_trials},
        "upstream_DEG_leakage_caveat":
            "Input GENE may be preselected upstream; this script cannot "
            "prove train-only limma without the original 469-subject data "
            "and archived subject-level split.",
        "exploratory_fixed_final_roc":
            "Original fixed M44 ROC is not a fold-nested unbiased ROC.",
        "positive_label_semantics": "Must be verified against diagnosis codebook.",
    }
    (out/"workflow_metadata.json").write_text(
        json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    _cv_holdout_run(data,args,out)
    print("\n" + "="*72)
    print("FINISHED. All stage-specific outputs:",out)
    print("  ROC is a single independent 4:1 holdout, not five-fold CV.")
    print("  The chosen module and hubs come from development only.")
    print("  No test-set information enters model/hub selection.")
    print("  Train-only limma screening must be separately documented.")
    print("="*72)
    return out
