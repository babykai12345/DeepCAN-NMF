"""Shared settings and fixed default hyperparameters."""
from pathlib import Path

# ============================================================
# 【配置区】
# ============================================================
MY_MODALITY_FILES = [
    str(Path("输入文件") / "c.csv"),
    str(Path("输入文件") / "a.csv"),
    str(Path("输入文件") / "b.csv"),
    str(Path("输入文件") / "d.csv")
]
MODALITY_NAMES = ["SNP", "MRI", "PET", "GENE"]

N_TRIALS           = 50
VAL_RATIO          = 0.2
OPTUNA_EPOCHS      = 50
FINAL_EPOCHS       = 150
STUDY_NAME         = "DeepCANNMF_HPO"
STORAGE            = None
SEED               = 42
TOP_N_MODULES_SHOW = 5

FEATURE_SELECT_STRATEGY = "cumsum"
ZSCORE_THRESH   = 1.0
CUMSUM_THRESH   = 0.80
ELBOW_MIN_N     = 3
FEATURE_MIN_N   = 3
FEATURE_MAX_N   = 50
TRIAL_LOG_PATH  = "optuna_trial_log.csv"
SPLIT_DATA_DIR  = "split_data"

# ★ 跨模态互作分析阈值
CROSSMODAL_CORR_THRESH = 0.30   # 显著互作的 |Pearson r| 最低阈值
CROSSMODAL_PVAL_THRESH = 0.05   # 显著互作的 p 值上限

# ★ 标签文件配置
LABEL_FILE      = str(Path("输入文件") / "label.csv")   # 标签文件路径（首列为样本ID或设为索引）
LABEL_COL       = "label"                    # 标签列名；若不存在则自动取第一列
# 阳性类名称（ROC 正类）；设为 None 则自动取字典序靠后的类
POSITIVE_CLASS  = None
# ROC 图中每个子图最多显示几条命名曲线（超出部分以灰线显示）
ROC_MAX_LEGEND  = 12

# ============================================================
# ★ 全局 SCI 字体设置（放大版）
# ============================================================
SCI_RC = {
    "font.family":           "Arial",
    "font.size":             18,
    "axes.titlesize":        22,
    "axes.labelsize":        20,
    "xtick.labelsize":       17,
    "ytick.labelsize":       17,
    "legend.fontsize":       17,
    "legend.title_fontsize": 18,
    "axes.linewidth":        1.4,
    "xtick.major.width":     1.4,
    "ytick.major.width":     1.4,
    "xtick.minor.width":     0.8,
    "ytick.minor.width":     0.8,
    "xtick.major.size":      6,
    "ytick.major.size":      6,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "legend.frameon":        False,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "figure.dpi":            150,
}

# 绘图配色
COLOR_TRAIN       = "#2166AC"
COLOR_VAL         = "#D6604D"
COLOR_RECON_TRAIN = "#4DAC26"
COLOR_RECON_VAL   = "#E08214"
MOD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
              "#56B4E9", "#D55E00", "#F0E442", "#000000"]

# ROC 曲线每子图调色板（Tab20 扩展，支持多特征区分）
_ROC_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
]

_CV_DEFAULT_PARAMS = {
    "hidden_dim": 128, "latent_dim": 60,
    "lr": 0.001, "weight_decay": 1e-5,
    "lambda_cca": 0.2, "lambda_graph": 0.03,
    "lambda_ortho": 0.2, "lambda_l1": 1e-5,
}
_CV_PARAM_KEYS = tuple(_CV_DEFAULT_PARAMS)
_CV_MODES = ("holdout", "both", "cv", "final", "legacy")
