from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
)

ROOT = Path("/data/xuewz/WSI_PRE/CLAM_0423")
CLAM = ROOT / "code/CLAM"

DATASET_CSV = CLAM / "dataset_csv/hzey_stage1_aonly_bm.csv"
SELECTED_CSV = ROOT / "data/meta/hzey_stage1_aonly_selected.csv"
SPLIT_DIR = CLAM / "splits/task_hzey_stage1_aonly_bm_100"
EVAL_DIR = CLAM / "eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_eval"

OUT_DIR = ROOT / "analysis/stage1_clam_tables_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_LABEL = 1.0  # malignant

def safe_div(a, b):
    return float(a) / float(b) if b else np.nan

def mean_sd_ci(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
    # n=5 folds; t_0.975, df=4 = 2.776445
    t = 2.7764451051977987 if len(arr) == 5 else 1.96
    half = t * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, sd, mean - half, mean + half

# -------------------------
# Table 1
# -------------------------
dataset = pd.read_csv(DATASET_CSV)
selected = pd.read_csv(SELECTED_CSV)

rows = []
rows.append({"item": "total_cases", "value": dataset["case_id"].nunique()})
rows.append({"item": "total_slides", "value": dataset["slide_id"].nunique()})

for label in ["benign", "malignant"]:
    sub = dataset[dataset["label"] == label]
    rows.append({"item": f"{label}_cases", "value": sub["case_id"].nunique()})
    rows.append({"item": f"{label}_slides", "value": sub["slide_id"].nunique()})

if "selection_category" in selected.columns:
    vc = selected["selection_category"].value_counts()
    rows.append({"item": "routine_A_only_slides", "value": int(vc.get("routine_A_only", 0))})
    rows.append({"item": "A_plus_H_secondary_slides", "value": int(vc.get("A_plus_H_secondary", 0))})

split_summary_rows = []
for p in sorted(SPLIT_DIR.glob("splits_*_descriptor.csv")):
    fold = int(p.stem.split("_")[1])
    d = pd.read_csv(p)
    # first column is label name
    label_col = d.columns[0]
    for _, r in d.iterrows():
        split_summary_rows.append({
            "fold": fold,
            "label": r[label_col],
            "train": int(r["train"]),
            "val": int(r["val"]),
            "test": int(r["test"]),
        })

split_summary = pd.DataFrame(split_summary_rows)
split_summary.to_csv(OUT_DIR / "table1_split_summary_by_fold.csv", index=False)

# compact split summary
for split in ["train", "val", "test"]:
    total_each_fold = split_summary.groupby("fold")[split].sum()
    rows.append({
        "item": f"{split}_slides_per_fold",
        "value": "/".join(map(str, sorted(total_each_fold.unique())))
    })

table1 = pd.DataFrame(rows)
table1.to_csv(OUT_DIR / "table1_dataset_characteristics.csv", index=False)

# -------------------------
# Table 2
# -------------------------
fold_dfs = []
for p in sorted(EVAL_DIR.glob("fold_*.csv")):
    fold = int(p.stem.split("_")[1])
    f = pd.read_csv(p)
    f["fold"] = fold
    fold_dfs.append(f)

pred = pd.concat(fold_dfs, ignore_index=True)
pred.to_csv(OUT_DIR / "all_fold_predictions.csv", index=False)

perf_rows = []
for fold, g in pred.groupby("fold"):
    y_true = g["Y"].astype(int).values
    y_pred = g["Y_hat"].astype(int).values
    p1 = g["p_1"].astype(float).values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    perf_rows.append({
        "fold": int(fold),
        "n_test": len(g),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "AUC": auc(*roc_curve(y_true, p1)[:2]),
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Specificity": safe_div(tn, tn + fp),
        "PPV": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "NPV": safe_div(tn, tn + fn),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    })

table2_fold = pd.DataFrame(perf_rows)
table2_fold.to_csv(OUT_DIR / "table2_classification_performance_by_fold.csv", index=False)

summary_rows = []
for metric in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_Accuracy"]:
    mean, sd, ci_low, ci_high = mean_sd_ci(table2_fold[metric].values)
    summary_rows.append({
        "metric": metric,
        "mean": mean,
        "sd": sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_sd": f"{mean:.4f} ± {sd:.4f}",
        "ci95": f"{ci_low:.4f}–{ci_high:.4f}",
    })

table2_summary = pd.DataFrame(summary_rows)
table2_summary.to_csv(OUT_DIR / "table2_classification_performance_summary.csv", index=False)

# Combined confusion matrix
tn, fp, fn, tp = confusion_matrix(pred["Y"].astype(int), pred["Y_hat"].astype(int), labels=[0, 1]).ravel()
combined = pd.DataFrame([{
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "ACC": safe_div(tp + tn, tp + tn + fp + fn),
    "Sensitivity": safe_div(tp, tp + fn),
    "Specificity": safe_div(tn, tn + fp),
    "PPV": safe_div(tp, tp + fp),
    "NPV": safe_div(tn, tn + fn),
    "Balanced_Accuracy": (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)) / 2,
}])
combined.to_csv(OUT_DIR / "combined_confusion_metrics.csv", index=False)

# -------------------------
# Figure 1: ROC curves
# -------------------------
mean_fpr = np.linspace(0, 1, 200)
interp_tprs = []
fold_aucs = []

plt.figure(figsize=(6, 6))

for fold, g in pred.groupby("fold"):
    y_true = g["Y"].astype(int).values
    p1 = g["p_1"].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, p1)
    fold_auc = auc(fpr, tpr)
    fold_aucs.append(fold_auc)

    interp = np.interp(mean_fpr, fpr, tpr)
    interp[0] = 0.0
    interp_tprs.append(interp)

    plt.plot(fpr, tpr, linewidth=1.0, alpha=0.35, label=f"Fold {fold} AUC={fold_auc:.3f}")

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
sd_auc = np.std(fold_aucs, ddof=1)

plt.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f"Mean ROC AUC={mean_auc:.3f} ± {sd_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stage-1 CLAM ROC Curves")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure1_roc_curves.png", dpi=300)
plt.savefig(OUT_DIR / "figure1_roc_curves.pdf")
plt.close()

# -------------------------
# Figure 2 representative candidates
# -------------------------
tp = pred[(pred["Y"] == 1) & (pred["Y_hat"] == 1)].sort_values("p_1", ascending=False).head(1)
tn = pred[(pred["Y"] == 0) & (pred["Y_hat"] == 0)].sort_values("p_1", ascending=True).head(1)
fp = pred[(pred["Y"] == 0) & (pred["Y_hat"] == 1)].sort_values("p_1", ascending=False).head(1)
fn = pred[(pred["Y"] == 1) & (pred["Y_hat"] == 0)].sort_values("p_1", ascending=True).head(1)

rep = pd.concat([
    tp.assign(case_type="TP_high_conf"),
    tn.assign(case_type="TN_high_conf"),
    fp.assign(case_type="FP_high_conf"),
    fn.assign(case_type="FN_high_conf"),
], ignore_index=True)

rep.to_csv(OUT_DIR / "figure2_representative_cases.csv", index=False)

# Human-readable report
with open(OUT_DIR / "stage1_metrics_report.md", "w", encoding="utf-8") as f:
    f.write("# Stage-1 CLAM Results\n\n")
    f.write("## Table 1. Dataset characteristics\n\n")
    f.write(table1.to_markdown(index=False))
    f.write("\n\n## Split summary\n\n")
    f.write(split_summary.to_markdown(index=False))
    f.write("\n\n## Table 2. Classification performance by fold\n\n")
    f.write(table2_fold.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n\n## Table 2. Summary\n\n")
    f.write(table2_summary.to_markdown(index=False))
    f.write("\n\n## Combined confusion metrics\n\n")
    f.write(combined.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n\n## Figure 2 representative cases\n\n")
    f.write(rep.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n")

print("Saved outputs to:", OUT_DIR)
print("Main report:", OUT_DIR / "stage1_metrics_report.md")
print("ROC:", OUT_DIR / "figure1_roc_curves.png")
print("Representative cases:", OUT_DIR / "figure2_representative_cases.csv")
