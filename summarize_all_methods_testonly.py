import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

METHOD_DIRS = {
    "CLAM-SB baseline": Path("eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy"),
    "ECLAM-SB constant": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_constant_check_energy"),
    "ECLAM-SB linear": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy"),
    "ECLAM-SB sigmoid": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_energy"),
}

OUT_DIR = Path("eval_results/FINAL_TEST_ONLY_BASELINE_CONSTANT_LINEAR_SIGMOID")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def auc_safe(y_true, score):
    y_true = pd.Series(y_true)
    if y_true.nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, score)

def read_method(method, d):
    files = sorted(d.glob("energy_scores_fold_*.csv"))
    if len(files) != 5:
        raise RuntimeError(f"{method}: expected 5 CSVs, found {len(files)} in {d}")

    dfs = []
    for f in files:
        fold = int(f.stem.split("_")[-1])
        df = pd.read_csv(f)
        df["fold"] = fold
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    print(f"\n\n================ {method}: split counts ================")
    print(all_df["split"].value_counts(dropna=False).to_string())

    test_df = all_df[all_df["split"].astype(str).str.lower() == "test"].copy()

    print(f"\n================ {method}: TEST ONLY shape ================")
    print(test_df.shape)

    if len(test_df) != 50:
        raise RuntimeError(f"{method}: expected 50 test rows, got {len(test_df)}")

    required = [
        "slide_id", "label", "pred", "prob_benign", "prob_malignant",
        "energy", "entropy", "confidence", "margin", "correctness", "error_type"
    ]
    missing = [c for c in required if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"{method}: missing columns: {missing}")

    print(f"\n================ {method}: NaN check ================")
    print(test_df[["energy", "entropy", "confidence", "margin"]].isna().sum().to_string())

    return test_df

def metrics_for_df(method, df):
    cm = confusion_matrix(df["label"], df["pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "method": method,
        "n": len(df),
        "AUC": auc_safe(df["label"], df["prob_malignant"]),
        "ACC": accuracy_score(df["label"], df["pred"]),
        "F1": f1_score(df["label"], df["pred"], zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "Balanced_ACC": balanced_accuracy_score(df["label"], df["pred"]),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

def per_fold_metrics(method, df):
    rows = []
    for fold, g in df.groupby("fold"):
        row = metrics_for_df(method, g)
        row["fold"] = fold
        rows.append(row)
    cols = ["method", "fold", "n", "AUC", "ACC", "F1", "Sensitivity",
            "Specificity", "PPV", "NPV", "Balanced_ACC", "TN", "FP", "FN", "TP"]
    return pd.DataFrame(rows)[cols]

def uncertainty_detection(method, df):
    out = {"method": method}

    # energy 越高越不确定；这里 energy 本身越接近 0 越高，所以直接用 energy。
    df = df.copy()
    df["is_error"] = (df["pred"] != df["label"]).astype(int)
    df["is_fn"] = ((df["label"] == 1) & (df["pred"] == 0)).astype(int)
    df["is_fp"] = ((df["label"] == 0) & (df["pred"] == 1)).astype(int)

    out["error_detection_AUROC_energy"] = auc_safe(df["is_error"], df["energy"])
    out["error_detection_AUROC_entropy"] = auc_safe(df["is_error"], df["entropy"])
    out["error_detection_AUROC_low_margin"] = auc_safe(df["is_error"], -df["margin"])

    out["FN_detection_AUROC_energy"] = auc_safe(df["is_fn"], df["energy"])
    out["FN_detection_AUROC_entropy"] = auc_safe(df["is_fn"], df["entropy"])
    out["FN_detection_AUROC_low_margin"] = auc_safe(df["is_fn"], -df["margin"])

    out["FP_detection_AUROC_energy"] = auc_safe(df["is_fp"], df["energy"])
    out["FP_detection_AUROC_entropy"] = auc_safe(df["is_fp"], df["entropy"])
    out["FP_detection_AUROC_low_margin"] = auc_safe(df["is_fp"], -df["margin"])

    for k in [5, 10, 15, 20]:
        top = df.sort_values("energy", ascending=False).head(k)
        out[f"top{k}_error_hits"] = int(top["is_error"].sum())
        out[f"top{k}_FN_hits"] = int(top["is_fn"].sum())
        out[f"top{k}_FP_hits"] = int(top["is_fp"].sum())
        out[f"top{k}_error_rate"] = float(top["is_error"].mean())
        out[f"top{k}_FN_rate"] = float(top["is_fn"].mean())
        out[f"top{k}_FP_rate"] = float(top["is_fp"].mean())

    return out

all_test = {}
pooled_rows = []
per_fold_rows = []
mean_std_rows = []
unc_rows = []

for method, d in METHOD_DIRS.items():
    if not d.exists():
        print(f"\n[SKIP] missing dir for {method}: {d}")
        continue

    df = read_method(method, d)
    all_test[method] = df

    df.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_test_only_predictions.csv", index=False)

    pooled = metrics_for_df(method, df)
    pooled_rows.append(pooled)

    print(f"\n================ {method}: TEST ONLY pooled metrics ================")
    print(f"TN FP FN TP = {pooled['TN']} {pooled['FP']} {pooled['FN']} {pooled['TP']}")
    for c in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        print(f"{c}={pooled[c]:.4f}")

    pf = per_fold_metrics(method, df)
    per_fold_rows.append(pf)

    print(f"\n================ {method}: TEST ONLY per-fold metrics ================")
    print(pf.to_string(index=False))

    print(f"\n================ {method}: TEST ONLY mean ± std ================")
    for c in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        mean = pf[c].mean()
        std = pf[c].std(ddof=1)
        mean_std_rows.append({"method": method, "metric": c, "mean": mean, "std": std})
        print(f"{c}: {mean:.4f} ± {std:.4f}")

    energy_by_type = (
        df.groupby("error_type")[["energy", "entropy", "confidence", "margin"]]
        .agg(["count", "mean", "std", "min", "max"])
    )
    print(f"\n================ {method}: TEST ONLY energy by error_type ================")
    print(energy_by_type.to_string())
    energy_by_type.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_energy_by_error_type.csv")

    cols = [
        "fold", "slide_id", "label", "pred", "prob_benign", "prob_malignant",
        "energy", "entropy", "margin", "confidence", "error_type"
    ]
    top = df.sort_values("energy", ascending=False)[cols].head(20)
    print(f"\n================ {method}: TEST ONLY top high-energy cases ================")
    print(top.to_string(index=False))
    top.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_top_high_energy_cases.csv", index=False)

    unc = uncertainty_detection(method, df)
    unc_rows.append(unc)

pooled_df = pd.DataFrame(pooled_rows)
per_fold_df = pd.concat(per_fold_rows, ignore_index=True) if per_fold_rows else pd.DataFrame()
mean_std_df = pd.DataFrame(mean_std_rows)
unc_df = pd.DataFrame(unc_rows)

print("\n\n================ FINAL TABLE 1: pooled classification ================")
print(pooled_df.to_string(index=False))

print("\n\n================ FINAL TABLE 1b: per-fold mean ± std ================")
print(mean_std_df.to_string(index=False))

print("\n\n================ FINAL TABLE 2: uncertainty detection ================")
print(unc_df.to_string(index=False))

pooled_df.to_csv(OUT_DIR / "final_pooled_classification.csv", index=False)
per_fold_df.to_csv(OUT_DIR / "final_per_fold_classification.csv", index=False)
mean_std_df.to_csv(OUT_DIR / "final_mean_std_classification.csv", index=False)
unc_df.to_csv(OUT_DIR / "final_uncertainty_detection.csv", index=False)

print(f"\nSaved all outputs to: {OUT_DIR.resolve()}")
