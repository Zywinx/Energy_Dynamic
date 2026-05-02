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
    "ECLAM-SB linear_warmup": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy"),
}

OUT_DIR = Path("eval_results/TEST_ONLY_COMPARISON_baseline_vs_linear")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def auc_safe(y, p):
    if pd.Series(y).nunique() < 2:
        return float("nan")
    return roc_auc_score(y, p)

def load_test_only(method, d):
    files = sorted(d.glob("energy_scores_fold_*.csv"))
    if len(files) != 5:
        raise RuntimeError(f"{method}: expected 5 fold CSVs, found {len(files)} in {d}")

    dfs = []
    for f in files:
        fold = int(f.stem.split("_")[-1])
        df = pd.read_csv(f)
        df["fold"] = fold
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    print(f"\n================ {method}: split counts ================")
    print(all_df["split"].value_counts(dropna=False).to_string())

    test_df = all_df[all_df["split"].astype(str).str.lower() == "test"].copy()

    print(f"\n================ {method}: TEST ONLY shape ================")
    print(test_df.shape)

    if len(test_df) != 50:
        raise RuntimeError(f"{method}: expected 50 test rows, got {len(test_df)}")

    need_cols = [
        "slide_id", "label", "pred",
        "prob_benign", "prob_malignant",
        "energy", "entropy", "confidence", "margin",
        "correctness", "error_type",
    ]
    missing = [c for c in need_cols if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"{method}: missing columns: {missing}")

    print(f"\n================ {method}: NaN check ================")
    print(test_df[["energy", "entropy", "confidence", "margin"]].isna().sum().to_string())

    return test_df

def compute_metrics(method, df):
    cm = confusion_matrix(df["label"], df["pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    row = {
        "method": method,
        "n": len(df),
        "AUC": auc_safe(df["label"], df["prob_malignant"]),
        "ACC": accuracy_score(df["label"], df["pred"]),
        "F1": f1_score(df["label"], df["pred"], zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
        "NPV": tn / (tn + fn) if (tn + fn) else float("nan"),
        "Balanced_ACC": balanced_accuracy_score(df["label"], df["pred"]),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    return row

def per_fold_metrics(method, df):
    rows = []
    for fold, g in df.groupby("fold"):
        cm = confusion_matrix(g["label"], g["pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "method": method,
            "fold": fold,
            "n": len(g),
            "AUC": auc_safe(g["label"], g["prob_malignant"]),
            "ACC": accuracy_score(g["label"], g["pred"]),
            "F1": f1_score(g["label"], g["pred"], zero_division=0),
            "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
            "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
            "NPV": tn / (tn + fn) if (tn + fn) else float("nan"),
            "Balanced_ACC": balanced_accuracy_score(g["label"], g["pred"]),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })
    return pd.DataFrame(rows)

all_pooled = []
all_per_fold = []

for method, d in METHOD_DIRS.items():
    if not d.exists():
        print(f"\n[SKIP] {method}: directory not found: {d}")
        continue

    df = load_test_only(method, d)
    df.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_test_only_predictions.csv", index=False)

    pooled = compute_metrics(method, df)
    all_pooled.append(pooled)

    print(f"\n================ {method}: TEST ONLY pooled confusion matrix ================")
    print(f"TN FP FN TP = {pooled['TN']} {pooled['FP']} {pooled['FN']} {pooled['TP']}")

    print(f"\n================ {method}: TEST ONLY pooled metrics ================")
    for k in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        print(f"{k}={pooled[k]:.4f}")

    pf = per_fold_metrics(method, df)
    all_per_fold.append(pf)

    print(f"\n================ {method}: TEST ONLY per-fold metrics ================")
    print(pf.to_string(index=False))

    print(f"\n================ {method}: TEST ONLY mean ± std ================")
    for k in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        print(f"{k}: {pf[k].mean():.4f} ± {pf[k].std(ddof=1):.4f}")

    energy_table = (
        df.groupby("error_type")[["energy", "entropy", "confidence", "margin"]]
        .agg(["count", "mean", "std", "min", "max"])
    )
    print(f"\n================ {method}: TEST ONLY energy by error_type ================")
    print(energy_table.to_string())
    energy_table.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_energy_by_error_type.csv")

    cols = [
        "fold", "slide_id", "label", "pred",
        "prob_benign", "prob_malignant",
        "energy", "entropy", "margin", "confidence", "error_type",
    ]
    top = df.sort_values("energy", ascending=False)[cols].head(20)
    print(f"\n================ {method}: TEST ONLY top high-energy cases ================")
    print(top.to_string(index=False))
    top.to_csv(OUT_DIR / f"{method.replace(' ', '_')}_top_high_energy_cases.csv", index=False)

if all_pooled:
    pooled_df = pd.DataFrame(all_pooled)
    print("\n\n================ POOLED COMPARISON ================")
    print(pooled_df.to_string(index=False))
    pooled_df.to_csv(OUT_DIR / "pooled_comparison.csv", index=False)

if all_per_fold:
    pf_df = pd.concat(all_per_fold, ignore_index=True)
    print("\n\n================ PER-FOLD MEAN ± STD COMPARISON ================")
    rows = []
    for method, g in pf_df.groupby("method"):
        for k in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
            rows.append({
                "method": method,
                "metric": k,
                "mean": g[k].mean(),
                "std": g[k].std(ddof=1),
            })
    ms = pd.DataFrame(rows)
    print(ms.to_string(index=False))
    pf_df.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)
    ms.to_csv(OUT_DIR / "mean_std_comparison.csv", index=False)

print(f"\nSaved to: {OUT_DIR.resolve()}")
