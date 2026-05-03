#!/usr/bin/env bash
set -euo pipefail

cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM

export CUDA_VISIBLE_DEVICES=0

TASK="task_hzey_stage1_aonly_bm"
DATA_ROOT="/data/xuewz/WSI_PRE/CLAM_0423/data/features"
SPLIT_DIR="splits/task_hzey_stage1_aonly_bm_100"
BASELINE_CKPT_DIR="/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM/results/hzey_stage1_aonly_bm_clam_sb_r50_s1"
SAVE_DIR="eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy"

mkdir -p "${SAVE_DIR}"

echo "===== Export baseline CLAM-SB energy scores: fold 0-4 ====="

for f in 0 1 2 3 4; do
  CKPT="${BASELINE_CKPT_DIR}/s_${f}_checkpoint.pt"

  if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Missing checkpoint: ${CKPT}"
    exit 1
  fi

  echo "===== fold ${f} ====="
  python tools/export_energy_scores.py \
    --task "${TASK}" \
    --data_root_dir "${DATA_ROOT}" \
    --split_dir "${SPLIT_DIR}" \
    --ckpt_path "${CKPT}" \
    --model_type clam_sb \
    --temperature 1.0 \
    --save_dir "${SAVE_DIR}" \
    --fold "${f}" \
    --embed_dim 1024 \
    --drop_out 0.25
done

echo "===== Summarize baseline test-only energy ====="

python - <<'PY'
import math
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

BASELINE_DIR = Path("eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy")
LINEAR_DIR = Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy")
OUT_DIR = BASELINE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "fold", "split", "slide_id", "label", "pred",
    "prob_benign", "prob_malignant",
    "confidence", "entropy", "margin", "energy",
    "correctness", "error_type",
]

def safe_auc(y_true, y_score):
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")

def summarize_energy_dir(d: Path, name: str):
    files = sorted(d.glob("energy_scores_fold_*.csv"))
    if len(files) != 5:
        raise SystemExit(f"[ERROR] {name}: expected 5 energy_scores_fold_*.csv, found {len(files)} in {d}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        fold = int(f.stem.split("_")[-1])
        if "fold" not in df.columns:
            df["fold"] = fold
        else:
            df["fold"] = fold
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    missing = [c for c in REQUIRED_COLS if c not in all_df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {name}: missing required columns: {missing}")

    print(f"\n\n================ {name}: split counts ================")
    print(all_df["split"].value_counts(dropna=False).to_string())

    test_df = all_df[all_df["split"].astype(str).str.lower() == "test"].copy()

    print(f"\n================ {name}: TEST ONLY shape ================")
    print(test_df.shape)

    if len(test_df) != 50:
        raise SystemExit(f"[ERROR] {name}: test-only rows should be 50, got {len(test_df)}")

    print(f"\n================ {name}: TEST ONLY NaN check ================")
    print(test_df[["energy", "entropy", "confidence", "margin"]].isna().sum().to_string())

    if test_df[["energy", "entropy", "confidence", "margin"]].isna().any().any():
        raise SystemExit(f"[ERROR] {name}: NaN detected in energy/entropy/confidence/margin")

    cm = confusion_matrix(test_df["label"], test_df["pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pooled = {
        "method": name,
        "n": len(test_df),
        "AUC[object Object],[object Object],[object Object],[object Object]": safe_auc(test_df["label"], test_df["prob_malignant"]),
        "ACC": accuracy_score(test_df["label"], test_df["pred"]),
        "F1": f1_score(test_df["label"], test_df["pred"], zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
        "NPV": tn / (tn + fn) if (tn + fn) else float("nan"),
        "Balanced_ACC": balanced_accuracy_score(test_df["label"], test_df["pred"]),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    print(f"\n================ {name}: TEST ONLY pooled confusion matrix ================")
    print(f"TN FP FN TP = {tn} {fp} {fn} {tp}")

    print(f"\n================ {name}: TEST ONLY pooled metrics ================")
    for k in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        print(f"{k}={pooled[k]:.4f}")

    rows = []
    for fold, g in test_df.groupby("fold"):
        cm = confusion_matrix(g["label"], g["pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "method": name,
            "fold": fold,
            "n": len(g),
            "AUC": safe_auc(g["label"], g["prob_malignant"]),
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

    per_fold = pd.DataFrame(rows)

    print(f"\n================ {name}: TEST ONLY per-fold metrics ================")
    print(per_fold.to_string(index=False))

    print(f"\n================ {name}: TEST ONLY mean ± std ================")
    mean_std_rows = []
    for c in ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_ACC"]:
        mean = per_fold[c].mean()
        std = per_fold[c].std(ddof=1)
        mean_std_rows.append({"method": name, "metric": c, "mean": mean, "std": std})
        print(f"{c}: {mean:.4f} ± {std:.4f}")

    mean_std = pd.DataFrame(mean_std_rows)

    energy_by_type = (
        test_df.groupby("error_type")[["energy", "entropy", "confidence", "margin"]]
        .agg(["count", "mean", "std", "min", "max"])
    )

    print(f"\n================ {name}: TEST ONLY energy by error_type ================")
    print(energy_by_type.to_string())

    cols_show = [
        "fold", "slide_id", "label", "pred",
        "prob_benign", "prob_malignant",
        "energy", "entropy", "margin", "confidence", "error_type",
    ]
    top_high_energy = test_df.sort_values("energy", ascending=False)[cols_show].head(30)

    print(f"\n================ {name}: TEST ONLY top high-energy cases ================")
    print(top_high_energy.to_string(index=False))

    # Save artifacts
    prefix = name.replace(" ", "_").replace("/", "_")
    test_df.to_csv(OUT_DIR / f"{prefix}_test_only_predictions.csv", index=False)
    per_fold.to_csv(OUT_DIR / f"{prefix}_test_only_per_fold_metrics.csv", index=False)
    mean_std.to_csv(OUT_DIR / f"{prefix}_test_only_mean_std_metrics.csv", index=False)
    energy_by_type.to_csv(OUT_DIR / f"{prefix}_test_only_energy_by_error_type.csv")
    top_high_energy.to_csv(OUT_DIR / f"{prefix}_test_only_top_high_energy_cases.csv", index=False)

    return {
        "pooled": pooled,
        "per_fold": per_fold,
        "mean_std": mean_std,
        "energy_by_type": energy_by_type,
        "test_df": test_df,
    }

baseline = summarize_energy_dir(BASELINE_DIR, "CLAM-SB baseline")

if LINEAR_DIR.exists():
    linear = summarize_energy_dir(LINEAR_DIR, "ECLAM-SB linear_warmup")

    print("\n\n================ baseline vs linear_warmup: mean ± std comparison ================")
    comp = pd.concat([baseline["mean_std"], linear["mean_std"]], ignore_index=True)
    pivot = comp.pivot(index="metric", columns="method", values=["mean", "std"])
    print(pivot.to_string())

    comp.to_csv(OUT_DIR / "baseline_vs_linear_test_only_mean_std_comparison.csv", index=False)

    print("\n================ baseline vs linear_warmup: pooled comparison ================")
    pooled_comp = pd.DataFrame([baseline["pooled"], linear["pooled"]])
    print(pooled_comp.to_string(index=False))
    pooled_comp.to_csv(OUT_DIR / "baseline_vs_linear_test_only_pooled_comparison.csv", index=False)
else:
    print(f"\n[INFO] Linear warmup energy dir not found, skip comparison: {LINEAR_DIR}")

print("\n===== Saved outputs in =====")
print(OUT_DIR.resolve())
PY

echo "===== Done ====="
