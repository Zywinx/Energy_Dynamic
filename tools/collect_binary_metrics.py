from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def binary_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Rank-based Mann-Whitney U AUC with average ranks for ties.
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def metrics_for_df(df):
    y = df["label"].astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    score = df["prob_malignant"].astype(float).to_numpy()
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    bal = (sens + spec) / 2.0
    auc = binary_auc(y, score)
    return {
        "AUC": auc,
        "ACC": acc,
        "F1": f1,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "Balanced_Accuracy": bal,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "N": int(len(df)),
    }


def main(args):
    files = sorted(Path().glob(args.glob))
    if not files:
        files = sorted(Path(args.input_dir).glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob={args.glob} under input_dir={args.input_dir}")

    rows = []
    all_df = []
    for f in files:
        df = pd.read_csv(f)
        if args.which_split != "all" and "split" in df.columns:
            df = df[df["split"] == args.which_split].copy()
        if df.empty:
            print(f"[skip-empty] {f}")
            continue
        m = metrics_for_df(df)
        m["file"] = str(f)
        if "fold" in df.columns:
            m["fold"] = int(df["fold"].iloc[0])
        rows.append(m)
        all_df.append(df)

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise RuntimeError("No non-empty files after filtering.")

    metric_cols = ["AUC", "ACC", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_Accuracy"]
    summary_rows = []
    for col in metric_cols:
        summary_rows.append({
            "metric": col,
            "mean": per_fold[col].mean(skipna=True),
            "std": per_fold[col].std(skipna=True, ddof=1),
        })
    summary = pd.DataFrame(summary_rows)

    pooled = pd.concat(all_df, ignore_index=True)
    pooled_metrics = metrics_for_df(pooled)
    pooled_conf = pd.DataFrame([{
        "TN": pooled_metrics["TN"],
        "FP": pooled_metrics["FP"],
        "FN": pooled_metrics["FN"],
        "TP": pooled_metrics["TP"],
        "N": pooled_metrics["N"],
    }])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "metrics_per_fold.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    pooled_conf.to_csv(out_dir / "pooled_confusion_matrix.csv", index=False)
    pd.DataFrame([pooled_metrics]).to_csv(out_dir / "pooled_metrics.csv", index=False)
    print(f"[saved] {out_dir / 'metrics_per_fold.csv'}")
    print(f"[saved] {out_dir / 'metrics_summary.csv'}")
    print(f"[saved] {out_dir / 'pooled_confusion_matrix.csv'}")
    print(summary)
    print(pooled_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect binary metrics from energy_scores_fold_*.csv files.")
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--glob", type=str, default="energy_scores_fold_*.csv")
    parser.add_argument("--which_split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--out_dir", type=str, required=True)
    main(parser.parse_args())
