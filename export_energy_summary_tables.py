import pandas as pd
from pathlib import Path

METHOD_DIRS = {
    "CLAM-SB baseline": Path("eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy"),
    "ECLAM-SB constant": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_constant_check_energy"),
    "ECLAM-SB linear": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy"),
    "ECLAM-SB sigmoid": Path("eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_energy"),
}

OUT_DIR = Path("/data/xuewz/WSI_PRE/CLAM_0423/analysis/ECLAM_key_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "split",
    "label",
    "pred",
    "energy",
    "entropy",
    "confidence",
    "margin",
    "correctness",
    "error_type",
]

ERROR_TYPE_ORDER = ["TP", "TN", "FP", "FN"]

def load_method_csvs(method, energy_dir):
    files = sorted(energy_dir.glob("energy_scores_fold_*.csv"))
    if len(files) != 5:
        raise RuntimeError(
            f"{method}: expected 5 energy_scores_fold_*.csv files, "
            f"found {len(files)} in {energy_dir}"
        )

    dfs = []
    for f in files:
        fold = int(f.stem.split("_")[-1])
        df = pd.read_csv(f)
        df["fold"] = fold
        df["method"] = method

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{method}, {f}: missing columns: {missing}")

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

all_dfs = []
for method, d in METHOD_DIRS.items():
    if not d.exists():
        raise RuntimeError(f"Missing energy directory for {method}: {d}")
    all_dfs.append(load_method_csvs(method, d))

df = pd.concat(all_dfs, ignore_index=True)

# Normalize correctness in case it is bool / int / string.
if df["correctness"].dtype == bool:
    df["correctness"] = df["correctness"].map({True: "correct", False: "incorrect"})
else:
    df["correctness"] = df["correctness"].astype(str)
    df["correctness"] = df["correctness"].replace({
        "1": "correct",
        "0": "incorrect",
        "True": "correct",
        "False": "incorrect",
        "true": "correct",
        "false": "incorrect",
    })

# Ensure error_type is one of TP/TN/FP/FN if needed.
def infer_error_type(row):
    y = int(row["label"])
    p = int(row["pred"])
    if y == 1 and p == 1:
        return "TP"
    if y == 0 and p == 0:
        return "TN"
    if y == 0 and p == 1:
        return "FP"
    if y == 1 and p == 0:
        return "FN"
    return str(row.get("error_type", "NA"))

df["error_type"] = df.apply(infer_error_type, axis=1)

# Save combined raw table for audit.
df.to_csv(OUT_DIR / "all_methods_energy_scores_combined.csv", index=False)

# ---------------------------------------------------------------------
# Table 1:
# method, split, correctness,
# energy_mean, energy_std, entropy_mean, entropy_std,
# confidence_mean, margin_mean
# ---------------------------------------------------------------------
summary_split_correctness = (
    df.groupby(["method", "split", "correctness"], dropna=False)
      .agg(
          count=("energy", "size"),
          energy_mean=("energy", "mean"),
          energy_std=("energy", "std"),
          entropy_mean=("entropy", "mean"),
          entropy_std=("entropy", "std"),
          confidence_mean=("confidence", "mean"),
          margin_mean=("margin", "mean"),
      )
      .reset_index()
)

# Keep exactly requested leading columns, with count included for sanity.
summary_split_correctness = summary_split_correctness[
    [
        "method",
        "split",
        "correctness",
        "count",
        "energy_mean",
        "energy_std",
        "entropy_mean",
        "entropy_std",
        "confidence_mean",
        "margin_mean",
    ]
]

# ---------------------------------------------------------------------
# Table 2:
# method, error_type, count,
# energy_mean, energy_std, entropy_mean, confidence_mean, margin_mean
# error_type: TP/TN/FP/FN
# ---------------------------------------------------------------------
summary_error_type = (
    df.groupby(["method", "error_type"], dropna=False)
      .agg(
          count=("energy", "size"),
          energy_mean=("energy", "mean"),
          energy_std=("energy", "std"),
          entropy_mean=("entropy", "mean"),
          confidence_mean=("confidence", "mean"),
          margin_mean=("margin", "mean"),
      )
      .reset_index()
)

summary_error_type["error_type"] = pd.Categorical(
    summary_error_type["error_type"],
    categories=ERROR_TYPE_ORDER,
    ordered=True,
)
summary_error_type = summary_error_type.sort_values(["method", "error_type"])

summary_error_type = summary_error_type[
    [
        "method",
        "error_type",
        "count",
        "energy_mean",
        "energy_std",
        "entropy_mean",
        "confidence_mean",
        "margin_mean",
    ]
]

# Also make TEST ONLY versions, because your main paper comparison uses test-only.
test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

summary_split_correctness_test_only = (
    test_df.groupby(["method", "split", "correctness"], dropna=False)
      .agg(
          count=("energy", "size"),
          energy_mean=("energy", "mean"),
          energy_std=("energy", "std"),
          entropy_mean=("entropy", "mean"),
          entropy_std=("entropy", "std"),
          confidence_mean=("confidence", "mean"),
          margin_mean=("margin", "mean"),
      )
      .reset_index()
)

summary_split_correctness_test_only = summary_split_correctness_test_only[
    [
        "method",
        "split",
        "correctness",
        "count",
        "energy_mean",
        "energy_std",
        "entropy_mean",
        "entropy_std",
        "confidence_mean",
        "margin_mean",
    ]
]

summary_error_type_test_only = (
    test_df.groupby(["method", "error_type"], dropna=False)
      .agg(
          count=("energy", "size"),
          energy_mean=("energy", "mean"),
          energy_std=("energy", "std"),
          entropy_mean=("entropy", "mean"),
          confidence_mean=("confidence", "mean"),
          margin_mean=("margin", "mean"),
      )
      .reset_index()
)

summary_error_type_test_only["error_type"] = pd.Categorical(
    summary_error_type_test_only["error_type"],
    categories=ERROR_TYPE_ORDER,
    ordered=True,
)
summary_error_type_test_only = summary_error_type_test_only.sort_values(["method", "error_type"])

summary_error_type_test_only = summary_error_type_test_only[
    [
        "method",
        "error_type",
        "count",
        "energy_mean",
        "energy_std",
        "entropy_mean",
        "confidence_mean",
        "margin_mean",
    ]
]

# Save CSVs.
summary_split_correctness.to_csv(
    OUT_DIR / "energy_summary_by_split_correctness.csv",
    index=False,
)
summary_error_type.to_csv(
    OUT_DIR / "energy_summary_by_error_type.csv",
    index=False,
)

summary_split_correctness_test_only.to_csv(
    OUT_DIR / "energy_summary_by_split_correctness_TEST_ONLY.csv",
    index=False,
)
summary_error_type_test_only.to_csv(
    OUT_DIR / "energy_summary_by_error_type_TEST_ONLY.csv",
    index=False,
)

# Print useful preview.
print("\n===== Saved files =====")
for p in [
    OUT_DIR / "all_methods_energy_scores_combined.csv",
    OUT_DIR / "energy_summary_by_split_correctness.csv",
    OUT_DIR / "energy_summary_by_error_type.csv",
    OUT_DIR / "energy_summary_by_split_correctness_TEST_ONLY.csv",
    OUT_DIR / "energy_summary_by_error_type_TEST_ONLY.csv",
]:
    print(p)

print("\n===== energy_summary_by_split_correctness_TEST_ONLY.csv =====")
print(summary_split_correctness_test_only.to_string(index=False))

print("\n===== energy_summary_by_error_type_TEST_ONLY.csv =====")
print(summary_error_type_test_only.to_string(index=False))

print("\n===== split counts by method =====")
print(
    df.groupby(["method", "split"])
      .size()
      .reset_index(name="count")
      .to_string(index=False)
)

print("\n===== test-only error_type counts by method =====")
print(
    test_df.groupby(["method", "error_type"])
      .size()
      .reset_index(name="count")
      .to_string(index=False)
)
