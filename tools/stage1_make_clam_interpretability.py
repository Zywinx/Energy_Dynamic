import sys
from pathlib import Path

CLAM_ROOT = Path("/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM")
sys.path.insert(0, str(CLAM_ROOT))

from pathlib import Path
import re
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.model_clam import CLAM_SB

ROOT = Path("/data/xuewz/WSI_PRE/CLAM_0423")
CLAM = ROOT / "code/CLAM"

OUT_DIR = ROOT / "analysis/stage1_clam_tables_figures/figure2_attention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_CSV = ROOT / "analysis/stage1_clam_tables_figures/figure2_representative_cases.csv"
SELECTED_CSV = ROOT / "data/meta/hzey_stage1_aonly_selected.csv"

FEATURE_DIR = ROOT / "data/features/hzey_sdpc_resnet50/pt_files"
PATCH_DIR = ROOT / "data/patching/hzey_sdpc/patches"
MODEL_DIR = CLAM / "results/hzey_stage1_aonly_bm_clam_sb_r50_s1"

TOP_K = 20
PATCH_SIZE = 256
TARGET_PATCH_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def safe_name(s):
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.+-]+", "_", str(s))

def load_model(fold):
    model = CLAM_SB(
        dropout=True,
        n_classes=2,
        embed_dim=1024,
        size_arg="small"
    )
    ckpt = torch.load(MODEL_DIR / f"s_{fold}_checkpoint.pt", map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    # handle DataParallel prefix if needed
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def compute_attention(model, features):
    features = features.to(DEVICE)
    with torch.no_grad():
        A_raw, h = model.attention_net(features)
        A = torch.transpose(A_raw, 1, 0)
        A = torch.softmax(A, dim=1).squeeze(0)
    return A.detach().cpu().numpy()

def read_coords(slide_id):
    h5_path = PATCH_DIR / f"{slide_id}.h5"
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
    return coords

def try_export_top_patch_images(slide_id, coords, scores, out_dir, source_path):
    patch_dir = out_dir / "top_patches"
    patch_dir.mkdir(exist_ok=True)

    try:
        from wsi_core.slide_backend import open_wsi, read_region_rgb
    except Exception as e:
        print("[WARN] Cannot import sdpc slide backend:", e)
        return

    if not source_path or str(source_path) == "nan":
        print("[WARN] Missing source_path for", slide_id)
        return

    try:
        slide = open_wsi(str(source_path))
    except Exception as e:
        print("[WARN] Cannot open WSI:", source_path, e)
        return

    order = np.argsort(scores)[::-1][:TOP_K]
    for rank, idx in enumerate(order, start=1):
        x, y = coords[idx]
        try:
            img = read_region_rgb(slide, (int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE))
            if TARGET_PATCH_SIZE:
                img = img.resize((TARGET_PATCH_SIZE, TARGET_PATCH_SIZE))
            img.save(patch_dir / f"rank_{rank:02d}_idx_{idx}_x{x}_y{y}_attn_{scores[idx]:.6f}.png")
        except Exception as e:
            print("[WARN] Failed patch crop:", slide_id, rank, e)

def main():
    rep = pd.read_csv(REP_CSV)
    selected = pd.read_csv(SELECTED_CSV)

    meta_cols = [c for c in ["slide_id", "case_id", "label", "source_path", "orig_filename", "selection_category"] if c in selected.columns]
    rep = rep.merge(selected[meta_cols], on="slide_id", how="left")
    rep.to_csv(OUT_DIR / "representative_cases_with_meta.csv", index=False, encoding="utf-8-sig")

    for _, r in rep.iterrows():
        slide_id = str(r["slide_id"])
        fold = int(r["fold"])
        case_type = str(r["case_type"])
        label = int(float(r["Y"]))
        pred = int(float(r["Y_hat"]))
        p1 = float(r["p_1"])
        source_path = r.get("source_path", "")

        print("Processing:", case_type, "fold", fold, slide_id)

        out_case = OUT_DIR / f"{case_type}_fold{fold}_{safe_name(slide_id)}"
        out_case.mkdir(parents=True, exist_ok=True)

        feat_path = FEATURE_DIR / f"{slide_id}.pt"
        features = torch.load(feat_path, map_location="cpu")
        if isinstance(features, dict):
            # fallback for unusual saved format
            features = features.get("features", next(iter(features.values())))
        features = features.float()

        coords = read_coords(slide_id)
        model = load_model(fold)
        scores = compute_attention(model, features)

        n = min(len(scores), len(coords))
        scores = scores[:n]
        coords = coords[:n]

        attn_df = pd.DataFrame({
            "idx": np.arange(n),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "attention": scores,
        })
        attn_df = attn_df.sort_values("attention", ascending=False)
        attn_df.to_csv(out_case / "attention_scores.csv", index=False)
        attn_df.head(TOP_K).to_csv(out_case / "top_attention_patches.csv", index=False)

        plt.figure(figsize=(6, 6))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=scores, s=2, cmap="hot")
        plt.gca().invert_yaxis()
        plt.colorbar(sc, label="attention")
        plt.title(f"{case_type} | fold={fold} | Y={label}, pred={pred}, p_mal={p1:.3f}\n{slide_id}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_case / "attention_coord_heatmap.png", dpi=300)
        plt.close()

        try_export_top_patch_images(slide_id, coords, scores, out_case, source_path)

    print("Saved Figure 2 interpretability outputs to:", OUT_DIR)

if __name__ == "__main__":
    main()
