import sys
from pathlib import Path

CLAM_ROOT = Path(__file__).resolve().parents[1]
if str(CLAM_ROOT) not in sys.path:
    sys.path.insert(0, str(CLAM_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from models.model_clam import CLAM_SB
from vis_utils.heatmap_utils import drawHeatmap
from wsi_core.WholeSlideImage import WholeSlideImage


BASE = Path("/data/xuewz/WSI_PRE/CLAM_0423")
ECLAM = BASE / "code/CLAM_ECLAM"

REP_CSV = BASE / "analysis/ECLAM_key_logs/Representative Case Heatmap/representative_case_selection_TOP1.csv"

RAW_SLIDE_DIR = BASE / "data/raw/formal_sdpc_flat"
SLIDE_EXT = ".sdpc"

FEATURE_DIR = BASE / "data/features/hzey_sdpc_resnet50/pt_files"
COORD_DIR = BASE / "data/patching/hzey_sdpc/patches"

OUT_ROOT = BASE / "analysis/ECLAM_key_logs/Representative Case Heatmap/reuse_pt_h5_final"

METHODS = {
    "baseline": {
        "display": "CLAM-SB baseline",
        "ckpt": BASE / "code/CLAM/results/hzey_stage1_aonly_bm_clam_sb_r50_s1/s_0_checkpoint.pt",
    },
    "sigmoid": {
        "display": "ECLAM-SB sigmoid",
        "ckpt": ECLAM / "results/hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_s1/s_0_checkpoint.pt",
    },
}

PATCH_SIZE = 256
TOPK = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path):
    model = CLAM_SB(
        gate=True,
        size_arg="small",
        dropout=0.25,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=None,
        subtyping=False,
        embed_dim=1024,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    clean = {}
    for k, v in ckpt.items():
        clean[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[load ckpt] {ckpt_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    model.to(DEVICE)
    model.eval()
    return model


def load_coords(h5_path: Path):
    with h5py.File(str(h5_path), "r") as f:
        if "coords" not in f:
            raise RuntimeError(f"{h5_path} missing coords")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)

    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)

    patch_size = attrs.get("patch_size", PATCH_SIZE)
    try:
        patch_size = int(patch_size)
    except Exception:
        patch_size = PATCH_SIZE

    return coords, patch_size


def rank_percentile(scores):
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(np.argsort(scores))
    pct = 100.0 * order / max(len(scores) - 1, 1)
    return pct.astype(np.float32)


def compute_attention(model, pt_path: Path):
    features = torch.load(str(pt_path), map_location="cpu")

    if isinstance(features, dict):
        if "features" in features:
            features = features["features"]
        else:
            raise RuntimeError(f"{pt_path} is dict but missing features key")

    features = torch.as_tensor(features).float()

    with torch.inference_mode():
        A = model(features.to(DEVICE), attention_only=True)
        if isinstance(A, tuple):
            A = A[0]
        A = A.detach().cpu().reshape(-1).numpy()

    return A


def save_attention_h5(out_h5: Path, scores_pct, coords):
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_h5), "w") as f:
        f.create_dataset("attention_scores", data=scores_pct.reshape(-1, 1))
        f.create_dataset("coords", data=coords.astype(np.int64))
    return out_h5


def save_top_patches(slide_path: Path, coords, scores_pct, out_dir: Path, patch_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    wsi = WholeSlideImage(str(slide_path)).getOpenSlide()
    idxs = np.argsort(scores_pct)[::-1][:TOPK]

    saved = []
    for rank, idx in enumerate(idxs):
        x, y = coords[idx]
        score = scores_pct[idx]
        img = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
        out = out_dir / f"{rank}_{slide_path.stem}_x_{int(x)}_y_{int(y)}_a_{score:.3f}.png"
        img.save(out)
        saved.append(out)

    return saved


def render_heatmap(slide_path: Path, coords, scores_pct, out_jpg: Path, patch_size: int):
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    heatmap = drawHeatmap(
        scores=scores_pct.reshape(-1, 1),
        coords=coords,
        slide_path=str(slide_path),
        vis_level=-1,
        patch_size=(patch_size, patch_size),
        alpha=0.4,
        blur=False,
        overlap=0.0,
        segment=False,
        use_holes=False,
        convert_to_percentiles=False,
        binarize=False,
        blank_canvas=False,
        cmap="jet",
    )

    if isinstance(heatmap, Image.Image):
        heatmap.save(out_jpg)
    else:
        Image.fromarray(np.asarray(heatmap)).save(out_jpg)

    return out_jpg


def main():
    if not REP_CSV.exists():
        raise FileNotFoundError(REP_CSV)

    rep = pd.read_csv(REP_CSV)

    required = ["selection_category", "slide_id", "label", "pred_baseline", "pred_sigmoid"]
    missing = [c for c in required if c not in rep.columns]
    if missing:
        raise RuntimeError(f"representative_case_selection_TOP1.csv missing columns: {missing}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for method_key, minfo in METHODS.items():
        model = load_model(minfo["ckpt"])

        for _, r in rep.iterrows():
            category = str(r["selection_category"])
            sid = str(r["slide_id"])

            slide_path = RAW_SLIDE_DIR / f"{sid}{SLIDE_EXT}"
            pt_path = FEATURE_DIR / f"{sid}.pt"
            coord_h5 = COORD_DIR / f"{sid}.h5"

            if not slide_path.exists():
                raise FileNotFoundError(f"Missing raw slide: {slide_path}")
            if not pt_path.exists():
                raise FileNotFoundError(f"Missing feature pt: {pt_path}")
            if not coord_h5.exists():
                raise FileNotFoundError(f"Missing coords h5: {coord_h5}")

            print("\n" + "=" * 100)
            print(f"[{method_key}] {category} | {sid}")
            print("slide:", slide_path)
            print("pt:", pt_path)
            print("coords:", coord_h5)

            coords, patch_size = load_coords(coord_h5)
            A_raw = compute_attention(model, pt_path)

            if len(A_raw) != len(coords):
                raise RuntimeError(f"{sid}: attention N={len(A_raw)} != coords N={len(coords)}")

            A_pct = rank_percentile(A_raw)

            case_dir = OUT_ROOT / category / sid / method_key
            h5_out = case_dir / f"{sid}_attention_scores_reuse_pt_h5.h5"
            jpg_out = case_dir / f"{sid}_{method_key}_attention_heatmap.jpg"
            patch_dir = case_dir / "top_patches"

            save_attention_h5(h5_out, A_pct, coords)
            render_heatmap(slide_path, coords, A_pct, jpg_out, patch_size)
            top_patch_files = save_top_patches(slide_path, coords, A_pct, patch_dir, patch_size)

            print("[saved h5]", h5_out)
            print("[saved heatmap]", jpg_out)
            print("[saved top patches]", len(top_patch_files), patch_dir)

            manifest_rows.append({
                "selection_category": category,
                "slide_id": sid,
                "method": method_key,
                "method_display": minfo["display"],
                "label": r["label"],
                "pred_baseline": r["pred_baseline"],
                "pred_sigmoid": r["pred_sigmoid"],
                "attention_h5": str(h5_out),
                "heatmap_jpg": str(jpg_out),
                "top_patches_dir": str(patch_dir),
                "patch_size": patch_size,
                "n_coords": len(coords),
                "attention_min": float(A_pct.min()),
                "attention_max": float(A_pct.max()),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_ROOT / "reuse_pt_h5_heatmap_manifest.csv", index=False, encoding="utf-8-sig")
    print("\n===== DONE =====")
    print(OUT_ROOT)
    print(OUT_ROOT / "reuse_pt_h5_heatmap_manifest.csv")


if __name__ == "__main__":
    main()
