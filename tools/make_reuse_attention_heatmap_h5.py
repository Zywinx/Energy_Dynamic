import sys
from pathlib import Path

_CLAM_ROOT = Path(__file__).resolve().parents[1]
if str(_CLAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLAM_ROOT))

import os
import h5py
import torch
import numpy as np
from pathlib import Path

from models.model_clam import CLAM_SB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SLIDES = {
    "F24-03483A01+A02": {
        "pt": "/data/xuewz/WSI_PRE/CLAM_0423/data/features/hzey_sdpc_resnet50/pt_files/F24-03483A01+A02.pt",
        "h5": "/data/xuewz/WSI_PRE/CLAM_0423/data/patching/hzey_sdpc/patches/F24-03483A01+A02.h5",
    },
    "F24-05463A01": {
        "pt": "/data/xuewz/WSI_PRE/CLAM_0423/data/features/hzey_sdpc_resnet50/pt_files/F24-05463A01.pt",
        "h5": "/data/xuewz/WSI_PRE/CLAM_0423/data/patching/hzey_sdpc/patches/F24-05463A01.h5",
    },
    "F23-01910A01+H01": {
        "pt": "/data/xuewz/WSI_PRE/CLAM_0423/data/features/hzey_sdpc_resnet50/pt_files/F23-01910A01+H01.pt",
        "h5": "/data/xuewz/WSI_PRE/CLAM_0423/data/patching/hzey_sdpc/patches/F23-01910A01+H01.h5",
    },
    "23-48187A01+A02": {
        "pt": "/data/xuewz/WSI_PRE/CLAM_0423/data/features/hzey_sdpc_resnet50/pt_files/23-48187A01+A02.pt",
        "h5": "/data/xuewz/WSI_PRE/CLAM_0423/data/patching/hzey_sdpc/patches/23-48187A01+A02.h5",
    },
}

METHODS = {
    "REP_CASE_BASELINE_ATTENTION_TOP1": {
        "ckpt": "/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM/results/hzey_stage1_aonly_bm_clam_sb_r50_s1/s_0_checkpoint.pt",
    },
    "REP_CASE_SIGMOID_ATTENTION_TOP1": {
        "ckpt": "/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM/results/hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_s1/s_0_checkpoint.pt",
    },
}

RESULT_ROOT = Path("/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM/heatmaps/results")


def load_coords(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise RuntimeError(f"{h5_path} missing coords")
        coords = f["coords"][:]
    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    return coords


def load_model(ckpt_path):
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

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    clean = {}
    for k, v in ckpt.items():
        nk = k.replace("module.", "")
        clean[nk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[load] {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def percentile_scores(x):
    # top-k 排序不变；转成 0-100，文件名里的 a_ 会更直观
    x = np.asarray(x).reshape(-1)
    order = np.argsort(np.argsort(x))
    pct = 100.0 * order / max(len(x) - 1, 1)
    return pct.reshape(-1, 1).astype(np.float32)


for method, minfo in METHODS.items():
    model = load_model(minfo["ckpt"])

    for sid, paths in SLIDES.items():
        pt_path = Path(paths["pt"])
        h5_path = Path(paths["h5"])

        if not pt_path.exists():
            raise FileNotFoundError(pt_path)
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)

        features = torch.load(pt_path, map_location="cpu")
        if isinstance(features, dict):
            if "features" in features:
                features = features["features"]
            else:
                raise RuntimeError(f"{pt_path} is dict but has no 'features' key")

        features = torch.as_tensor(features).float()
        coords = load_coords(h5_path)

        if features.shape[0] != coords.shape[0]:
            raise RuntimeError(
                f"{sid}: features N={features.shape[0]} != coords N={coords.shape[0]}"
            )

        with torch.inference_mode():
            A = model(features.to(device), attention_only=True)

            if isinstance(A, tuple):
                A = A[0]

            A = A.detach().cpu()

            if A.ndim == 2 and A.shape[0] == 1:
                A = A.view(-1, 1)
            elif A.ndim == 2 and A.shape[1] == 1:
                pass
            else:
                A = A.reshape(-1, 1)

        A_np = A.numpy()
        A_pct = percentile_scores(A_np)

        out_dir = RESULT_ROOT / method / "Unspecified" / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        # create_heatmaps.py 采样时找的就是这个文件
        out_h5 = out_dir / f"{sid}_0.5_roi_False.h5"

        if out_h5.exists() or out_h5.is_symlink():
            out_h5.unlink()

        with h5py.File(out_h5, "w") as f:
            f.create_dataset("attention_scores", data=A_pct)
            f.create_dataset("coords", data=coords)

        print(f"[saved] {out_h5}")
        print(f"        attention_scores={A_pct.shape}, coords={coords.shape}, min/max={A_pct.min():.3f}/{A_pct.max():.3f}")
