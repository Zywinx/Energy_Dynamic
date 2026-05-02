from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Allow running as: python tools/export_energy_scores.py -h
# Without this, Python puts tools/ on sys.path but not the CLAM root.
_CLAM_ROOT = Path(__file__).resolve().parents[1]
if str(_CLAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLAM_ROOT))

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_SB
from models.model_eclam import EnergyCLAM_SB
from utils.energy_utils import (
    compute_active_score,
    compute_confidence,
    compute_free_energy,
    compute_margin,
    compute_softmax_entropy,
)
from utils.utils import get_simple_loader,  get_split_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(args):
    if args.task == "task_hzey_stage1_aonly_bm":
        return Generic_MIL_Dataset(
            csv_path="dataset_csv/hzey_stage1_aonly_bm.csv",
            data_dir=os.path.join(args.data_root_dir, "hzey_sdpc_resnet50"),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={"benign": 0, "malignant": 1},
            label_col="label",
            ignore=[],
        )
    if args.task == "task_hzey_quick_bm":
        return Generic_MIL_Dataset(
            csv_path="dataset_csv/hzey_quick_done_dataset.csv",
            data_dir=os.path.join(args.data_root_dir, "hzey_sdpc_resnet50"),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={"benign": 0, "malignant": 1},
            label_col="label",
            ignore=[],
        )
    raise NotImplementedError(f"Unsupported task: {args.task}")


def resolve_split_csv(split_dir: str, fold: int) -> str:
    split_path = Path(split_dir)
    if split_path.is_file():
        return str(split_path)
    candidates = [
        split_path / f"splits_{fold}.csv",
        Path("splits") / split_path / f"splits_{fold}.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find split CSV for fold={fold}. Tried: {[str(p) for p in candidates]}"
    )


def build_instance_loss(args):
    if args.inst_loss == "svm":
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        return loss_fn.cuda() if DEVICE.type == "cuda" else loss_fn
    return nn.CrossEntropyLoss()


def build_model(args):
    common = dict(
        dropout=args.drop_out,
        n_classes=args.n_classes,
        embed_dim=args.embed_dim,
        size_arg=args.model_size,
        k_sample=args.B,
        instance_loss_fn=build_instance_loss(args),
    )
    if args.model_type == "eclam_sb":
        model = EnergyCLAM_SB(
            **common,
            energy_enable=True,
            energy_temperature=args.temperature,
        )
    elif args.model_type == "clam_sb":
        # Baseline CLAM-SB can also be assigned slide-level energy post hoc
        # because energy is computed from slide-level logits.
        model = CLAM_SB(**common)
    else:
        raise ValueError("--model_type must be clam_sb or eclam_sb")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing checkpoint keys:", missing)
    if unexpected:
        print("[WARN] Unexpected checkpoint keys:", unexpected)
    return model.to(DEVICE).eval()


def iter_named_splits(args, dataset) -> Iterable[Tuple[str, object]]:
    split_csv = resolve_split_csv(args.split_dir, args.fold)
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=split_csv)
    split_map = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if args.which_split == "all":
        for name in ["train", "val", "test"]:
            yield name, split_map[name]
    else:
        yield args.which_split, split_map[args.which_split]


def get_slide_id(split_dataset, idx: int) -> str:
    if hasattr(split_dataset, "slide_data") and "slide_id" in split_dataset.slide_data.columns:
        return str(split_dataset.slide_data["slide_id"].iloc[idx])
    if hasattr(split_dataset, "slide_data") and "case_id" in split_dataset.slide_data.columns:
        return str(split_dataset.slide_data["case_id"].iloc[idx])
    return str(idx)


def error_type(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return "NA"


def scalar_from_tensor(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().view(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


def infer_one(model, args, data, label):
    # Do not pass return_energy here; baseline CLAM_SB does not support it.
    logits, y_prob, y_hat, _, _ = model(data, label=label, instance_eval=False)
    energy = compute_free_energy(logits, temperature=args.temperature)
    entropy = compute_softmax_entropy(y_prob)
    confidence = compute_confidence(y_prob)
    margin = compute_margin(y_prob)
    return logits, y_prob, y_hat, energy, entropy, confidence, margin


def export_scores(args):
    dataset = build_dataset(args)
    model = build_model(args)
    model.eval()

    rows = []

    for split_name, split_dataset in iter_named_splits(args, dataset):
        # Use deterministic sequential iteration over the full selected split.
        # Do not use CLAM's testing=True subset sampling here.
        loader = get_split_loader(split_dataset, testing=False)

        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            with torch.inference_mode():
                _, y_prob, y_hat, energy, entropy, confidence, margin = infer_one(
                    model, args, data, label
                )

            probs = y_prob.detach().cpu().view(-1).numpy()
            label_i = int(label.detach().cpu().item())
            pred_i = int(y_hat.detach().cpu().item())

            # Recover slide_id from the split dataset row order.
            if hasattr(split_dataset, "slide_data"):
                row_data = split_dataset.slide_data.iloc[batch_idx]
                if "slide_id" in row_data.index:
                    slide_id = row_data["slide_id"]
                elif "case_id" in row_data.index:
                    slide_id = row_data["case_id"]
                else:
                    slide_id = row_data.iloc[0]
            else:
                slide_id = str(batch_idx)

            energy_v = float(energy.detach().cpu().view(-1)[0].item())
            entropy_v = float(entropy.detach().cpu().view(-1)[0].item())
            confidence_v = float(confidence.detach().cpu().view(-1)[0].item())
            margin_v = float(margin.detach().cpu().view(-1)[0].item())

            correctness = int(pred_i == label_i)

            if label_i == 1 and pred_i == 1:
                error_type = "TP"
            elif label_i == 0 and pred_i == 0:
                error_type = "TN"
            elif label_i == 0 and pred_i == 1:
                error_type = "FP"
            elif label_i == 1 and pred_i == 0:
                error_type = "FN"
            else:
                error_type = "NA"

            prob_benign = float(probs[0]) if len(probs) > 0 else float("nan")
            prob_malignant = float(probs[1]) if len(probs) > 1 else float("nan")

            rows.append({
                "fold": int(args.fold),
                "split": split_name,
                "slide_id": slide_id,
                "label": label_i,
                "pred": pred_i,
                "prob_benign": prob_benign,
                "prob_malignant": prob_malignant,
                "confidence": confidence_v,
                "entropy": entropy_v,
                "margin": margin_v,
                "energy": energy_v,
                "active_score_energy": float(compute_active_score(
                    torch.tensor([energy_v]), mode="energy"
                ).view(-1)[0].item()),
                "active_score_entropy": float(compute_active_score(
                    torch.tensor([energy_v]), entropy=torch.tensor([entropy_v]), mode="entropy"
                ).view(-1)[0].item()),
                "active_score_energy_entropy": float(compute_active_score(
                    torch.tensor([energy_v]), entropy=torch.tensor([entropy_v]), mode="energy_entropy"
                ).view(-1)[0].item()),
                "active_score_energy_low_margin": float(compute_active_score(
                    torch.tensor([energy_v]), margin=torch.tensor([margin_v]), mode="energy_low_margin"
                ).view(-1)[0].item()),
                "correctness": correctness,
                "error_type": error_type,
            })

    if len(rows) == 0:
        raise RuntimeError("No rows were exported. Check split_dir, fold, and selected split.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_csv = save_dir / f"energy_scores_fold_{args.fold}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(f"[rows] {len(rows)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Export slide-level energy, uncertainty, and active-learning scores for CLAM-SB/ECLAM-SB."
    )
    p.add_argument("--task", type=str, required=True,
                   choices=["task_hzey_stage1_aonly_bm", "task_hzey_quick_bm"])
    p.add_argument("--data_root_dir", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--model_type", type=str, default="eclam_sb", choices=["clam_sb", "eclam_sb"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--which_split", type=str, default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--drop_out", type=float, default=0.25)
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--model_size", type=str, default="small", choices=["small", "big"])
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--inst_loss", type=str, default="svm", choices=["svm", "ce"])
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    export_scores(parse_args())
