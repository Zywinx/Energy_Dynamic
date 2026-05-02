from pathlib import Path
import os
import re
import pandas as pd

ROOT = Path("/data/xuewz/WSI_PRE/CLAM_0423")
LINK_DIR = ROOT / "data/raw/formal_sdpc_flat"
META_DIR = ROOT / "data/meta"

SRCS = [
    (Path("/data/zhangsj/hzey_data/病理切片/恶性"), "malignant"),
    (Path("/data/zhangsj/hzey_data/病理切片/良性"), "benign"),
]

LINK_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

def split_doctor(stem: str):
    stem = stem.strip()
    m = re.match(r"^(.*?)[,，]\s*([^,，]+)$", stem)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return stem, ""

def normalize_slide_core(stem_no_doc: str):
    s = stem_no_doc.strip().replace(" ", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("/", "_").replace("\\", "_")
    return s

def parse_case_id(slide_core: str):
    # 兼容: 22-36133..., F22-06001..., F24-02672...
    m = re.match(r"^((?:[FQ])?\d{2})[-_]?(\d+)(.*)$", slide_core, flags=re.IGNORECASE)
    if not m:
        return None, None, None
    year_prefix = m.group(1).upper()
    path_no = m.group(2)
    rest = m.group(3)
    case_id = f"{year_prefix}-{path_no}"

    # rest 里前面通常是 A01 / A01+A02 / A01+H01 之类
    specimen_m = re.match(r"^([A-Z]\d+(?:\+[A-Z]\d+)*)", rest)
    specimen_token = specimen_m.group(1) if specimen_m else ""
    return case_id, year_prefix, specimen_token

rows = []
used_slide_ids = {}

for src_dir, label in SRCS:
    files = sorted(src_dir.glob("*.sdpc"))
    for fp in files:
        orig_filename = fp.name
        raw_stem = fp.stem

        stem_no_doc, doctor_tag = split_doctor(raw_stem)
        slide_core = normalize_slide_core(stem_no_doc)

        case_id, year_prefix, specimen_token = parse_case_id(slide_core)
        if case_id is None:
            print(f"[WARN] 无法解析 case_id: {fp}")
            continue

        slide_id = slide_core
        if slide_id in used_slide_ids and used_slide_ids[slide_id] != str(fp):
            idx = 2
            base = slide_id
            while f"{base}__dup{idx}" in used_slide_ids:
                idx += 1
            slide_id = f"{base}__dup{idx}"

        used_slide_ids[slide_id] = str(fp)

        link_path = LINK_DIR / f"{slide_id}.sdpc"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(str(fp), str(link_path))

        rows.append({
            "label": label,
            "case_id": case_id,
            "slide_id": slide_id,
            "year_prefix": year_prefix,
            "specimen_token": specimen_token,
            "doctor_tag": doctor_tag,
            "orig_filename": orig_filename,
            "source_path": str(fp),
            "link_path": str(link_path),
        })

df = pd.DataFrame(rows).sort_values(["label", "case_id", "slide_id"]).reset_index(drop=True)
out_csv = META_DIR / "hzey_manifest_all.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"saved: {out_csv}")
print(df.head(20).to_string(index=False))
print("\nslides by label:")
print(df.groupby("label")["slide_id"].nunique())

print("\ncases by label:")
print(df.groupby("label")["case_id"].nunique())