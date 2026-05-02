from pathlib import Path
import numpy as np
from PIL import Image

def open_wsi(path):
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".sdpc":
        try:
            import opensdpc
            return opensdpc.OpenSdpc(path)
        except Exception:
            import sdpc
            return sdpc.Sdpc(path)

    import openslide
    return openslide.open_slide(path)

def get_properties(slide):
    return getattr(slide, "properties", {})

def get_level_dimensions(slide):
    return slide.level_dimensions

def get_level_downsamples(slide):
    if hasattr(slide, "level_downsamples"):
        return slide.level_downsamples
    if hasattr(slide, "level_downsample"):
        return slide.level_downsample
    raise AttributeError("slide has no level_downsamples / level_downsample")

def read_region_rgb(slide, location, level, size):
    img = slide.read_region(location, level, size)
    if hasattr(img, "convert"):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    return Image.fromarray(arr).convert("RGB")