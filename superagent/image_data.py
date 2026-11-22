"""Small, CPU-only image preprocessing for the structured baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_SIZE = (16, 16)


def flat_images(directory):
    return [p for p in sorted(Path(directory).glob('*')) if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]


def folder_samples(directory):
    directory = Path(directory)
    pairs = [
        (path, path.parent.name)
        for path in sorted(directory.glob("*/*"))
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not pairs:
        raise ValueError(f"{directory.name}: expected images under class directories.")
    return pairs


def pixels(samples):
    from PIL import Image

    rows = []
    for sample in samples:
        if isinstance(sample, Path):
            with Image.open(sample) as image:
                row = np.asarray(image.convert("RGB").resize(_SIZE), dtype=np.float32)
        else:
            values = np.asarray(sample)
            if values.ndim == 3 and values.shape[-1] not in (1, 3, 4) and values.shape[0] in (1, 3, 4):
                values = np.moveaxis(values, 0, -1)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            values = values.astype(float)
            if not np.isfinite(values).all():
                raise ValueError("Image tensor contains nonfinite values.")
            if values.min() >= 0 and values.max() <= 1:
                values *= 255
            image = Image.fromarray(values.clip(0, 255).astype(np.uint8))
            row = np.asarray(image.convert("RGB").resize(_SIZE), dtype=np.float32)
        rows.append(row.ravel() / 255)
    return pd.DataFrame(np.asarray(rows), columns=[f"px{i}" for i in range(_SIZE[0] * _SIZE[1] * 3)])
