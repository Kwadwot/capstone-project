"""Download + extract UTKFace and Tiny ImageNet from Kaggle into ./data/.

Public entry point: ``ensure_datasets()``. Idempotent — safe to re-run.

Kaggle credentials must be at ``~/.kaggle/kaggle.json``. If absent, the
helper prints setup instructions before re-raising.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import kagglehub


UTKFACE_SLUG = "jangedoo/utkface-new"
TINY_IMAGENET_SLUG = "akash2sharma/tiny-imagenet"

_UTKFACE_FILENAME_RE = re.compile(r"^(\d+)_([01])_([0-4])_")

_CREDENTIALS_HELP = """
Kaggle credentials not found. To set up:
  1. Visit https://www.kaggle.com/settings -> 'Create New API Token'
     (downloads kaggle.json).
  2. Place it at: C:\\Users\\Savag\\.kaggle\\kaggle.json
  3. Re-run this helper.
""".strip()


def _kagglehub_download(slug: str) -> Path:
    try:
        return Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        msg = str(exc).lower()
        if "credentials" in msg or "authenticat" in msg or "401" in msg or "403" in msg:
            print(_CREDENTIALS_HELP)
        raise


def _mirror_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _find_subdir(root: Path, name: str) -> Path:
    if root.name == name:
        return root
    direct = root / name
    if direct.is_dir():
        return direct
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not locate '{name}' under {root}. Kaggle dataset layout may have changed."
        )
    return matches[0]


def _verify_utkface(dst: Path) -> None:
    files = list(dst.glob("*.jpg"))
    if len(files) < 20_000:
        raise RuntimeError(
            f"UTKFace at {dst} has only {len(files)} .jpg files (expected >=20000). "
            "Extract looks incomplete — delete the folder and re-run."
        )
    if not any(_UTKFACE_FILENAME_RE.match(p.name) for p in files[:100]):
        sample = [p.name for p in files[:3]]
        raise RuntimeError(
            f"UTKFace filenames at {dst} do not match expected '{{age}}_{{gender}}_{{race}}_...' "
            f"pattern. Wrong dataset variant? Samples: {sample}"
        )


def _verify_tiny_imagenet(dst: Path) -> None:
    train = dst / "train"
    if not train.is_dir():
        raise RuntimeError(f"Tiny ImageNet missing train/ directory at {dst}.")
    class_dirs = [p for p in train.iterdir() if p.is_dir()]
    if len(class_dirs) != 200:
        raise RuntimeError(
            f"Tiny ImageNet at {dst} has {len(class_dirs)} class dirs under train/ (expected 200). "
            "Extract looks incomplete — delete the folder and re-run."
        )
    wnids = dst / "wnids.txt"
    if not wnids.is_file():
        raise RuntimeError(f"Tiny ImageNet missing wnids.txt at {dst}.")


def ensure_utkface(data_dir: Path = Path("./data")) -> Path:
    """Ensure UTKFace is at ``<data_dir>/utkface/UTKFace/``. Returns that path."""
    data_dir = Path(data_dir).resolve()
    target = data_dir / "utkface" / "UTKFace"
    if target.is_dir():
        _verify_utkface(target)
        return target

    print(f"Downloading UTKFace via kagglehub ({UTKFACE_SLUG})...")
    cached = _kagglehub_download(UTKFACE_SLUG)
    src = _find_subdir(cached, "UTKFace")
    print(f"Copying {src} -> {target}")
    _mirror_into(src, target)
    _verify_utkface(target)
    return target


def ensure_tiny_imagenet(data_dir: Path = Path("./data")) -> Path:
    """Ensure Tiny ImageNet is at ``<data_dir>/tiny-imagenet-200/``. Returns that path."""
    data_dir = Path(data_dir).resolve()
    target = data_dir / "tiny-imagenet-200"
    if target.is_dir():
        _verify_tiny_imagenet(target)
        return target

    print(f"Downloading Tiny ImageNet via kagglehub ({TINY_IMAGENET_SLUG})...")
    cached = _kagglehub_download(TINY_IMAGENET_SLUG)
    src = _find_subdir(cached, "tiny-imagenet-200")
    print(f"Copying {src} -> {target}")
    _mirror_into(src, target)
    _verify_tiny_imagenet(target)
    return target


def ensure_datasets(data_dir: Path = Path("./data")) -> dict[str, Path]:
    return {
        "utkface": ensure_utkface(data_dir),
        "tiny_imagenet": ensure_tiny_imagenet(data_dir),
    }


if __name__ == "__main__":
    paths = ensure_datasets()
    for name, p in paths.items():
        print(f"{name}: {p}")
