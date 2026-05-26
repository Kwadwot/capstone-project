"""Local replacement for ``mdl.lab2.TrainingDatasetLoader`` + ``get_test_faces``.

Decodes UTKFace + Tiny ImageNet once into ``./data/cache/`` as uint8 ``.npy``
files, then exposes a loader matching MIT's API so the notebook's training
cells need no changes.

Public entry: :func:`prepare_datasets` -> ``(loader, test_faces, keys)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data_download import ensure_datasets


IMAGE_SIZE = 64

# UTKFace label conventions (per project memory):
#   gender: 0=Male, 1=Female
#   race:   0=White, 1=Black, 2=Asian, 3=Indian, 4=Other
RACE_NAMES = ["White", "Black", "Asian", "Indian", "Other"]
GENDER_NAMES_BY_CODE = {0: "Male", 1: "Female"}

# Race-major ordering with Female before Male inside each race, so adjacent
# bars in downstream plots compare M vs. F within the same race.
TEST_KEYS = [
    f"{race} {GENDER_NAMES_BY_CODE[g]}"
    for race in RACE_NAMES
    for g in (1, 0)
]
# Each entry: (race_id, gender_id). Iteration order matches TEST_KEYS.
_TEST_GROUP_DEFS = [
    (race_id, gender_id)
    for race_id in range(len(RACE_NAMES))
    for gender_id in (1, 0)
]

_UTK_NAME_RE = re.compile(r"^(\d+)_([01])_([0-4])_")


def _parse_utkface(name: str) -> tuple[int, int, int] | None:
    m = _UTK_NAME_RE.match(name)
    if not m:
        return None
    age, gender, race = m.groups()
    return int(age), int(gender), int(race)


def _decode_resize(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def _build_utkface_cache(
    utk_dir: Path, cache_dir: Path, image_size: int
) -> tuple[Path, Path]:
    images_path = cache_dir / f"utkface_images_{image_size}.npy"
    labels_path = cache_dir / f"utkface_labels_{image_size}.npy"
    if images_path.exists() and labels_path.exists():
        return images_path, labels_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(utk_dir.glob("*.jpg"))
    images: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    skipped = 0
    for f in tqdm(files, desc="Decoding UTKFace"):
        parsed = _parse_utkface(f.name)
        if parsed is None:
            skipped += 1
            continue
        _, gender, race = parsed
        try:
            img = _decode_resize(f, image_size)
        except Exception:
            skipped += 1
            continue
        images.append(img)
        labels.append((gender, race))

    images_arr = np.stack(images, axis=0)
    labels_arr = np.array(labels, dtype=np.int8)
    np.save(images_path, images_arr)
    np.save(labels_path, labels_arr)
    print(f"UTKFace cache: {len(images_arr)} kept, {skipped} skipped")
    return images_path, labels_path


def _build_tiny_imagenet_cache(
    tin_root: Path, cache_dir: Path, image_size: int, n_samples: int, seed: int
) -> Path:
    images_path = cache_dir / f"tiny_imagenet_images_{image_size}_n{n_samples}_s{seed}.npy"
    if images_path.exists():
        return images_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    train = tin_root / "train"
    class_dirs = sorted(p for p in train.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    per_class = max(1, n_samples // len(class_dirs))

    images: list[np.ndarray] = []
    for cls in tqdm(class_dirs, desc="Decoding Tiny ImageNet"):
        files = sorted((cls / "images").glob("*.JPEG"))
        if not files:
            continue
        idx = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        for i in idx:
            try:
                images.append(_decode_resize(files[i], image_size))
            except Exception:
                continue

    images_arr = np.stack(images, axis=0)
    np.save(images_path, images_arr)
    print(f"Tiny ImageNet cache: {len(images_arr)} samples")
    return images_path


def _make_test_split(
    utk_images: np.ndarray,
    utk_labels: np.ndarray,
    n_per_group: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    genders = utk_labels[:, 0]
    races = utk_labels[:, 1]

    test_faces: list[np.ndarray] = []
    test_indices: list[int] = []
    for race_id, gender_id in _TEST_GROUP_DEFS:
        mask = (races == race_id) & (genders == gender_id)
        candidates = np.where(mask)[0]
        if len(candidates) < n_per_group:
            raise RuntimeError(
                f"Not enough UTKFace samples for race={RACE_NAMES[race_id]} "
                f"gender={GENDER_NAMES_BY_CODE[gender_id]}: {len(candidates)} < {n_per_group}"
            )
        chosen = rng.choice(candidates, size=n_per_group, replace=False)
        test_indices.extend(chosen.tolist())
        group = utk_images[chosen].astype(np.float32) / 255.0
        # channels-last (N, H, W, C) -> channels-first (N, C, H, W)
        test_faces.append(np.transpose(group, (0, 3, 1, 2)).copy())

    train_mask = np.ones(len(utk_images), dtype=bool)
    train_mask[test_indices] = False
    return test_faces, train_mask


class TrainingDatasetLoader:
    """Drop-in replacement for ``mdl.lab2.TrainingDatasetLoader``.

    Holds float32 face and non-face arrays in RAM in ``(N, H, W, C)`` layout.
    ``get_batch`` transposes to ``(B, C, H, W)`` for the model;
    ``get_all_train_faces`` returns the channels-last array unchanged (the
    notebook's ``get_latent_mu`` does its own ``permute(0, 3, 1, 2)``).
    """

    def __init__(self, faces_hwc: np.ndarray, nonfaces_hwc: np.ndarray):
        assert faces_hwc.dtype == np.float32 and nonfaces_hwc.dtype == np.float32
        assert faces_hwc.ndim == 4 and faces_hwc.shape[-1] == 3
        assert nonfaces_hwc.ndim == 4 and nonfaces_hwc.shape[-1] == 3
        self._faces_hwc = faces_hwc
        self._nonfaces_hwc = nonfaces_hwc
        self.n_faces = len(faces_hwc)
        self.n_nonfaces = len(nonfaces_hwc)

    def get_train_size(self) -> int:
        return self.n_faces + self.n_nonfaces

    def get_batch(
        self, batch_size: int, p_pos: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        if p_pos is None:
            face_idx = np.random.choice(self.n_faces, n_pos)
        else:
            face_idx = np.random.choice(self.n_faces, n_pos, p=p_pos)
        nonface_idx = np.random.choice(self.n_nonfaces, n_neg)

        faces = np.transpose(self._faces_hwc[face_idx], (0, 3, 1, 2))
        nonfaces = np.transpose(self._nonfaces_hwc[nonface_idx], (0, 3, 1, 2))
        x = np.concatenate([faces, nonfaces], axis=0)
        y = np.concatenate(
            [
                np.ones((n_pos, 1), dtype=np.float32),
                np.zeros((n_neg, 1), dtype=np.float32),
            ],
            axis=0,
        )
        perm = np.random.permutation(batch_size)
        return x[perm].copy(), y[perm]

    def get_all_train_faces(self) -> np.ndarray:
        return self._faces_hwc


def prepare_datasets(
    data_dir: Path = Path("./data"),
    image_size: int = IMAGE_SIZE,
    n_nonfaces: int = 25_000,
    test_per_group: int = 200,
    seed: int = 0,
) -> tuple[TrainingDatasetLoader, list[np.ndarray], list[str]]:
    """Build (or load cached) datasets and return loader + test split.

    Returns:
        loader: TrainingDatasetLoader instance.
        test_faces: list of 10 arrays, each ``(test_per_group, 3, image_size, image_size)``,
            float32 in [0, 1], order matches ``TEST_KEYS``.
        keys: race-major demographic labels, e.g. ``["White Female", "White Male",
            "Black Female", "Black Male", ...]`` — see :data:`TEST_KEYS`.
    """
    data_dir = Path(data_dir).resolve()
    paths = ensure_datasets(data_dir)
    cache_dir = data_dir / "cache"

    utk_img_path, utk_lbl_path = _build_utkface_cache(paths["utkface"], cache_dir, image_size)
    tin_img_path = _build_tiny_imagenet_cache(
        paths["tiny_imagenet"], cache_dir, image_size, n_nonfaces, seed
    )

    utk_images = np.load(utk_img_path)   # uint8 (N, H, W, 3)
    utk_labels = np.load(utk_lbl_path)   # int8 (N, 2): [gender, race]
    tin_images = np.load(tin_img_path)   # uint8 (M, H, W, 3)

    test_faces, train_mask = _make_test_split(
        utk_images, utk_labels, test_per_group, seed
    )

    train_faces = utk_images[train_mask].astype(np.float32) / 255.0
    train_nonfaces = tin_images.astype(np.float32) / 255.0

    loader = TrainingDatasetLoader(train_faces, train_nonfaces)
    return loader, test_faces, list(TEST_KEYS)


if __name__ == "__main__":
    loader, test_faces, keys = prepare_datasets()
    print(f"Train size: {loader.get_train_size()}  "
          f"(faces={loader.n_faces}, nonfaces={loader.n_nonfaces})")
    x, y = loader.get_batch(32)
    print(f"get_batch(32) -> x.shape={x.shape} x.dtype={x.dtype} "
          f"x.min={x.min():.3f} x.max={x.max():.3f}  y.shape={y.shape} "
          f"pos_frac={float(y.mean()):.3f}")
    all_faces = loader.get_all_train_faces()
    print(f"get_all_train_faces -> shape={all_faces.shape} dtype={all_faces.dtype}")
    for k, group in zip(keys, test_faces):
        print(f"test '{k}': shape={group.shape} dtype={group.dtype} "
              f"min={group.min():.3f} max={group.max():.3f}")
