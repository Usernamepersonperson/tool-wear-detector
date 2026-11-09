"""Fetch and load the public NASA milling-wear data set (never committed, cached locally)."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
ZIP_MD5 = "4da3afb0aa50cb3dcdd8e20ed1ed1c7c"  # S3 ETag of the published archive
CITATION = (
    'A. Agogino and K. Goebel (2007). BEST Lab, UC Berkeley. "Milling Data Set", '
    "NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA."
)
SENSOR_KEYS = ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle")


@dataclass
class Run:
    case: int  # cutter (insert) id; the unit we split on
    run: int
    vb: float  # measured flank wear, mm (nan = not measured)
    time: float  # cumulative cutting time, min
    doc: float  # depth of cut, mm
    feed: float  # mm/rev
    material: int  # 1 = cast iron, 2 = steel
    signals: dict[str, np.ndarray]


def md5_of(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the published ETag only
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cache: Path, url: str = URL) -> Path:
    """Download and unpack mill.mat into cache/. Returns the path to mill.mat."""
    cache = Path(cache)
    mat = cache / "mill.mat"
    if mat.exists():
        return mat
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "milling.zip"
    if not zpath.exists():
        print(f"downloading {url} (about 15 MB) ...")
        urllib.request.urlretrieve(url, zpath)  # noqa: S310 - fixed https URL
    if md5_of(zpath) != ZIP_MD5:
        raise RuntimeError("downloaded archive does not match the published checksum")
    with zipfile.ZipFile(zpath) as outer:
        inner_bytes = outer.read("3. Milling/mill.zip")
    inner = cache / "mill_inner.zip"
    inner.write_bytes(inner_bytes)
    with zipfile.ZipFile(inner) as z:
        mat.write_bytes(z.read("mill.mat"))
    return mat


def load_runs(mat_path: Path) -> list[Run]:
    from scipy.io import loadmat

    m = loadmat(str(mat_path))["mill"][0]
    runs = []
    for r in m:
        runs.append(
            Run(
                case=int(r["case"].item()),
                run=int(r["run"].item()),
                vb=float(r["VB"].item()),
                time=float(r["time"].item()),
                doc=float(r["DOC"].item()),
                feed=float(r["feed"].item()),
                material=int(r["material"].item()),
                signals={k: np.asarray(r[k]).ravel().astype(float) for k in SENSOR_KEYS},
            )
        )
    return runs
