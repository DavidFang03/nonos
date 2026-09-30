import os
import re
import sys
from pathlib import Path
from typing import Any, final

import numpy as np

from nonos._types import F, FArray1D, ParticlesData
from nonos.geometry import Geometry

f64 = np.float64

if sys.version_info >= (3, 13):
    from copy import replace
else:
    from dataclasses import replace


@final
class Fargo3DReaderHelper:
    @staticmethod
    def parse_snapshot_uid_and_filename(
        file_or_uid: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,  # noqa ARG004
    ) -> tuple[int, Path]:
        directory = Path(directory).resolve()
        if isinstance(file_or_uid, int):
            snapshot_uid = file_or_uid
            file = directory / f"particles{snapshot_uid:04d}.npy"
        else:
            file = Path(file_or_uid)
            if file == Path(file.name):
                file = directory / file
            if len(matches := re.findall(r"\d+", file.name)) == 1:
                snapshot_uid = int(matches[0])
            elif len(matches) == 0:
                raise RuntimeError(
                    rf"Failed to guess an output number from {file_or_uid!r}"
                )
            else:
                raise RuntimeError(rf"Ambiguous output number from {file_or_uid!r}")
        return snapshot_uid, file

    @staticmethod
    def get_particles_files(directory: os.PathLike[str], /) -> list[Path]:
        directory = Path(directory)
        return [
            fn
            for fn in sorted(directory.glob("particles*.npy"))
            if re.search(r"particles\d+.npy$", str(fn)) is not None
        ]

    @staticmethod
    def _get_snapshot_uid_and_dir_from(
        file: str | os.PathLike[str],
    ) -> tuple[int, Path]:
        _in_file = Path(file).resolve()
        directory = _in_file.parent
        if (match := re.search(r"(?P<on>\d+).npy$", _in_file.name)) is not None:
            snapshot_uid = int(match.group("on"))
        else:
            raise ValueError(f"Failed to parse filename {file!r}")

        return snapshot_uid, directory


@final
class Fargo3DReader:
    @staticmethod
    def parse_snapshot_uid_and_filename(
        file_or_uid: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,
    ) -> tuple[int, Path]:
        return Fargo3DReaderHelper.parse_snapshot_uid_and_filename(
            file_or_uid, directory=directory, prefix=prefix
        )

    @staticmethod
    def get_particles_files(directory: os.PathLike[str], /) -> list[Path]:
        return Fargo3DReaderHelper.get_particles_files(directory)

    @staticmethod
    def read(
        file: os.PathLike[str],
        /,
        **meta: Any,
    ) -> ParticlesData[f64]:
        snapshot_uid, directory = Fargo3DReaderHelper._get_snapshot_uid_and_dir_from(
            file
        )

        V = ParticlesData.default_init(dtype=np.dtype("float64"))
        geometry_str = meta["geometry"]

        reversed_pairs = enumerate(
            [
                "uid",
                "x1",  # only (r,phi) atm
                "x2",  # only (r,phi) atm
                "vx1",
                "vx2",
                "a_code",
                "a_cm",
                "St",
                "mass_code",
                "grains_represented",
                "alive",
            ]
        )
        if geometry_str in ("cylindrical", "polar"):
            V = replace(
                V,
                geometry=Geometry.POLAR,
            )

        elif geometry_str == "spherical":
            V = replace(
                V,
                geometry=Geometry.SPHERICAL,
            )
        else:
            raise NotImplementedError(f"Geometry {geometry_str!r} is not supported")

        def _read_array(file: Path) -> FArray1D[F]:
            return np.load(file)

        file = directory / f"particles{snapshot_uid}.npy"
        array = _read_array(file)
        for idx, key in reversed_pairs:
            if not file.is_file():
                continue
            V.data[key] = array[:, idx]

        V.data["x3"] = V.data["x1"] * 0.0  # trick
        V.data["uid"] = V.data["uid"].astype(int)  # trick

        if not V.data:
            raise FileNotFoundError(
                f"No file matches the pattern 'particles{snapshot_uid}.npy'"
            )

        return V.finalize()
