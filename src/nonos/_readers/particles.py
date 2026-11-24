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
class IdefixVTKReader:
    NATIVE_COORDINATE_REGEXP = re.compile(r"X(1|2|3)(L|C)_NATIVE_COORDINATES")

    @staticmethod
    def parse_snapshot_uid_and_filename(
        file_or_uid: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,  # noqa: ARG004
    ) -> tuple[int, Path]:
        if isinstance(file_or_uid, int):
            snapshot_uid = int(file_or_uid)
            file = Path(f"part.{snapshot_uid:04d}.vtk")
        else:
            file = Path(file_or_uid)
            if (m := re.search(r"\d+", file.name)) is None:
                raise ValueError(
                    f"Failed to parse output number from filename {file.name}"
                )
            else:
                snapshot_uid = int(m.group())

        if file == Path(file.name):
            file = Path(directory) / file

        return snapshot_uid, file

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        directory = Path(directory)
        return sorted(directory.glob("part.*.vtk"))

    @staticmethod
    def read(file: str | os.PathLike[str], /, **meta: Any) -> ParticlesData[f64]:
        """
        Adapted from Geoffroy Lesur
        """
        geometry_str = meta["geometry"]

        fh = open(file, "rb")  # noqa: SIM115

        f_64_be = np.dtype(">f4")  # Big endian single precision floats
        V = ParticlesData.default_init(dtype=f_64_be)

        if meta["geometry"] is not None:
            V = replace(V, geometry=Geometry(meta["geometry"]))

        data = {}

        dint = np.dtype(">i4")  # Big endian integer
        # Load header
        fh.seek(0)
        # skip over the first 3 lines which normally contains
        # VTK DataFile Version x.x
        # <Comments>
        # BINARY
        for _ in range(3):
            fh.readline()

        line = fh.readline().decode("utf-8")
        assert line.startswith("DATASET ")
        _dataset_type = line.split()[1]

        ref_position = fh.tell()
        line = fh.readline().decode("utf-8")  # DIMENSIONS NX NY NZ or FIELD
        if line.startswith("DIMENSIONS"):
            # Idefix < 0.8
            fh.seek(ref_position)
        elif line.startswith("FIELD"):
            # Idefix >= 0.8
            nfield = int(line.split()[2])
            vtk_sizes = {"int": 4, "float": 4, "double": 8}
            for _ in range(nfield):
                d = fh.readline().decode("utf-8")

                _, ncomp, ntuples, vtype = d.split()
                fh.seek(int(ncomp) * int(ntuples) * vtk_sizes[vtype], os.SEEK_CUR)

                # if d.startswith("GEOMETRY"):
                #     geom_flag = np.fromfile(fh, dint, 1)[0]
                #     geometry_from_data = KNOWN_GEOMETRIES.get(geom_flag)
                #     if geometry_from_data is None:
                #         raise RuntimeError(
                #             "Found unknown geometry value %d" % geom_flag
                #         )
                #     elif geometry is not None and geometry_from_data != geometry:
                #         warnings.warn(
                #             "Received geometry argument '%s' inconsistent with data geometry '%s'."
                #             % (geometry, geometry_from_data),
                #             stacklevel=3,
                #         )
                #     geometry = geometry_from_data
                # elif d.startswith("TIME"):
                #     t = np.fromfile(fh, dt, 1)
                # elif d.startswith("PERIODICITY"):
                #     periodicity = np.fromfile(fh, dtype=dint, count=3).astype(bool)
                # elif d.startswith("HAVE_TEST_PARTICLES"):
                #     have_test_particles = np.fromfile(
                #         fh, dtype=dint, count=1
                #     ).astype(bool)
                # elif NATIVE_COORDINATE_REGEXP.match(d):
                #     native_name, _ncomp, native_dim, _dtype = d.split()
                #     native_coordinates[native_name] = np.fromfile(
                #         fh, dtype=dt, count=int(native_dim)
                #     )
                # else:
                #     warnings.warn("Found unknown field %s" % d)
                fh.readline()  # skip extra linefeed (empty line)

        # if geometry is None:
        #     raise ValueError(
        #         "No GEOMETRY metadata was found, "
        #         "please provide a geometry argument to this function"
        #     )

        # Load particles
        line = fh.readline()
        npoints = int(line.split()[1])
        x, y, z = np.fromfile(fh, ">f", 3 * npoints).reshape(npoints, 3).T

        fh.readline()
        fh.readline()  # extra line feed
        line = fh.readline().decode("utf-8")
        assert line.startswith("POINT_DATA ")
        assert int(line.split()[-1]) == npoints

        fh.readline()  # extra line feed

        while True:
            line = fh.readline()
            if len(line) < 2:
                break

            keyword, varname, datatype = line.decode("utf-8").split()
            assert keyword == "SCALARS"
            if datatype == "float":
                dtype = ">f"
            elif datatype == "int":
                dtype = ">i4"
            else:
                raise RuntimeError(f"Got unsupported datatype '{datatype}'")
            line = fh.readline().decode("utf-8")
            keyword, value = line.split()
            assert keyword == "LOOKUP_TABLE"
            assert value == "default"
            data[varname] = np.fromfile(fh, dtype, npoints)
            fh.readline()

        x1: FArray1D[f64]
        x2: FArray1D[f64]
        x3: FArray1D[f64]

        if geometry_str == "cartesian":
            x1, x2, x3 = x, y, z
            V = replace(
                V,
                geometry=Geometry.CARTESIAN,
            )

        # reconstruct native coordinates for curvilinear geometries
        if geometry_str in ("cylindrical", "polar"):
            x1 = np.sqrt(x**2 + y**2)  # cylindrical radius
            x2 = np.arctan2(y, x)
            x3 = z
            V = replace(
                V,
                geometry=Geometry.POLAR,
            )

        elif geometry_str == "spherical":
            r = np.sqrt(x**2 + y**2 + z**2)

            if np.all(z == 0):  # 2D data
                phi = np.arctan2(z, x)
                theta = np.arccos(y / r)
            else:
                phi = np.arctan2(y, x)
                theta = np.arccos(z / r)
            x1, x2, x3 = r, theta, phi
            V = replace(
                V,
                geometry=Geometry.SPHERICAL,
            )
        else:
            raise NotImplementedError(f"Geometry {geometry_str!r} is not supported")

        V.data["x1"] = x1
        V.data["x2"] = x2
        V.data["x3"] = x3

        for key, item in data.items():
            V.data[key.lower()] = item

        return V.finalize()


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
