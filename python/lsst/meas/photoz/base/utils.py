# This file is part of meas_photoz_base.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = [
    "chunk_iterator",
]

from collections.abc import Generator

import numpy as np


def chunk_iterator(
    chunk_size: int,
    data: dict[str, np.ndarray],
    n_values: int,
) -> Generator[tuple[dict, int]]:
    """Yield chunks of a dict of equal-length arrays in a fixed chunk size.

    Parameters
    ----------
    chunk_size : int
        The size of each chunk.
    data : dict[str, np.ndarray]
        The input data.
    n_values : int
        Length of each array in data.

    Yields
    ------
    a_chunk : dict[str, np.array]
        A chunk of data.

    chunk_size: int
        The number of objects in the chunk.
    """
    if not chunk_size > 0:
        raise ValueError(f"{chunk_size=} must be >0")
    for key, values in data.items():
        if (n_key := len(values)) != n_values:
            raise ValueError(f"{key=} has len={n_key} != {n_values=}")

    chunk_start = 0
    chunk_stop = 0

    while chunk_stop < n_values:
        chunk_stop = min(chunk_start + chunk_size, n_values)
        this_chunk_size = chunk_stop - chunk_start
        this_chunk = {key: val[chunk_start:chunk_stop] for key, val in data.items()}
        chunk_start += this_chunk_size
        yield this_chunk, this_chunk_size
