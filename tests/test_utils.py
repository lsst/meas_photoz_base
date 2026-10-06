# This file is part of meas_photoz_base
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from lsst.meas.photoz.base.utils import chunk_iterator


@pytest.fixture(scope="module")
def size():
    """Return the size of data arrays."""
    return 12


@pytest.fixture(scope="module")
def data(size):
    """Return some arbitrary columns."""
    return {
        "a": np.arange(size),
        "b": np.arange(size + 1, 1, -1) + 0.5,
    }


def test_chunk_iterator(size, data):
    """Test the chunk_iterator generator utility."""
    keys = list(data.keys())

    for chunk_size in (3, 5):
        start = 0
        remaining = size
        iterator = chunk_iterator(chunk_size, data, size)
        while remaining > 0:
            chunk, n_chunk = next(iterator)
            assert n_chunk == min(remaining, chunk_size)
            assert list(chunk.keys()) == keys
            for key, values in chunk.items():
                assert len(values) == n_chunk
                np.testing.assert_array_equal(data[key][start : start + n_chunk], values)
            remaining -= n_chunk
            start += n_chunk
        with pytest.raises(StopIteration):
            next(iterator)
