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

import os

import astropy.units as u
import numpy as np
from astropy.table import Table
from rail.core.model import Model as PhotoZModel

from lsst.meas.photoz.base.estimate_photoz_task_trainz import (
    EstimatePhotozTrainZConfig,
    EstimatePhotozTrainZTask,
)

TESTDIR = os.path.abspath(os.path.dirname(__file__))


def test_algo_config():
    """Test default initialization of TrainZ config class."""
    config = EstimatePhotozTrainZConfig()
    config.validate()


def test_run_trainz() -> None:
    """Test that trainz runs on plausible data."""
    path = os.path.join(TESTDIR, "data", "model_inform_train_z_wrap.pickle")
    model = PhotoZModel.read(path)
    config = EstimatePhotozTrainZConfig()
    task = EstimatePhotozTrainZTask(config=config, initInputs={})

    algo_config = task.photoz_algo.config
    n_data = 173
    algo_config.chunk_size = int(round(n_data / 3.1))
    config.freeze()
    mags = np.linspace(19, 26, n_data)
    mag_names = algo_config.get_mag_names()
    data = {column: mags + 0.5 * idx for idx, column in enumerate(mag_names.values())}
    for band, column in algo_config.get_flux_names().items():
        data[column] = (data[mag_names[band]] * u.ABmag).to(u.nJy).value

    data["ebv"] = 0.05 * (1 + np.sin(np.arange(n_data)))
    data = Table(data)

    result = task.run(photoz_model=model, fluxes=data)
    yvals = result.photoz_ensemble.objdata["yvals"]
    assert yvals.shape[0] == n_data
    np.testing.assert_array_compare(np.greater, np.sum(yvals, axis=1), 0)
