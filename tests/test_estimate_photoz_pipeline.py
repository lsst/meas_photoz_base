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
import tempfile
import unittest
from typing import Any

import lsst.meas.photoz.base.all_algos as all_algos
from lsst.daf.butler import Butler, Config
from lsst.daf.butler.tests import DatastoreMock
from lsst.daf.butler.tests.utils import makeTestTempDir, removeTestTempDir
from lsst.meas.photoz.base.estimate_photoz_task import EstimatePhotozConnections, photozAlgoRegistry
from lsst.pipe.base.tests.pipelineStepTester import PipelineStepTester

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


class MeasPzPipelineTestCase(unittest.TestCase):
    """Test the PZ pipeline plumbing for fully supported algorithms.

    This uses the `PipelineStepTester` to test
    a test pipeline define in tests/data/photoz_pipeline_hsc.yaml

    This should include any algorithms that are
    including in the rubin-env environment.

    For now that is knn and trainz.
    """

    def setUp(self) -> None:
        self.root = makeTestTempDir(TEST_DATA_DIR)
        self.maxDiff = None

    def tearDown(self) -> None:
        removeTestTempDir(self.root)

    def makeButler(self, **kwargs: Any) -> Butler:
        """Return new Butler instance on each call."""
        config = Config()

        # make separate temporary directory for registry of this instance
        tmpdir = tempfile.mkdtemp(dir=self.root)
        config["registry", "db"] = f"sqlite:///{tmpdir}/gen3.sqlite3"
        config = Butler.makeRepo(self.root, config)
        butler = Butler.from_config(config, **kwargs)
        DatastoreMock.apply(butler)
        return butler

    def test_photoz_pipeline(self) -> None:
        butler = self.makeButler(writeable=True)

        expected_inputs = ["object"]
        expected_outputs = []
        inputs = [("object", {"skymap", "tract"}, "ArrowAstropy", False)]
        names = list(photozAlgoRegistry.keys())
        tasks = list(photozAlgoRegistry.values())
        all_tasks = [x for x in all_algos.__all__ if x != "photozAlgoRegistry"]
        assert len(names) == len(all_tasks)
        assert set(tasks) == set([getattr(all_algos, attr) for attr in all_tasks])

        pipeline_patches = {}
        chunk_size = 10

        for algo in names:
            dataset = EstimatePhotozConnections.photoz_model.name.format(algo=algo)
            expected_inputs.append(dataset)
            expected_outputs.append(EstimatePhotozConnections.photoz_ensemble.name.format(algo=algo))
            inputs.append((dataset, {"instrument"}, "PhotozModel", True))
            pipeline_patches[f"photoz_{algo}:photoz_algo.active.chunk_size"] = chunk_size
            chunk_size *= 5

        tester = PipelineStepTester(
            os.path.join(PIPELINES_DIR, "photoz.yaml"),
            ["#photoz_all"],
            inputs,
            expected_inputs=set(expected_inputs),
            expected_outputs=set(expected_outputs),
            pipeline_patches=pipeline_patches,
        )
        tester.run(butler, self)
