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
    "EstimatePhotozAlgoConfigBase",
    "EstimatePhotozAlgoTask",
    "EstimatePhotozTask",
    "EstimatePhotozTaskConfig",
    "photozAlgoRegistry",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import qp
from astropy.table import Table
from ceci.config import StageConfig as CeciStageConfig
from ceci.config import StageParameter as CeciParam
from rail.core.model import Model
from rail.estimation.estimator import CatEstimator
from rail.interfaces import PZFactory

import lsst.pex.config as pexConfig
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    Task,
)

from .utils import chunk_iterator


class EstimatePhotozConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract"),
    defaultTemplates={"algo": "trainz"},
):
    """Connections for tasks that make p(z) estimates.

    These will take pickled model file as a "calibration-like" input,
    an object table as input, and create a p(z) file in 'qp' format.
    """

    photoz_model = cT.PrerequisiteInput(
        doc="Model for PZ Estimation",
        name="photoz_model_{algo}",
        storageClass="PhotozModel",
        dimensions=["instrument"],
        isCalibration=True,
    )
    objects = cT.Input(
        doc="Object table",
        name="object",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
        deferLoad=True,
    )

    photoz_ensemble = cT.Output(
        doc="Per-object p(z) estimates",
        name="photoz_ensemble_{algo}",
        storageClass="QPEnsemble",
        dimensions=("skymap", "tract"),
    )


class EstimatePhotozAlgoConfigBase(
    pexConfig.Config,
):
    """Base class for configurations of algorithm-specific p(z)
    estimation tasks.

    This class mostly just translates the RAIL configuration
    parameters to pex.config parameters.

    Subclasses will just have to set `estimator_class` and `stage_name`
    and invoke `_make_fields` once in the module.
    """

    @classmethod
    @abstractmethod
    def estimator_class(cls) -> type[CatEstimator]:
        """Return the type of the estimator's RAIL class."""
        raise NotImplementedError("Subclasses must specify an estimator class")

    # This should be a property but py3.13+ don't allow it
    @classmethod
    @abstractmethod
    def stage_name(cls) -> str:
        """Return the RAIL stage name for the estimator."""
        raise NotImplementedError("Subclasses must define a RAIL stage name")

    # Extinction coefficients; see https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract
    # Also in rail.utils.catalog_utils.RubinCatalogConfig.a_env
    default_a_env_values = dict(
        u=4.81,
        g=3.64,
        r=2.70,
        i=2.06,
        z=1.58,
        y=1.31,
    )

    # These appear in many DESC repos, for example:
    # https://github.com/LSSTDESC/TXPipe/blob/00ebe7476fd5d9529f5bbc4d73fcef0629d134c7/examples/dp0.2/config.yml#L47
    # They seem to be 10y WFD limits. Origin unclear.
    default_mag_limit_10y_values = dict(
        u=27.79,
        g=29.04,
        r=29.06,
        i=28.62,
        z=27.98,
        y=27.05,
    )

    # These appear to be from Roman-Rubin simulations:
    # https://github.com/LSSTDESC/rail_base/blob/v1.2.1/src/rail/utils/catalog_utils.py#L207
    # Presumably max 5y depth, and more useful for now
    default_mag_limit_values = dict(
        u=24.0,
        g=27.66,
        r=27.25,
        i=26.6,
        z=26.24,
        y=25.35,
    )

    def get_band_a_env_dict(self):
        """Return the set of a_envs to use."""
        return {band_: self.default_a_env_values[band_] for band_ in self.bands_to_convert}

    def get_mag_lim_dict(self):
        """Return the set of maglims to use."""
        return {
            self.mag_template.format(band=band): self.default_mag_limit_values[band]
            for band in self.bands_to_convert
        }

    def get_flux_names(self) -> dict[str, str]:
        """Return a dict mapping band to flux column name."""
        return {band: self.flux_column_template.format(band=band) for band in self.bands_to_convert}

    def get_flux_err_names(self) -> dict[str, str]:
        """Return a dict mapping band to flux error column name."""
        return {band: self.flux_err_column_template.format(band=band) for band in self.bands_to_convert}

    def get_mag_names(self) -> dict[str, str]:
        """Return a dict mapping band to mag column name."""
        return {band: self.mag_template.format(band=band) for band in self.bands_to_convert}

    def get_mag_err_names(self) -> dict[str, str]:
        """Return a dict mapping band to mag error column name."""
        return {band: self.mag_err_template.format(band=band) for band in self.bands_to_convert}

    mag_offset = pexConfig.Field(doc="Magnitude offset", dtype=float, default=31.4)
    deredden = pexConfig.Field[bool](
        doc="Apply dereddening",
        default=True,
    )
    band_ref = pexConfig.Field[str](
        doc="Name of the most reliable reference band, if needed",
        default="i",
    )
    bands_to_convert = pexConfig.ListField[str](
        doc="Names of bands to convert fluxs to mags for RAIL",
        default=["u", "g", "r", "i", "z", "y"],
    )
    flux_column_template = pexConfig.Field[str](
        doc="Template for flux column names",
        default="{band}_gaap1p0Flux",
        # default="{band}_cModelFlux",
    )
    flux_err_column_template = pexConfig.Field[str](
        doc="Template for flux error column names",
        default="{band}_gaap1p0FluxErr",
        # default="{band}_cModelFluxErr",
    )
    mag_template = pexConfig.Field[str](
        doc="Template for magnitude names",
        default="{band}_gaap1p0Mag",
        # default="{band}_cModelMag",
    )
    mag_err_template = pexConfig.Field[str](
        doc="Template for magntitude error names",
        default="{band}_gaap1p0MagErr",
        # default="{band}_cModelMagErr",
    )
    nondetect_val = pexConfig.Field[float](
        doc="Magnitude to set for non-detections",
        default=np.nan,
    )
    band_a_env = pexConfig.DictField[str, float](
        doc="Reddening parameters",
        default=default_a_env_values,
    )

    def freeze(self):
        if not self._frozen:
            self._finalize()
        super().freeze()

    def _finalize(self):
        # These calls will fail if it's already frozen.
        if hasattr(self, "ref_band"):
            self.ref_band = self.mag_template.format(band=self.band_ref)
        if hasattr(self, "bands"):
            # This is a list of mag columns in RAIL, not bands
            self.bands = list(self.get_mag_names().values())
        if hasattr(self, "err_bands"):
            self.err_bands = list(self.get_mag_err_names().values())
        if hasattr(self, "mag_limits"):
            self.mag_limits = self.get_mag_lim_dict()
        if hasattr(self, "band_a_env"):
            self.band_a_env = self.get_band_a_env_dict()

    @classmethod
    def _make_fields(cls) -> None:
        """Import the RAIL estimation stage.

        This method loops through the stage config parameters and converts
        RAIL/Ceci parameters to corresponding pex.config parameters.

        It should be called exactly once, immediately after the definition
        of every subclass of this base class.
        """
        if hasattr(cls, "__fields_made__"):
            if cls.__fields_made__ is not True:
                raise RuntimeError(f"{cls.__fields_made__=} exists but is not True")
            raise RuntimeError(f"{cls=} called _make_fields twice")
        stage_class = cls.estimator_class()
        for key, val in stage_class.config_options.items():
            if isinstance(val, CeciStageConfig):
                val = val.get(key)
            if isinstance(val, CeciParam):
                if val.dtype in [bool, int, float, str]:
                    if (attr := getattr(cls, key, None)) is not None:
                        if not isinstance(attr, pexConfig.Field):
                            raise RuntimeError(f"{cls=} {key=} exists but is of {type(key)=}, not Field")
                        elif attr.dtype != val.dtype:
                            raise RuntimeError(f"{cls=} {key=} exists but {attr.dtype=} != {val.dtype=}")
                        attr.default = val.default
                        attr.doc = f"{val.msg} (overriding base doc='{attr.doc}')"
                    else:
                        setattr(
                            cls,
                            key,
                            pexConfig.Field(doc=val.msg, dtype=val.dtype, default=val.default),
                        )
                elif val.dtype in [list]:
                    # this is a hack, but it works.
                    if val.default:
                        item_type = type(val.default[0])
                    else:
                        item_type = str
                    setattr(
                        cls,
                        key,
                        pexConfig.ListField(doc=val.msg, dtype=item_type, default=val.default),
                    )
                elif val.dtype in [dict]:
                    setattr(
                        cls,
                        key,
                        pexConfig.DictField(doc=val.msg, keytype=str, default=val.default),
                    )
        cls.__fields_made__ = True


photozAlgoRegistry = pexConfig.makeRegistry(
    doc="A registry of photometric redshift estimation algorithm subtasks",
)


class EstimatePhotozAlgoTask(Task, ABC):
    """Task for algorithm-specific p(z) estimation.

    This provides almost all of the functionality
    needed to run RAIL p(z) algorithms.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to super().__init__.
    """

    ConfigClass = EstimatePhotozAlgoConfigBase

    mag_conv = np.log(10) * 0.4

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @staticmethod
    def _flux_to_mag(
        flux_vals: np.ndarray,
        mag_offset: float,
        nondetect_val: float,
    ) -> np.ndarray:
        """Convert flux to magnitude.

        Parameters
        ----------
        flux_vals : np.array
            Input flux values (units?)

        mag_offset : float
            Magnitude offset (corresponding to a flux of 1.)

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags : np.array
            Magnitude values
        """
        vals = np.empty_like(flux_vals)
        positive = flux_vals >= 0
        vals[positive] = -2.5 * np.log10(flux_vals[positive]) + mag_offset
        vals[~positive] = nondetect_val
        return vals

    @staticmethod
    def _flux_err_to_mag_err(
        flux_vals: np.ndarray,
        flux_err_vals: np.ndarray,
        mag_conv: float,
        nondetect_val: float = np.nan,
    ) -> np.ndarray:
        """Config flux error to magnitude error.

        Parameters
        ----------
        flux_vals : np.array
            Input flux values (units?)

        flux_err_vals : np.array
            Input flux errors (units?)

        mag_conv : float
            Magnitude to flux conversion (typically np.log(10)*0.4)

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags_errs : np.array
            Magnitude errors
        """
        vals = np.empty_like(flux_vals)
        positive = flux_vals >= 0
        vals[positive] = flux_err_vals[positive] / (flux_vals[positive] * mag_conv)
        vals[~positive] = nondetect_val
        return vals

    @staticmethod
    def _deredden_mags(
        data: dict[str, np.ndarray],
        a_env_dict: dict[str, float],
        mag_names: dict[str, str],
        nondetect_val: float,
    ) -> dict[str, np.ndarray]:
        """Deredden the magnitdues

        Parameters
        ----------
        data: dict[str, np.array]
            Input data

        a_env_dict: dict[str, float],
            Redenning parameters for bands

        mag_names: dict[str, str]
            Mapping from bands to magnitudes

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags: dict[str, np.array]
            Udpated dict with dereddened mags
        """
        ebv = data["ebv"]
        for band_, a_env_ in a_env_dict.items():
            mag_name = mag_names[band_]
            raw_mag = data[mag_name]
            dered_mag = np.where(
                np.isfinite(raw_mag),
                raw_mag - ebv * a_env_,
                nondetect_val,
            )
            data[mag_name] = dered_mag
        return data

    def _get_mags_and_errs(
        self,
        fluxes: Table,
        mag_offset: float,
    ) -> dict[str, np.ndarray]:
        """Fill and return a numpy dict with mags and mag errors.

        Parameters
        ----------
        fluxes : Table
            Input fluxes and flux errors

        mag_offset : float
            Magnitude offset (corresponding to a flux of 1.)

        Returns
        -------
        mags: dict[str, np.array]
            Numpy dict with mags and mag errors
        """
        # get all the column names we will use
        flux_names = self.config.get_flux_names()
        mag_names = self.config.get_mag_names()
        flux_err_names = self.config.get_flux_err_names()
        mag_err_names = self.config.get_mag_err_names()
        nondetect_val = self.config.nondetect_val
        # output dict
        mag_dict = {}
        if object_id_name := self.config.id_col:
            mag_dict[object_id_name] = fluxes[object_id_name]

        # loop over bands, make mags and mag errors and fill dict
        for band in flux_names.keys():
            fluxVals = np.asarray(fluxes[flux_names[band]])
            fluxErrVals = np.asarray(fluxes[flux_err_names[band]])
            mag_dict[mag_names[band]] = self._flux_to_mag(
                fluxVals,
                mag_offset,
                nondetect_val,
            )
            if flux_err_names:
                mag_dict[mag_err_names[band]] = self._flux_err_to_mag_err(
                    fluxVals,
                    fluxErrVals,
                    self.mag_conv,
                    nondetect_val,
                )

        # return the dict with the mags
        return mag_dict

    def init(
        self,
        photoz_model: Model,
    ) -> None:
        """Set up the RAIL stage to compute photo-zs.

        Parameters
        ----------
        photoz_model : Model
            Model used by the p(z) estimation algorithm.
        """
        # pop the pipeline task config options
        # so that we can pass the rest to RAIL
        rail_kwargs = self.config.toDict().copy()
        for key in ["saveLogOutput", "stage_name", "mag_offset", "connections"]:
            rail_kwargs.pop(key, None)
        rail_kwargs["output_mode"] = "return"

        # Build the RAIL stage
        self._stage = PZFactory.build_stage_instance(
            self.config.stage_name(),
            self.config.estimator_class(),
            model_path=photoz_model.data,
            input_path="dummy.in",
            **rail_kwargs,
        )
        self._stage._initialize_run()

    def col_names(
        self,
    ) -> list[str]:
        """Get the list of column names to read from the input data."""
        columns = list(self.config.get_flux_names().values()) + list(
            self.config.get_flux_err_names().values()
        )
        if self.config.deredden:
            columns += ["ebv"]

        return columns

    def run(
        self,
        fluxes: Table,
    ) -> Struct:
        """Run a p(z) estimation algorithm.

        Parameters
        ----------
        fluxes : Table
            Fluxes used to compute the redshifts.

        Returns
        -------
        photoz_pdfs : qp.Ensemble
            Object with the p(z) PDFs.
        """
        n_obj = len(fluxes)
        # Convert fluxes to mags
        mags = self._get_mags_and_errs(fluxes, self.config.mag_offset)
        nondetect_val = self.config.nondetect_val if hasattr(self.config, "nondetect_val") else np.nan

        # De-redden
        if self.config.deredden:
            # asarray will convert an astropy column to an array w/o units
            mags["ebv"] = np.asarray(fluxes["ebv"])
            mags = self._deredden_mags(
                mags,
                self.config.band_a_env,
                self.config.get_mag_names(),
                nondetect_val,
            )

        pz_ensembles: list[qp.Ensemble] = []

        # split processing into chunks to avoid memory issues in rail
        iterator = chunk_iterator(self.config.chunk_size, mags, n_obj)
        for a_chunk, chunk_size in iterator:
            # Pass the mags to RAIL and get back the p(z) pdfs
            # as a qp.Ensemble object
            pz_ensembles.append(PZFactory.estimate_single_pz(self._stage, a_chunk, chunk_size))

        photoz_pdfs = qp.concatenate(pz_ensembles)
        return Struct(photoz_ensemble=photoz_pdfs)


class EstimatePhotozTaskConfig(PipelineTaskConfig, pipelineConnections=EstimatePhotozConnections):
    """Configuration for EstimatePhotozTask PipelineTask."""

    photoz_algo = photozAlgoRegistry.makeField(
        doc="Algorithm specific configuration p(z) estimation task",
    )


class EstimatePhotozTask(PipelineTask):
    """PipelineTask for p(z) estimation.

    Parameters
    ----------
    initInputs
        Initialization inputs to pass to super().__init__.
    **kwargs
        Additional keyword arguments to pass to super().__init__.
    """

    ConfigClass = EstimatePhotozTaskConfig
    _DefaultName = "estimatePhotoz"

    def __init__(self, initInputs: dict, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self._initialized = False
        self.makeSubtask("photoz_algo")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["fluxes"] = inputs.pop("objects").get(
            parameters=dict(columns=self.photoz_algo.col_names()),
        )
        outputs = self.run(**inputs, skip_init=self._initialized)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        *,
        photoz_model: Model,
        fluxes: Table,
        skip_init: bool = False,
    ) -> Struct:
        if not skip_init:
            self._initialized = True
            self.photoz_algo.init(photoz_model)

        ret_struct = self.photoz_algo.run(fluxes)
        return Struct(photoz_ensemble=ret_struct.photoz_ensemble)
