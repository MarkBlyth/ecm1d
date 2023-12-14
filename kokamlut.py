#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import scipy.interpolate
import pandas as pd
import ecm


_R0_PARS = "./parameters/kokam_pars/R0.csv"
_Ri_PARS = "./parameters/kokam_pars/Ri.csv"
_Ci_PARS = "./parameters/kokam_pars/Ci.csv"
_OCV_PARS = "./parameters/kokam_pars/OCV-SoC.csv"
_dVdT_PARS = "./parameters/kokam_pars/dVdT-SoC.csv"


def check_result(func):
    """
    Optional decorator to add to Parameters class methods. Checks
    returns for NaN, and gives a printout of the parameters.
    """
    def checked(*args, **kwargs):
        ret = func(*args, **kwargs)
        if any(np.isnan(ret.ravel())):
            print(f"{func.__name__} gave nan with args")
            for arg, name in zip(args[1:], ["SOC", "Temperature"]):
                print(name, arg)
        return ret

    return checked


class KokamParameters(ecm.BaseParameters):
    def __init__(self, nlayers):
        diffusivity = 0.9048e-6
        heat_capacity = 880
        line_density = 11.13
        thickness = 0.0115
        capacity_Ah = 5

        super().__init__(
            nlayers, diffusivity, heat_capacity, line_density, thickness, capacity_Ah
        )

        ocv_df = pd.read_csv(_OCV_PARS)
        entropy_df = pd.read_csv(_dVdT_PARS)
        r0_df = pd.read_csv(_R0_PARS)
        ri_df = pd.read_csv(_Ri_PARS)
        ci_df = pd.read_csv(_Ci_PARS)

        self._ocv_interp = scipy.interpolate.CubicSpline(
            ocv_df["SoC"].to_numpy()[::-1],
            ocv_df["OCV"].to_numpy()[::-1],
        )
        self._entropy_interp = scipy.interpolate.CubicSpline(
            entropy_df["SoC"].to_numpy()[::-1],
            entropy_df["dVdT"].to_numpy()[::-1],
        )
        self._r0_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(r0_df, "R0")
        )
        self._r1_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ri_df, "R1")
        )
        self._r2_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ri_df, "R2")
        )
        self._r3_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ri_df, "R3")
        )
        self._c1_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ci_df, "C1")
        )
        self._c2_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ci_df, "C2")
        )
        self._c3_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(ci_df, "C3")
        )

    @staticmethod
    def _get_grid_interpolator(points, values):
        unique_x = np.unique(points[0])
        unique_y = np.unique(points[1])
        values = values.reshape((unique_x.size, unique_y.size))
        return scipy.interpolate.RegularGridInterpolator(
            (unique_x, unique_y),
            values,
            bounds_error=False,
            method="linear",
        )

    def get_entropy(self, soc: float | np.ndarray) -> float | np.ndarray:
        return self._entropy_interp(soc)

    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        return self._ocv_interp(soc)

    def get_unscaled_r0(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        return self._r0_interp((temperature, soc))

    def get_unscaled_ris(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        r1 = self._r1_interp((temperature, soc))
        r2 = self._r2_interp((temperature, soc))
        r3 = self._r3_interp((temperature, soc))
        return np.vstack((r1, r2, r3))

    def get_unscaled_cis(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        c1 = self._c1_interp((temperature, soc))
        c2 = self._c2_interp((temperature, soc))
        c3 = self._c3_interp((temperature, soc))
        return np.vstack((c1, c2, c3))

    @staticmethod
    def _get_soc_temp_predictors(df, target):
        sortings = ["T", "SoC"]
        sort_df = df.sort_values(by=sortings)
        temps = sort_df["T"].to_numpy()
        socs = sort_df["SoC"].to_numpy()
        targetdata = sort_df[target].to_numpy()
        mask = np.logical_not(
            np.any(np.c_[np.isnan(temps), np.isnan(socs), np.isnan(targetdata)], axis=1)
        )
        return np.vstack((temps[mask], socs[mask])), targetdata[mask]
