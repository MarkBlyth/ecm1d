#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import ecm
import scipy.interpolate
import pandas as pd


_PARAMETER_FILE = "./parameters/lgm50/lgm50pars.csv"


class LGM50Parameters(ecm.BaseParameters):
    eps = 1e-4

    def __init__(self, nlayers):
        # TODO these are taken from Kokam data, for lack of a better guess
        diffusivity = 0.9048e-6
        heat_capacity = 880
        line_density = 11.13
        thickness = 0.0115
        capacity_Ah = 5

        super().__init__(
            nlayers, diffusivity, heat_capacity, line_density, thickness, capacity_Ah
        )

        keys = _get_data_keys(2)
        df = pd.read_csv(_PARAMETER_FILE)

        self._ocv_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "OCV")
        )
        self._r0_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "R0")
        )
        self._r1_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "R1")
        )
        self._r2_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "R2")
        )
        self._c1_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "C1")
        )
        self._c2_interp = self._get_grid_interpolator(
            *self._get_soc_temp_predictors(df, keys, "C2")
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
        # TODO allow entropy to take temperature argument
        # def get_entropy(self, soc: float | np.ndarray, temperature: float | np.ndarray) -> float | np.ndarray:

        t0 = self._ocv_interp((25, soc))
        t1 = self._ocv_interp((25 + self.eps, soc))
        return (t1 - t0) / self.eps

    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        # TODO allow OCV to take temperature argument
        return self._ocv_interp((25, soc))

    def get_unscaled_r0(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        return self._r0_interp((temperature, soc))

    def get_unscaled_ris(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        r1 = self._r1_interp((temperature, soc))
        r2 = self._r2_interp((temperature, soc))
        return np.r_[r1, r2]

    def get_unscaled_cis(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        c1 = self._c1_interp((temperature, soc))
        c2 = self._c2_interp((temperature, soc))
        return np.r_[c1, c2]

    @staticmethod
    def _get_soc_temp_predictors(df, keys, target):
        sortings = [keys["TempC"], keys["SoC"]]
        sort_df = df.sort_values(by=sortings)
        temps = sort_df[keys["TempC"]].to_numpy()
        socs = sort_df[keys["SoC"]].to_numpy()
        targetdata = sort_df[keys[target]].to_numpy()
        return np.vstack((temps, socs)), targetdata


def _get_data_keys(n_rc):
    # Map from csv column names, to something clear to us
    return {
        "SoC": "SOC",
        "TempC": "T_degC",
        "OCV": "V_Vsrc0_V",
        "R0": "R_R0_Ohm",
        **{f"R{i}": f"R_R{i}_Ohm" for i in range(1, n_rc + 1)},
        **{f"C{i}": f"C_C{i}_F" for i in range(1, n_rc + 1)},
    }
