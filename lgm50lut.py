#!/usr/bin/env python3

import numpy as np
import ecm
import scipy.interpolate
import pandas as pd


_PARAMETER_FILE = "./parameters/lgm50/lgm50pars.csv"


class LGM50LUT(ecm.BaseLUT):
    eps = 1e-4

    def __init__(self):

        keys = _get_data_keys(2)
        df = pd.read_csv(_PARAMETER_FILE)

        self._ocv_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "OCV")
        )
        self._r0_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "R0")
        )
        self._r1_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "R1")
        )
        self._r2_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "R2")
        )
        self._c1_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "C1")
        )
        self._c2_interp = scipy.interpolate.CloughTocher2DInterpolator(
            *self._get_soc_temp_predictors(df, keys, "C2")
        )

    @ecm.check_nan
    def get_entropy(self, cell_state: ecm.ElectricalState) -> float:
        t0 = self._ocv_interp(
            cell_state.layer_temperature, cell_state.layer_soc,
        ).squeeze()
        t1 = self._ocv_interp(
            cell_state.layer_temperature + self.eps, cell_state.layer_soc,
        ).squeeze()
        return (t1 - t0) / self.eps

    @ecm.check_nan
    def get_ocv(self, cell_state: ecm.ElectricalState) -> float:
        return self._ocv_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()

    @ecm.check_nan
    def get_unscaled_r0(self, cell_state: ecm.ElectricalState) -> float:
        return self._r0_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()

    @ecm.check_nan
    def get_unscaled_ris(self, cell_state: ecm.ElectricalState) -> np.ndarray:
        r1 = self._r1_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()
        r2 = self._r2_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()
        return np.r_[r1, r2]

    @ecm.check_nan
    def get_unscaled_cis(self, cell_state: ecm.ElectricalState) -> np.ndarray:
        c1 = self._c1_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()
        c2 = self._c2_interp(
            cell_state.layer_temperature, cell_state.layer_soc
        ).squeeze()
        return np.r_[c1, c2]

    @staticmethod
    def _get_soc_temp_predictors(df, keys, target):
        sortings = [keys["TempC"], keys["SoC"]]
        sort_df = df.sort_values(by=sortings)
        temps = sort_df[keys["TempC"]].to_numpy()
        socs = sort_df[keys["SoC"]].to_numpy()
        targetdata = sort_df[keys[target]].to_numpy()
        mask = np.logical_not(
            np.any(np.c_[np.isnan(temps), np.isnan(socs), np.isnan(targetdata)], axis=1)
        )
        return np.vstack((temps[mask], socs[mask])).T, targetdata[mask]


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
