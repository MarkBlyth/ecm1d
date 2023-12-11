import scipy.interpolate
import pandas as pd
import numpy as np
import ecm


_R0_PARS = "./parameters/kokam_pars/R0.csv"
_Ri_PARS = "./parameters/kokam_pars/Ri.csv"
_Ci_PARS = "./parameters/kokam_pars/Ci.csv"
_OCV_PARS = "./parameters/kokam_pars/OCV-SoC.csv"
_dVdT_PARS = "./parameters/kokam_pars/dVdT-SoC.csv"


class KokamLUT(ecm.BaseLUT):
    def __init__(self):
        super().__init__()

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
            method="cubic",
        )

    @ecm.check_nan
    def get_entropy(self, cell_state: ecm.ElectricalState) -> float:
        return self._entropy_interp(cell_state.layer_soc).squeeze()

    @ecm.check_nan
    def get_ocv(self, cell_state: ecm.ElectricalState) -> float:
        return self._ocv_interp(cell_state.layer_soc).squeeze()

    @ecm.check_nan
    def get_unscaled_r0(self, cell_state: ecm.ElectricalState) -> float:
        return self._r0_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()

    @ecm.check_nan
    def get_unscaled_ris(self, cell_state: ecm.ElectricalState) -> np.ndarray:
        r1 = self._r1_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        r2 = self._r2_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        r3 = self._r3_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        return np.r_[r1, r2, r3]

    @ecm.check_nan
    def get_unscaled_cis(self, cell_state: ecm.ElectricalState) -> np.ndarray:
        c1 = self._c1_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        c2 = self._c2_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        c3 = self._c3_interp(
            (cell_state.layer_temperature, cell_state.layer_soc)
        ).squeeze()
        return np.r_[c1, c2, c3]

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
