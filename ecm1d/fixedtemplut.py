#!/usr/bin/env python3

from __future__ import annotations
from os import path
import numpy as np
import scipy.interpolate
import pandas as pd
from .ecm import BaseParameters


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


class FixedTemperatureLUT(BaseParameters):
    def __init__(
        self, pars_filename, ocv_filename, nlayers=1, method="linear", s=None
    ):
        methods = ["linear", "pchip", "cubicspline", "smoothingspline"]
        if method not in methods:
            raise ValueError(f"Method must be one of {', '.join(methods)}")

        diffusivity = 0.9048e-6
        heat_capacity = 880
        line_density = 11.13
        thickness = 0.0115
        capacity_Ah = 2.16

        super().__init__(
            nlayers,
            diffusivity,
            heat_capacity,
            line_density,
            thickness,
            capacity_Ah,
        )

        df = pd.read_csv(pars_filename)
        ocv_df = pd.read_csv(ocv_filename)

        self._ocv_interp = self._get_interpolator(
            ocv_df, "OCV[V]", method, s
        )
        self._r0_interp = self._get_interpolator(df, "R0", method, s)
        self._r1_interp = self._get_interpolator(df, "R1", method, s)
        self._r2_interp = self._get_interpolator(df, "R2", method, s)
        self._c1_interp = self._get_interpolator(df, "C1", method, s)
        self._c2_interp = self._get_interpolator(df, "C2", method, s)

    @staticmethod
    def _get_interpolator(df, header, method, s):
        xs = df["SOC"]
        ys = df[header]
        if method == "linear":
            return lambda soc: np.interp(soc, xs, ys)
        if method == "pchip":
            return scipy.interpolate.PchipInterpolator(
                xs, ys, extrapolate=False
            )
        if method == "cubicspline":
            return scipy.interpolate.CubicSpline(xs, ys, extrapolate=False)
        if method == "smoothingspline":
            spline = scipy.interpolate.UnivariateSpline(
                xs, ys, s=s, ext="raise"
            )

            def interpolate(socs):
                try:
                    return spline(socs)
                except ValueError:
                    return np.nan * np.ones_like(socs)

            return interpolate
        raise ValueError("Invalid interpolation method requested")

    @check_result
    def get_entropy(self, soc: float | np.ndarray) -> float | np.ndarray:
        return np.array((0))

    @check_result
    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        return self._ocv_interp(soc)

    @check_result
    def get_unscaled_r0(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        return self._r0_interp(soc)

    @check_result
    def get_unscaled_ris(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        r1 = self._r1_interp(soc)
        r2 = self._r2_interp(soc)
        return np.vstack((r1, r2))

    @check_result
    def get_unscaled_cis(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        c1 = self._c1_interp(soc)
        c2 = self._c2_interp(soc)
        return np.vstack((c1, c2))
