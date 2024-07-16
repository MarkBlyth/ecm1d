#!/usr/bin/env python3

from __future__ import annotations
from os import path
import numpy as np
import scipy.interpolate
import pandas as pd
from .ecm import BaseParameters


_OCV_PARS = "parameters/pyecn_kokam/OCV-SoC.csv"
_dVdT_PARS = "parameters/pyecn_kokam/dVdT-SoC.csv"


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


class IdealisedParameters(BaseParameters):
    def __init__(
        self,
        nlayers,
        temp_min=0,
        temp_max=40,
        charged_r0_hot=0.003,
        charged_r0_cold=0.015,
        discharged_r0_hot=0.015,
        discharged_r0_cold=0.03,
        lambda_soc=2,
        lambda_temp=2,
    ):
        diffusivity = 0.9048e-6
        heat_capacity = 880
        line_density = 11.13
        thickness = 0.0115
        capacity_Ah = 5

        super().__init__(
            nlayers,
            diffusivity,
            heat_capacity,
            line_density,
            thickness,
            capacity_Ah,
        )

        self.temp_min = temp_min
        self.temp_max = temp_max
        self._charged_r0_cold = charged_r0_cold
        self._charged_r0_hot = charged_r0_hot
        self._discharged_r0_cold = discharged_r0_cold
        self._discharged_r0_hot = discharged_r0_hot
        self._lambda_soc = lambda_soc
        self._lambda_temp = lambda_temp
        self._find_abgd()

        ocv_df = pd.read_csv(path.join(path.dirname(__file__), _OCV_PARS))
        entropy_df = pd.read_csv(
            path.join(path.dirname(__file__), _dVdT_PARS)
        )

        self._ocv_interp = scipy.interpolate.CubicSpline(
            ocv_df["SoC"].to_numpy()[::-1],
            ocv_df["OCV"].to_numpy()[::-1],
        )
        self._entropy_interp = scipy.interpolate.CubicSpline(
            entropy_df["SoC"].to_numpy()[::-1],
            entropy_df["dVdT"].to_numpy()[::-1],
        )

    def _find_abgd(self):
        # R0 = alpha + beta*exp(-lambda_T * T)
        #      + gamma * exp(-lambda_SOC * SOC)
        #      + delta * exp(-lambda_T * T - lambda_SOC * SOC)
        # This finds alpha, beta, gamma, delta. Thanks to sympy for code gen.
        c1 = self._discharged_r0_cold
        c2 = self._charged_r0_cold
        c3 = self._discharged_r0_hot
        c4 = self._charged_r0_hot
        es = np.exp(-self._lambda_soc)
        et = np.exp(-self._lambda_temp)
        ets = np.exp(-self._lambda_soc - self._lambda_temp)
        a = (
            c1 * es**2 * et
            + c1 * es * et**2
            - c1 * es * et * ets
            - 2 * c1 * es * et
            + c1 * ets
            - c2 * et**2
            + c2 * et * ets
            + c2 * et
            - c2 * ets
            - c3 * es**2
            + c3 * es * ets
            + c3 * es
            - c3 * ets
            - c4 * es * et
            + c4 * es
            + c4 * et
            - c4
        ) / (
            es**2 * et
            - es**2
            + es * et**2
            - es * et * ets
            - 3 * es * et
            + es * ets
            + 2 * es
            - et**2
            + et * ets
            + 2 * et
            - ets
            - 1
        )
        b = (
            -c1 * es
            + c1 * ets
            - c2 * et
            + c2
            + c3 * es
            - c3 * ets
            + c4 * et
            - c4
        ) / (es * et - es + et**2 - et * ets - 2 * et + ets + 1)
        g = (
            -c1 * et
            + c1 * ets
            + c2 * et
            - c2 * ets
            - c3 * es
            + c3
            + c4 * es
            - c4
        ) / (es**2 + es * et - es * ets - 2 * es - et + ets + 1)
        d = (-c1 + c2 + c3 - c4) / (es + et - ets - 1)
        self._abgd = [a, b, g, d]

    # @check_result
    def get_entropy(self, soc: float | np.ndarray) -> float | np.ndarray:
        return self._entropy_interp(soc)

    # @check_result
    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        return self._ocv_interp(soc)

    @check_result
    def get_unscaled_r0(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        if any(soc > 1) or any(soc < 0):
            return np.array([np.nan])
        if any(temperature < self.temp_min) or any(
            temperature > self.temp_max
        ):
            return np.array([np.nan])
        rel_temp = (temperature - self.temp_min) / (  # Relative temperature
            self.temp_max - self.temp_min  # in range [0,1]
        )
        exp_T = np.exp(-self._lambda_temp * rel_temp)
        exp_SOC = np.exp(-self._lambda_soc * soc)
        expexp = exp_T * exp_SOC
        alpha, beta, gamma, delta = self._abgd
        return alpha + beta * exp_T + gamma * exp_SOC + delta * expexp

    def get_unscaled_ris(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        return np.array([1e-9])

    def get_unscaled_cis(
        self, soc: float | np.ndarray, temperature: float | np.ndarray
    ) -> np.ndarray:
        return np.array([1e-6])
