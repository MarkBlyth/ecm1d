from __future__ import annotations
from typing import Callable
import numpy as np
import scipy.integrate
from . import ecm


class HomogenousECM(ecm._BaseECM):
    def __init__(self, parameters: ecm.BaseParameters):
        super().__init__(parameters)

    def _ode_rhs(
        self,
        t: float,
        x: np.ndarray,
        thermal_coeff: float,
        current_func: Callable,
        convection: float | list | np.ndarray,
        temp_inf: float | list | np.ndarray,
    ) -> np.ndarray:
        temperature = x[0]
        soc = x[1]
        rc_voltages = x[2:]
        current = current_func(t)
        (
            _,
            series_resistance,
            rc_resistances,
            rc_capacitances,
            entropy,
        ) = self._get_parameters(soc, temperature)
        heat_gen = self._get_heating_rate(
            current,
            temperature,
            series_resistance,
            rc_resistances.squeeze(),
            rc_voltages,
            entropy,
        )
        cooling = self._get_cooling_rate(temperature, convection, temp_inf)
        d_temperature_dt = heat_gen * thermal_coeff + cooling
        d_soc_dt = current / (self.parameters.capacity_Ah * 3600)
        d_rcvoltages_dt = (current * rc_resistances.squeeze() - rc_voltages) / (
            rc_capacitances.squeeze() * rc_resistances.squeeze()
        )
        ret = np.r_[d_temperature_dt, d_soc_dt, d_rcvoltages_dt]
        if any(np.isnan(ret)):
            raise ecm.ParameterException
        return ret

    @staticmethod
    def _get_heating_rate(
        current: float | np.ndarray,
        temperature: float | np.ndarray,
        series_resistance: float | np.ndarray,
        rc_resistances: np.ndarray,
        rc_voltages: np.ndarray,
        entropy: float | np.ndarray,
    ) -> float:
        reversible_heat = -current * temperature * entropy
        # current**2 * R0 + v_RC **2 / R_RC
        irreversible_heat = current**2 * series_resistance + (
            rc_voltages**2 / rc_resistances
        ).sum(axis=0)
        return reversible_heat + irreversible_heat

    @staticmethod
    def _get_cooling_rate(temperature, convection_coeffs, temp_inf) -> float:
        try:
            lhs_loss = convection_coeffs[0] * (temp_inf[0] - temperature)
            rhs_loss = convection_coeffs[1] * (temp_inf[1] - temperature)
            return lhs_loss + rhs_loss
        except TypeError:
            return convection_coeffs * (temp_inf - temperature)

    def run(
        self,
        currentdraw: Callable | float,
        convection_coeffs: float | list[float] | np.ndarray,
        temp_inf: float | list[float] | np.ndarray,
        initial_rc_voltages: np.ndarray,
        initial_soc: float | np.ndarray = 0,
        initial_temp: float | np.ndarray = 25,
        solver=scipy.integrate.BDF,
        **kwargs,
    ) -> tuple[list[float], list[float]]:
        thermal_coeff = 1 / (
            self.parameters.heat_capacity
            * self.parameters.line_density
            * self.parameters.thickness
        )
        initial_cond = np.r_[initial_temp, initial_soc, initial_rc_voltages]
        try:
            currentdraw(0)
            currentfunc = currentdraw
        except TypeError:
            currentfunc = lambda x: currentdraw

        ts, states = self._integrator(
            solver,
            np.inf,
            initial_cond,
            lambda t, x: self._ode_rhs(
                t, x, thermal_coeff, currentfunc, convection_coeffs, temp_inf
            ),
            **kwargs,
        )
        if ts is None:
            return None
        return self._postprocess(ts, states, currentfunc)

    def _postprocess(
        self, ts: list[float], states: list[np.ndarray], currentfunc: Callable
    ):
        states = np.array(states).T
        ts = np.array(ts)

        temperatures = states[0]
        socs = states[1]
        rc_voltages = states[2:]

        currents = np.fromiter([currentfunc(t) for t in ts], dtype=float)
        series_resistances = self.parameters.get_r0(socs, temperatures)
        entropies = self.parameters.get_entropy(socs)
        rc_resistances = self.parameters.get_ris(socs, temperatures)
        heatgens = self._get_heating_rate(
            currents,
            temperatures,
            series_resistances,
            rc_resistances,
            rc_voltages,
            entropies,
        )
        vs = (
            self.parameters.get_ocv(socs)
            + currents * series_resistances
            + rc_voltages.sum(axis=0)
        )
        return ecm.Solution(ts, vs, currents, temperatures, socs, heatgens, rc_voltages)
