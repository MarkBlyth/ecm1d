from __future__ import annotations
from typing import Callable
import abc
import warnings
import collections
import numpy as np
import scipy.integrate


Solution = collections.namedtuple(
    "Solution",
    ["t", "v", "currents", "temperatures", "socs", "heatgens", "rc_voltages"],
)


class ParameterException(Exception):
    pass


class BaseParameters(abc.ABC):
    """
    Interface for implementing an ECM-compatible lookup-table
    (regression model!). User is responsible for handling data, and
    implementing interpolation (probably scipy interpolate).
    """

    def __init__(
        self,
        nlayers: int,
        diffusivity: float,
        heat_capacity: float,
        line_density: float,
        thickness: float,
        capacity_Ah: float,
    ):
        self.nlayers = nlayers
        self.diffusivity = diffusivity
        self.heat_capacity = heat_capacity
        self.line_density = line_density
        self.thickness = thickness
        self.capacity_Ah = capacity_Ah
        self.layer_capacity_As = capacity_Ah * 3600 / nlayers

    @abc.abstractmethod
    def get_entropy(self, soc: float | np.ndarray) -> float | np.ndarray:
        pass

    @abc.abstractmethod
    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        pass

    @abc.abstractmethod
    def get_unscaled_r0(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        pass

    @abc.abstractmethod
    def get_unscaled_ris(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> np.ndarray:
        # one row = one RC pair; one col = one layer
        pass

    @abc.abstractmethod
    def get_unscaled_cis(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> np.ndarray:
        # one row = one RC pair; one col = one layer
        pass

    def get_r0(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        return self.get_unscaled_r0(soc, temperature) * self.nlayers

    def get_ris(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> np.ndarray:
        # one row = one RC pair; one col = one layer
        return self.get_unscaled_ris(soc, temperature) * self.nlayers

    def get_cis(
        self,
        soc: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> np.ndarray:
        # one row = one RC pair; one col = one layer
        return self.get_unscaled_cis(soc, temperature) / self.nlayers


class _HeatEquationMOL:
    """
    Method-of-lines discretisation of a 1d inhomogenous heat equation.
    Discretises in space, using finite differences on an equispaced
    mesh. This defines a system of ODEs, defined over the mesh, which
    can subsequently be integrated by an ODE solver. Robin BCs are
    used by default, of form [c_left_side, c_right_side]. Taking c=0
    results in a Dirichlet BC; c=np.inf produces a Neumann BC; c
    finite gives a Robin BC. Neumann and Robin BCs are defined by the
    pinned or convective surrounding temperature defined by
    [temperature_infinity_left, temperature_infinity_right].

    Note that taking c=np.inf effectively means the respective ECM has
    heat pulled out of it infinitely fast. That's not physically
    meaningful, therefore the routine will raise a warning. Expect the
    ODE solver to struggle with converging.
    """

    def __init__(
        self,
        diffusivity: float,
        heat_capacity: float,
        line_density: float,
        convection_coeffs: list[float] | np.ndarray,
        temperature_infinity: list[float] | np.ndarray | Callable,
        domain_size: float = 1,
    ):
        if any([np.isinf(cc) for cc in convection_coeffs]):
            warnings.warn(
                "Dirichlet BCs are not physically meaningful due to internal heat generation, even when modelling batteries under thermal control. Consider switching to Robin BCs with high convection rates, or expect the ODE solver to struggle or fail."
            )
        self._diffusivity = diffusivity
        self._forcing_coeff = 1 / (line_density * heat_capacity)
        self._convection_coeffs = list([np.nan_to_num(cc) for cc in convection_coeffs])
        try:
            temperature_infinity(0)
            self._temp_inf = temperature_infinity
        except TypeError:
            self._temp_inf = lambda t: temperature_infinity
        self.domain_size = domain_size

    def get_step_max(self, n_points):
        dx = self.domain_size / (n_points - 1)
        return 0.5 * dx**2 / self._diffusivity

    def _get_boundary_conds(self, time, lhs, rhs, dx):
        """
        Construct fictitious mesh points to implement boudary
        conditions
        """
        temp_inf = self._temp_inf(time)
        lhs_fictitious = lhs - dx * self._convection_coeffs[0] * (lhs - temp_inf[0])
        rhs_fictitious = rhs - dx * self._convection_coeffs[1] * (rhs - temp_inf[1])
        return lhs_fictitious, rhs_fictitious

    def get_ode_rhs(
        self, time: float, state: np.ndarray, forcing: np.ndarray
    ) -> np.ndarray:
        """
        state captures temperature at any given point across the cell.
        This is driven by some forcing term. In this context, forcing
        is the heat generated by any given unit ECM, which drives its
        associated mesh point.
        """
        dx = self.domain_size / (state.size - 1)
        lhs_fictitious, rhs_fictitious = self._get_boundary_conds(
            time, state[0], state[-1], dx
        )
        augmented_state = np.r_[lhs_fictitious, state, rhs_fictitious]
        forward_diffs = (
            augmented_state[:-2] + augmented_state[2:] - 2 * augmented_state[1:-1]
        )
        return (
            self._diffusivity * forward_diffs / (dx**2)
            + self._forcing_coeff * forcing
        )


class _BaseECM(abc.ABC):
    def __init__(self, parameters: BaseParameters):
        self.parameters = parameters

    def _get_parameters(self, layer_socs: np.ndarray, layer_temperatures: np.ndarray):
        ocvs = self.parameters.get_ocv(layer_socs)
        series_resistances = self.parameters.get_r0(layer_socs, layer_temperatures)
        rc_resistances = self.parameters.get_ris(layer_socs, layer_temperatures)
        rc_capacitances = self.parameters.get_cis(layer_socs, layer_temperatures)
        entropies = self.parameters.get_entropy(layer_socs)
        return ocvs, series_resistances, rc_resistances, rc_capacitances, entropies

    @abc.abstractmethod
    def run(
        self,
        currentdraw: Callable | float,
        convection_coeffs: list[float] | np.ndarray,
        temp_inf: list[float] | np.ndarray | Callable,
        initial_rc_voltages: np.ndarray,
        initial_soc: float | np.ndarray,
        initial_temp: float | np.ndarray,
        solver=scipy.integrate.BDF,
        **kwargs,
    ):
        pass

    @staticmethod
    def _integrator(
        solver: Callable,
        step_max: float,
        initial_cond: np.ndarray,
        ode_func: Callable,
        **kwargs,
    ):
        """
        ODE solver must stop when a ParameterException is raised, so
        the standard scipy.integrate routines can't be used. Instead,
        use the lower level API, whereby a scipy integrator is passed
        ('solver' arg), and stepped manually.
        """
        ts, states = [], []
        try:
            solverinstance = solver(
                ode_func,
                0,
                initial_cond,
                np.inf,
                max_step=step_max,
                #options={"max_step": step_max},
                **kwargs,
            )
            while solverinstance.status == "running":
                ts.append(solverinstance.t)
                states.append(solverinstance.y)
                solverinstance.step()
        except ParameterException:
            pass
        if len(ts) == 0:
            return None, None
        return ts, states


class ECM(_BaseECM):
    @staticmethod
    def _get_layer_currents(
        layer_rc_voltages: np.ndarray,  # One row = one RC pair; one col = one layer
        terminal_voltage: float,
        ocvs: np.ndarray,
        series_resistances: np.ndarray,
    ) -> np.ndarray:
        return (
            terminal_voltage - ocvs - layer_rc_voltages.sum(axis=0)
        ) / series_resistances

    def _get_soc_odes(self, layer_currents: np.ndarray) -> np.ndarray:
        return layer_currents / self.parameters.layer_capacity_As

    @staticmethod
    def _get_rc_odes(
        layer_currents: np.ndarray,
        layer_rc_voltages: np.ndarray,  # one row = one RC pair; one col = one layer
        rc_resistances: np.ndarray,  # One row = one RC pair; one col = one layer
        rc_capacitances: np.ndarray,  # one row = one RC pair; one col = one layer
    ) -> np.ndarray:
        ret = np.zeros_like(layer_rc_voltages)
        for i, (voltages, resistances, capacitances) in enumerate(
            zip(layer_rc_voltages, rc_resistances, rc_capacitances)
        ):
            ret[i] = (layer_currents - voltages / resistances) / capacitances
        return ret

    @staticmethod
    def _get_heat_gen(
        layer_currents: np.ndarray,
        layer_temperatures: np.ndarray,
        series_resistances: np.ndarray,
        rc_resistances: np.ndarray,
        rc_voltages: np.ndarray,
        entropies: np.ndarray,
    ) -> np.ndarray:
        reversible_heat = -layer_currents * layer_temperatures * entropies
        # current**2 * R0 + v_RC **2 / R_RC
        irreversible_heat = layer_currents**2 * series_resistances + (
            rc_voltages**2 / rc_resistances
        ).sum(axis=0)
        return reversible_heat + irreversible_heat

    def _ode_rhs(
        self,
        t: float,
        x: np.ndarray,
        currentfunc: Callable,
        heat_equation: _HeatEquationMOL,
    ) -> np.ndarray:
        total_current = currentfunc(t)
        if total_current is None:
            raise ParameterException

        temperatures, socs, rc_voltages = self._unpack_state(x, self.parameters.nlayers)
        (
            ocvs,
            series_resistances,
            rc_resistances,
            rc_capacitances,
            entropies,
        ) = self._get_parameters(socs, temperatures)

        terminal_voltage = self._get_output_voltage(
            total_current, rc_voltages, ocvs, series_resistances
        )
        layer_currents = self._get_layer_currents(
            rc_voltages, terminal_voltage, ocvs, series_resistances
        )

        heat_gen = self._get_heat_gen(
            layer_currents,
            temperatures,
            series_resistances,
            rc_resistances,
            rc_voltages,
            entropies,
        )
        forcing = heat_gen * self.parameters.nlayers / self.parameters.thickness
        d_temperature_dt = heat_equation.get_ode_rhs(t, temperatures, forcing)
        d_soc_dt = self._get_soc_odes(layer_currents)
        d_rc_dt = self._get_rc_odes(
            layer_currents, rc_voltages, rc_resistances, rc_capacitances
        )

        ode_rhs = np.empty_like(x).reshape((-1, self.parameters.nlayers))
        ode_rhs[0] = d_temperature_dt
        ode_rhs[1] = d_soc_dt
        ode_rhs[2:] = d_rc_dt

        if any(np.isnan(ode_rhs.ravel())):
            raise ParameterException

        return ode_rhs.ravel()

    @staticmethod
    def _unpack_state(state, n_ecms):
        # state = col vectors of [temperature, soc, rc vals] for each layer
        unit_ecm_states = state.reshape((-1, n_ecms))
        temperatures = unit_ecm_states[0]
        socs = unit_ecm_states[1]
        rc_voltages = unit_ecm_states[2:]
        return temperatures, socs, rc_voltages

    @staticmethod
    def _get_output_voltage(
        total_current: float,
        layer_rc_voltages: np.ndarray,  # one row = one RC pair; one col = one layer
        layer_ocvs: np.ndarray,
        layer_series_resistances: np.ndarray,
    ) -> float:
        rc_contribution = (
            layer_rc_voltages.sum(axis=0) / layer_series_resistances
        ).sum()
        ocv_contribution = (layer_ocvs / layer_series_resistances).sum()
        series_conductance = (1 / layer_series_resistances).sum()
        return (ocv_contribution + rc_contribution + total_current) / series_conductance

    def _postprocess(
        self, ts: list[float], states: list[np.ndarray], currentfunc: Callable
    ):
        n_rcs = int(states[0].size / self.parameters.nlayers) - 2
        terminal_voltages = np.zeros(len(ts))
        layer_currents = np.zeros((self.parameters.nlayers, len(ts)))
        layer_temps = np.zeros((self.parameters.nlayers, len(ts)))
        layer_heatgen = np.zeros((self.parameters.nlayers, len(ts)))
        layer_socs = np.zeros((self.parameters.nlayers, len(ts)))
        layer_rcs = np.zeros((self.parameters.nlayers, len(ts), n_rcs))

        for i, (t, state) in enumerate(zip(ts, states)):
            # TODO avoid repetition with ODEs
            total_current = currentfunc(t)
            temperatures, socs, rc_voltages = self._unpack_state(
                state, self.parameters.nlayers
            )
            (
                ocvs,
                series_resistances,
                rc_resistances,
                rc_capacitances,
                entropies,
            ) = self._get_parameters(socs, temperatures)

            terminal_voltage = self._get_output_voltage(
                total_current, rc_voltages, ocvs, series_resistances
            )
            currents = self._get_layer_currents(
                rc_voltages, terminal_voltage, ocvs, series_resistances
            )

            heat_gen = self._get_heat_gen(
                currents,
                temperatures,
                series_resistances,
                rc_resistances,
                rc_voltages,
                entropies,
            )

            terminal_voltages[i] = terminal_voltage
            layer_currents[:, i] = currents
            layer_temps[:, i] = temperatures
            layer_socs[:, i] = socs
            layer_heatgen[:, i] = heat_gen
            layer_rcs[:, i, :] = rc_voltages.T
        return Solution(
            np.array(ts),
            terminal_voltages,
            layer_currents,
            layer_temps,
            layer_socs,
            layer_heatgen,
            layer_rcs,
        )

    def run(
        self,
        currentdraw: Callable | float,
        convection_coeffs: list[float] | np.ndarray,
        temp_inf: list | np.ndarray | Callable,
        initial_rc_voltages: np.ndarray,
        initial_soc: float | np.ndarray = 0,
        initial_temp: float | np.ndarray = 25,
        solver=scipy.integrate.BDF,
        step_max=None,
        **kwargs,
    ):
        # Build initial condition
        socs = initial_soc * np.ones(self.parameters.nlayers)
        temps = initial_temp * np.ones(self.parameters.nlayers)
        if initial_rc_voltages.ndim == 1:
            ones = np.ones((initial_rc_voltages.size, self.parameters.nlayers))
            initial_rc_voltages = initial_rc_voltages.reshape((-1, 1)) * ones
        initial_cond = np.empty(
            initial_rc_voltages.size + 2 * self.parameters.nlayers
        ).reshape((-1, self.parameters.nlayers))
        initial_cond[0] = temps
        initial_cond[1] = socs
        initial_cond[2:] = initial_rc_voltages
        initial_cond = initial_cond.ravel()

        # Build MOL heat equation
        heat_equation = _HeatEquationMOL(
            self.parameters.diffusivity,
            self.parameters.heat_capacity,
            self.parameters.line_density,
            convection_coeffs,
            temp_inf,
            self.parameters.thickness,
        )
        if step_max is None:
            step_max = (
                np.inf
                if self.parameters.nlayers == 1
                else 0.9 * heat_equation.get_step_max(self.parameters.nlayers)
            )

        try:
            currentdraw(0)
            currentfunc = currentdraw
        except TypeError:
            currentfunc = lambda x: currentdraw

        ts, states = self._integrator(
            solver,
            step_max,
            initial_cond,
            lambda t, x: self._ode_rhs(t, x, currentfunc, heat_equation),
            **kwargs,
        )
        if ts is None:
            return None
        return self._postprocess(ts, states, currentfunc)
