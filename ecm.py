from __future__ import annotations
import abc
import numpy as np
import collections
import warnings


ElectricalState = collections.namedtuple("ElectricalState", [])

TimeStep = collections.namedtuple(
    "TimeStep",
    [
        "time",
        "soc",
        "current",
        "terminal_voltage",
        "cell_resistance",
        "heat_generation",
        "layer_entropy_coeffs",
        "layer_currents",
        "layer_socs",
        "layer_temperatures",
        "layer_heat_generation",
        "layer_ocvs",
    ],
)


class ECMResults:
    """
    Stacked ECM data storage.

    Addressable as output[i] to get TimeStep(time, soc, ...), or
    output.time, output.current, ...
    """

    def __init__(self, n_timestamps: int, n_layers: int):
        self._n_entries = 0
        self._ts = np.array(n_timestamps)
        self._soc = np.array(n_timestamps)
        self._current = np.array(n_timestamps)
        self._terminal_voltage = np.array(n_timestamps)
        self._cell_resistance = np.array(n_timestamps)
        self._heat_generation = np.array(n_timestamps)
        self._layer_entropy_coeffs = np.array((n_layers, n_timestamps))
        self._layer_currents = np.array((n_layers, n_timestamps))
        self._layer_socs = np.array((n_layers, n_timestamps))
        self._layer_temperatures = np.array((n_layers, n_timestamps))
        self._layer_heat_generation = np.array((n_layers, n_timestamps))
        self._layer_ocvs = np.array((n_layers, n_timestamps))

    def append_state(
        self,
        time: float,
        soc: float,
        current: float,
        terminal_voltage: float,
        cell_resistance: float,
        heat_generation: float,
        layer_entropy_coeffs: np.ndarray,
        layer_currents: np.ndarray,
        layer_socs: np.ndarray,
        layer_temperatures: np.ndarray,
        layer_heat_generation: np.ndarray,
        layer_ocvs: np.ndarray,
    ) -> None:
        """
        Record new data.
        """
        if self._n_entries == self._ts.size:
            raise ValueError(
                "Cannot append additional items. Data structure has reached initialised size."
            )
        self._ts[self._n_entries] = time
        self._soc[self._n_entries] = soc
        self._current[self._n_entries] = current
        self._terminal_voltage[self._n_entries] = terminal_voltage
        self._cell_resistance[self._n_entries] = cell_resistance
        self._heat_generation[self._n_entries] = heat_generation
        self._layer_entropy_coeffs[:, self._n_entries] = layer_entropy_coeffs
        self._layer_currents[:, self._n_entries] = layer_currents
        self._layer_socs[:, self._n_entries] = layer_socs
        self._layer_temperatures[:, self._n_entries] = layer_temperatures
        self._layer_heat_generation[:, self._n_entries] = layer_heat_generation
        self._layer_ocvs[:, self._n_entries] = layer_ocvs
        self._n_entries = self._n_entries + 1

    def __getitem__(self, i: int) -> TimeStep:
        """
        ecmresults[i] returns a TimeStep object of datapoints at index
        i
        """
        if i <= self._n_entries:
            raise ValueError(f"Index {i} exceeds number of entries ({self._n_entries})")
        return TimeStep(
            self._ts[i],
            self._soc[i],
            self._current[i],
            self._terminal_voltage[i],
            self._cell_resistance[i],
            self._heat_generation[i],
            self._layer_entropy_coeffs[:, i],
            self._layer_socs[:, i],
            self._layer_temperatures[:, i],
            self._layer_heat_generation[:, i],
            self._layer_ocvs[:, i],
        )

    def __getattribute__(self, item):
        """
        Only return the first n_entries datapoints when doing (eg.)
        results.time etc.
        """
        item = object.__getattribute__(self, item)
        if item.ndim == 1:
            return item[: self._n_entries]
        return item[:, : self._n_entries]


class BaseECMUnit(abc.ABC):
    def get_electrical_parameters(
        self, voltage: float, temperature: float
    ) -> ElectricalState:
        pass

    def get_ode_rhs(
        self, voltage: float, temperature: float, ode_state: np.ndarray
    ) -> np.ndarray:
        pass

    def set_soc(self, new_soc: float) -> None:
        pass

    def set_rc_voltages(self, new_rc_voltages: np.ndarray) -> None:
        pass

    def get_heat_gen(self, battery_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_ocv(self, battery_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_r0(self, battery_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_ris(self, battery_state: ElectricalState) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_cis(self, battery_state: ElectricalState) -> np.ndarray:
        pass


class HeatEquationMOL:
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
        self, diffusivity: float, convection_coeffs: list, temperature_infinity: list
    ):
        if any([np.isinf(cc) for cc in convection_coeffs]):
            warnings.warn(
                "Neumann BCs are not physically meaningful due to internal heat generation, even when modelling batteries under thermal control. Consider switching to Robin BCs with high convection rates, or expect the ODE solver to struggle or fail."
            )
        self._diffusivity = diffusivity
        self._convection_coeffs = list([np.nan_to_num(cc) for cc in convection_coeffs])
        self._temp_inf = temperature_infinity

    def _get_boundary_conds(self, lhs, rhs, dx):
        """
        Construct fictitious mesh points to implement boudary
        conditions
        """
        lhs_fictitious = lhs - dx * self._convection_coeffs[0] * (
            lhs - self._temp_inf[0]
        )
        rhs_fictitious = rhs - dx * self._convection_coeffs[1] * (
            rhs - self._temp_inf[1]
        )
        return lhs_fictitious, rhs_fictitious

    def get_ode_rhs(self, state: np.ndarray, forcing: np.ndarray) -> np.ndarray:
        """
        state captures temperature at any given point across the cell.
        This is driven by some forcing term. In this context, forcing
        is the heat generated by any given unit ECM, which drives its
        associated mesh point.
        """
        dx = 1 / (state.size - 1)
        lhs_fictitious, rhs_fictitious = self._get_boundary_conds(
            state[0], state[-1], dx
        )
        augmented_state = np.r_[lhs_fictitious, state, rhs_fictitious]
        forward_diffs = (
            augmented_state[:-2] + augmented_state[2:] - 2 * augmented_state[1:-1]
        )
        return self._diffusivity * forward_diffs / (dx**2) + forcing


class ECMStack:
    def __init__(self, unit_ecms: list[BaseECMUnit]):
        self._ecms = unit_ecms

    def run(
        self,
        initial_soc: np.ndarray | float,
        initial_temp: np.ndarray | float,
        ts: np.ndarray,
        currents: np.ndarray,
    ) -> ECMResults:
        pass

    def _timestep(self, timestep: float, current: float) -> list[ElectricalState]:
        pass

    def _get_temperature(
        self, position_x: np.ndarray | float, time_t: np.ndarray | float
    ) -> np.ndarray | float:
        pass


def _unpack_state(state, n_ecms):
    # state = [temperature, soc, rc vals] * n_ecms
    unit_ecm_states = state.reshape((n_ecms, -1))
    temperatures = unit_ecm_states[:, 0]
    socs = unit_ecm_states[:, 1]
    rc_vals = unit_ecm_states[:, 2:]
    return temperatures, socs, rc_vals
