import abc
import numpy as np
import collections


ElectricalState = collections.namedtuple("ElectricalState", [])
TimeStep = collections.namedtuple("TimeStep", [])


class ECMResults:
    """
    Stacked ECM data storage.

    should be addressable as output[i] to get (time, current, ...), or
    output.time, output.current, ...
    """

    def __init__(self, n_timestamps: int):
        # TODO
        # Store data as np arrays of pre-determined size
        pass

    def append_state(
        self,
        time: float,
        soc: float,
        current: float,
        terminal_voltage: float,
        cell_resistance: float,
        heat_generation: float,
        entropy_coeffs: np.ndarray,
        layer_currents: np.ndarray,
        layer_socs: np.ndarray,
        layer_temperatures: np.ndarray,
        layer_heat_generation: np.ndarray,
        layer_ocvs: np.ndarray,
    ) -> None:
        pass

    def __getitem__(self, index: int) -> TimeStep:
        pass


class HeatEquationMOL:

    def __init__(self, initial_cond: np.ndarray):
        pass

    def get_ode_rhs(self, heat_state: np.ndarray, heat_gen: np.ndarray):
        pass


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
