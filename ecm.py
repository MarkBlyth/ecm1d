from __future__ import annotations
import abc
import numpy as np
import collections
import warnings
import scipy.integrate

try:
    import progressbar

    PROGRESSBAR = True
except ImportError:
    PROGRESSBAR = False


ElectricalState = collections.namedtuple(
    "ElectricalState",
    ["terminal_voltage", "layer_temperature", "layer_soc", "layer_rc_voltages"],
)
CellODE = collections.namedtuple(
    "CellODE", ["d_soc_dt", "d_rcvoltages_dt", "layer_heatgen", "layer_current"]
)
TimeStep = collections.namedtuple(
    "TimeStep",
    [
        "time",
        "soc",
        "current",
        "terminal_voltage",
        "heat_generation",
        "layer_entropy_coeffs",
        "layer_currents",
        "layer_socs",
        "layer_temperatures",
        "layer_heat_generation",
        "layer_ocvs",
        "electrical_states",
    ],
)


class ECMResults:
    """
    Nothing but data storage. Maintains arrays of lots of relevant
    variables, and allows them to be accessed in nice ways.

    Addressable as output[i] to get TimeStep(time, soc, ...), or
    output.time, output.current, ...
    """

    def __init__(self, n_timestamps: int, n_layers: int):
        self._n_entries = 0
        self._ts = np.array(n_timestamps)
        self._soc = np.array(n_timestamps)
        self._current = np.array(n_timestamps)
        self._terminal_voltage = np.array(n_timestamps)
        self._heat_generation = np.array(n_timestamps)
        self._layer_entropy_coeffs = np.array((n_layers, n_timestamps))
        self._layer_currents = np.array((n_layers, n_timestamps))
        self._layer_socs = np.array((n_layers, n_timestamps))
        self._layer_temperatures = np.array((n_layers, n_timestamps))
        self._layer_heat_generation = np.array((n_layers, n_timestamps))
        self._layer_ocvs = np.array((n_layers, n_timestamps))
        self._electrical_states: list[list[ElectricalState]] = []

    def append_state(
        self,
        time: float,
        soc: float,
        current: float,
        terminal_voltage: float,
        heat_generation: float,
        layer_entropy_coeffs: np.ndarray,
        layer_currents: np.ndarray,
        layer_socs: np.ndarray,
        layer_temperatures: np.ndarray,
        layer_heat_generation: np.ndarray,
        layer_ocvs: np.ndarray,
        electrical_states: list[ElectricalState],
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
        self._heat_generation[self._n_entries] = heat_generation
        self._layer_entropy_coeffs[:, self._n_entries] = layer_entropy_coeffs
        self._layer_currents[:, self._n_entries] = layer_currents
        self._layer_socs[:, self._n_entries] = layer_socs
        self._layer_temperatures[:, self._n_entries] = layer_temperatures
        self._layer_heat_generation[:, self._n_entries] = layer_heat_generation
        self._layer_ocvs[:, self._n_entries] = layer_ocvs
        self._electrical_states.append(electrical_states)
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
            self._heat_generation[i],
            self._layer_entropy_coeffs[:, i],
            self._layer_currents[i],
            self._layer_socs[:, i],
            self._layer_temperatures[:, i],
            self._layer_heat_generation[:, i],
            self._layer_ocvs[:, i],
            self._electrical_states[i],
        )

    def __getattribute__(self, item):
        """
        Only return the first n_entries datapoints when doing (eg.)
        results.time etc.
        """
        item = object.__getattribute__(self, item)
        try:
            if item.ndim == 1:
                return item[: self._n_entries]
        except AttributeError:
            return item[: self._n_entries]
        return item[:, : self._n_entries]


class BaseLUT(abc.ABC):
    """
    Interface for implementing an ECM-compatible lookup-table
    (regression model!). User is responsible for handling data, and
    implementing interpolation (probably scipy interpolate).
    """

    @abc.abstractmethod
    def __init__(self, ocv_data, entropy_data, r0_data, ri_datasets, ci_datasets):
        pass

    @abc.abstractmethod
    def get_entropy(self, cell_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_ocv(self, cell_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_r0(self, cell_state: ElectricalState) -> float:
        pass

    @abc.abstractmethod
    def get_ris(self, cell_state: ElectricalState) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_cis(self, cell_state: ElectricalState) -> np.ndarray:
        pass


class UnitECM(abc.ABC):
    def __init__(self, lookup_table: BaseLUT, capacity_Ah: float):
        self.parameters = lookup_table
        self._capacity = capacity_Ah * 3600

    def get_odes(self, cell_state: ElectricalState) -> CellODE:
        series_resistance = self.parameters.get_r0(cell_state)
        rc_resistances = self.parameters.get_ris(cell_state)
        rc_capacitances = self.parameters.get_cis(cell_state)
        ocv = self.parameters.get_ocv(cell_state)
        my_current = self._get_layer_current(
            cell_state.terminal_voltage,
            cell_state.layer_rc_voltages,
            ocv,
            series_resistance,
        )
        d_soc_dt = my_current / self._capacity
        heat_gen = self._get_heat_gen(
            cell_state, series_resistance, rc_resistances, my_current
        )
        d_layer_rc_voltages_dt = (
            my_current - cell_state.layer_rc_voltages / rc_resistances
        ) / rc_capacitances
        return CellODE(d_soc_dt, d_layer_rc_voltages_dt, heat_gen, my_current)

    def get_layer_current(self, cell_state: ElectricalState) -> float:
        ocv = self.parameters.get_ocv(cell_state)
        series_resistance = self.parameters.get_r0(cell_state)
        return self._get_layer_current(
            cell_state.terminal_voltage,
            cell_state.layer_rc_voltages,
            ocv,
            series_resistance,
        )

    def get_heat_gen(self, cell_state: ElectricalState) -> float:
        series_resistance = self.parameters.get_r0(cell_state)
        rc_resistances = self.parameters.get_ris(cell_state)
        ocv = self.parameters.get_ocv(cell_state)
        my_current = self._get_layer_current(
            cell_state.terminal_voltage,
            cell_state.layer_rc_voltages,
            ocv,
            series_resistance,
        )
        return self._get_heat_gen(
            cell_state, series_resistance, rc_resistances, my_current
        )

    def _get_heat_gen(
        self, cell_state: ElectricalState, r0: float, ris: np.ndarray, my_current: float
    ) -> float:
        entropy = self.parameters.get_entropy(cell_state)
        reversible_heat = my_current * cell_state.layer_temperature * entropy
        irreversible_heat = my_current**2 * (r0 + ris.sum())
        return reversible_heat + irreversible_heat

    @staticmethod
    def _get_layer_current(
        terminal_voltage: float,
        rc_voltages: np.ndarray,
        ocv: float,
        series_resistance: float,
    ) -> float:
        working_voltage = ocv - terminal_voltage - rc_voltages.sum()
        return working_voltage / series_resistance


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

        # TODO let T_infinity be time-dependent!

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
    def __init__(
        self, unit_ecm: UnitECM, heat_equation: HeatEquationMOL, stacksize: int
    ):
        self._ecm = unit_ecm
        self._heat_equation = heat_equation
        self._nstack = stacksize

    def run(
        self,
        ts: np.ndarray,
        currents: np.ndarray,
        initial_rc_voltages: np.ndarray,
        initial_socs: np.ndarray | float = 0,
        initial_temps: np.ndarray | float = 0,
        progress: bool = False,
        **kwargs,
    ) -> ECMResults:
        if progress and not PROGRESSBAR:
            warnings.warn(
                "No progressbar2 module available, proceeding without progress bar."
            )
        elif progress:
            bar = progressbar.ProgressBar(max_value=ts.size - 1)
        else:
            bar = None
        results = ECMResults(ts.size, self._nstack)
        socs = initial_socs * np.ones(self._nstack)
        temps = initial_temps * np.ones(self._nstack)
        initial_cond = np.empty(initial_rc_voltages.size + 2 * self._nstack).reshape(
            (self._nstack, -1)
        )
        for i in range(self._nstack):
            initial_cond[i][0] = temps[i]
            initial_cond[i][1] = socs
            initial_cond[i][2:] = initial_rc_voltages[i]
        self._append_state_to_results(ts[0], initial_cond, currents[0], results)
        for i, (t, dt, current) in enumerate(zip(ts, np.diff(ts), currents)):
            initial_cond, electrical_states = self._timestep(
                dt, current, initial_cond, **kwargs
            )
            self._append_state_to_results(
                t, initial_cond, current, results, electrical_states
            )
            if bar is not None:
                bar.update(i)
        return results

    def _append_state_to_results(
        self,
        t0: float,
        statevec: np.ndarray,
        current: float,
        results: ECMResults,
        electrical_states: list[ElectricalState] | None = None,
    ) -> None:
        temperatures, socs, rc_voltages = self._unpack_state(statevec, self._nstack)
        if electrical_states is None:
            electrical_states = self._get_electrical_states(statevec, current)

        terminal_voltage = electrical_states[0].terminal_voltage
        heatgen = np.fromiter(
            (self._ecm.get_heat_gen(s) for s in electrical_states), float
        )
        total_heatgen = scipy.integrate.trapezoid(
            heatgen, np.linspace(0, 1, self._nstack)
        )
        entropies = np.fromiter(
            (self._ecm.parameters.get_entropy(s) for s in electrical_states), float
        )
        layer_currents = np.fromiter(
            (self._ecm.get_layer_current(s) for s in electrical_states), float
        )
        layer_socs = np.fromiter(
            (self._ecm.parameters.get_ocv(s) for s in electrical_states), float
        )

        results.append_state(
            t0,
            socs.mean(),
            current,
            terminal_voltage,
            total_heatgen,
            entropies,
            layer_currents,
            socs,
            temperatures,
            heatgen,
            layer_socs,
            electrical_states,
        )

    def _timestep(
        self,
        timestep: float,
        current: float,
        initial_state: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, list[ElectricalState]]:
        soln = scipy.integrate.solve_ivp(
            self._ode_rhs, [0, timestep], initial_state, args=current, **kwargs
        )
        finalstate = soln.y[:, -1]
        electrical_states = self._get_electrical_states(finalstate, current)
        return finalstate, electrical_states

    def _ode_rhs(self, t: float, x: np.ndarray, total_current: float) -> np.ndarray:
        electrical_states = self._get_electrical_states(x, total_current)
        layer_odes = [self._ecm.get_odes(state) for state in electrical_states]
        heatstate = np.fromiter(
            (layer.layer_temperature for layer in electrical_states), float
        )
        heatgen = np.fromiter((layer.layer_heatgen for layer in layer_odes), float)
        heat_equation_rhs = self._heat_equation.get_ode_rhs(heatstate, heatgen)
        ode_rhs = np.empty_like(x).reshape((self._nstack, -1))
        for i in range(self._nstack):
            ode_rhs[i][0] = heat_equation_rhs[i]
            ode_rhs[i][1] = layer_odes[i].d_soc_dt
            ode_rhs[i][2:] = layer_odes[i].d_rcvoltages_dt
        return ode_rhs.squeeze()

    def _get_electrical_states(
        self, ode_state: np.ndarray, total_current: float
    ) -> list[ElectricalState]:
        layer_temperatures, layer_socs, layer_rc_voltages = self._unpack_state(
            ode_state, self._nstack
        )
        ocvs = np.empty(self._nstack)
        series_resistances = np.empty(self._nstack)
        for i in range(self._nstack):
            layer_i_dummystate = ElectricalState(
                None, layer_temperatures[i], layer_socs[i], layer_rc_voltages[i]
            )
            ocvs[i] = self._ecm.parameters.get_ocv(layer_i_dummystate)
            series_resistances[i] = self._ecm.parameters.get_r0(layer_i_dummystate)
        output_voltage = self._get_output_voltage(
            total_current, layer_rc_voltages, ocvs, series_resistances
        )
        return [
            ElectricalState(output_voltage, temp, soc, rcvs)
            for temp, soc, rcvs in zip(
                layer_temperatures, layer_socs, layer_rc_voltages
            )
        ]

    @staticmethod
    def _unpack_state(state, n_ecms):
        # state = [temperature, soc, rc vals] * n_ecms
        unit_ecm_states = state.reshape((n_ecms, -1))
        temperatures = unit_ecm_states[:, 0]
        socs = unit_ecm_states[:, 1]
        rc_voltages = unit_ecm_states[:, 2:]
        return temperatures, socs, rc_voltages

    @staticmethod
    def _get_output_voltage(
        total_current: float,
        layer_rc_voltages: np.ndarray,
        layer_ocvs: np.ndarray,
        layer_series_resistances: np.ndarray,
    ) -> float:
        # layer_rc_voltages : np array where row i is vector of rc overpotentials for ECM i; taken from ODE state
        rc_contribution = (
            layer_rc_voltages.sum(axis=1) / layer_series_resistances
        ).sum()
        ocv_contribution = (layer_ocvs / layer_series_resistances).sum()
        series_conductance = (1 / layer_series_resistances).sum()
        return (ocv_contribution - rc_contribution - total_current) / series_conductance
