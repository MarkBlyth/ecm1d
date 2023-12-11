#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import ecm
import kokamlut


"""
Thermal data from https://www.sciencedirect.com/science/article/pii/S1359431122005233?via%3Dihub
"""


def main():
    lut = kokamlut.KokamLUT()
    heat_equation = ecm.HeatEquationMOL(
        0.9048e-6,
        [0, 1e4],
        [0, 25],
        domain_size=0.0115,
    )

    currents = 5 * np.r_[0, np.ones(3600)]
    ts = np.r_[-1, np.linspace(0, 3600, currents.size - 1)]

    _, [(ax, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3)
    battery = ecm.ECMStack(lut, heat_equation, stacksize=10, capacity_Ah=5)
    results = battery.run(
        ts, currents, np.r_[0, 0, 0], 0.1, 25, True, atol=1e-3, rtol=1e-3, method="BDF"
    )

    ax.plot(results.ts, results.terminal_voltage)
    ax.set_ylabel("Terminal voltage [V]")

    final_tempdist = results.layer_temperatures[:, -1]
    ax2.plot(np.linspace(0, 1, final_tempdist.size), final_tempdist)
    ax2.set_xlabel("Nondimensional position")
    ax2.set_ylabel("Temperature [degC]")

    for current in results.layer_currents:
        ax3.plot(results.ts, current)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Layer currents [A]")

    for socs in results.layer_socs:
        ax4.plot(results.ts, socs)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Layer SOC")

    for heatgen in results.layer_heat_generation:
        ax5.plot(results.ts, heatgen)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Layer heat generation [W]")

    for temp in results.layer_temperatures:
        ax6.plot(results.ts, temp)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Layer temperature [oC]")

    plt.show()


if __name__ == "__main__":
    main()
