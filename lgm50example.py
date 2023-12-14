#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import ecm
import lgm50lut


def currentdraw(ts: float | np.ndarray) -> float | np.ndarray:
    if ts == 0:
        return 0
    return 5


def main():
    external_temperature = 25
    convection_rate = 1e4
    parameters = lgm50lut.LGM50Parameters(10)
    battery = ecm.ECM(parameters)

    (
        ts,
        terminal_voltage,
        layer_currents,
        layer_temperatures,
        layer_socs,
        layer_heatgen,
        layer_rcs,
    ) = battery.run(
        currentdraw,
        [0, convection_rate],
        [external_temperature, external_temperature],
        np.r_[0, 0, 0],
        0.1,
        external_temperature,
        scipy.integrate.BDF,
        atol=1e-5,
        rtol=1e-5,
    )

    _, [(ax, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3)

    ax.plot(ts, terminal_voltage)
    ax.set_ylabel("Terminal voltage [V]")

    final_tempdist = layer_temperatures[:, -1]
    ax2.plot(np.linspace(0, 1, final_tempdist.size), final_tempdist)
    ax2.set_xlabel("Nondimensional position")
    ax2.set_ylabel("Temperature [degC]")

    for soc in layer_socs:
        ax3.plot(ts, soc)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Layer socs")

    for heatgen in layer_heatgen:
        ax4.plot(ts, heatgen)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Layer heat generation [W]")

    for temp in layer_temperatures:
        ax5.plot(ts, temp)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Layer temperature [oC]")

    ax6.plot(ts, convection_rate * (layer_temperatures[-1] - external_temperature))
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Convective heat loss")

    plt.show()


if __name__ == "__main__":
    main()
