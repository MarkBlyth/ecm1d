#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import ecm1d


def currentdraw(ts: float | np.ndarray) -> float | np.ndarray:
    if ts == 0:
        return 0
    if ts < 1500:
        return -10
    return 10


def main():
    nlayer = 10
    parameters = ecm1d.IdealisedParameters(nlayer)
    battery = ecm1d.ECM(parameters)

    socs = np.linspace(0, 1, 20)
    temps = np.linspace(parameters.temp_min, parameters.temp_max, 20)
    xs, ys = np.meshgrid(socs, temps)
    r0s = parameters.get_r0(xs.ravel(), ys.ravel()).reshape(xs.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xs, ys, r0s, cmap="coolwarm", linewidth=0, antialiased=False
    )
    plt.show()

    external_temperature = 10.1
    convection_rate = 1e4
    soln = battery.run(
        currentdraw,
        [0, convection_rate],
        [external_temperature, external_temperature],
        np.array([0]),
        0.99,
        external_temperature,
    )

    _, [(ax, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3)

    ax.plot(soln.t, soln.v)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal voltage [V]")

    final_tempdist = soln.temperatures[:, -1]
    ax2.plot(np.linspace(0, 1, final_tempdist.size), final_tempdist)
    ax2.set_xlabel("Nondimensional position")
    ax2.set_ylabel("Temperature [degC]")

    for current in soln.currents:
        ax3.plot(soln.t[1:], np.abs(current[1:]))
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Absolute layer currents [A]")

    for heatgen in soln.heatgens:
        ax4.plot(soln.t, heatgen)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Layer heat generation [W]")

    for temp in soln.temperatures:
        ax5.plot(soln.t, temp)
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Layer temperature [oC]")

    ax6.plot(
        soln.t,
        convection_rate
        * (soln.temperatures[-1] - external_temperature)
        * 0.0425
        * 0.142,
    )
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Convective heat loss [W]")

    plt.show()


if __name__ == "__main__":
    main()
