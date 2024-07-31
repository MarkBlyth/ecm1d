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
    ax.set_xlabel("SOC")
    ax.set_ylabel("Temperature [degC]")
    ax.set_zlabel("Synthetic R0 [Ohm]")
    plt.show()

    external_temperature = 10.1
    convection_rate = 1e4
    soln = battery.run(
        currentdraw,
        [0, convection_rate],
        [external_temperature, external_temperature],
        np.array([]),
        0.99,
        external_temperature,
    )

    _, ax = plt.subplots(2,2, figsize=(12, 5))
    ax[0][0].plot(soln.t, soln.v)
    ax[0][0].set_xlabel("Time [s]")
    ax[0][0].set_ylabel("Terminal voltage [V]")

    for current in soln.currents:
        ax[0][1].plot(soln.t[1:], np.abs(current[1:]))
    ax[0][1].set_xlabel("Time [s]")
    ax[0][1].set_ylabel("Absolute layer currents [A]")

    mean_socs = soln.socs.mean(axis=0)
    for soc in soln.socs:
        ax[1][0].plot(soln.t, soc - mean_socs)
    ax[1][0].set_xlabel("Time [s]")
    ax[1][0].set_ylabel("Layer SOC - overall cell SOC")

    homo_ocv = parameters.get_ocv(soln.socs.mean(axis=0))
    for soc in soln.socs:
        ax[1][1].plot(soln.t, parameters.get_ocv(soc) - homo_ocv)
    ax[1][1].set_xlabel("Time [s]")
    ax[1][1].set_ylabel("Layer OCV - homogeneous cell's OCV [V]")

    plt.show()


if __name__ == "__main__":
    main()
