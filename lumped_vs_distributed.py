#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import ecm1d

"""
Thermal data from https://www.sciencedirect.com/science/article/pii/S1359431122005233?via%3Dihub
"""

def currentdraw(ts: float | np.ndarray) -> float | np.ndarray:
    if ts == 0:
        return 0
    if ts < 1500:
        return -10
    return 10


def main():
    nlayer = 10
    distributed_parameters = ecm1d.KokamParameters(nlayer, "linear")
    distributed = ecm1d.ECM(distributed_parameters)
    lumped_parameters = ecm1d.KokamParameters(1, "linear")
    lumped = ecm1d.HomogenousECM(lumped_parameters)

    external_temperature = 20
    convection_rate = 1e4

    _, ax = plt.subplots()
    for model, title in zip((lumped, distributed), ["Lumped", "Distributed"]):
        soln = model.run(
            currentdraw,
            [0, convection_rate],
            [external_temperature, external_temperature],
            np.r_[0, 0, 0],
            0.99,
            external_temperature,
            scipy.integrate.BDF,
            atol=1e-6,
            rtol=1e-6,
        )
        ax.plot(soln.t, soln.v, label=title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal voltage [v]")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
