#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import ecm1d


"""
Thermal data from https://www.sciencedirect.com/science/article/pii/S1359431122005233?via%3Dihub
"""

# TODO work out what thermal coeffs to use.
# Currently, it's D * area

"""
TODO get validating!

Run inhomogeneity paper procedures, compare to data and PyECN
Run on Kokam Validation procedure
Try on Jasper's LG M50 parameters
[Try new Kokam parameters]
"""
# 


def currentdraw(ts: float | np.ndarray) -> float | np.ndarray:
    if ts == 0:
        return 0
    if ts < 1500:
        return -10
    return 10


def main():
    nlayer = 10
    parameters = ecm1d.KokamParameters(nlayer, "linear")
    battery = ecm1d.ECM(parameters)

    external_temperature = 10
    convection_rate = 1e4
    soln = battery.run(
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

    _, [(ax, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3)

    ax.plot(soln.t, soln.v)
    ax.set_ylabel("Terminal voltage [V]")

    final_tempdist = soln.temperatures[:, -1]
    ax2.plot(np.linspace(0, 1, final_tempdist.size), final_tempdist)
    ax2.set_xlabel("Nondimensional position")
    ax2.set_ylabel("Temperature [degC]")

    for soc in soln.socs:
        ax3.plot(soln.t, soc)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Layer socs")

    for heatgen in soln.heatgens:
        ax4.plot(soln.t, heatgen)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Layer heat generation [W]")

    for temp in soln.temperatures:
        ax5.plot(soln.t, temp)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Layer temperature [oC]")

    ax6.plot(soln.t, convection_rate * (soln.temperatures[-1] - external_temperature))
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Convective heat loss")

    plt.show()




if __name__ == "__main__":
    main()
