#!/usr/bin/env python3

import itertools
import numpy as np
import ecm
import lgm50lut
import matplotlib.pyplot as plt


def main():
    lut = lgm50lut.LGM50LUT()
    # TAKEN FROM KOKAM:
    heat_equation = ecm.HeatEquationMOL(
        0.01, [1e3, 1e3], [25, 25],
    )
    battery = ecm.ECMStack(lut, heat_equation, 8, 5)

    currents = np.r_[0, list(itertools.chain.from_iterable([[5]*300 + [0]*300]*20))]
    ts = np.r_[0, np.linspace(0, currents.size-1, currents.size-1)]

    results = battery.run(ts, currents, np.r_[0, 0], 0.1, 25, True, atol=1e-3, rtol=1e-3, method="BDF")

    _, (ax, ax2, ax3) = plt.subplots(3)

    ax.plot(results.ts, results.terminal_voltage)
    ax.set_ylabel("Terminal voltage [V]")

    for data in results.layer_currents:
        ax2.plot(results.ts, data)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Layer currents")

    final_tempdist = results.layer_temperatures[:, -1]
    ax3.plot(np.linspace(0, 1, final_tempdist.size), final_tempdist, marker="o")
    ax3.set_xlabel("Nondimensional position")
    ax3.set_ylabel("Temperature [degC]")

    plt.show()


if __name__ == "__main__":
    main()
