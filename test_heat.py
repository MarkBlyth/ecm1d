#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import ecm

# TODO use analytical solution to automatically test!

def main():
    mol = ecm.HeatEquationMOL(1, [0,10], [1,1])
    ic = np.zeros(20)

    def ode_rhs(_, state):
        return mol.get_ode_rhs(state, np.ones(state.size))

    soln = scipy.integrate.solve_ivp(ode_rhs, [0, 5], ic)

    _, ax = plt.subplots(2)
    for row in soln.y:
        ax[0].plot(soln.t, row)
    ax[1].plot(np.linspace(0, 1, ic.size), soln.y[:,-1])
    plt.show()


if __name__ == '__main__':
    main()
