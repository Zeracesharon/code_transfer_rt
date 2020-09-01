"""
Constant-pressure, adiabatic kinetics simulation.

Requires: cantera >= 2.5.0, matplotlib >= 2.0
"""

import sys

import cantera as ct
from time import perf_counter

t0_start = perf_counter()
gas = ct.Solution('ic8_ver3_mech.yaml')

gas.TPX = 1200.0, ct.one_atm*20, 'IC8H18:1.0,O2:12.5,N2:47.00'
r = ct.IdealGasConstPressureReactor(gas)

sim = ct.ReactorNet([r])
sim.verbose = True

# limit advance when temperature difference is exceeded
#delta_T_max = 20.
#r.set_advance_limit('temperature', delta_T_max)

dt_max = 1.e-6
t_end = 100000 * dt_max
states = ct.SolutionArray(gas, extra=['t'])

print('{:10s} {:10s} {:10s} {:14s}'.format(
    't [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
while sim.time < t_end:
    sim.advance(sim.time + dt_max)
    states.append(r.thermo.state, t=sim.time*1e3)
    print('{:10.3e} {:10.3f} {:10.3f} {:14.6f}'.format(
            sim.time, r.T, r.thermo.P, r.thermo.u))
    
t_stop = perf_counter()
print('time spent {:.1e} [s]'.format(t_stop - t0_start))
# Plot the results if matplotlib is installed.
# See http://matplotlib.org/ to get it.
if '--plot' in sys.argv[1:]:
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(states.t, states.T)
    plt.xlabel('Time (ms)')
    plt.ylabel('Temperature (K)')
    plt.subplot(2, 2, 2)
    plt.plot(states.t, states.X[:, gas.species_index('OH')])
    plt.xlabel('Time (ms)')
    plt.ylabel('OH Mole Fraction')
    plt.subplot(2, 2, 3)
    plt.plot(states.t, states.X[:, gas.species_index('H')])
    plt.xlabel('Time (ms)')
    plt.ylabel('H Mole Fraction')
    plt.subplot(2, 2, 4)
    plt.plot(states.t, states.X[:, gas.species_index('H2')])
    plt.xlabel('Time (ms)')
    plt.ylabel('H2 Mole Fraction')
    plt.tight_layout()
    plt.show()
else:
    print("To view a plot of these results, run this script with the option --plot")
