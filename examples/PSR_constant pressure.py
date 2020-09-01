"""
Created on Mon Nov 14 14:11:59 2016

Reactor PSR
"""
###############################################################################
#import 
###############################################################################
import cantera as ct
import numpy as np
import csv 
import sys
import matplotlib.pylab as plt 

###############################################################################
#Début du programme
###############################################################################
# Création des différents objets gas associés à ces mécanismes
gas1 = ct.Solution('GRI30.cti')                     # Mécanisme détaillé

# Set initial states
Pini=ct.one_atm
Tini=1200.0
gas1.TPX = Tini, Pini, 'CH4:0.5,O2:1,N2:3.76'

# Set reactor
r=ct.IdealGasConstPressureReactor(gas1)
sim = ct.ReactorNet([r])

# Set timestep 
time = 0.0
nt = 100000             # total timesteps
dt = 1.e-6              
tim = np.zeros(nt,'d')
temp_cas = np.zeros(nt,'d')
mfrac_cas = np.zeros([nt,gas1.n_species],'d')

for n in range(nt):
    time += dt
    sim.advance(time)
    tim[n] = time
    temp_cas[n] = r.T
    mfrac_cas[n,:] = r.thermo.Y

#################################################################
# Sauvergarde des résultats dans fichier csv
#################################################################
csv_file = 'T-Y_Profile.csv'
with open(csv_file, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['time','Temperature'] + gas1.species_names)
    for i in range(nt):
        writer.writerow([tim[i], temp_cas[i]] + list(mfrac_cas[i,:]))
print ('output written to '+csv_file)


#################################################################
# Plot results
#################################################################
# create plot
plt.subplot(2,2,1)
plt.plot(tim,temp_cas)
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('Temperature (K)', fontsize=13)
plt.subplot(2,2,2)
plt.plot(tim,mfrac_cas[:,gas1.species_index('OH')])
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('OH Mass Fraction', fontsize=13)
plt.show()