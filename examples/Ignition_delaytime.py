# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:37:42 2020

@author: Zerace
"""
import cantera as ct
import numpy as np
import csv 
import sys
import matplotlib.pylab as plt 
from time import perf_counter

#Inputfile_name='CH4_MP1'
Inputfile_name='chem'
gas1 = ct.Solution(Inputfile_name+'.yaml')   #mechanism input file setting

# temperature range setting
Tmin = 900                #unit k              
Tmax = 1200                #unit k    
npoints = 8                #every 50k for one point           
P = ct.one_atm * 20        #constant presure 

  
#time integration parameter
nt = 100000             
dt = 1.e-6              


Tem = np.zeros(npoints,'d')    # initial temperature state series
tim = np.zeros(nt,'d')         # time series
temp_res = np.zeros(nt,'d')    # temperature during reactions
dTdt = np.zeros(nt-1,'d')      # dT/dt [gradient for discrete value] 


Autoignition = np.zeros(npoints,'d')     #ignition time record for each temp states 
FinalTemp_res = np.zeros(npoints,'d')    # final temperature of reactions for each temp states
mfrac_res = np.zeros([npoints,gas1.n_species],'d') #species mass fraction(Y)
#for each initial temperature state

############################################
t0_start = perf_counter()
############################################

for j in range(npoints):    #for loop every intial temperature state
    Tem[j] = Tmin + (Tmax - Tmin)*j/(npoints - 1)
    
    #gas1.TPX = Tem[j], P, 'CH4:0.5,O2:1,N2:3.76'   #phi=1
    gas1.TPX = Tem[j], P, 'IC8H18:1.0,O2:12.5,N2:47.00'  #phi=1
    # reactor of constant pressure
    r=ct.IdealGasConstPressureReactor(gas1)
    # simulations of networks of reactors
    sim = ct.ReactorNet([r])
    # initialization
    time = 0.0
   
    for n in range(nt):            #for loop for time integration
        time += dt
        sim.advance(time)
        tim[n] = time
        temp_res[n] = r.T
        mfrac_res[j][:] = r.thermo.Y

        #################################################################
        #determination of the ignition delay time for each initial temp states
        #################################################################
    
    
    
    plt.plot(tim,temp_res, '--', color = 'orange') #target plot
    plt.xlabel(r'Time [s]', fontsize=12)
    plt.ylabel(r'temperature', fontsize=12)
    plt.grid()
    plt.show()
    str1=Inputfile_name+'%d' %j+'T_t.png'
    plt.savefig(str1, bbox_inches='tight')  
    
    
    
    
    
    Dtmax = [0,0.0] #[index_timestep,max(dTdt)]

    for n in range(nt-1): #for loop for every time step
        dTdt[n] = (temp_res[n+1]-temp_res[n])/dt
        if (dTdt[n]>Dtmax[1]):
            Dtmax[0] = n
            Dtmax[1] = dTdt[n]
            #print for time(/ms)
            Autoignition[j] = (tim[Dtmax[0]]+tim[Dtmax[0] + 1])*1000/2.
                        
    FinalTemp_res[j] = temp_res[nt-1]
    print ('For ' +str(Tem[j]) +', Autoignition time = (ms) ' + str(Autoignition[j]))
#################################################################
t_stop = perf_counter()
print('time spent {:.1e} [s]'.format(t_stop - t0_start))
# Save file for later analysis
#################################################################
csv_file = Inputfile_name+'Ignitiontime_T_Y.csv'
with open(csv_file, 'w') as outfile:   #write data to the file
    writer = csv.writer(outfile)
    writer.writerow(['Auto ignition time','Final Temperature'] + gas1.species_names)
    for i in range(npoints):
        writer.writerow(['Initial Temperature = ', Tem[i]])
        writer.writerow([Autoignition[i], FinalTemp_res[i]] + list(mfrac_res[i,:]))
print ('succeeded in writing output to '+csv_file)


#################################################################
# Plot and save results
#################################################################

plt.semilogy(Tem,Autoignition, '^', color = 'orange') #target plot
plt.xlabel(r'Temp [K]', fontsize=12)
plt.ylabel(r'Ignition delaytime [ms]', fontsize=12)
plt.grid()
plt.show()
plt.savefig(Inputfile_name+'.png', bbox_inches='tight')           