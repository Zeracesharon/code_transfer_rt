"""
Created on Mon Nov 14 14:11:59 2016

TD n°3: Réacteur PSR
Exercice 1: Premiers calculs d'auto-alluamge avec le methane

Solution pour itération sur la température


Cours Combustion 2
Département Energétique et Propulsion
INSA Rouen-Normandie
Année scolaire 2016-2017


Author: Bruno Renou
"""

###############################################
# Exercice
###############################################
# Soit un mélange méthane/air.
# On cherche à déterminer le délai d'auto-inflammation avec la température
# Mélange stoechiométrique

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
gas1 = ct.Solution('CH4_MP1.yaml')                     # Mécanisme détaillé

# On represente le delai en fonction de 1000/T avec T en Kelvin
# On rentre ici les bornes de 1000/T (min et max)
Tmin = 0.9091             # 1000/T min     
Tmax = 0.6667             # 1000/T max
npoints = 8            # Nombre de valeur de 1000/T
Pini=101325*20

#Valeurs des paramètres d'intégration
nt = 100000             # Nombre de points
dt = 1.e-6              # Pas de temps en s

# Enregistrement des données de températures
Ti = np.zeros(npoints,'d')     # Valeurs de température  (K)
Ti2 = np.zeros(npoints,'d')    # Valeurs de 1000/T

# Enregistrement des variables qui seront écrasés par chaque température 
tim = np.zeros(nt,'d')
temp_cas = np.zeros(nt,'d')
dtemp_cas = np.zeros(nt-1,'d')

#AEnregistrement des données finales pour chaque température ou cas
Autoignition_cas = np.zeros(npoints,'d')        # 
FinalTemp_cas = np.zeros(npoints,'d')
mfrac_cas = np.zeros([npoints,gas1.n_species],'d')

############################################
# Début de la boucle sur les températures
############################################

for j in range(npoints):
    Ti2[j] = Tmin + (Tmax - Tmin)*j/(npoints - 1)
    Ti[j] = 1000/Ti2[j]
    #A ssignation des propriétés du gas. 1 atm, phi=1
    gas1.TPX = Ti[j], Pini, 'CH4:0.5,O2:1,N2:3.76'
    # Création du réacteur à Pression constante et T fixée
    r=ct.IdealGasConstPressureReactor(gas1)
    # Définition de l'intégrateur de la classe cantera (préfixe ct.) associé au réacteur r
    # Préparation du calcul
    sim = ct.ReactorNet([r])
    # Temps initial de la solution
    time = 0.0
    # Boucle sur tous les pas de temps dt
    # print 'time [s] , T [K] , p [Pa] , u [J/kg]'
    for n in range(nt):
        time += dt
        sim.advance(time)
        tim[n] = time
        temp_cas[n] = r.T
        mfrac_cas[j][:] = r.thermo.Y

        #################################################################
        # Determination du délai d'auto-inflammation
        #################################################################

    Dtmax = [0,0.0]
    Autoignition = 0
    for n in range(nt-1):
        dtemp_cas[n] = (temp_cas[n+1]-temp_cas[n])/dt
        if (dtemp_cas[n]>Dtmax[1]):
            Dtmax[0] = n
            Dtmax[1] = dtemp_cas[n]
            # Local print
            Autoignition = (tim[Dtmax[0]]+tim[Dtmax[0] + 1])/2.
            # Posterity
            Autoignition_cas[j] = Autoignition*1000 #ms
            FinalTemp_cas[j] = temp_cas[nt-1]
    print ('For ' +str(Ti[j]) +', Autoignition time = (ms) ' + str(Autoignition*1000))
#################################################################
# Sauvergarde des résultats dans fichier csv
#################################################################
csv_file = 'Phi-1_P-1_Trange.csv'
with open(csv_file, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Auto ignition time','Final Temperature'] + gas1.species_names)
    for i in range(npoints):
        writer.writerow(['cas Ti = ', Ti[i]])
        writer.writerow([Autoignition_cas[i], FinalTemp_cas[i]] + list(mfrac_cas[i,:]))
print ('output written to '+csv_file)


#################################################################
# Plot results
#################################################################
# create plot
plt.semilogy(Ti2,Autoignition_cas, '^', color = 'orange')
plt.xlabel(r'Temp [1000/K]', fontsize=13)
plt.ylabel(r'Autoignition delay [ms]', fontsize=13)
#plt.axis('equal')
plt.grid()
plt.show()
plt.savefig('Phi-1_P-1_Trange_UV.png', bbox_inches='tight')