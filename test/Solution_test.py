# from multiprocessing import Pool
from time import perf_counter

import cantera as ct
import torch

import reactorch as rt
import numpy as np

cpu = torch.device('cpu')

cuda = torch.device('cuda:0')

device = cpu
###################修改输入文件
mech_yaml = '../data/IC8H18_reduced.yaml'
# mech_yaml = '../data/gri30.yaml'
# mech_yaml = '../data/nc7_ver3.1_mech_chem.yaml'
# composition = 'IC8H18:0.5,O2:12.5,N2:34.0'
# composition='CH4:0.5,O2:12,N2:40.0'
composition = 'IC8H18:0.08,O2:11.0,N2:39.0'
# composition='NC7H16:0.5,O2:11.0,N2:40.0'

sol = rt.Solution(mech_yaml=mech_yaml, device=device)#here vectorize seems not change the results

#print(sol.species_names,sol.nasa_low[4,:],sol.nasa_high[4,:])


gas = sol.gas
gas.TPX = 1800, 5 * ct.one_atm, composition


# r = ct.IdealGasReactor(gas)
r=ct.IdealGasConstPressureReactor(gas)

sim = ct.ReactorNet([r])

# time = 0.0
# t_end=10
t_end = 1e-3
idt = 0
states = ct.SolutionArray(gas, extra=['t'])


T0 = gas.T

# print('%10s %10s %10s %14s' % ('t [s]', 'T [K]', 'P [atm]', 'u [J/kg]'))

while sim.time < t_end:

    time=sim.step()
  

    states.append(r.thermo.state, t=time)   
           

    # if r.thermo.T > T0 + 600 and idt < 1e-10:
    #     idt = sim.time
    idt+=1
    # if idt > 1e-10 and sim.time > 4 * idt:
    #     break
    # # print('zerace',time)
# print('%10.3e %10.3f %10.3f %14.6e' % (sim.time,
#                                         r.T,
#                                         r.thermo.P / ct.one_atm,
#                                         r.thermo.u))

# print('idt = {:.2e} [s] number of points {}'.format(idt, states.t.shape[0]))
print('idt = {:.2e} [s] number of points {}'.format(idt, states.t.shape))
# TP = torch.stack((torch.Tensor(states.T), torch.Tensor(states.P)), dim=-1)
T=torch.Tensor(states.T)
P=torch.Tensor(states.P)
TP = torch.stack((T,P), dim=-1)

Y = torch.Tensor(states.Y)
# print(Y,type(Y),Y.size(),TP,TP.size())
TPY = torch.cat([TP, Y], dim=-1).to(device)
# print('zerace',"T",T.size(),"P",P.size(),"TP",TP.size(),'Y',Y.size(),'TPY',TPY.size())
# print('zerace',TPY.shape[1])
t0_start = perf_counter()

sol.set_states(TPY)

t1_stop = perf_counter()


print('sol set_states time spent {:.1e} [s]'.format(t1_stop - t0_start))


reaction_equation = gas.reaction_equations()
species_name=gas.species_names
frates_ct=states.forward_rates_of_progress
rrates_ct=states.reverse_rates_of_progress
net_ct= states.net_rates_of_progress
netrates_ct=states.net_production_rates
kf = states.forward_rate_constants
kc = states.equilibrium_constants
kr = states.reverse_rate_constants
# print('zerace',np.shape(net_ct),np.shape(netrates_ct))

net_rt= sol.qdot.detach().cpu().numpy()
frates_rt=sol.forward_rates_of_progress.detach().cpu().numpy()
rrates_rt=sol.reverse_rates_of_progress.detach().cpu().numpy()
netrates_rt=sol.net_production_rates.detach().cpu().numpy()
kf_rt = sol.forward_rate_constants.detach().cpu().numpy()
kc_rt = sol.equilibrium_constants.detach().cpu().numpy()
kr_rt = sol.reverse_rate_constants.detach().cpu().numpy()

# print("kf_rt negtive happens",np.where(kf_rt<0))
# print("kr_rt negtive happens",np.where(kf_rt<0))
# print(sol.Y.size(),sol.reactant_orders.size())
# print('zerace',gas.n_reactions,gas.n_species)
# print('zerace',np.shape(kf),np.shape(kc),np.shape(kr),np.shape(net_rt),np.shape(netrates_rt))
# zerace1=sol.reactant_orders.detach().cpu().numpy()
zerace2=states.reactant_stoich_coeffs()
# zerace3=sol.product_stoich_coeffs.detach().cpu().numpy()
zerace4=states.product_stoich_coeffs()

C_rt=sol.C.detach().cpu().numpy()
C_ct=states.concentrations
Y_rt=sol.Y.detach().cpu().numpy()
Y_ct=states.Y
negtiveY_ct=np.where(Y_ct<0)
index_species=[]
abnormal_species=[]
abnormal_reaction=[]
# print("negtive happens",np.where(Y_ct<0))

# print("negtive happens",np.where(C_ct<0))
density_rt=sol.density_mass.detach().cpu().numpy()
density_ct=states.density_mass
reverse_reaction=sol.is_reversible
# print('size match',np.shape(density_ct),np.shape(density_rt))

Ydot_rt=sol.Ydot.detach().cpu().numpy()
Tdot_rt=sol.Tdot.detach().cpu().numpy()
mean_W_rt=sol.mean_molecular_weight.detach().cpu().numpy()
mean_W_ct=states.mean_molecular_weight
H_rt=sol.partial_molar_enthalpies.numpy()
H_ct=states.partial_molar_enthalpies
# print('size match',np.shape(mean_W_ct),np.shape(mean_W_rt))
#################################cantera TYdot calculation
rho = states.DP[:][0]
wdot = netrates_ct
temp_entro=np.dot(states.partial_molar_enthalpies, wdot.T)
Tdot_ct = - ( np.diag(temp_entro)/
                  (rho * states.cp_mass))

temp_ydot=wdot * states.molecular_weights
Ydot_ct=np.zeros_like(temp_ydot)
# Tdot_ct=np.zeros((idt,1))
for i in range(idt):

    Ydot_ct[i,:] = temp_ydot[i,:] / rho[i]
    



################################################所需判断数组中是否存在无穷或者Nan
if np.isfinite(net_rt).all()==False:
    print('reactorch net_rt qdot is not finite')
if np.isfinite(frates_rt).all()==False:
    print('reactorch frates_rt forward rates of progress is not finite')
if np.isfinite(rrates_rt).all()==False:
    print('reactorch rrates_rt reverse rates of progress is not finite')
if np.isfinite(netrates_rt).all()==False:
    print('reactorch netrates_rt net production rates is not finite')
if np.isfinite(kf_rt).all()==False:
    print('reactorch kf_rt forward rate constant is not finite')
if np.isfinite(kc_rt).all()==False:
    print('reactorch kc_rt equilibrium constant is not finite')
if np.isfinite(kr_rt).all()==False:
    print('reactorch kr_rt reverse rate constant is not finite')   
if np.isfinite(C_rt).all()==False:
    print('reactorch C_rt concentration is not finite')
if np.isfinite(Y_rt).all()==False:
    print('reactorch Y_rt mass fraction is not finite')
if np.isfinite(density_rt).all()==False:
    print('reactorch mass density is not finite')
if np.isfinite(Ydot_rt).all()==False:
    print('reactorch Ydot is not finite')
if np.isfinite(Tdot_rt).all()==False:
    print('reactorch Tdot is not finite')
if np.isfinite(mean_W_rt).all()==False:
    print('reactorch mean_W_rt is not finite')
if np.isfinite(H_rt).all()==False:
    print('reactorch,  partial_molar_enthalpies is not finite')                       

####################################################################

def check_rates(i):
    global ratio_fr
    eps = 1e-300
    delta = 1e-3
    global counter_abnormal_reactions

    ratio = (kf[:, i] + eps) / (kf_rt[:, i] + eps)

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("forward constants {} {} {:.4e} {:.4e}".format(
            i, reaction_equation[i], ratio.min(), ratio.max()))

    ratio = (kc[:, i] + eps) / (kc_rt[:, i] + eps)

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("equilibrium constants {} {} {:.4e} {:.4e}".format(
            i, reaction_equation[i], ratio.min(), ratio.max()))

    ratio = (kr[:, i] + eps) / (kr_rt[:, i] + eps)

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("reverse constants {} {} {:.4e} {:.4e}".format(
            i, reaction_equation[i], ratio.min(), ratio.max()))
        
        
    ratio = (net_ct[:, i] + eps) / (net_rt[:, i] + eps)  

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        counter_abnormal_reactions+=1
  
        print("net rates of progress {} {:.4e} {:.4e}".format(
            i, ratio.min(), ratio.max()))
        abnormal_reaction.append((i,reaction_equation[i],np.argmin(ratio)))
        
    ratio_fr = (frates_ct[:, i] + eps) / (frates_rt[:, i] + eps)  

    if ratio_fr.min() < 1 - delta or ratio_fr.max() > 1 + delta:
        
        # print(np.argmin(ratio_fr))
        # if frates_ct[np.argmin(ratio_fr),i]!=0: # pass
        #     # print('rt',frates_rt[np.argmin(ratio_fr), i])
        print("forward rates of progress {}{} {:.4e} {:.4e}".format(
            i, reaction_equation[i],ratio_fr.min(), ratio_fr.max()))
        # if frates_ct[np.argmin(ratio_fr),i]==0 and frates_rt[np.argmin(ratio_fr),i]<0:
        #     print('attention forward negtive in rt zero in ct',
        #           i,reaction_equation[i],np.argmin(ratio))
            
            
    ratio = (rrates_ct[:, i] + eps) / (rrates_rt[:, i] + eps)  

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
   
        print("reverse rates of progress{} {} {:.4e} {:.4e}".format(
            i, reaction_equation[i],ratio.min(), ratio.max()))           
    return i

def check_rates_production(j):
    eps = 1e-300
    delta = 1e-3
    global ratio_yd
    index_species.append((j,species_name[j]))
    ratio = (netrates_ct[:, j] + eps) / (netrates_rt[:, j] + eps)  

    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("net rates of production {} {} {:.4e} {:.4e}".format(
            j, species_name[j], ratio.min(), ratio.max()))
        abnormal_species.append((j,species_name[j],np.argmin(ratio),np.argmax(ratio)))
    
    ratio = (C_ct[:, j] + eps) / (C_rt[:, j] + eps) 
    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("concentration {} {} {:.4e} {:.4e}".format(
            j, species_name[j], ratio.min(), ratio.max()))
        
    ratio = (Y_ct[:, j] + eps) / (Y_rt[:, j] + eps) 
    
    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print("mass fraction {} {} {:.4e} {:.4e}".format(
            j, species_name[j], ratio.min(), ratio.max()))
        
    ratio_yd = (Ydot_ct[:, j] + eps) / (Ydot_rt[:, j] + eps) 
    
    if ratio_yd.min() < 1 - delta or ratio_yd.max() > 1 + delta:
        # pass
        print("Ydot {} {} {:.4e} {:.4e}".format(
            j, species_name[j], ratio_yd.min(), ratio_yd.max()))
   
    ratio = (H_ct[:, j] + eps) / (H_rt[:, j] + eps) 
    
    if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
        # pass
        print(" partial_molar_enthalpies {} {} {:.4e} {:.4e}".format(
            j, species_name[j], ratio.min(), ratio.max()))
    return j


    
    
    
counter_abnormal_reactions=0
for i in range(gas.n_reactions):
    check_rates(i)
    if reverse_reaction[i]==0:
        # krr=kr_rt[:,i]
        # krc=kr[:,i]
        krr=np.nonzero(kr_rt[:,i])
        krc=np.nonzero(kr[:,i])
        if np.nonzero(krr)[0].size !=0:
            print('nonzero irreverse constants detected in reactorch')
        if np.nonzero(krc)[0].size !=0:
            print('nonzero irreverse constants detected in reactorch')
        # print('nonzero irreverse constants_reactorch',np.nonzero(krr)[0].size)
        # print('nonzero irreverse constants_cantera',np.nonzero(krc)[0].size)

for j in range(gas.n_species):
    check_rates_production(j)   
    
eps = 1e-300
delta = 1e-3
ratio_density = (density_ct + eps) / (density_rt[:,0] + eps) 
if ratio_density.min() < 1 - delta or ratio_density.max() > 1 + delta:
    # pass
    print("mass density {:.4e} {:.4e}".format(
             ratio_density.min(), ratio_density.max()))
# print('min index value',np.argmin(ratio),ratio[(np.argmin(ratio))])
# print('max index value',np.argmax(ratio),ratio[(np.argmax(ratio))]) 
# print('Tdot shape',np.shape(Tdot_ct),np.shape(Tdot_rt))
ratio_td = (Tdot_ct + eps) / (Tdot_rt[:,0] + eps) 
if ratio_td.min() < 1 - delta or ratio_td.max() > 1 + delta:
    # pass
    print("Tdot {:.4e} {:.4e}".format(
             ratio_td.min(), ratio_td.max()))
    
ratio_mean = (mean_W_ct + eps) / (mean_W_rt[:,0] + eps) 
if ratio_mean.min() < 1 - delta or ratio_mean.max() > 1 + delta:
    # pass
    print("mean W {:.4e} {:.4e}".format(
             ratio_mean.min(), ratio_mean.max()))


t1_stop = perf_counter()
print('sol check_rates time spent {:.1e} [s]'.format(t1_stop - t0_start))
print('abnormal reactions detected',counter_abnormal_reactions)