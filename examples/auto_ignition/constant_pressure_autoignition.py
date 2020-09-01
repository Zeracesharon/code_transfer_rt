"""
Solve a constant pressure ignition problem where the governing equations are
implemented in Python.

This demonstrates an approach for solving problems where Cantera's reactor
network model cannot be configured to describe the system in question. Here,
Cantera is used for evaluating thermodynamic properties and kinetic rates while
an external ODE solver is used to integrate the resulting equations. In this
case, the SciPy wrapper for VODE is used, which uses the same variable-order BDF
methods as the Sundials CVODES solver used by Cantera.
"""

# TODO: the reactorch class seems to be very slow here, will figure out later

import cantera as ct
import numpy as np
import reactorch as rt
from scipy.integrate import solve_ivp
import torch
from torch.autograd.functional import jacobian as jacobian
from time import perf_counter

device=torch.device('cuda:0')
torch.set_default_tensor_type('torch.DoubleTensor')
class ReactorOde(object):
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P
       
        self.counter=0
        

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """
        # print('cantera',np.shape(y))
        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.counter+=1
       # y=np.clip(y,0,np.max(y))
        # y1=np.zeros_like(y)
        # y=np.where(y<0,y1,y)
        # print("cantera negtive happens",np.where(y<0))

        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP=y[0], self.P
        #self.gas.TPY = y[0], self.P,y[1:]
        #if (self.gas.concentrations<0).any():
         #   print("cantera concentration happens",np.where(self.gas.Y<0))
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho
        
        return np.hstack((dTdt, dYdt))


class ReactorOdeRT(object):
    def __init__(self, sol):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.sol = sol
        self.gas = sol.gas
        self.counter=0
  

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """
        # print('reactorch',np.shape(y))
        # print("reactorch negtive happens",np.where(y<0))
        self.counter+=1
        TPY = torch.Tensor(y).T.to(device)
        # if (TPY<0).any():
        #    print("reactorch negtive happens",torch.where(TPY<0))
        # print(TPY.size())
        with torch.no_grad():

            self.sol.set_states(TPY)

            TYdot = self.sol.TYdot_func()
            # TYdot_zero=torch.zeros_like(TYdot)
            # TYdot=torch.where(abs(TYdot)>1e-290,TYdot,TYdot_zero)

        # return TYdot.T.detach().cpu().numpy()
        return TYdot.T.cpu().numpy()



    # def __call__(self, t, y):
    #     """the ODE function, y' = f(t,y) """
    #     # print(np.size(y))
    #     self.counter+=1
    #     T=torch.tensor([[y[0]]])
       
    #     Y=torch.tensor([y[1:]])
        
    #     TPY = torch.cat((T, Y), dim=1)
    #     # print(TPY.size())

    #     with torch.no_grad():

    #         self.sol.set_states(TPY)
    #         TYdot = self.sol.TYdot_func()
    #         # TYdot_zero=torch.zeros_like(TYdot)
    #         # TYdot=torch.where(abs(TYdot)>1e-20,TYdot,TYdot_zero)
    #         # self.sol.Tdot_func()
    #         # self.sol.Ydot_func()            
    #         # TYdot = torch.cat((self.sol.Tdot, self.sol.Ydot), dim=1)
    #         TYdot=TYdot.squeeze()           
    #     return TYdot.T.detach().cpu().numpy()

    def TYdot_jac(self, TPY):

        TPY = torch.Tensor(TPY).unsqueeze(0)

        sol.set_states(TPY)

        return sol.TYdot_func().squeeze(0)

    def jac(self, t, y):

        TPY = torch.Tensor(y)

        TPY.requires_grad = True

        jac_ = jacobian(self.TYdot_jac, TPY, create_graph=False)
        # print(jac_,np.size(jac_))


        return jac_

################################modified input
t0_start = perf_counter()

mech_yaml = '../../data/IC8H18_reduced.yaml'
# mech_yaml = '../../data/nc7_ver3.1_mech_chem.yaml'
#mech_yaml = '../../data/gri30.yaml'

sol = rt.Solution(mech_yaml=mech_yaml, device=device,vectorize=True)

gas = ct.Solution(mech_yaml)



# Initial condition
P = ct.one_atm * 1
T = 1800
composition = 'IC8H18:0.5,O2:12.5,N2:34.0'
#composition = 'IC8H18:0.08,O2:1.0,N2:3.76'
#composition = 'CH4:0.5,O2:11,N2:40'
# composition='NC7H16:0.5,O2:11.0,N2:40.0'
# composition='NC7H16:0.4,O2:11.0,N2:41.36'
gas.TPX = T, P, composition
if (gas.Y<0).any():
    print("cantera before clipping",np.where(gas.Y<0))
if (np.sum(gas.Y)!=1):
    print('unnormalization happen')
gas.Y=np.clip(gas.Y,0,np.max(gas.Y))

y0 = np.hstack((gas.T, gas.Y))


# Set up objects representing the ODE and the solver
ode = ReactorOde(gas)

# Ydot_cantera=[]
# Tdot_cantera=[]
# TYdot_reactorch=[]
######################################################
# ode.gas.TPY = y0[0], ode.P,y0[1:]
# rho = ode.gas.density

# wdot = ode.gas.net_production_rates
# dTdt = - (np.dot(ode.gas.partial_molar_enthalpies, wdot) /
#                   (rho * ode.gas.cp))
# dYdt = wdot * ode.gas.molecular_weights / rho
# Ydot_cantera.append([dYdt.T])
# Tdot_cantera.append([dTdt.T])
# print('cantera',dTdt,dYdt)
##################################################################

# Integrate the equations using Cantera
t_end = 1e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-5
t = 0
ode_success = True
y = y0
# counter1=0

sol.gas.TPX = T, P, composition
sol.set_pressure(sol.gas.P)
ode_rt = ReactorOdeRT(sol=sol)
states_rt = ct.SolutionArray(sol.gas, 1, extra={'t': [0.0]})

njev=np.zeros((int(t_end/dt)+1,2),dtype=int)
nfev=np.zeros((int(t_end/dt)+1,2),dtype=int)
nlu=np.zeros((int(t_end/dt)+1,2),dtype=int)

###################################################################
# T=torch.tensor([[y0[0]]])
       
# Y=torch.tensor([y0[1:]])
        
# TPY = torch.cat((T, Y), dim=1)

# with torch.no_grad():

#     ode_rt.sol.set_states(TPY)

#     TYdot = ode_rt.sol.TYdot_func()
# TYdot=TYdot.T.numpy()
# TYdot_reactorch.append([TYdot])
# # print('reactorch',TYdot)
#####################################################
i,j=0,0
t1 = perf_counter()
print('before integration','time spent {:.1e} [s]'.format(t1 - t0_start))
while ode_success and t < t_end:
    odesol = solve_ivp(ode,
                       t_span=(t, t + dt),
                       y0=y,
                       method='BDF',
                       vectorized=False, jac=None)

    t = odesol.t[-1]
    y = odesol.y[:, -1]
    # y=np.clip(y,0,np.max(y))
    # print('t {:.2e} [s]'.format(t),'nfev',odesol.nfev,'njev',odesol.njev,'nlu',odesol.nlu)
    # print('counter',ode.counter)
    nfev[i,0]=odesol.nfev
    njev[i,0]=odesol.njev
    nlu[i,0]=odesol.nlu
    i+=1
    #print('t {} T {}'.format(t, y[0]))
    ###########################################################
   
    # ode.gas.TPY = y[0], ode.P,y[1:]
    # rho = ode.gas.density

    # wdot = ode.gas.net_production_rates
    # dTdt = - (np.dot(ode.gas.partial_molar_enthalpies, wdot) /
    #               (rho * ode.gas.cp))
    # dYdt = wdot * ode.gas.molecular_weights / rho
    # Ydot_cantera.append([dYdt.T])
    # Tdot_cantera.append([dTdt.T])
    # print('cantera',dTdt,dYdt)
##################################################################
    ode_successful = odesol.success

    gas.TPY = odesol.y[0, -1], P, odesol.y[1:, -1]
    states.append(gas.state, t=t)
    # if counter1 <1:
    #     sol.gas.TPY = gas.T, P, gas.Y
    #     states_rt.append(sol.gas.state, t=t)
    #     counter1=counter1+1



t_stop1 = perf_counter()
print('finish cantera integration')
print('time spent {:.1e} [s]'.format(t_stop1 - t1))


# # Integrate the equations using ReacTorch

t = 0*dt
ode_success = True
#y0 = np.hstack((states_rt[1].T,states_rt[1].Y))
y = y0

# Diable AD for jacobian seems more effient for this case.
print('reacotrch')
while ode_success and t < t_end:
    odesol = solve_ivp(ode_rt,
                        t_span=(t, t + dt),
                        y0=y,
                        method='BDF',
                        vectorized=True, jac=None)#ode_rt.jac

    t = odesol.t[-1]
    y = odesol.y[:, -1]
    # y=np.clip(y,0,np.max(y))
    ode_successful = odesol.success
    # print('reactorch',np.shape(y))
    ####################################################################
    # T=torch.tensor([[y[0]]])
           
    # Y=torch.tensor([y[1:]])
        
    # TPY = torch.cat((T, Y), dim=1)

    # with torch.no_grad():

    #     ode_rt.sol.set_states(TPY)

    #     TYdot = ode_rt.sol.TYdot_func()

        
    # TYdot=TYdot.T.numpy()
    # TYdot_reactorch.append([TYdot])
    # # print('reactorch',TYdot)    
#####################################################

    #print('t {} T {}'.format(t, y[0]))
    
    # print('t {:.2e} [s]'.format(t),'nfev',odesol.nfev,'njev',odesol.njev,'nlu',odesol.nlu)
    # print('counter',ode_rt.counter)
    nfev[j,1]=odesol.nfev
    njev[j,1]=odesol.njev
    nlu[j,1]=odesol.nlu
    j+=1
    # ode_rt.counter=0
    sol.gas.TPY = odesol.y[0, -1], P, odesol.y[1:, -1]
    states_rt.append(sol.gas.state, t=t)
    
t_stop = perf_counter()
print('time spent {:.1e} [s]'.format(t_stop - t_stop1))
###########################################################
# eps = 1e-300
# delta = 1e-4
# ratiot=np.ones(j)
# for i in range(j):
#     TYdot_rt=np.array(TYdot_reactorch[i])
#     Tdot_ct=np.array(Tdot_cantera[i])
#     Ydot_ct=np.array(Ydot_cantera[i])
#     ratiot[i] = (TYdot_rt[0,0] + eps) / (Tdot_ct[0] + eps) 
    

#     ratio = (TYdot_rt[0,1:] + eps) / (Ydot_ct[:] + eps)
#     if ratio.min() < 1 - delta or ratio.max() > 1 + delta:
#     # pass
#         print("Ydot {:.4e} {:.4e}".format(
#               ratio.min(), ratio.max()))
        
# if ratiot.min()< 1 - delta or ratiot.max()> 1 + delta:
#     # pass
#         print("Tdot",ratiot.min(),ratiot.max())
    
# if np.isfinite(np.array(TYdot_reactorch)).all()==False:
#     print('reactorch TYdot_rt qdot is not finite')

#Plot the results
try:
    import matplotlib.pyplot as plt
    L1 = plt.plot(states.t, states.T, ls='--',
                  color='r', label='T Cantera', lw=1)
    L1_rt = plt.plot(states_rt.t, states_rt.T, ls='-',
                      color='r', label='T ReacTorch', lw=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')

    plt.twinx()
    L2 = plt.plot(states.t, states('OH').Y, ls='--', label='OH Cantera', lw=1)
    L2_rt = plt.plot(states_rt.t, states_rt('OH').Y,
                      ls='-',
                      label='OH ReacTorch',
                      lw=1)
    plt.ylabel('Mass Fraction')

    plt.legend(L1+L2+L1_rt+L2_rt, [line.get_label()
                                    for line in L1+L2+L1_rt+L2_rt], loc='best')

    plt.savefig('cantera_reactorch_validation.png', dpi=300)
    plt.show()
except ImportError:
    print('Matplotlib not found. Unable to plot results.')
len_n=np.shape(states.t)
njevp=njev[0:len_n,:]
nfevp=nfev[0:len_n,:]
nlup=nlu[0:len_n,:]
plt.subplot(311)
plt.plot(states.t,njevp[:,0],ls='-.',label='njev_ct')
plt.plot(states.t,njevp[:,1],ls='-',label='njev_rt')
plt.legend()
plt.subplot(312)
plt.plot(states.t,nfevp[:,0],ls='-.',label='nfev_ct')
plt.plot(states.t,nfevp[:,1],ls='-',label='nfev_rt')
plt.legend()
plt.subplot(313)
plt.plot(states.t,nlup[:,0],ls='-.',label='nlu_ct')
plt.plot(states.t,nlup[:,1],ls='-',label='nlu_rt')
plt.legend()
plt.savefig('njev.png', dpi=300)
plt.show()
