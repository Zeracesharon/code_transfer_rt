# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:58:32 2020

@author: Zerace
"""
import torch 
import numpy as np
import cantera as ct

mech_yaml = '../../data/IC8H18_reduced.yaml'
gas = ct.Solution(mech_yaml)

# Initial condition
P = ct.one_atm * 1
T = 1800
composition = 'IC8H18:0.5,O2:12.5,N2:34.0'

gas.TPX = T, P, composition
print('cantera setting by TPX')
if (gas.Y<0).any():
    print("no clipping for Y",np.where(gas.Y<0))
if (gas.concentrations<0).any():
    print("no clipping for concentration",np.where(gas.concentrations<0))
if (np.sum(gas.Y)!=1):
    print('unnormalization happen for Y')
if (np.sum(gas.X)!=1):
    print('unnormalization happen for X') 
    
    
print('cantera setting by unnormalized way')
Y=np.zeros(gas.n_species)
Y[0]=-0.4
Y[5]=-0.3
Y[7]=0.1
Y[10]=1.4
gas.set_unnormalized_mass_fractions(Y)
if (gas.Y<0).any():
    print("no clipping for Y",np.where(gas.Y<0))
if (gas.concentrations<0).any():
    print("no clipping for concentration",np.where(gas.concentrations<0))
if (gas.forward_rates_of_progress<0).any():
    print('forward rates of progress calculation without clipping',np.where(gas.forward_rates_of_progress<0))
if (np.sum(gas.Y)!=1):
    print('mass fraction unnormalization happen')
if (np.sum(gas.X)!=1):
    print('unnormalization happen for X') 

print('cantera setting by TPY')
gas.Y = Y
if (gas.Y<0).any():
    print("no clipping for Y",np.where(gas.Y<0))
if (gas.concentrations<0).any():
    print("no clipping for concentration",np.where(gas.concentrations<0))
if (np.sum(gas.Y)!=1):
    print('unnormalizations for Y')
if (np.sum(gas.X)!=1):
    print('unnormalization happen for X') 
# C=torch.tensor([[1,2,1e-289],[1e-301,1e-219,1],[2,0,-1e-300]])
# eps=1e-300
# C_c=C+eps
# # C_eps=torch.zeros_like(C).fill_(eps)
# # C_temp=torch.where(C==0,C_eps,C)
# # print(C,C_eps,C_temp)
# c=torch.log(C)
# c_c=torch.log(C_c)
# Ratio=c/c_c
# print(C,C_c,c,c_c,Ratio)
# delta=1e-5
# if Ratio.min()<1-delta or Ratio.max()>1+delta:
#     print('inf nan runs out')
# result=torch.exp(torch.mm(c,C))
# print(result)

# C=torch.tensor([[1,2,1e-289],[1e-301,1e-219,1],[2,0,-1e-300]]).numpy()
# eps=1e-300
# C_c=C+eps
# # C_eps=torch.zeros_like(C).fill_(eps)
# # C_temp=torch.where(C==0,C_eps,C)
# # print(C,C_eps,C_temp)
# c=np.log(C)
# c_c=np.log(C_c)
# Ratio=(c+eps)/(c_c+eps)
# print(C,C_c,c,c_c)
# delta=1e-5
# if Ratio.min()<1-delta or Ratio.max()>1+delta:
#     print('inf nan runs out')
# result=np.exp(np.dot(c,C))
# print('result',result)
# a=np.array([[-1,-2,2,3],[-3,-4,-5,6]])
# b=np.array([[1,2,2,3],[3,4,5,6]])
# if (a<0).any():
#     print('a negtive happen')
# if (b<0).any():
#     print('a negtive happen')
C=[]
C.append(1)
C.append(2)
print(len(C))
