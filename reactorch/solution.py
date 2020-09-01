#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Weiqi Ji"
__copyright__ = "Copyright 2020, DENG"

__version__ = "0.1"
__email__ = "weiqiji@mit.edu"
__status__ = "Development"

import cantera as ct
import torch
from ruamel.yaml import YAML
from torch import nn

torch.set_default_tensor_type("torch.DoubleTensor") 
#64位浮点数矩阵定义


class Solution(nn.Module): #将nn.module作为一个基类
    from .import_kinetics import set_nasa
    from .import_kinetics import set_reactions
    from .import_kinetics import set_transport

    from .kinetics import forward_rate_constants_func
    from .kinetics import forward_rate_constants_func_vec
    from .kinetics import equilibrium_constants_func
    from .kinetics import reverse_rate_constants_func
    from .kinetics import wdot_func
    from .kinetics import Ydot_func
    from .kinetics import Xdot_func
    from .kinetics import Tdot_func
    from .kinetics import TXdot_func
    from .kinetics import TYdot_func

    from .thermo import cp_mole_func
    from .thermo import cp_mass_func
    from .thermo import enthalpy_mole_func
    from .thermo import enthalpy_mass_func
    from .thermo import entropy_mole_func
    from .thermo import entropy_mass_func

    from .transport import update_transport
    from .transport import viscosities_func
    from .transport import thermal_conductivity_func
    from .transport import binary_diff_coeffs_func

    from .magic_function import C2X, Y2X, Y2C, X2C, X2Y

    def __init__(self, mech_yaml=None, device=None, vectorize=False):
        super(Solution, self).__init__()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # whether the computation of reaction rate of type4 will be vectorized
        self.vectorize = vectorize#非向量化

        self.gas = ct.Solution(mech_yaml)

        self.R = ct.gas_constant #R气体常数

        self.one_atm = torch.Tensor([ct.one_atm]).to(self.device)
        #ct.one_atm是一个常数，其值为标准大气压，创建一个tensor张量,且使其在cpu/gpu中存储
        #其值为tensor([101325])

        self.n_species = self.gas.n_species#气体的物质数目

        self.n_reactions = self.gas.n_reactions#反应数目

        self.uq_A = nn.Parameter(torch.Tensor(self.n_reactions).fill_(1.0).to(self.device))
        #创建一全为1的n_reactions的横向向量，创建一个parameter类，其中包含全为一的张量矩阵

        self.molecular_weights = torch.Tensor([self.gas.molecular_weights]).T.to(self.device)
        #molecular-weights是一个矩阵，类型为numpy.ndarray，将其转换为tensor类型，且tensor.T是
        #转置的意思，即将原来的行向量转化为列向量
        with open(mech_yaml, 'r') as stream:
        #打开名为mech_yaml的文件，'r'只读，作为stream 打开，yaml对文件进行操作和读取
            yaml = YAML()
            #创建yaml对象

            model_yaml = yaml.load(stream)
            #将mech_yaml中的所有东西加载进来给model_yaml

            self.model_yaml = model_yaml

            self.set_nasa() #读取nasa的七个系数值，有高有低

            self.set_reactions()

    def set_pressure(self, P):
        self.P_ref = torch.Tensor([P]).to(self.device)
        #该函数将p作为tensor然后设置在p_ref中，一个数的张量
         
       

    def set_states(self, TPY):
       

    
        self.T = torch.clamp(TPY[:, 0:1], min=200, max=None)
        #将TPY这个张量的第一列元素取出来，最小值为200，若小于200
        #数据改为200

        self.logT = torch.log(self.T)#对每一个元素取对数,返回一个tensor类型
        

        if TPY.shape[1] == self.n_species + 2:
            self.P = TPY[:, 1:2]
            #TPY第二列给P，第三列及以后的值赋给Y,最小值是0       
            # self.Y= torch.clamp(TPY[:, 2:], min=0, max=None)
            self.Y= TPY[:, 2:]
            
            

        if TPY.shape[1] == self.n_species + 1:
            self.P = torch.ones_like(self.T) * self.P_ref.to(self.device)
            #压力=一个与T维度一样的张量*参考压强
            # self.Y= torch.clamp(TPY[:, 1:], min=0, max=None)
            self.Y= TPY[:, 1:]
            #TPY中第二列及以后的值赋给Y,最小值为0
        
        # if (TPY<0).any():
        #     print("reactorch negtive happens",torch.where(TPY<0))
        # self.Y = (self.Y.T / self.Y.sum(dim=1)).T
        #装置后除以转置前行向量的和，最后在转置

        self.mean_molecular_weight = 1 / torch.mm(self.Y, 1 / self.molecular_weights)
        #torch.mm表示两个张量的乘法，平均分子重量
        self.density_mass = self.P / self.R / self.T * self.mean_molecular_weight

        self.Y2X()
        self.Y2C()

        self.cp_mole_func()
        self.cp_mass_func()

        self.enthalpy_mole_func()
        self.enthalpy_mass_func()

        self.entropy_mole_func()
        self.entropy_mass_func()

        # concentration of M in three-body reaction (type 2)
        self.C_M = torch.mm(self.C, self.efficiencies_coeffs)
        #1*n_reaction的矩阵

        self.identity_mat = torch.ones_like(self.C_M)

        # for batch computation
        self.C_M2 = (self.C_M * self.is_three_body +
                     self.identity_mat * (1 - self.is_three_body))
        

        if self.vectorize is True:
            # for type 4
            self.C_M_type4 = torch.mm(self.C, self.efficiencies_coeffs_type4)
            self.forward_rate_constants_func_vec()

        else:

            self.forward_rate_constants_func()

        self.equilibrium_constants_func()
        self.reverse_rate_constants_func()

        self.wdot_func()
        #self.TYdot_func()# set this on when TYdot checking is required for solution_test
    
        
