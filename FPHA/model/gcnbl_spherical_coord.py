from cProfile import label
import math
from multiprocessing.sharedctypes import Value
from re import I
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from torch import linalg as LA
import scipy.special

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A

        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        
        return y

class symmetry_module(nn.Module):
    def __init__(self):
        super(symmetry_module, self).__init__()

    def colatitude(self,x):
        eps = 0.0000001
        angle =  torch.atan2(torch.sqrt(x[:, 0]**2+ x[:, 1]**2+eps),x[:, 2]) # sqrt of neg value is a complex number -> NaN

        return angle
    
    def azimuth(self,x):
        eps = 0.0000001
        angle = torch.atan2(x[:, 1]+eps, x[:, 0]+eps)
        
        return angle
        
    def radius(self,x):
        eps = 0.0000001
        p = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 +eps) # magnitude of vector
        
        return p

    def local_coord(self,x):
        _, _, _, V = x.size()
        # new dim: 128 x 3 x 64 x 25 x 25
        x = torch.stack([x] * V, axis=4) - torch.stack([x] * V, axis=3)     
        return x

    def Spherical_harm(self,x,l):
        x_tran = x.detach().numpy()
        result_r = None
        result_i = None
        result_m = None
        L = np.arange(1, l+1,1, dtype=int)
        for l in L:
            M = np.arange(-l, l+1,1, dtype=int)
            for m in M:
                #raise ValueError(m.dtype, x_tran.dtype, l.dtype)
                test = scipy.special.sph_harm(m, l, x_tran[:,2],x_tran[:,1]) # theta: azimuth, phi = colatitude
                #raise ValueError(type(test), test.dtype)
                #test_real = test.real
                #test_imag= test.imag
                test_mag = np.absolute(test)
                #test_real = torch.from_numpy(test_real).unsqueeze(1).float()
                #test_imag = torch.from_numpy(test_imag).unsqueeze(1).float()
                test_mag = torch.from_numpy(test_mag).unsqueeze(1).float()
                #result_r = torch.cat((result_r, test_real),dim =1) if result_r is not None else test_real
                #result_i= torch.cat((result_i, test_imag),dim =1) if result_i is not None else test_imag
                result_m= torch.cat((result_m, test_mag),dim =1) if result_m is not None else test_mag
                # result: torch.Size([128, 21, 64, 25, 25]))
        return result_m#, result_r, result_i


    def forward(self, x, l):
        x = self.local_coord(x) # 128,3,64,8,8
        
        # convert from catesian coordinates to cylindrical
        azimuth = self.azimuth(x) # input [128,3,64,25], output [128, 64, 25]
        longitude = self.colatitude(x)
        radius = self.radius(x)
        x = torch.cat((azimuth.unsqueeze(1), longitude.unsqueeze(1)), dim = 1)

        #x = self.Spherical_harm(x, l) # 128, 8,64,8,8
        #x0 = x.real.float() # take real part of SHT # 128, 8,64,8,8
        #x1 = x.imag.float() # take imaginary part of SHT # 128, 8,64,8,8
        #x = torch.cat((x0,x1), dim = 1) # 128, 16,64,8,8
        
        N,_ , T, V,_ = x.size()
        x = x.permute(0,1,4,2,3).contiguous().view(N,-1,T,V)# 128, 128,64,8
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=45, num_point=21, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0) #create 3 times identity matrix and stack them into 3D array, matching input dims -> when adaptive = TRUE: learnable
        self.num_class = num_class
        self.num_point = num_point
        self.SHT = 21*2
        self.data_bn = nn.BatchNorm1d(num_person * num_point * in_channels)
        
        self.sym = symmetry_module()
        self.l1 = TCN_GCN_unit(self.SHT+in_channels, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        N, C, T, V, M = x.size()

        # Code from original paper (x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) and continue resp.)
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N, M * C * V, T)
        # order is now N,(M,V,C),T -> print(x.shape) -> 64, 150, 64
        x = self.data_bn(x)
        #print(x.shape) -> shape stays the same
        x = x.view(N, M, C, V, T).permute(0, 1, 2, 4, 3).contiguous().view(N * M, C, T, V)
        # x is now 4 D: N*M, C, T,V
        # print(x.shape) -> 128, 3, 64, 25

        # All skeletons should be normed, i.e. joint #1 should be on the origine. Not always the case -> corrected 
        x1 = torch.stack([x[:,:,:,1]]*V, dim = 3)
        x = x - x1
        x1 = None

           # send data to symmetry module
        sym = self.sym(x,2) # 128, 128,64,8
        _, C, _,_ = sym.size()
        interim = torch.zeros(N*M,C, T,V) # 128, 128,64,25
        #raise ValueError(interim.shape, sym.shape) 
        for i in range(V):
            interim[:,:,:,i] = sym[:,:,:,i]
            i +=1
        #raise ValueError(x.shape, sym.shape, interim[0,0,0,:])
        #x = torch.cat((x,interim), dim = 1)  # 128, 131,64,25  
        #raise ValueError(torch.min(x), torch.max(x)) 
        x = torch.cat((x,interim), dim = 1)  # 128, 131,64,25  
        #raise ValueError(torch.min(x), torch.max(x))  
        

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
   
        return self.fc(x)
