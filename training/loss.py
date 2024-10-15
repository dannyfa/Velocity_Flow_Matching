# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Loss functions used. 

Containts both original loss fns (from EDM paper)
and inflationary flow loss fns (for toy and image data).
"""

import torch
from torch_utils import persistence
from torch_cfm import conditional_flow_matching as cfm
import numpy as np 


#####################################################################
#Legacy code from EDM repo
#Will be removed eventually

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
    


#############################################################################

#--------------------------------------------------------
# Loss fn for VFM models         
@persistence.persistent_class
class VFMToyLoss:
    def __init__(self,
                 flow_matcher_type='exactot', 
                 sigma=0.1, 
                 cd_eps=1e-7):
        
        #get flow matcher for u 
        assert flow_matcher_type in ['exactot', 'regular']
        
        self.flow_matcher_type = flow_matcher_type
        
        if flow_matcher_type=='regular': 
            self.tau_flowmatcher = cfm.ConditionalFlowMatcher(sigma=sigma)
            
        elif flow_matcher_type=='exactot':
            self.tau_flowmatcher = cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else: 
            raise NotImplementedError('Only ifs or exactot fm  supported!')
        
        #get flow matcher for v 
        #this is always just regular flow matcher! 
        self.t_flowmatcher = cfm.ConditionalFlowMatcher(sigma=sigma)
        
        self.eps = cd_eps
    
    def calc_lie_derivative(self, net, x0_tau, xt_tau, taus, u, v):
        #get Jacobian for all net outputs w.r.t. inputs! 
        taus.requires_grad=True
        net_jac = torch.autograd.functional.jacobian(net, (x0_tau, xt_tau, taus))
        
        #get \nabla u 
        nabla_u = torch.sum(net_jac[0][0], dim=2).transpose(2,1) #bs, d, d
        
        #get \nabla v, \partial_tau v 
        nabla_v = torch.sum(net_jac[1][1], dim=2).transpose(2,1) #bs, d, d
        partial_tau_v = torch.sum(net_jac[1][2], dim=2) #bs, d
        
        
        # calc whole Lie derivative 
        # \partial_tau_v + u \cdot \nabla_v - v \cdot \nabla_u 
        lie_derivative = partial_tau_v #bs, dim
        lie_derivative += torch.einsum('bij, bjk -> bik', u.unsqueeze(1), nabla_v).squeeze(1) #bs, dim
        lie_derivative -= torch.einsum('bij, bjk -> bik', v.unsqueeze(1), nabla_u).squeeze(1) #bs, dim 
        return lie_derivative
        
                
    def __call__(self, net, x0_1, xdt_1, dt):
        
        #get x0_0, xdt_0
        x0_0 = net.module.get_x0s(x0_1)
        xdt_0 = net.module.get_x0s(xdt_1)
        
        if net.module.dims_to_keep < net.module.data_dim:
            #this is a PRR case! 
            #sample last dim 
            x0_0_cd_sample = torch.randn(x0_1.shape[0], (net.module.data_dim - net.module.dims_to_keep)).to(x0_1.device) * np.sqrt(self.eps)
            xdt_0_cd_sample = torch.randn(xdt_1.shape[0], (net.module.data_dim - net.module.dims_to_keep)).to(xdt_1.device) * np.sqrt(self.eps)
            x0_0_fullsample = torch.cat([x0_0, x0_0_cd_sample], dim=-1)
            xdt_0_fullsample = torch.cat([xdt_0, xdt_0_cd_sample], dim=-1)
            
        else:
            #prp case
            #ok to keep encoder outputs as is 
            x0_0_fullsample = x0_0
            xdt_0_fullsample = xdt_0 
       
        #sample taus
        taus = torch.rand(x0_0_fullsample.shape[0], device=x0_0_fullsample.device) # \tau \sim U[0,1]
        #sample ts 
        ts = torch.rand(xdt_0_fullsample.shape[0], device=xdt_0_fullsample.device) * dt #\t \sim U[0, dt]
        
        
        #get x0_tau, u0_tau
        _, x0_tau, u0_tau = self.tau_flowmatcher.sample_location_and_conditional_flow(x0_0_fullsample, x0_1, t=taus)
        _, xdt_tau, _ = self.tau_flowmatcher.sample_location_and_conditional_flow(xdt_0_fullsample, xdt_1, t=taus)
        
        
        #get xt_tau, ut_tau 
        _, xt_tau, ut_tau = self.t_flowmatcher.sample_location_and_conditional_flow(x0_tau, xdt_tau, t=(ts/dt))
                
        #now get net estimates for u,v ... 
        u, v = net(x0_tau, xt_tau, taus)
        
        #calc different loss pieces 
        flow_loss = (u - u0_tau)**2 #bs, dim
        dyn_loss = (v - ut_tau) **2 #bs, dim
        
        #lie derivative 
        lie_loss = (self.calc_lie_derivative(net, x0_tau, xt_tau, taus, u, v))**2 #bs, dim
        
        return torch.cat([flow_loss.unsqueeze(0), dyn_loss.unsqueeze(0), lie_loss.unsqueeze(0)], dim=0) # 3, bs, dim  


