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
    

#--------------------------------------------------------------------------#
#Toy implementation of EDM loss in IS and in ES
#These are NOT used in paper experiments
#but can be used for sanity checks.

@persistence.persistent_class
class EDMToyLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, samples):
        rnd_normal = torch.randn(samples.shape[0], device=samples.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(samples) * sigma[:, None]
        D_yn = net(samples + n, sigma)
        loss = weight[:, None] * ((D_yn - samples) ** 2)
        return loss

@persistence.persistent_class
class EDMToyESLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2):
        self.P_mean=P_mean
        self.P_std=P_std
        
    def __call__(self, net, samples):
        #sample noise in IS (as in original)
        rnd_normal = torch.randn(samples.shape[0], device=samples.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        n = torch.randn_like(samples) * sigma[:, None]
        D_yn, weight = net(samples + n, sigma)
        loss = (weight * (D_yn - samples))**2 
        return loss 
#--------------------------------------------------------------------------#
        

#----------------------------------------------------------------------------
#Def IFs Losses for HD data and toys

@persistence.persistent_class
class IFsLoss:
    def __init__(self, g=None, W=None, space='ES', t_min=1e-7, t_max=15.01, rho=1., \
                 gamma0=5e-4, device=torch.device('cuda'), t_sampling='uniform'):
        """
        Computes model loss for IFs PreCond in either eigenspace 'ES' or image space 'IS'
        for HD data.
        """
        self.g = g
        self.W = torch.from_numpy(W).type(torch.float32).to(device) 
        self.space = space
        self.t_min=t_min
        self.t_max = t_max
        self.rho=rho
        self.gamma0 = gamma0
        self.device=device
        self.t_sampling=t_sampling
    
    def _get_Sigman(self, ts): 
        exp_term = (torch.ones(self.g.shape[0]).to(self.device) + self.g)[None, :] * ts[:, None] #bs, dim
        sigma_n = self.gamma0*torch.exp(self.rho*exp_term) 
        return sigma_n 
    
    def __call__(self, net, images, labels=None, augment_pipe=None):
        if self.t_sampling == 'uniform':
            #sample ts uniformly from entire interval between tmin-tmax
            ts = torch.rand(images.shape[0], device=images.device) * (self.t_max - self.t_min) + self.t_min
        else: 
            #sample ts from N(12.0, 1.2std) - over critical region for D_x/scores 
            ts = (torch.randn(images.shape[0], device=images.device) * 1.2) + 12.
            
        sigmas = self._get_Sigman(ts)

        #handle data,label augmentation 
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        #sample noise in ES
        n_es = torch.randn_like(y.reshape(y.shape[0], -1)) * torch.sqrt(sigmas) #bs, dim, in ES!! 
        #pass (potentially augmented) samples to ES before adding n_es in
        samples_es = torch.einsum('ij, bjk -> bik', self.W.T, y.reshape(y.shape[0], -1).unsqueeze(-1)).squeeze(-1) #bs, D in ES
        
        
        #feed perturbed samples in ES and reshaped to bs, C, H, W to net 
        #net handles different space options by itself ... 
        D_yn, weight = net((samples_es + n_es).view(images.shape), ts, labels, augment_labels=augment_labels) #bs, D 
        
        if self.space =='ES': 
            #compute loss in ES 
            loss = (weight * (D_yn - samples_es))**2 #this is still bs, dim 
        else: 
            #compute loss in IS 
            loss = torch.einsum('ij, bjk -> bik', self.W.T, (D_yn - y.reshape(y.shape[0], -1)).unsqueeze(-1)).squeeze(-1) #bs, D, in ES 
            loss *= weight #bs, D, in ES 
            loss = torch.einsum('ij, bjk -> bik', self.W, loss.unsqueeze(-1)).squeeze(-1) #bs, D, in IS
            loss = loss**2 
            
        return loss    


@persistence.persistent_class
class IFsToyLoss:
    def __init__(self, g=None, W=None, space='ES', t_min=1e-7, t_max=15.01, rho=1., gamma0=5e-4, device=torch.device('cuda')):
        """
        Computes model loss for IFs PreCond in either eigenspace 'ES' or image space 'IS' for toy data.
        """
        self.g = g
        self.W = torch.from_numpy(W).type(torch.float32).to(device) 
        self.space = space
        self.t_min=t_min
        self.t_max = t_max
        self.rho=rho
        self.gamma0 = gamma0
        self.device=device
    
    def _get_Sigman(self, ts): 
        exp_term = (torch.ones(self.g.shape[0]).to(self.device) + self.g)[None, :] * ts[:, None] #bs, dim
        sigma_n = self.gamma0*torch.exp(self.rho*exp_term) 
        return sigma_n 
    
    def __call__(self, net, samples):
        #sample ts
        ts = torch.rand(samples.shape[0], device=samples.device) * (self.t_max - self.t_min) + self.t_min
        #sample sigmas in ES 
        sigmas = self._get_Sigman(ts)
        #sample noise in ES
        n = torch.randn_like(samples) * torch.sqrt(sigmas) #bs, dim 
        #call net to get D_yn and weight matrix
        #Net method handles IS/ES choice on its own...
        D_yn, weight = net(samples + n, ts)
        
        if self.space =='ES': 
            #compute loss in ES 
            loss = (weight * (D_yn - samples))**2 #this is still bs, dim 
        else: 
            #compute loss in IS 
            samples_is = torch.einsum('ij, bjk -> bik', self.W, samples.unsqueeze(-1)).squeeze(-1) #bs, D
            loss = torch.einsum('ij, bjk -> bik', self.W.T, (D_yn - samples_is).unsqueeze(-1)).squeeze(-1)
            loss *= weight
            loss = torch.einsum('ij, bjk -> bik', self.W, loss.unsqueeze(-1)).squeeze(-1)
            loss = loss**2 
            
        return loss
    
#-------------------------------------------------------
#Def loss for toy IFs schedule CFM case

#Toys -- this still includes eary tests with an IFs flow matching 
@persistence.persistent_class
class IFsCFMToyLoss:
    def __init__(self, 
                 flow_matcher_type, 
                 data_dim, 
                 dims_to_keep, 
                 data_eigs=None, 
                 W=None, 
                 rho=1., 
                 gamma0=5e-4, 
                 ODE_type='scaled', 
                 space='ES', 
                 g_scaling=1.0,
                 t_min=1e-7, 
                 t_max=15.01,
                 sigma=0.1, 
                 apply_precond=False):
        self.t_min=t_min
        self.t_max=t_max
        self.flow_matcher_type = flow_matcher_type
        self.precond = apply_precond
        if flow_matcher_type=='ifs': 
            self.flowmatcher = cfm.IFsConditionalFlowMatcher(rho=rho, gamma0=gamma0, data_dim=data_dim, 
                                                            dims_to_keep=dims_to_keep, ODE_type=ODE_type, \
                                                                space=space, data_eigs=data_eigs, W=W, g_scaling=g_scaling)
        elif flow_matcher_type=='exactot':
            self.flowmatcher = cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else: 
            raise NotImplementedError('Only ifs or exactot fm  supported!')
    
    def _get_Cout(self, ts):
        """
        Computes Cout for a given set of time pts
        """
        sigma_dot = self.flowmatcher.compute_sigma_t_dot(ts)
        cout = (sigma_dot**2)
        
        if self.flowmatcher.ODE_type == 'scaled':
            alpha_dot = self.flowmatcher.get_alpha_dot(ts)
            cout +=  (alpha_dot**2)*self.flowmatcher.data_eigs[None, :]
       
        return torch.reciprocal(torch.sqrt(cout)) 

    def _get_Cin(self, ts):
        """
        Computes Cin for a given set of time pts
        """
        sigma = self.flowmatcher.compute_sigma_t(ts)
        if self.flowmatcher.ODE_type == 'scaled':
            alpha = self.flowmatcher.get_alpha(ts)
            cin = torch.sqrt((sigma**2) + (alpha**2)*self.flowmatcher.data_eigs[None, :])
        else: 
            cin = torch.sqrt((sigma**2) + self.flowmatcher.data_eigs[None, :])
        
        return torch.reciprocal(cin)
            
    def __call__(self, net, x1, x0=None):
        #sample ts
        ts = torch.rand(x1.shape[0], device=x1.device) * (self.t_max - self.t_min) + self.t_min
        
        if self.flow_matcher_type == 'ifs':
            #flip ts to account for different notation
            ts = self.t_max - ts 
        
        #get xt, u_t 
        _, xt, ut = self.flowmatcher.sample_location_and_conditional_flow(x0, x1, t=ts)
        
        
        #apply precond, if desired
        if self.precond:
            #get Cin, Cout
            cout = self._get_Cout(ts)
            cin = self._get_Cin(ts) 
            #apply these while preserving space of inputs/outputs 
            if self.flowmatcher.space == 'ES':
                xt = cin * xt 
                ut = cout * ut 
            else: 
                #construct net_in in IS 
                xt = torch.einsum('ij, bjk -> bik', self.flowmatcher.W.T, xt.unsqueeze(-1)).squeeze(-1) 
                xt *= cin 
                xt = torch.einsum('ij, bjk -> bik', self.flowmatcher.W, xt.unsqueeze(-1)).squeeze(-1) 
                #construct norm ut in IS 
                ut = torch.einsum('ij, bjk -> bik', self.flowmatcher.W.T, ut.unsqueeze(-1)).squeeze(-1) 
                ut *= cout
                ut = torch.einsum('ij, bjk -> bik', self.flowmatcher.W, ut.unsqueeze(-1)).squeeze(-1)            
        
        #get network output 
        vt = net(xt, ts)
        #compute MSE loss 
        loss = (vt - ut)**2 #bs, dim 
        return loss 
    
    
#HD 
#this uses only OT-FM which is what worked in practice for toys
#note that here we can either use ExactOT or Sinkhorn
#also, this assumes both x1, x0 are samples in IS.

@persistence.persistent_class
class IFsCFMLoss:
    def __init__(self, 
                 ot_method, 
                 sigma=0.1, 
                 apply_precond=False):
        
        self.flowmatcher = cfm.FlexibleOptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)
        self.precond = apply_precond
                
    def __call__(self, net, x1, x0, labels=None, augment_pipe=None):
        
        #get augmented imgs, labels (if desired)
        y, augment_labels = augment_pipe(x1) if augment_pipe is not None else (x1, None)

        #get ts, xt, u_t --> note that we use aug x1's (if augmentation != None)
        ts, xt, ut = self.flowmatcher.sample_location_and_conditional_flow(x0, y)
        
        #get network output 
        vt = net(xt, ts, labels, augment_labels=augment_labels) #do ONLY unconditional gen for now... 
        
        #compute MSE loss 
        loss = (vt - ut)**2 #bs, C, H, W  
        
        return loss 

