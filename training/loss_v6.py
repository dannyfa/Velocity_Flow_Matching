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


def g_only(old, new):
    return (old - new).detach() + new

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
    def __init__(self, flow_matcher_type='exactot',
                 sigma_dynamics=0.1, 
                 sigma_compression=0.0,
                 normalize_lie=False,
                 cmp_is=True,              # IS for compression (τ)
                 dyn_is=True,              # IS for dynamics (τ,t)
                 is_bins_tau=16,           # τ bins
                 is_bins_t=16,             # t bins (for t_dyn in [0,1])
                 is_ema=0.05,              # EMA step for second-moment tables
                 is_eps=1e-8):

        self._q_tau_cache = None          # cached mixed τ proposal (K_tau,)
        self._q_tau_raw   = None          # last raw τ proposal (K_tau,)
        self._row_probs_cache = None      # cached mixed per-τ t proposals (K_tau, K_t)
        self._row_probs_raw   = None      # last raw per-τ t proposals (K_tau, K_t)
        self.drift_tol = 0.02             # reuse cache if max |new-old| < drift_tol

        self.normalize_lie = normalize_lie
        #get flow matcher for u 
        assert flow_matcher_type in ['exactot', 'regular', 'sinkhorn']
        
        self.flow_matcher_type = flow_matcher_type
        
        if flow_matcher_type=='regular': 
            self.tau_flowmatcher = cfm.ConditionalFlowMatcher(sigma=sigma_compression)
            
        elif flow_matcher_type=='exactot':
            self.tau_flowmatcher = cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=sigma_compression)
        elif flow_matcher_type=='sinkhorn':
            self.tau_flowmatcher = cfm.GeneralOptimalTransportConditionalFlowMatcher(sigma=sigma_compression, OTmethod = 'sinkhorn')
        else: 
            raise NotImplementedError('Only ifs, exactot or sinkhorn fm  supported!')
        
        #get flow matcher for v 
        #this is always just regular flow matcher! 
        self.t_flowmatcher = cfm.ConditionalFlowMatcher(sigma=sigma_dynamics)

        self.cmp_is = bool(cmp_is)
        self.dyn_is = bool(dyn_is)
        self.K_tau  = int(is_bins_tau)
        self.K_t    = int(is_bins_t)
        self.is_ema = float(is_ema)
        self.is_eps = float(is_eps)

        self.m_tau = torch.ones(self.K_tau)
        self.m_dyn = torch.ones(self.K_tau, self.K_t)
        
    
    def __call__(self, net, x0_1, xdt_1, dt, pre_training=False, 
                 cov_dynamic_0=None, cov_dynamic_dt=None, cov_static=None):

        
        #sample taus, ts
        B = x0_1.shape[0]
        device = x0_1.device
        dt = float(dt)

        # lazy move second-moment tables to device
        if self.m_tau.device != device:
            self.m_tau = self.m_tau.to(device=device, dtype=torch.float32)
        if self.m_dyn.device != device:
            self.m_dyn = self.m_dyn.to(device=device, dtype=torch.float32)

        # ---------- τ sampling (IS if cmp_is, else uniform) ----------
        if self.cmp_is:
            if self.dyn_is:
                tau_score = (self.m_dyn.sum(dim=1) + self.is_eps).sqrt()     # [K_tau]
            else:
                tau_score = (self.m_tau + self.is_eps).sqrt()                # [K_tau]
            q_raw = tau_score / (tau_score.sum() + self.is_eps)              # [K_tau]
        
            # reuse cached mixed proposal if drift is tiny
            reuse = False
            if self._q_tau_raw is not None:
                drift = torch.max(torch.abs(q_raw - self._q_tau_raw)).item()
                reuse = drift < self.drift_tol
        
            if reuse and (self._q_tau_cache is not None):
                q_mix = self._q_tau_cache
            else:
                lam = 0.9
                q_mix = (1 - lam) / float(self.K_tau) + lam * q_raw          # [K_tau]
                self._q_tau_cache = q_mix.detach()
                self._q_tau_raw   = q_raw.detach()
        
            # batched CDF sampling for τ (one uniform per example)
            cdf_tau = torch.cumsum(q_mix, dim=0)
            cdf_tau[-1] = 1.0
            u_tau = torch.rand(B, device=device)
            i_tau = torch.searchsorted(cdf_tau, u_tau, right=False).long().clamp_(0, self.K_tau - 1)
            taus  = (i_tau.to(torch.float32) + torch.rand(B, device=device)) / float(self.K_tau)
            w_tau = 1.0 / (q_mix[i_tau] * float(self.K_tau) + self.is_eps)
        else:
            taus  = torch.rand(B, device=device)
            i_tau = torch.clamp((taus * self.K_tau).long(), 0, self.K_tau - 1)
            w_tau = torch.ones(B, device=device)

        # ---------- t sampling (IS if dyn_is, else uniform) ----------
        if self.dyn_is:
            # Build per-τ proposal over t once, then gather rows for the batch
            row_scores = (self.m_dyn + self.is_eps).sqrt()                                              # [K_tau, K_t]
            row_probs  = row_scores / (row_scores.sum(dim=1, keepdim=True) + self.is_eps)               # [K_tau, K_t]
            lam = 0.9
            row_mix   = (1 - lam) / self.K_t + lam * row_probs                                          # [K_tau, K_t]
        
            # Select rows for current τ indices
            row_b = row_mix.index_select(0, i_tau)                                                      # [B, K_t]
        
            # Batched CDF sampling: one uniform per example
            cdf = torch.cumsum(row_b, dim=1)
            cdf[:, -1] = 1.0                                                                            # numeric guard
            u = torch.rand(B, device=device).unsqueeze(1)                                               # [B, 1]
            j_t = torch.searchsorted(cdf, u, right=False).squeeze(1).long().clamp_(0, self.K_t - 1)     # [B]
        
            # Dequantized t in [0, dt]
            tnorm = (j_t.to(torch.float32) + torch.rand(B, device=device)) / float(self.K_t)
            ts = tnorm * dt
        
            # Importance weights for chosen cells
            sel = row_b.gather(1, j_t.view(-1, 1)).squeeze(1)                                           # [B]
            w_t = 1.0 / (sel * float(self.K_t) + self.is_eps)                                           # [B]
        else:
            tnorm = torch.rand(B, device=device)                                                        # U[0,1]
            ts = tnorm * dt
            j_t = torch.clamp((tnorm * self.K_t).long(), 0, self.K_t - 1)
            w_t = torch.ones(B, device=device)

        # ---------------------------
        # Forward pass through network (unchanged API)
        # ---------------------------
        u0_tau, ut_tau, u, v, x0_0, xdt_0 = net(
            x0_1, xdt_1, taus, ts, dt,
            self.tau_flowmatcher, self.t_flowmatcher,
            cov_dynamic_0=cov_dynamic_0, cov_dynamic_dt=cov_dynamic_dt,
            cov_static=cov_static
        )

        flow_sq = (u - u0_tau) ** 2               # [B, D]
        dyn_sq  = (v - ut_tau) ** 2               # [B, D]
        flow_scalar = flow_sq.mean(dim=1)         # [B] for EMA
        dyn_scalar  = dyn_sq.mean(dim=1)

        # option 1: vanilla apply IS weights only for the dimensions we importance-sampled
        # flow_loss = flow_sq * (w_tau.view(-1, 1) if self.cmp_is else 1.0)
        # dyn_w = torch.ones(B, 1, device=device)
        # if self.cmp_is:
        #     dyn_w *= w_tau.view(-1, 1)
        # if self.dyn_is:
        #     dyn_w *= w_t.view(-1, 1)
        # dyn_loss = dyn_sq * dyn_w

        # option 2: batch-normalized weights for gradient
        eps = self.is_eps
        if self.cmp_is:
            w_tau_g = w_tau / (w_tau.mean() + eps)
            w_tau_g = torch.clamp(w_tau_g, max=10.0)  # tune 2–10 if needed
        
        if self.dyn_is:
            w_t_g = w_t / (w_t.mean() + eps)
            w_t_g = torch.clamp(w_t_g, max=10.0)
        
        # Compression (u)
        if self.cmp_is:
            flow_loss = g_only(w_tau.view(-1,1), w_tau_g.view(-1,1)) * flow_sq
        else:
            flow_loss = flow_sq
        
        # Dynamics (v)
        dyn_w = torch.ones(B, 1, device=device)
        if self.cmp_is: dyn_w = dyn_w * w_tau.view(-1,1)
        if self.dyn_is: dyn_w = dyn_w * w_t.view(-1,1)
        
        if self.cmp_is or self.dyn_is:
            dyn_w_g = torch.ones_like(dyn_w)
            if self.cmp_is: dyn_w_g = dyn_w_g * w_tau_g.view(-1,1)
            if self.dyn_is: dyn_w_g = dyn_w_g * w_t_g.view(-1,1)
            dyn_loss = g_only(dyn_w, dyn_w_g) * dyn_sq
        else:
            dyn_loss = dyn_sq
        

        # ---------------------------------------------------------------------------
        
        if pre_training:
            #compute reconstruction loss
            enc_pt_loss = (x0_1 - x0_0)**2

            
            #set conditional Lie loss to zero 
            enc_lie_loss = torch.zeros(x0_1.shape[0], x0_1.shape[1]).type(torch.float32).to(x0_1.device)

            # don' train dynamic flow during pre_training?? (Let's first make sure compression flow is correct...)
            # dyn_loss = dyn_loss*0
            # flow_loss = flow_loss*0
            
            
        else:
            #set reconstruction loss to zero
            enc_pt_loss = (x0_1 - x0_0)**2
            
            #compute conditional Lie loss 
            enc_lie_loss = (xdt_1 - x0_1 - xdt_0 + x0_0)**2 #bs, dim 

        # IS bins weight update
        with torch.no_grad():
            a = self.is_ema
        
            # τ table: use bincount to aggregate per-bin means
            if self.cmp_is:
                idx_tau = i_tau                                # [B]
                cnt_tau = torch.bincount(idx_tau, minlength=self.K_tau)                     # [K_tau]
                sumsq_tau = torch.bincount(idx_tau, weights=(flow_scalar ** 2),
                                           minlength=self.K_tau)                             # [K_tau]
                mean_tau = torch.zeros_like(self.m_tau)
                nonzero_tau = cnt_tau > 0
                mean_tau[nonzero_tau] = sumsq_tau[nonzero_tau] / cnt_tau[nonzero_tau].to(sumsq_tau.dtype)
                # EMA only where updated
                self.m_tau = torch.where(nonzero_tau,
                                         (1.0 - a) * self.m_tau + a * mean_tau,
                                         self.m_tau)
        
            # (τ,t) table: flatten (i,j) → k = i*K_t + j, then bincount once
            if self.dyn_is:
                flat = i_tau * self.K_t + j_t                                                      # [B]
                cnt_flat = torch.bincount(flat, minlength=self.K_tau * self.K_t)                  # [K_tau*K_t]
                sumsq_flat = torch.bincount(flat, weights=(dyn_scalar ** 2),
                                            minlength=self.K_tau * self.K_t)                      # [K_tau*K_t]
                mean_flat = torch.zeros_like(sumsq_flat)
                nonzero_flat = cnt_flat > 0
                mean_flat[nonzero_flat] = sumsq_flat[nonzero_flat] / cnt_flat[nonzero_flat].to(sumsq_flat.dtype)
        
                m_flat = self.m_dyn.view(-1)
                m_flat = torch.where(nonzero_flat,
                                     (1.0 - a) * m_flat + a * mean_flat,
                                     m_flat)
                self.m_dyn = m_flat.view(self.K_tau, self.K_t)

        

        return torch.cat([flow_loss.unsqueeze(0), dyn_loss.unsqueeze(0), enc_pt_loss.unsqueeze(0), \
                          enc_lie_loss.unsqueeze(0)], dim=0) # 4, bs, dim 
        