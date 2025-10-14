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
# import numpy as np 


LAMBDA_MIX = 0.8
def _mix_from_scores_local(scores, K):
    probs = scores / (scores.sum() + 1e-12)
    return (1.0 - LAMBDA_MIX) / float(K) + LAMBDA_MIX * probs


# def g_only(old, new):
#     return (old - new).detach() + new
# e.g., g_only(w_tau.view(-1,1), w_tau_g.view(-1,1))


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
    def __init__(self, flow_matcher_type='regular',
                 sigma_dynamics=0.1, 
                 sigma_compression=0.0,
                 normalize_lie=False,
                 cmp_is=True,              # IS for compression (τ)
                 dyn_is=True,              # IS for dynamics (τ,t)
                 is_bins_tau=16,           # τ bins
                 is_bins_t=16,             # t bins (for t_dyn in [0,1])
                 is_ema=0.05,              # EMA step for second-moment tables
                 is_eps=1e-5,
                 is_sqrt=False,
                 uniform_tau_v=False,
                 pre_no_flow = True):

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

        # IS 
        self.cmp_is = bool(cmp_is)
        self.dyn_is = bool(dyn_is)
        self.K_tau  = int(is_bins_tau)
        self.K_t    = int(is_bins_t)
        self.is_ema = float(is_ema)
        self.is_eps = float(is_eps)
        self.is_sqrt = bool(is_sqrt)
        self.uniform_tau_v = bool(uniform_tau_v)
        self.pre_no_flow = bool(pre_no_flow)
        self.m_tau = torch.ones(self.K_tau)
        self.m_dyn = torch.ones(self.K_tau, self.K_t)
        
    
    def __call__(self, net, x0_1, xdt_1, dt, pre_training=False, 
                 cov_dynamic_0=None, cov_dynamic_dt=None, cov_static=None):

        
        #sample taus, ts
        B = x0_1.shape[0]
        device = x0_1.device
        dt = float(dt)

        # if pre-train don't have flow loss, don't calculate it to save time
        if self.pre_no_flow and pre_training:
            mod = getattr(net, 'module', net)
            enc = getattr(mod, 'encoder', None)
            if enc is None:
                raise RuntimeError("Expected `net` to have .encoder during pre-train.")
            # encoder forwards only
            x0_0  = enc.rsample(x0_1)
            xdt_0 = enc.rsample(xdt_1)
            enc_pt_loss  = (x0_1 - x0_0)**2
            enc_lie_loss = (xdt_1 - x0_1 - xdt_0 + x0_0)**2
            zeros = torch.zeros_like(enc_pt_loss)
            return torch.stack([zeros, zeros, enc_pt_loss, enc_lie_loss], dim=0)


        # ----------------------------------------------------------------------------
        # IS
        # lazy move second-moment tables to device
        if self.m_tau.device != device:
            self.m_tau = self.m_tau.to(device=device, dtype=torch.float32)
        if self.m_dyn.device != device:
            self.m_dyn = self.m_dyn.to(device=device, dtype=torch.float32)

        # split the batch used for IS for u_net and v_net
        # lazy split...
        B_u = (B // 2)
        B_v = B - B_u

        taus  = torch.empty(B, device=device)
        i_tau = torch.empty(B, dtype=torch.long, device=device)
        tnorm = torch.empty(B, device=device)
        j_t   = torch.empty(B, dtype=torch.long, device=device)

        w_tau_u = torch.ones(B, device=device)   # used on [:B_u]
        w_tau_v = torch.ones(B, device=device)   # used on [B_u:]
        w_t     = torch.ones(B, device=device)   # used on [B_u:]

        # ---- compression flow (u_net) subset (compression; tau only) ----
        if B_u > 0:
            if self.cmp_is:
                s_u = self.m_tau + self.is_eps
                if self.is_sqrt: s_u = s_u.sqrt()
                q_tau_u = _mix_from_scores_local(s_u, self.K_tau)
            else:
                q_tau_u = torch.full((self.K_tau,), 1.0 / self.K_tau, device=device)

            cdf = torch.cumsum(q_tau_u, dim=0); cdf[-1] = 1.0
            u = torch.rand(B_u, device=device)
            i_u = torch.searchsorted(cdf, u, right=False).clamp_(0, self.K_tau - 1)
            taus[:B_u] = (i_u.float() + torch.rand(B_u, device=device)) / float(self.K_tau)
            i_tau[:B_u] = i_u
            w_tau_u[:B_u] = 1.0 / (q_tau_u[i_u] * float(self.K_tau) + self.is_eps)

            # t not used by L_u; keep uniform placeholders
            tnorm[:B_u] = torch.rand(B_u, device=device)
            j_t[:B_u]   = torch.clamp((tnorm[:B_u] * self.K_t).long(), 0, self.K_t - 1)

        # ---- dynamic flow (v_net) subset (dynamics; tau and t_dyn) ----
        if B_v > 0:
            start, end = B_u, B

            # tau for v: learned unless uniform_tau_v=True or dyn_is=False
            if self.dyn_is and (not self.uniform_tau_v):
                tau_marg = self.m_dyn.sum(dim=1)
                s_v = tau_marg + self.is_eps
                if self.is_sqrt: s_v = s_v.sqrt()
                q_tau_v = _mix_from_scores_local(s_v, self.K_tau)
            else:
                q_tau_v = torch.full((self.K_tau,), 1.0 / self.K_tau, device=device)

            cdf = torch.cumsum(q_tau_v, dim=0); cdf[-1] = 1.0
            u = torch.rand(B_v, device=device)
            i_v = torch.searchsorted(cdf, u, right=False).clamp_(0, self.K_tau - 1)
            taus[start:end] = (i_v.float() + torch.rand(B_v, device=device)) / float(self.K_tau)
            i_tau[start:end] = i_v
            w_tau_v[start:end] = 1.0 / (q_tau_v[i_v] * float(self.K_tau) + self.is_eps)

            # t_dyn | tau for v
            if self.dyn_is:
                row_scores = self.m_dyn + self.is_eps                              # [K_tau, K_t]
                if self.is_sqrt: row_scores = row_scores.sqrt()
                row_probs = row_scores / (row_scores.sum(dim=1, keepdim=True) + self.is_eps)
                row_mix  = (1.0 - LAMBDA_MIX) / float(self.K_t) + LAMBDA_MIX * row_probs

                rows = row_mix.index_select(0, i_v)                                 # [B_v, K_t]
                cdf_t = torch.cumsum(rows, dim=1); cdf_t[:, -1] = 1.0
                uu = torch.rand(B_v, device=device).unsqueeze(1)
                jv = torch.searchsorted(cdf_t, uu, right=False).squeeze(1).clamp_(0, self.K_t - 1)

                tnorm[start:end] = (jv.float() + torch.rand(B_v, device=device)) / float(self.K_t)
                j_t[start:end]   = jv
                sel = rows.gather(1, jv.view(-1, 1)).squeeze(1)
                w_t[start:end]   = 1.0 / (sel * float(self.K_t) + self.is_eps)
            else:
                tnorm[start:end] = torch.rand(B_v, device=device)
                j_t[start:end]   = torch.clamp((tnorm[start:end] * self.K_t).long(), 0, self.K_t - 1)

        # map normalized t to real seconds
        ts = tnorm * dt

        # ---------------------------
        # Forward pass through network (unchanged API)
        # ---------------------------
        u0_tau, ut_tau, u, v, x0_0, xdt_0 = net(
            x0_1, xdt_1, taus, ts, dt,
            self.tau_flowmatcher, self.t_flowmatcher,
            cov_dynamic_0=cov_dynamic_0, cov_dynamic_dt=cov_dynamic_dt,
            cov_static=cov_static,
            split_bu=B_u
        )

        flow_sq = (u - u0_tau) ** 2               # [B, D]
        dyn_sq  = (v - ut_tau) ** 2               # [B, D]
        flow_scalar = flow_sq.mean(dim=1)         # [B] for EMA
        dyn_scalar  = dyn_sq.mean(dim=1)

        flow_loss = flow_sq.clone()
        dyn_loss  = dyn_sq.clone()

        # split samples for two flows.
        if self.cmp_is and self.dyn_is:
            flow_loss[B_u:] = 0.0
            dyn_loss[:B_u]  = 0.0

        # compression weights
        if self.cmp_is and B_u > 0:
            flow_loss[:B_u] = flow_loss[:B_u] * w_tau_u[:B_u].view(-1, 1)

        # dynamics weights
        if self.dyn_is and B_v > 0:
            eff = (w_tau_v[B_u:] * w_t[B_u:]).view(-1, 1)
            dyn_loss[B_u:] = dyn_loss[B_u:] * eff
            
        # encoder loss: reconstruction & Lie
        enc_pt_loss  = (x0_1 - x0_0)**2
        enc_lie_loss = (xdt_1 - x0_1 - xdt_0 + x0_0)**2

        # IS bins weight update
        with torch.no_grad():
            a = self.is_ema

            # tau table from U subset
            if self.cmp_is and B_u > 0:
                idx = i_tau[:B_u]
                cnt = torch.bincount(idx, minlength=self.K_tau)
                meas = (flow_scalar[:B_u]**2) if self.is_sqrt else flow_scalar[:B_u]
                sums = torch.bincount(idx, weights=meas, minlength=self.K_tau)
                mean = torch.zeros_like(self.m_tau)
                nz = cnt > 0
                mean[nz] = sums[nz] / cnt[nz].to(sums.dtype)
                self.m_tau = torch.where(nz, (1.0 - a) * self.m_tau + a * mean, self.m_tau)

            # (tau,t_dyn) table from V subset
            if self.dyn_is and B_v > 0:
                idx_tau = i_tau[B_u:]; idx_t = j_t[B_u:]
                flat = idx_tau * self.K_t + idx_t
                cnt  = torch.bincount(flat, minlength=self.K_tau * self.K_t)
                meas = (dyn_scalar[B_u:]**2) if self.is_sqrt else dyn_scalar[B_u:]
                sums = torch.bincount(flat, weights=meas, minlength=self.K_tau * self.K_t)
                mean_flat = torch.zeros_like(sums)
                nz = cnt > 0
                mean_flat[nz] = sums[nz] / cnt[nz].to(sums.dtype)
                m_flat = self.m_dyn.view(-1)
                m_flat = torch.where(nz, (1.0 - a) * m_flat + a * mean_flat, m_flat)
                self.m_dyn = m_flat.view(self.K_tau, self.K_t)
        

        return torch.cat([flow_loss.unsqueeze(0), dyn_loss.unsqueeze(0), enc_pt_loss.unsqueeze(0), \
                          enc_lie_loss.unsqueeze(0)], dim=0) # 4, bs, dim 
        