"""Implements Conditional Flow Matcher Losses."""

import math
import warnings
from typing import Union

import torch

from .optimal_transport import OTPlanSampler
from dnnlib.util import get_g 
from torch_utils import persistence


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

@persistence.persistent_class
class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


@persistence.persistent_class
class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1

@persistence.persistent_class
class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)

@persistence.persistent_class
class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for Schrödinger bridge conditional flow matching method. This class implements
    the SB-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and
    sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        r"""Initialize the SchrodingerBridgeConditionalFlowMatcher class. It requires the hyper-
        parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
            we use exact as the default as we found this to perform better
            (more accurate and faster) in practice for reasonable batch sizes.
            We note that as batchsize --> infinity the correct choice is the
            sinkhorn method theoretically.
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) with respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise: bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch entropic OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1

@persistence.persistent_class
class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """Albergo et al. 2023 trigonometric interpolants class. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t and compute_conditional_flow functions in
    order to compute [3]'s trigonometric interpolants.

    [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def compute_mu_t(self, x0, x1, t):
        r"""Compute the mean of the probability path (Eq.5) from [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: cos(pi t/2)x0 + sin(pi t/2)x1

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        r"""Compute the conditional vector field similar to [3].

        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(\pi*t/2) x0)

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)

#----------------------------------------------------------------------------------------#
#Def class for IFsConditionalFlowMatcher
#this is with independent coupling between x_0, x_1

@persistence.persistent_class
class IFsConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Child class for implementing conditional flow matching using 
    Inflationary Flows schedule and scaling. 
    For now, coupling is sassumed independent  - i.e., 
    q(x_1, x_0) = q(x_1)*q(x_0)
    """
    def __init__(self, sigma: Union[float, int] = 0.0, rho=1.0, gamma0=5e-4, \
                 data_dim=2, dims_to_keep=2, ODE_type='scaled', space='ES', data_eigs=None, \
                     W=None, g_scaling=1.0):
        """
        Initialize the IFsConditionalFlowMatcher class (and its parent).
        Parameters:
        -----------
        sigma: Union[float, int] --> needed for parent class 
        rho: float. Exponential growth constant for IFs schedule.
        gamma0: float. Minimal noise kernel width for IFs schedule.
        data_dim:int. Dimensionality of data.
        dims_to_keep:int. How many dims (of total) to preserve in IFs schedule. 
        ODE_type:std. Whether to simulate xt, vt(x|x1) for scaled or unscaled pfODEs.
        space: str. Whether to return results in image space (IS) or eigenspace (ES). 
        Regardless of choice of basis, inputs should be given in eigenspace!
        data_eigs: torch.Tensor [dim]. Tensor containing eigenvalues for the original 
        data.
        W: torch.Tensor [dim, dim]. Tensor containing eigenvectors of original data
        as its columns.
        g_scaling: float. Scaling to be applied to g tensor. Defaults to 1.0 (no scaling)
        """
        super().__init__(sigma)
        self.rho=rho
        self.gamma0=gamma0
        self.ODE_type=ODE_type
        self.space=space 
        self.data_eigs=data_eigs
        self.W=W
        g = get_g(data_dim, dims_to_keep, data_eigs.device)
        g*=g_scaling
        self.g = torch.ones(g.shape[0]).to(g.device) + g 
        
    def get_gamma_sqrd(self, ts):
        """
        Computes diag(C_t) = \gamma^2 
        for IFs schedule

        Parameters
        ----------
        ts : torch.Tensor [bs]. Contains batch times 
        for which we wish to compute gammas.

        Returns
        -------
        torch.Tensor [bs, dim]. Tensor containing
        gamma_t for each sampled/given t.

        """
        exp_term = self.rho*self.g[None, :]*ts[:, None] #bs, dim
        gamma = self.gamma0 * torch.exp(exp_term)
        return gamma #bs, dim 
    
    def get_gamma_sqrd_dot(self, ts):
        """
        Computes the time derivative of diag(C_t) = \gamma^2 
        for IFs schedule

        Parameters
        ----------
        ts : torch.Tensor [bs]. Contains batch times 
        for which we wish to compute gamma_dots.

        Returns
        -------
        torch.Tensor [bs, dim]. Tensor containing
        temporal derivative of gamma_t for each sampled/given t.

        """        
        gamma = self.get_gamma_sqrd(ts)
        gamma_dot = self.rho*self.g*gamma
        return gamma_dot #bs, dim 
    
    def get_alpha(self, ts, return_all=False):
        """
        Computes diag(A_t) = \alpha
        for IFs schedule. 
        
        Parameters
        ----------
        ts : torch.Tensor [bs]. Contains batch times 
        for which we wish to compute alphas.

        Returns
        -------
        torch.Tensor [bs, dim]. Tensor containing
        alpha_t for each sampled/given t.
        
        """
        #set up xi_star^2
        if len(torch.unique(self.g)) == 1: 
            #prp schedule 
            xi_sqrd = self.data_eigs
        else: 
            #prr schedule 
            xi_sqrd = torch.amax(self.data_eigs)*\
            torch.ones(self.data_eigs.shape[0]).to(self.data_eigs.device)
        
        #set up A0 
        A0 = torch.sqrt(xi_sqrd)
        
        #calc alpha
        g_star = torch.amax(self.g)*\
        torch.ones(self.g.shape[0]).to(self.g.device)
        gamma_star = self.gamma0 * torch.exp(self.rho*g_star[None, :]*ts[:, None]) #bs, dim 
        alpha = A0[None, :] * torch.reciprocal(torch.sqrt(xi_sqrd[None, :] + gamma_star)) #bs, dim 
        
        if return_all:
            return alpha, gamma_star, g_star, A0, xi_sqrd
        else: 
            return alpha
    
    def get_alpha_dot(self, ts):
        """
        Computes the temporal derivative of diag(A_t) = \alpha
        for IFs schedule. 
        
        Parameters
        ----------
        ts : torch.Tensor [bs]. Contains batch times 
        for which we wish to compute alpha_dots.

        Returns
        -------
        torch.Tensor [bs, dim]. Tensor containing
        temporal derivative of alpha_t for each sampled/given t.
        
        """        
        _, gamma_star, g_star, A0, xi_sqrd = self.get_alpha(ts, return_all=True)
        alpha_dot = -0.5*A0[None, :]*self.rho*g_star[None, :]*gamma_star
        alpha_dot *= (xi_sqrd[None, :] + gamma_star)**(-1.5)
        return alpha_dot #bs, dim 
    
    def compute_mu_t(self, x1, t):
        """
        Computes mean for IFs schedule affine trf.
        
        Parameters:
        ------
        x1: torch.Tensor [bs, dim]. Tensor containing 
        target distribution batch samples.
        t: torch.Tensor [bs]. Tensor containing 
        sampled times (one per each batch item).
        """
        
        if self.ODE_type == 'scaled': 
            alpha_t = self.get_alpha(t)
            mu_t = alpha_t * x1 
        else: 
            mu_t = x1 
        
        return mu_t
    
    def compute_mu_t_dot(self, x1, t):
        """
        Computes temporal derivative of mean 
        for IFs schedule affine trf. 

        Parameters:
        ------
        x1: torch.Tensor [bs, dim]. Tensor containing 
        target distribution batch samples.
        t: torch.Tensor [bs]. Tensor containing 
        sampled times (one per each batch item).
        """
        if self.ODE_type == 'scaled': 
            alpha_t_dot = self.get_alpha_dot(t)
            mu_t_dot = alpha_t_dot * x1 
        else: 
            mu_t_dot =  torch.zeros(x1.shape).to(x1.device)        
        return mu_t_dot

    def compute_sigma_t(self, t):
        """
        Computes diag(\Sigma^{1/2}) for IFs schedule.
        
        Parameters: 
        ----------
        t: torch.Tensor [bs]. Tensor containing 
        sampled times (one per each batch item).        
        """
        
        gamma_sqrd = self.get_gamma_sqrd(t)
        if self.ODE_type == 'scaled':
            alpha = self.get_alpha(t)
            sigma_t = torch.sqrt((alpha**2)*gamma_sqrd)
        else: 
            sigma_t = torch.sqrt(gamma_sqrd)
        return sigma_t
    
    def compute_sigma_t_dot(self, t):
        """
        Computes derivative of std dev for IFs schedule. 
        This is equivalent to: 
            0.5*\dot{C}*C^{-0.5} --> if unscaled ODE
            0.5*(A\dot{C}A^\top)*(ACA^\top)^{-0/5} --> if scaled ODE 
        
        Parameters: 
        ----------
        t: torch.Tensor [bs]. Tensor containing 
        sampled times (one per each batch item).   
        """
        
        gamma_sqrd_dot = self.get_gamma_sqrd_dot(t) #C_dot
        gamma_sqrd = self.get_gamma_sqrd(t) #C 
        if self.ODE_type=='scaled':
            alpha = self.get_alpha(t) #A 
            sigma_t_dot = 0.5*((alpha**2)*gamma_sqrd_dot) #0.5*(A\dot{{C}A^\top)
            sigma_t_dot *= ((alpha**2)*gamma_sqrd)**(-0.5) #(ACA^\top)^{-0.5}
        else: 
            sigma_t_dot = 0.5*gamma_sqrd_dot #0.5 \dot{C}
            sigma_t_dot *= (gamma_sqrd)**(-0.5) #C^{-0.5}
        return sigma_t_dot
    
    def sample_xt(self, x1, t, epsilon):
        """
        Draw a sample from the probability path N(\mu_t, \Sigma_t)

        Parameters
        ----------
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        """
        sigma_t = self.compute_sigma_t(t)
        mu_t = self.compute_mu_t(x1, t)
        xt = sigma_t * epsilon + mu_t
        return xt 
    
    def compute_conditional_flow(self, x1, t, xt):
        """
        Compute the conditional vector field vt(x1|x0) = (\dot{\sigma}/\sigma)*(xt - \mu_t) + \dot{\mut}_t

        Parameters
        ----------
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        vt : conditional vector field vt(x1|x0) 

        """
        sigma_t = self.compute_sigma_t(t)
        sigma_t_dot = self.compute_sigma_t_dot(t)
        mu_t = self.compute_mu_t(x1, t)
        mu_t_dot = self.compute_mu_t_dot(x1, t)
        
        vt = sigma_t_dot * torch.reciprocal(sigma_t)
        vt *= (xt - mu_t)
        vt += mu_t_dot
        
        return vt

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        """
        Compute the sample xt (drawn from N(\mu_t,  \Sigma_t))
        and the conditional vector field vt(x1|x0) = (\dot{\sigma}/\sigma)*(xt - \mu_t) + \dot{\mut}_t

        Parameters
        ----------
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        vt : conditional vector field. 

        """
        if t is None:
            t = torch.rand(x1.shape[0]).type_as(x1)
        assert len(t) == x1.shape[0], "t has to have batch size dimension"
        
        if x0 is None: 
            x0 = torch.randn_like(x1).type_as(x1).to(x1.device)
        
        xt = self.sample_xt(x1, t, x0)
        vt = self.compute_conditional_flow(x1, t, xt)
        
        if self.space=='IS':
            #convert outputs on image space basis
            xt = torch.einsum('ij, bjk -> bik', self.W, xt.unsqueeze(-1)).squeeze(-1)
            vt = torch.einsum('ij, bjk -> bik', self.W, vt.unsqueeze(-1)).squeeze(-1)
        
        return t, xt, vt
            
            
#-------------------------------------------------------------------------------# 
# Adding a flex version of OT-CFM class  
# This can use either OT-FM or sinkhorn -- migh be best for HD data (tbd) 

@persistence.persistent_class
class FlexibleOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    
    Only difference from ExactOptimalTransportConditionalFlowMatcher is that it
    allow either exact or sinkhorn approaches when computing OT-coupling.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, ot_method="exact"):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: which OT sampler we'd like to use. Defaults to exact
        b/c this usually trains best for not too big batch sizes.
        
        As batch size increases, we should opt for sinkhorn instead.
        
        
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)        

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1

#-----------------------------------------------------------------------------#