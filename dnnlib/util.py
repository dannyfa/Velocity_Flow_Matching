# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Miscellaneous utility classes and functions."""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import torch 
from torch.utils.data.dataset import Dataset
from sklearn.decomposition import PCA
from abc import ABC,abstractmethod
from scipy.ndimage import gaussian_filter

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Optional
from torchdyn.core import NeuralODE

#------------------------------------------------------------------------------------------#
# Utils for VFM
#------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------
#Dynamic Toy dataset utils 

class ToyData(ABC):

    """
    generic base class for toy datasets. implements
    generate, which all classes need, and defines methods needed
    by all inheriting classes
    """


    def __init__(self):

        pass

    @abstractmethod
    def f(self,x,t):
        raise NotImplementedError

    @abstractmethod
    def g(self,x,t):
        raise NotImplementedError

    @abstractmethod
    def dW(self,dt):
        raise NotImplementedError 

    @abstractmethod
    def init_conditions(self):
        raise NotImplementedError

    def dx(self,x,t,dt,sigma):

        fx = self.f(x,t)
        gx = self.g(x,t,sigma)
        dw = self.dW(dt)

        return fx * dt + gx @ dw
    
    def generate(self,n,T,dt,sigma):

        trajectories = []
        t = np.arange(dt,T+dt/2,dt)
        for ii in range(n):

            xnot = self.init_conditions()
            x = [xnot]

            for jj in range(1,len(t)):

                xx = x[jj-1]
                tt = t[jj-1]

                dx = self.dx(xx,tt,dt,sigma)
                
                x.append(xx + dx)
            
            x = np.vstack(x)
            trajectories.append(x)

        return trajectories

#vanderpol oscillator 
class Vanderpol(ToyData):

    def __init__(self,coeffs=[2,15],seed=1234):

        super(Vanderpol,self).__init__()
        self.rho,self.tau = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx1 = self.rho * self.tau * (x[0] - x[0]**3/3 - x[1])
        dx2 = self.tau/self.rho * x[0]
        return np.hstack([dx1,dx2])
    def g(self,x,t,sigma):
        return sigma*np.eye(2)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))

    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[1,1],cov=np.eye(2)*0.03)

#double circles data 
class DoubleCircles(ToyData):

    def __init__(self,coeffs=[3.5,4,2*np.pi],seed=1234):

        super(DoubleCircles,self).__init__()

        self.r0,self.a,self.omega = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        print("f unused in double circles")
        pass

    def ft(self,theta,r,t):
        #dtheta = omega
        if r > self.r0:
            return self.omega
        else:
            return -self.omega
    
    def fr(self,r,t):
        
        # potential function: (r - r0)^4 - a(r - r0)^2
        return -(4 * (r - self.r0)**3 - 2*self.a * (r - self.r0))

    def g(self,x,t,sigma):
        return sigma*np.eye(2)

    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov = dt*np.eye(2))
    
    def _polar_to_cartesian(self,r,theta):
    
        return np.hstack([r*np.cos(theta),r*np.sin(theta)])
    
    def _cartesian_to_polar(self,xy):
        
        r = np.linalg.norm(xy)
        theta = np.arctan2(xy[1],xy[0])
        return r,theta
    
    def init_conditions(self):
        r0 = self.r0 + self.gen.normal(loc=0,scale=0.1)
        t0 = self.gen.uniform(0,2*np.pi)

        return self._polar_to_cartesian(r0,t0)

    def dx(self,x,t,dt,sigma):
        
        r,theta = self._cartesian_to_polar(x)
        dr = self.fr(r,t)
        dtheta = self.ft(theta,r,t)
        r += dr*dt 
        theta += dtheta*dt

        newX = self._polar_to_cartesian(r,theta)
        dw_xy = self.g(x,t,sigma) @ self.dW(dt)
        xy2 = newX + dw_xy
        
        return xy2 - x

#Lorenz attractor - 3D ODE sys 
class Lorenz63(ToyData):

    def __init__(self,coeffs=[10,28,8/3],seed=1234):

        super(Lorenz63,self).__init__()
        self.sigma,self.rho,self.beta = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        
        dx = self.sigma * (x[1] - x[0]) #+ sample_dW[0]
        dy = (x[0] * (self.rho - x[2]) - x[1]) #+ sample_dW[1]
        dz = (x[0]*x[1]  - self.beta*x[2]) #+ sample_dW[2]
        return np.hstack([dx,dy,dz])
    
    def g(self,x,t,sigma):
        return sigma*np.eye(3)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=dt*np.eye(3))
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3))

#Alternative Lorenz attractor 
class Lorenz96(ToyData):

    def __init__(self,coeffs=[8],d=10,seed=1234):

        super(Lorenz96,self).__init__()
        self.F = coeffs
        self.d = d
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx = np.zeros(self.d)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(self.d):
            dx[i] = (x[(i + 1) % self.d] - x[i - 2]) * x[i - 1] - x[i] + self.F
        return dx
    
    def g(self,x,t,sigma):
        return np.eye(self.d)*sigma
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((self.d,)),cov=np.eye(self.d)*dt)
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=np.zeros((self.d,)),cov=np.eye(self.d))

#Rossler attractor 
class Rossler(ToyData):

    def __init__(self,coeffs=[0.1,0.1,14],seed=1234):

        super(Rossler,self).__init__()
        self.a,self.b,self.c = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx = -x[1] - x[2] #+ sample_dW[0]
        dy = x[0] + self.a*x[1] #+ sample_dW[1]
        dz = self.b + x[2]*(x[0] - self.c) #+ sample_dW[2]
        return np.hstack([dx,dy,dz])
    
    def g(self,x,t,sigma):
        return np.eye(3)*sigma
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3)*dt)
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[0,-9,0],cov=6*np.eye(3))

#Double SDE 
class DoubleSDE(ToyData):

    def __init__(self,coeffs=[np.pi,-np.pi,np.array([-0.5,0]),np.array([0.5,0])],seed=1234):

        super(DoubleSDE,self).__init__()
        self.omega1,self.omega2,self.center1,self.center2 = coeffs
        #self.center2 = -self.center1
        
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        print("f unused in diverging sdes")
        pass
    def ft(self,theta,x,t):
        if x[0] < 0:
            return self.omega1
        else:
            return self.omega2
        
    def g(self,x,t,sigma):
        return sigma * np.eye(2)#/(np.abs(x[0])+1/10)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=np.eye(2)*dt)
    
    def _polar_to_cartesian(self,r,theta):
    
        return np.hstack([r*np.cos(theta),r*np.sin(theta)])
    
    def _cartesian_to_polar(self,xy):
    
        r = np.linalg.norm(xy)
        theta = np.arctan2(xy[1],xy[0])
        return r,theta
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[0,-0.5],cov=np.eye(2)*0.01)

    def dx(self,x,t,dt,sigma):

        if x[0] < 0:
            v = x - self.center1
            r,theta = self._cartesian_to_polar(v)
            dtheta = self.ft(theta,x,t)
            theta += dtheta*dt
            xx2 = self._polar_to_cartesian(r,theta) + self.center1
        else:
            v = x - self.center2
            r,theta = self._cartesian_to_polar(v)
            dtheta = self.ft(theta,x,t)
            theta += dtheta*dt
            xx2 = self._polar_to_cartesian(r,theta) + self.center2

        return xx2 + self.g(x,t,sigma)@self.dW(dt) - x

#Moving balls dataset (movie)
class Balls(ToyData):

    def __init__(self,coeffs=np.array([[-1,-3],[-3,1]]),seed=1234):

        super(Balls,self).__init__()
        self.coeffs = np.array(coeffs)

        #self.center2 = -self.center1
        
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):

        return self.coeffs @ x 

    def g(self,x,t,sigma):

        return sigma*np.eye(2)

    def dW(self,dt):

        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[-1,0],cov=np.eye(2)*0.25**2)
    
    def traj_to_movie(self,trajectories,image_shape,radius,blur=False):
        """
        converts a latent trajectory to a movie
        """


        movies = []
        gX,gY = np.meshgrid(np.linspace(-1,1,2*radius),np.linspace(-1,1,2*radius))
        ball = (gX**2 + gY**2 < 1)
        image_shape = np.array(image_shape)
        
        for traj in trajectories:
            traj -= np.amin(traj)
            traj /= np.amax(np.abs(traj))

            frames = np.zeros((len(traj),1,image_shape[0],image_shape[1]))
        
            for point in range(len(traj)):
        
                ballCenter = (traj[point,:] *(image_shape - 2*radius)).astype(int) + radius
                frames[point,0,ballCenter[0]-radius:ballCenter[0] + radius, ballCenter[1]-radius:ballCenter[1] + radius] += ball
                if blur:
                    frames[point,0,:,:] = gaussian_filter(frames[point,0,:,:],sigma=radius*4,truncate=0.05)

            movies.append(frames)

        return movies


class ToyDsetDynamics(Dataset):

    """
    dataloader for toy datasets with dynamics. expects data in the form of that created by
    my toy data creation methods -- in other words, a list of np.arrays.
    flattens all arrays and creates a set of valid indices of that array to sample from.
    This set of valid indices is also based on nForward: the number of steps forward in time
    that we want our model to predict. 
    When sampling, will return samples of length nForward + 2 (last index is dt)
    """

    def __init__(self,data,dt,nForward=1) -> None:
        

        self.maxForward = nForward
        exampleInd = np.random.choice(len(data),1)[0] #choose a traj at random 
        self.exampleTraj = data[exampleInd] #get traj we sampled
        lens = list(map(len,data)) #list with lengths of trajs (should be all equal for toys) 
        lens2 = [0] + list(np.cumsum([l for l in lens][:-1])) #cumsum over total number of pts across trajs 
        #sets ranging from 0  to traj length -- constructs one per traj 
        sets = [np.vstack([np.arange(ii, l+ ii - self.maxForward) for ii in range(self.maxForward + 1)]).T for l in lens]
        sumSets = [p+l for p,l in zip(sets,lens2)] #shifts sets appropriately 
        validInds = np.vstack(sumSets) #stacks all sets
        self.data= np.vstack(data) #stacks all trajs, 
        self.data_inds = validInds 
        self.dt = dt
        self.length = len(validInds)

    def __len__(self):

        return self.length 
    
    def __getitem__(self, index):
        
        single_index = False
        result = []
        try:
            iterator = iter(index)
        except TypeError:
            index = [index]
            single_index = True

        for ii in index:
            inds = self.data_inds[ii]

            samples = [self.transform(self.data[ind]) for ind in inds]
            samples.append(self.dt)			
            result.append(samples)

        if single_index:
            return result[0]
        return result
    
    def transform(self,data):
        return torch.from_numpy(data).type(torch.FloatTensor)

#projection helper 
class projection():

    """
    projects things to a higher dimensional space, plus some additional nonlinearity. 
    should save out everything 

    """

    def __init__(self,origDim,newDim, projType='linear',temp=1,seed=1234) -> None:
        self.gen = np.random.default_rng(seed=seed)
        self.d1 = origDim
        self.d2 = newDim
        self.projType=projType 
        self.W = self.gen.normal(loc=0.0,scale=1.5,size=(origDim,newDim))
        self.temp=temp
        if projType == 'linear':
            

            self.projection = lambda x: x @ self.W 
        
        elif projType == 'softmax':

            assert self.temp >0, print('Temperature should be a positive number')
            self.projection = lambda x: _softmax(x @ self.W,temp=self.temp)

        elif projType == 'combine':

            self.projection = lambda x: _combine_dims(x @ self.W)

        elif projType == 'sigmoid':

            assert self.temp >0, print('Temperature should be a positive number')
            self.projection = lambda x: _sigmoid(x @ self.W,temp=self.temp)

        elif projType == 'swish':

            assert self.temp >0, print("Temperature should be a positive number")
            self.projection = lambda x: _swish(x @ self.W,temp=self.temp)
        elif projType == 'double_swish':

            assert self.temp >0, print("Temperature should be a positive number")
            self.projection = lambda x: _double_swish(x @ self.W,temp=self.temp)


        else:
            print('Method must be softmax,combine,sigmoid,swish,double swish,or linear')
            raise NotImplementedError

    
    def project(
        self,
        data: np.array,
        noise: float= 0.
     ) -> np.array:
        
        r"""
        Return batch projectiojns.
        

        Args: 
            data: what we are embedding in a higher-d space
            
        """
        if noise:
            proj = self.projection(data)
            return  proj + noise * np.random.normal(size=proj.shape)
        else:
            return self.projection(data)
          
def _swish(x:np.array,temp=1):

    xout = x * _sigmoid(x,temp)

    return xout

def _double_swish(x:np.array,temp=1):

    xout = np.sign(x) * _swish(np.abs(x),temp=temp)

    return xout

def _softmax(x:np.array,temp=1):

    m = np.amax (x,axis=1,keepdims=True)
    e_x = np.exp((x - m)/temp)
    return e_x/np.sum(e_x,axis=1,keepdims=True) 

def _combine_dims(x:np.array):

    xOut = x 
    for ii in range(x.shape[-1]-1):
        xOut[:,ii] = x[:,ii]*x[:,ii+1]

    return xOut

def _sigmoid(x:np.array,temp=1):

    xout = 1/(1 + np.exp(-x*temp))

    return xout


#wrapper to get toy dynamics dset by name 
def get_toy_dynamicdset(dset_name, n_trajs, T, dt, sigma, project=False, proj_specs=None):
    """
    Wrapper to construct desired toy dynamic dset from name, params.
    Can handle projections too if desired.
    """
    DATASETS = {'vanderpol': "Vanderpol()", 
            'doublecircles': "DoubleCircles()", 
            'rossler': "Rossler()", 
            'lorenz63': "Lorenz63()",
            'lorenz96': "Lorenz96()", 
            'doublesde': "DoubleSDE()"}
    dset_name = dset_name.lower()
    try: 
        dset_gen_obj = eval(DATASETS[dset_name])
    except KeyError:
        raise ValueError(f"Unknown Dataset: {dset_name}")
    #sample deserired trajs from it 
    dset_samples = dset_gen_obj.generate(n=n_trajs, T=T, dt=dt, sigma=sigma)
    #project data if desired 
    if project: 
        assert proj_specs != None, 'To project need projection specs!'
        orig_dim = dset_samples[0][0].shape[0]
        projection_obj = projection(orig_dim, proj_specs.project_to, \
                                    projType=proj_specs.proj_type, \
                                    temp=proj_specs.temp)
        proj_dset_samples = [[projection_obj.project(t) for t in traj] for traj in dset_samples]
        dset_samples = proj_dset_samples
    
    dset_obj = ToyDsetDynamics(dset_samples, dt, nForward=1)
    return dset_obj, dset_samples

#methods to sample (independently) xt,0's

def get_xt_zero_samples(dset_kwargs, n, device, W, concatenated=False):
    """
    Sample xt,0's from either full or low-rank MVN 
    and pass these samples to original IS/data space.

    Can either generate xt,0's for a single pt or for two points (independently)
    if concatenated ==True.
    """
    working_n = 2*n if concatenated else n
    x0 = torch.randn(working_n, dset_kwargs.working_data_dim).type(torch.float32).to(device)      
    if dset_kwargs.working_data_dim != dset_kwargs.dims_to_keep:
        scale =  torch.concatenate([torch.ones(dset_kwargs.dims_to_keep), \
                                                  torch.ones(dset_kwargs.working_data_dim - dset_kwargs.dims_to_keep)*dset_kwargs.eps], \
                                      dim=0).type(torch.float32).to(device)
        x0 *= torch.sqrt(scale)[None, :]
    x0 = torch.einsum('ij, bjk -> bik', W, x0.unsqueeze(-1)).squeeze(-1) #pass x0 to IS as well
    if concatenated: 
        return x0.reshape(n, -1) #n, 2d 
    else: 
        return x0


#helper to calc PCA decomp. 
#This will be replaced by streaming SVD 

def get_eigenvals_basis(X, n_comp=784):
    """
    Uses sklearn PCA method to obtain U_t matrix 
    containing eigenvectors of current covariance mat
    And its correspoding eigenvals.
    """
    pca = PCA(n_components=n_comp) 
    pca.fit(X)
    eigenvals = pca.explained_variance_
    U = pca.components_ #eigenvectors are rows of U 
    return eigenvals, pca, U.T #eigenvectors returned as cols of U


#------------------------------------------------------------------------------#
# methods to run trajectory simulation
#------------------------------------------------------------------------------#

class dyn_torch_wrapper(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format.
    
    Note that, for now, t==1 ALWAYS. 
    This simulates dynamics in IS only!
    
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        net_ts = torch.ones(x.shape[0]).type(torch.float32).to(x.device)
        out = self.model(x, net_ts)        
        return out


def calc_dyn_trajectories(model, init_samples, nt=100):
    """
    Simulates dyn net trajectories.
    
    Args
    -----
    model: torch.nn.Module. Instance of IFsCFMToyNet class. 
    init_samples: torch.Tensor. Contains starting points for ODE int.
    nt: int. Number of time pts to integrate over. This is per each 
    dt step in dyn trajectory.
    """
    #setup node 
    node = NeuralODE(dyn_torch_wrapper(model), solver='dopri5', \
                     sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    #get ts 
    ts = torch.linspace(0.0, 1.0, nt).to(init_samples.device)
    #now sim ODE 
    with torch.no_grad():
        traj = node.trajectory(init_samples, ts)
    return traj

#------------------------------------------------------------------------------#

# Util classes (EDM repo)

#-------------------------------------------------------------------------------#


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: Optional[str] = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


# Cache directories
# ------------------------------------------------------------------------------------------#

_dnnlib_cache_dir = None

def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

# Small util functions
# ------------------------------------------------------------------------------------------#


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------#

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------#

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------#

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)
