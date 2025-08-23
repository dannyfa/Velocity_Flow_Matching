# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""

Model architectures and preconditioning schemes used.

Includes original work from EDM repository 
along with modifications needed to implement inflationary flows
pre-conditioning for toy and image datasets.

"""

import numpy as np
import torch
import torch.distributions as D 
from torch_utils import persistence
from torch.nn.functional import silu
import os
from torch_cfm.models.unet.unet import UNetModelWrapper 

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.
# Added a stride arg here. Defaults to 2, which was original/cte val in EDM code.
# Shouldn't alter original code/archs but allows more flexibility for toys. 

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, stride=2, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride=stride
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.stride, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=self.stride)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.stride, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.stride, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad) #leave stride==1 (default!!)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


#----------------------------------------------------------------------------#

@persistence.persistent_class
class GaussianFourierProjection(torch.nn.Module):
  """
  Gaussian random features for encoding time steps.
  This is similar to implementation from Karras.
  """  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

#----------------------------------------------------------------------------#

@persistence.persistent_class
class Dense(torch.nn.Module):
  """
  A fully connected layer that reshapes outputs to feature maps.
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = torch.nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

#----------------------------------------------------------------------------#

#Toy Conv UNet for simple img data 
@persistence.persistent_class
class ToyConvUNet(torch.nn.Module):
  """A time-dependent model built upon U-Net architecture."""

  def __init__(self, channels=[32, 64, 128, 256], embed_dim=256, img_size=28, img_ch=1, input_size=784):
    """
    Initialize a time-dependent flow network.

    Args:
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
      
    """
    super().__init__()
    #set up basic shapes for img
    self.img_size=img_size
    self.img_ch=img_ch
    # Gaussian random feature embedding layer for time
    self.embed = torch.nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         torch.nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = torch.nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = torch.nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = torch.nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = torch.nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = torch.nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = torch.nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = torch.nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = torch.nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = torch.nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = torch.nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = torch.nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = torch.nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = torch.nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    #self.act = lambda x: x * torch.sigmoid(x)
  
  def forward(self, x, t): 
    # reshape flattened array 
    x = x.reshape(-1, self.img_ch, self.img_size, self.img_size)
    # Obtain the Gaussian random feature embedding for t   
    embed = silu(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = silu(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = silu(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = silu(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = silu(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = silu(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = silu(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = silu(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    #flatten output once again before returning 
    h = h.reshape(-1, self.img_ch*self.img_size**2)
      
    return h

#---------------------------------------------------------------------------#
#Equivalent ToyConvUNet adapted for non-image toys...

@persistence.persistent_class
class Adapted_ToyConvUNet(torch.nn.Module):

  def __init__(self, channels=[32, 64, 128, 256], embed_dim=256, img_size=8, img_ch=1, input_size=2):
    super().__init__()
    #set up basic shapes for img
    self.img_size=img_size
    self.img_ch=img_ch
    self.latent_size = img_ch*img_size**2 
    # Gaussian random feature embedding layer for time
    self.embed = torch.nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         torch.nn.Linear(embed_dim, embed_dim))
    #Up/Down-sampling linear layers for non-image toys... 
    self.up_fc = torch.nn.Linear(input_size, self.latent_size, bias=True)
    self.down_fc = torch.nn.Linear(self.latent_size, input_size, bias=True)
    # Encoding layers where the resolution decreases
    self.conv1 = torch.nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = torch.nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = torch.nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = torch.nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = torch.nn.GroupNorm(32, num_channels=channels[2])

    # Decoding layers where the resolution increases
    self.tconv3 = torch.nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=False)    
    self.dense4 = Dense(embed_dim, channels[1])
    self.tgnorm3 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = torch.nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=1, bias=False)    
    self.dense5 = Dense(embed_dim, channels[0])
    self.tgnorm2 = torch.nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = torch.nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

  def forward(self, x, t): 
    #apply up-sampling fc layer for x 
    x = self.up_fc(x)
    # reshape flattened array 
    x = x.reshape(-1, self.img_ch, self.img_size, self.img_size)
    # Obtain the Gaussian random feature embedding for t   
    embed = silu(self.embed(t))    
    
    # Encoding path
    h1 = self.conv1(x)
    h1 += self.dense1(embed)
    h1 = silu(self.gnorm1(h1))
      
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = silu(self.gnorm2(h2))
      
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = silu(self.gnorm3(h3))

    h4 = self.tconv3(h3)
    h4 += self.dense4(embed)
    h4 = silu(self.tgnorm3(h4))

    h5 = self.tconv2(torch.cat([h4, h2], dim=1))
    h5 += self.dense5(embed)
    h5 = silu(self.tgnorm2(h5))

    h6 = self.tconv1(torch.cat([h5, h1], dim=1))

    #now re-shape and downsample
    h6 = h6.reshape(-1, self.latent_size)
    h7 = self.down_fc(h6)
    return h7

#---------------------------------------------------------------------------#
#Simple toy MLP arch (from CFM repo)
#This can be used for any type of toy! 

@persistence.persistent_class
class ToyMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, n_hidden=2, w=64, time_varying=False):
        super().__init__()
        
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        
        net = [torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU()]
        for _ in range(n_hidden):
            net.append(torch.nn.Linear(w, w))
            net.append(torch.nn.SELU())
        net.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*net)

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], dim=-1))

#----------------------------------------------------------------------------
#MLP Encoder 
#Can be used for any toy type!!

#MLP VAE case is tested/works fine 
@persistence.persistent_class
class Latent_MLP_VAE(torch.nn.Module):
    
    def __init__(self, input_size, output_size, dims_to_keep, num_hidden, hidden_size=10, eps=1e-5, d_min=1e-10):

        super().__init__()

        #construct mu net -- this is still locally param
        mu = [torch.nn.Linear(input_size,hidden_size,bias=True), torch.nn.ReLU()] 
        for _ in range(num_hidden):
            mu.append(torch.nn.Linear(hidden_size,hidden_size))
            mu.append(torch.nn.ReLU())
        mu.append(torch.nn.Linear(hidden_size,output_size))


        self.mu = torch.nn.Sequential(*mu)
        
        #now construct GLOBAL u, D parameters 
        self.u = torch.nn.Parameter(torch.randn(output_size)) #dim 
        self.d = torch.nn.Parameter(torch.randn(output_size)) #dim 
        
        if dims_to_keep < output_size: 
            d_max_pds = torch.ones(dims_to_keep).unsqueeze(0) 
            d_max_cds = torch.ones(output_size - dims_to_keep).unsqueeze(0) * eps 
            self.d_max_diag = torch.cat([d_max_pds, d_max_cds], dim=-1) #1, dim
        else: 
            self.d_max_diag = torch.ones(output_size).unsqueeze(0) #1, dim 
        
        #set out min val for D diag
        #this is just to avoid chol from failing as model trains 
        self.d_min_diag = torch.ones(output_size).unsqueeze(0)*d_min #1, dim 
        
    def encode(self, x):
        
        mu = self.mu(x) #bs, dim 
        #set up L from global param u 
        u = self.u.unsqueeze(-1) #dim,1
        L = torch.einsum('ij, jk -> ik', u, torch.transpose(u, 1, 0)) #dim, dim
        L = torch.tril(L) #force L to be lower triangular (dim, dim)
        #now force diagonals of L to be == 1 
        L_diag = torch.diagonal(L, dim1=0, dim2=1) #dim 
        L_diag_ones = torch.ones(L_diag.shape).to(x.device) #dim
        L -= torch.diag_embed(L_diag) #remove original diagonal 
        L += torch.diag_embed(L_diag_ones) #force all diagonals to be ones 
        
        #now setup D from global param d 
        d = torch.exp(self.d) #dim 
        d = torch.clamp(d, min=self.d_min_diag.to(x.device), max=self.d_max_diag.to(x.device)) #clip d 
        d = torch.diag_embed(d).squeeze(0) #dim, dim 
        
        return mu, d, L

    
    def rsample(self, x):
        mu, d, L = self.encode(x)
        L_tril = torch.einsum('ij, jk -> ik', L, torch.sqrt(d))
        
        eps = torch.randn(mu.shape).to(mu.device)
        z = mu + eps 
        z = torch.einsum('ij, bjk -> bik', L_tril, z.unsqueeze(-1)).squeeze(-1)

        return z

        
    def forward(self,x):
        
        mu, d, L = self.encode(x)
        L_tril = torch.einsum('bij, bjk -> bik', L, torch.sqrt(d))
        latent_dist = D.MultivariateNormal(loc=mu, scale_tril=L_tril) 
        z = latent_dist.rsample() 
        #print(z.shape) 
        #print(latent_dist.entropy().shape) 
        kl_term = -0.5 * (torch.sum(torch.pow(z,2),axis=-1) + z.shape[-1] * np.log(2*np.pi)) 
        #assert False
        kl_term = kl_term.sum() + torch.sum(latent_dist.entropy()) 
        return z, -kl_term

    
#---------------------------------------------------------------------------

#CNN Encoder Arches for Image Toys

#needs testing on balls data 
@persistence.persistent_class
class Latent_CNN_VAE(torch.nn.Module):
    def __init__(self, img_resolution=28, img_ch=1, dims_to_keep=784, eps=1.0, d_min=1e-10):
        
        super().__init__()

        self.input_dim = img_ch*img_resolution**2
        self.latent_size = self.input_dim
        self.img_resolution= img_resolution
        self.img_ch = img_ch
        if dims_to_keep < self.input_dim: 
            d_max_pds = torch.ones(dims_to_keep).unsqueeze(0) 
            d_max_cds = torch.ones(self.input_dim - dims_to_keep).unsqueeze(0) * eps 
            self.d_max_diag = torch.cat([d_max_pds, d_max_cds], dim=-1) #1, dim    
        else: 
            self.d_max_diag = torch.ones(self.input_dim).unsqueeze(0) #1, dim 

        #set out min val for D diag
        #this is just to avoid chol from failing as model trains 
        self.d_min_diag = torch.ones(self.input_dim).unsqueeze(0)*d_min #1, dim 
    
        # For mean encoder
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear1 = torch.nn.Linear(32*4*4,300)
        self.mu = torch.nn.Linear(300, self.latent_size)
        
        # Add global parameters 
        self.u = torch.nn.Parameter(torch.randn(self.input_dim))
        self.d = torch.nn.Parameter(torch.randn(self.input_dim))

    def encode(self,x):
        #get mu, u, d 
        x = x.reshape(-1, self.img_ch, self.img_resolution, self.img_resolution)
        t = torch.nn.functional.relu(self.conv1(x))
        t = torch.nn.functional.relu(self.conv2(t))
        t = t.reshape((x.shape[0], -1))
        
        t = torch.nn.functional.relu(self.linear1(t))
        mu = self.mu(t)

        #get Ltril mat 
        u = self.u.unsqueeze(-1) # dim, 1
        L = torch.einsum('ij, jk -> ik', u, torch.transpose(u, 1, 0)) #dim, dim 
        L = torch.tril(L) #force L to be lower triangular (dim, dim)        
        
        #now force diagonals of L to be == 1 
        L_diag = torch.diagonal(L, dim1=0, dim2=1) #dim
        L_diag_ones = torch.ones(L_diag.shape).to(x.device) #dim 
        L -= torch.diag_embed(L_diag) #remove original diagonal 
        L += torch.diag_embed(L_diag_ones) #force all diagonals to be ones , dimxdim 

        #now setup D
        d = torch.exp(self.d) #d has to be pos
        d = torch.clamp(d, min=self.d_min_diag.to(x.device), max=self.d_max_diag.to(x.device)) #clip d 
        d = torch.diag_embed(d).squeeze(0) #dim,dim
        return mu, d, L
          
        
    def rsample(self, x):
        mu, d, L = self.encode(x)
        L_tril = torch.einsum('ij, jk -> ik', L, torch.sqrt(d))
        
        eps = torch.randn(mu.shape).to(mu.device)
        z = mu + eps 
        z = torch.einsum('ij, bjk -> bik', L_tril, z.unsqueeze(-1)).squeeze(-1)

        return z
    
    def forward(self, x):
        mu, d, L = self.encode(x)
        L_tril = torch.einsum('bij, bjk -> bik', L, torch.sqrt(d))
        latent_dist = D.MultivariateNormal(loc=mu, scale_tril=L_tril)
        z = latent_dist.rsample()
        
        kl_term = -0.5 * (torch.sum(torch.pow(z,2),axis=-1) + z.shape[-1] * np.log(2*np.pi))
        #assert False
        kl_term = kl_term.sum() + torch.sum(latent_dist.entropy())
        
        return z,-kl_term


#slightly larger/more powerful arch 
#also needs testing on balls data 
@persistence.persistent_class
class Latent_LargeCNN_VAE(torch.nn.Module):

  def __init__(self, channels=[32, 64, 128, 256], img_size=28, img_ch=1, dims_to_keep=784, eps=1.0, d_min=1e-15, input_size=784):
    super().__init__()
    
    #set up general attributes
    self.input_dim = img_ch*img_size**2
    self.img_size=img_size
    self.img_ch=img_ch

    #set d_max tensor
    if dims_to_keep < self.input_dim: 
        d_max_pds = torch.ones(dims_to_keep).unsqueeze(0) 
        d_max_cds = torch.ones(self.input_dim - dims_to_keep).unsqueeze(0) * eps 
        self.d_max_diag = torch.cat([d_max_pds, d_max_cds], dim=-1) #1, dim    
    else: 
        self.d_max_diag = torch.ones(self.input_dim).unsqueeze(0) #1, dim 

    #set d_min tensor
    self.d_min_diag = torch.ones(self.input_dim).unsqueeze(0)*d_min #1, dim 

    # Conv layers with decreasing res 
    self.conv1 = torch.nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.gnorm1 = torch.nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = torch.nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.gnorm2 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = torch.nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.gnorm3 = torch.nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = torch.nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.gnorm4 = torch.nn.GroupNorm(32, num_channels=channels[3])    

    #Linear Layers to extract mus
    self.mu = torch.nn.Linear(256*2*2, self.input_dim)
    
    #add in global params
    self.u = torch.nn.Parameter(torch.randn(self.input_dim))
    self.d = torch.nn.Parameter(torch.randn(self.input_dim))
  
  def encode(self, x): 
    # Reshape flattened array 
    x = x.reshape(-1, self.img_ch, self.img_size, self.img_size)
   
    # Encoder Conv/Gnorm BLocks
    h1 = silu(self.gnorm1(self.conv1(x)))
    h2 = silu(self.gnorm2(self.conv2(h1)))
    h3 = silu(self.gnorm3(self.conv3(h2)))
    h4 = silu(self.gnorm4(self.conv4(h3)))

    #Extract mu, d, u
    mu = self.mu(h4.reshape(x.shape[0], -1))


    #Compute desired D, L matrices  
    #L = uu^\top 
    u = self.u.unsqueeze(-1) #dim, 1
    L = torch.einsum('ij, jk -> ik', u, torch.transpose(u, 1, 0)) #dim, dim 
    #force L to be lower triangular
    L = torch.tril(L)        
    #force diagonals of L to be == 1 
    L_diag = torch.diagonal(L, dim1=0, dim2=1)
    L_diag_ones = torch.ones(L_diag.shape).to(x.device)
    L -= torch.diag_embed(L_diag) #remove original diagonal 
    L += torch.diag_embed(L_diag_ones) #force all diagonals to be ones 

    #D=exp(d), clamped by eps, d_min
    d = torch.exp(self.d)
    d = torch.clamp(d, min=self.d_min_diag.to(x.device), max=self.d_max_diag.to(x.device))
    d = torch.diag_embed(d).squeeze(0)

    #return encoder mean, d, L
    return mu, d, L 
    
  def rsample(self, x):
      mu, d, L = self.encode(x)
      L_tril = torch.einsum('ij, jk -> ik', L, torch.sqrt(d))
        
      eps = torch.randn(mu.shape).to(mu.device)
      z = mu + eps 
      z = torch.einsum('ij, bjk -> bik', L_tril, z.unsqueeze(-1)).squeeze(-1)

      return z 
    
  def forward(self, x):
      mu, d, L = self.encode(x)
      L_tril = torch.einsum('bij, bjk -> bik', L, torch.sqrt(d))
      latent_dist = D.MultivariateNormal(loc=mu, scale_tril=L_tril)
      z = latent_dist.rsample()
      kl_term = -0.5 * (torch.sum(torch.pow(z,2),axis=-1) + z.shape[-1] * np.log(2*np.pi))
      kl_term = kl_term.sum() + torch.sum(latent_dist.entropy())
      return z,-kl_term
#----------------------------------------------------------------------------

#version of LargeCNN Enc above for non-image toys
#needs testing on regular (non-image) toys ... 
@persistence.persistent_class
class Adapted_Latent_LargeCNN_VAE(torch.nn.Module):
  def __init__(self, channels=[32, 64, 128, 256], img_size=8, img_ch=1, input_size=2, dims_to_keep=2, eps=1.0, d_min=1e-15):
    super().__init__()
    #set up basic shapes for img
    self.img_size=img_size
    self.img_ch=img_ch
    self.latent_size = img_ch*img_size**2 
    self.input_size = input_size

    #set d_max tensor
    if dims_to_keep < self.input_size: 
        d_max_pds = torch.ones(dims_to_keep).unsqueeze(0) 
        d_max_cds = torch.ones(self.input_size - dims_to_keep).unsqueeze(0) * eps 
        self.d_max_diag = torch.cat([d_max_pds, d_max_cds], dim=-1) #1, dim    
    else: 
        self.d_max_diag = torch.ones(self.input_size).unsqueeze(0) #1, dim 

    #set d_min tensor
    self.d_min_diag = torch.ones(self.input_size).unsqueeze(0)*d_min #1, dim 
    
    #Upsampling linear layers for non-image toys... 
    self.up_fc = torch.nn.Linear(input_size, self.latent_size, bias=True)
      
    # Encoding layers where the resolution decreases
    self.conv1 = torch.nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.gnorm1 = torch.nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = torch.nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
    self.gnorm2 = torch.nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = torch.nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False)
    self.gnorm3 = torch.nn.GroupNorm(32, num_channels=channels[2])
      
    #general linear layer before extracting mu, d, u
    self.fc1 = torch.nn.Linear(128*2*2, 100)
      
    #mu layer 
    self.mu = torch.nn.Linear(100, input_size)
    
    #now add global params d, u
    self.u = torch.nn.Parameter(torch.randn(self.input_size))
    self.d = torch.nn.Parameter(torch.randn(self.input_size))    
    
  def encode(self, x): 
    # Apply up-sampling fc layer for x 
    x = self.up_fc(x)
    # Reshape flattened array 
    x = x.reshape(-1, self.img_ch, self.img_size, self.img_size)

    # Feed through CNN Encoding path
    h1 = silu(self.gnorm1(self.conv1(x)))      
    h2 = silu(self.gnorm2(self.conv2(h1)))      
    h3 = silu(self.gnorm3(self.conv3(h2)))

    # Extract mu, u, d
    h4 = self.fc1(h3.reshape(x.shape[0], -1))
    mu = self.mu(h4)

    #Compute desired D, L matrices  
    #L = uu^\top 
    u = self.u.unsqueeze(-1) #dim, 1
    L = torch.einsum('ij, jk -> ik', u, torch.transpose(u, 1, 0)) #dim, dim 
    #force L to be lower triangular
    L = torch.tril(L)        
    #force diagonals of L to be == 1 
    L_diag = torch.diagonal(L, dim1=0, dim2=1)
    L_diag_ones = torch.ones(L_diag.shape).to(x.device)
    L -= torch.diag_embed(L_diag) #remove original diagonal 
    L += torch.diag_embed(L_diag_ones) #force all diagonals to be ones 

    #D=exp(d), clamped by eps, d_min
    d = torch.exp(self.d)
    d = torch.clamp(d, min=self.d_min_diag.to(x.device), max=self.d_max_diag.to(x.device))
    d = torch.diag_embed(d).squeeze(0) #dim, dim 
      
    return mu, d, L

  def rsample(self, x):
      mu, d, L = self.encode(x)
      L_tril = torch.einsum('ij, jk -> ik', L, torch.sqrt(d))
        
      eps = torch.randn(mu.shape).to(mu.device)
      z = mu + eps 
      z = torch.einsum('ij, bjk -> bik', L_tril, z.unsqueeze(-1)).squeeze(-1)

      return z
    
  def forward(self, x):
      mu, d, L = self.encode(x)
      L_tril = torch.einsum('bij, bjk -> bik', L, torch.sqrt(d))
      latent_dist = D.MultivariateNormal(loc=mu, scale_tril=L_tril)
      z = latent_dist.rsample()
      kl_term = -0.5 * (torch.sum(torch.pow(z,2),axis=-1) + z.shape[-1] * np.log(2*np.pi))
      kl_term = kl_term.sum() + torch.sum(latent_dist.entropy())
      return z,-kl_term

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

##########################################
# Preconditionings from EDM paper
# This is legacy code and will be removed! 

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


#----------------------------------------------------------------------------
# Preconditioning corresponding to EDM (proposed) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".
@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

##################################################################################


#-------------------------------------------------------------------------------------
#Def network class for VFM

@persistence.persistent_class
class ConvVNetWrapper(torch.nn.Module):
    def __init__(self, base_model, img_ch, img_size):
        super().__init__()
        self.base_model = base_model
        self.img_ch = img_ch
        self.img_size = img_size
        # Use 1x1 convolution to project from 2*img_ch channels to img_ch channels
        self.channel_projection = torch.nn.Conv2d(2 * img_ch, img_ch, kernel_size=1)
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        # x has shape (batch, 2*data_dim) where data_dim = img_ch * img_size * img_size
        
        # Split the concatenated input back into two parts
        data_dim = self.img_ch * self.img_size * self.img_size
        xt_tau = x[:, :data_dim]  # First half: current position
        x0_tau = x[:, data_dim:]  # Second half: starting position
        
        # Reshape both parts to image format
        xt_tau_img = xt_tau.reshape(batch_size, self.img_ch, self.img_size, self.img_size)
        x0_tau_img = x0_tau.reshape(batch_size, self.img_ch, self.img_size, self.img_size)
        
        # Concatenate along channel dimension: (batch, 2*img_ch, img_size, img_size)
        concat_img = torch.cat([xt_tau_img, x0_tau_img], dim=1)
        
        # Project back to original number of channels using 1x1 conv
        # This preserves spatial structure while mixing information from both inputs
        projected_img = self.channel_projection(concat_img)  # (batch, img_ch, img_size, img_size)
        
        # Flatten back to the format expected by the base model
        projected_flat = projected_img.reshape(batch_size, -1)
        
        # Pass through the base model
        return self.base_model(projected_flat, t)

#-------------------------------------------------------------------------------------
#Def network class for VFM

@persistence.persistent_class
class  VFMToyNet(torch.nn.Module):
    """
    Wrapper for calling u (flow time)
    and v (dynamics time) nets. 
    
    Supports either MLP or ToyConvUNet architectures 
    """
    def __init__(self, 
                 channels = [32, 64, 128, 256], #Channels for feature maps of each resolution of conv layers
                 conv_embed_dim = 256,  # Dimensionality for Gaussian random feature embeddings for conv layers
                 data_dim=2, #dimensionality of data we will feed through Net 
                 dims_to_keep = 2, #number of nominal dimensions encoder should preserve 
                 dyn_model_type = "ToyMLP", #class name for underlying model
                 flow_model_type = "ToyMLP",
                 encoder_type = "Latent_MLP_VAE",
                 depth_encoder = 2, # number of hidden layers for MLP encoder 
                 width_encoder = 10, # with of hidden units (for each layer) of MLP encoder 
                 depth_mlp = 2,  # number of hidden layers for MLPs used for flow and dyn nets
                 width_mlp = 64, # number of hidden units (for each layer) of MLPs used in flow/dynamics nets.
                 cd_eps = 1.0, #max variance allowed for compressed dimensions in encoder output. 
                 img_size = 28, #size for img, if using balls or other img toy.
                 in_ch=1, #input channel if using toy img data.
                 d_min=1e-15, #min value to cap D across all dimensions
                 save_dir='' #dir where chol errors (if present) should be saved to. 
                 ):
        super().__init__()
        self.data_dim = data_dim
        self.dims_to_keep = dims_to_keep
        self.save_dir=save_dir
        
        #create u net (flow net)
        if flow_model_type in ["ToyConvUNet", "Adapted_ToyConvUNet"]:
            self.unet_model = globals()[flow_model_type](channels=channels, embed_dim=conv_embed_dim, img_size=img_size, img_ch=in_ch, \
                                                         input_size=data_dim)
        else:
            self.unet_model = globals()[flow_model_type](dim=data_dim, time_varying=True, n_hidden=depth_mlp, w=width_mlp)
        
        #create vnet (dynamics net)
        if dyn_model_type in ["ToyConvUNet", "Adapted_ToyConvUNet"]:
            base_vnet = globals()[dyn_model_type](channels=channels, embed_dim=conv_embed_dim, img_size=img_size, img_ch=in_ch, \
                                                  input_size=data_dim)
            self.vnet_model = ConvVNetWrapper(base_vnet, in_ch, img_size)
        else:
            self.vnet_model = globals()[dyn_model_type](dim=2*data_dim, out_dim=data_dim, time_varying=True, n_hidden=depth_mlp, w=width_mlp)
        
        #create encoder net 
        if encoder_type in ['Latent_LargeCNN_VAE', "Adapted_Latent_LargeCNN_VAE"]:
            self.encoder = globals()[encoder_type](channels=channels, img_size=img_size, img_ch=in_ch, dims_to_keep=dims_to_keep, eps=cd_eps, \
                                                   d_min=d_min, input_size=data_dim) 

        elif encoder_type == "Latent_MLP_VAE": 
            self.encoder = globals()[encoder_type](input_size=data_dim, output_size=data_dim, \
                                              dims_to_keep=dims_to_keep, num_hidden=depth_encoder, \
                                                  hidden_size=width_encoder, eps=cd_eps, d_min=d_min) 

        else:
            self.encoder = globals()[encoder_type](img_resolution=img_size, img_ch=in_ch, dims_to_keep=dims_to_keep, eps=cd_eps, \
                                                   d_min=d_min) 
        
                
    def forward(self, x0_1, xdt_1, taus, ts, dt, tau_flowmatcher, t_flowmatcher):
        # pre-training routine will be implemented at level of loss fn 
        # get proposals from encoder 
        x0_0 = self.encoder.rsample(x0_1)
        xdt_0 = self.encoder.rsample(xdt_1) 
            
        #get compression flow interpolated forms
        _, x0_tau, u0_tau = tau_flowmatcher.sample_location_and_conditional_flow(x0_0, x0_1, t=taus)
        _, xdt_tau, _ = tau_flowmatcher.sample_location_and_conditional_flow(xdt_0, xdt_1, t=taus)
            
        #get dynamics flow interpolated form 
        _, xt_tau, ut_tau = t_flowmatcher.sample_location_and_conditional_flow(x0_tau, xdt_tau, t=(ts/dt))
        
        # now get u, v
        u = self.unet_model(x0_tau, taus)
        v = self.vnet_model(torch.cat([xt_tau, x0_tau], dim=-1), taus)
        
        return u0_tau, ut_tau, u, v, x0_0, xdt_0
            
            

    
    