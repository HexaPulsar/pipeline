import torch
import torch.nn as nn 
import numpy as np
import torch
import torch.nn as nn
import math

class TimeFilmModified(nn.Module):
    def __init__(self, n_harmonics=4, embedding_size=64, Tmax=1000.0, input_size=1):
        super(TimeFilmModified, self).__init__() 
        self.linear_proj_only = False
        if self.linear_proj_only:
            print("USING LINEAR_PROJ ONLY!!!") 
            self.linear_proj = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = False))
        else:
            self.register_parameter('alpha_sin',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('alpha_cos',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('beta_sin',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('beta_cos',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            #self.linear_r = nn.Linear(in_features=n_harmonics, out_features=embedding_size, bias=False)
            self.n_harmonics = n_harmonics
            self.register_buffer("ar", torch.arange(1,n_harmonics+1).unsqueeze(0).unsqueeze(0))
            self.register_buffer('const', torch.tensor( 2 * torch.pi / Tmax))
            
            self.linear_proj = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = False))
            self.embedding_size = embedding_size
            self.dropout = nn.Dropout(p=0.0)
    def get_sin_cos(self, t,mask):
        return torch.sin(t) , torch.cos(t)

    def gelu_drop(self,coeffs):
        return nn.functional.gelu(self.dropout(coeffs))
    
    def forward(self, x, t,mask):
        if self.linear_proj_only:
            return self.linear_proj(x)
        else:
            sin_emb, cos_emb = self.get_sin_cos(self.const * self.ar*t.repeat(1,1,self.n_harmonics),mask)
            alpha = (torch.matmul(sin_emb, self.alpha_sin) + torch.matmul(cos_emb, self.alpha_cos)/2)
            beta = (torch.matmul(sin_emb, self.beta_sin) + torch.matmul(cos_emb, self.beta_cos))/2
            alpha = self.dropout(torch.clip(alpha,-1,1))
            #beta = self.dropout(torch.clip(beta,-1,1))
            x =  self.linear_proj(x)
            return x* alpha +  beta # + x.repeat(1,1,self.embedding_size)
        
    
class SpringEncoder(nn.Module):
    def __init__(self, dt=1e-3, steps=1000, mass=1.0, embedding_size=200,**kwargs):
        super(SpringEncoder, self).__init__()
        self.dt = dt
        self.steps = steps
        self.mass = mass  # Scalar mass (assumed the same for all objects)
        self.embedding_size = embedding_size
        self.Tmax = 1000.0
        self.min_k_clip = 1e-5
        self.max_k_clip = 10

    def forward(self, x0, t0,mask):
        """
        Simulates the motion of a spring-mass system over time.
        Arguments:
            x0: Initial positions (Tensor of shape [batch_size])
            t0: Periods (Tensor of shape [batch_size])
            v0: Initial velocities (Tensor of shape [batch_size])
        Returns:
            t_values: Time values (Tensor of shape [steps])
            x_values: Simulated positions (Tensor of shape [batch_size, steps])
        """
        v0 = 0
         
        batch_size ,seqlen, input_size= x0.shape  # Batch size
        #print(x0.shape)
       # print(batch_size,seqlen)
        # Ensure tensors are on the correct device
        x0 = (x0  * mask).squeeze(-1)
        t0 = (t0 * mask).squeeze(-1)
       # v0 = v0.to(self.device) * mask
        # Allocate memory for position and velocity
        x_values = torch.zeros((batch_size,seqlen, self.steps), device=x0.device)
        v_values = torch.zeros((batch_size,seqlen, self.steps), device=x0.device)
       #print(x_values.shape)
        # Compute spring constant k
        k_batch = self.calculate_k(t0) # Shape: [batch_size]
        m_batch = torch.ones_like(k_batch)
        # Time values
        t_values = torch.linspace(1e-5, self.dt * self.steps, self.steps, device=x0.device)

        # Set initial conditions
        x_values[:,:, 0] = x0
        v_values[:,:, 0] = v0

        # Time stepping loop using Euler's method
        for i in range(1, self.steps): 
            a = -(k_batch / m_batch) * x_values[:,:, i - 1]  # Acceleration (Hooke's law)
            v_values[:,:, i] =  a * self.dt  + v_values[:,:, i - 1]# Update velocity
            x_values[:,:, i] = x_values[:,:, i - 1] + v_values[:, :,i] * self.dt  # Update position
        #t_values =torch.nn.functional.interpolate(x_values, size=200, mode="linear", align_corners=False)
        x_values =  torch.nn.functional.interpolate(x_values, size=self.embedding_size, mode="linear", align_corners=False)
        return x_values
 



import torch
import numpy  as np
import torch.nn as nn

class SpringEncoderSTEP(nn.Module):
    def __init__(self, dt=1e-3, steps=1000, mass=1.0, embedding_size=128,Tmax = 1000.0,**kwargs):
        super(SpringEncoderSTEP, self).__init__()
        #self.register_parameter('dt',nn.Parameter(torch.tensor(dt)))
        ##self.register_parameter('Tmax',nn.Parameter(torch.tensor(Tmax)))
        #self.register_parameter('mass_coeff',nn.Parameter(torch.tensor(mass)))
        self.dt = dt
        self.Tmax = Tmax
        self.mass = mass
        self.steps = steps
        self.embedding_size = embedding_size
        self.min_k_clip = 1e-5
        self.max_k_clip = 10

    def forward(self, x0, t0,mask):
        """
        Simulates the motion of a spring-mass system over time.
        Arguments:
            x0: Initial positions (Tensor of shape [batch_size])
            t0: Periods (Tensor of shape [batch_size])
            v0: Initial velocities (Tensor of shape [batch_size])
        Returns:
            t_values: Time values (Tensor of shape [steps])
            x_values: Simulated positions (Tensor of shape [batch_size, steps])
        """
        v0 = 0
         
        batch_size ,seqlen, input_size= x0.shape  # Batch size
         
       # print(batch_size,seqlen)
        # Ensure tensors are on the correct device
        x0 = (x0  * mask).squeeze(-1)
        t0 = (t0 * mask).squeeze(-1)

       # v0 = v0.to(self.device) * mask
        # Allocate memory for position and velocity
        x_values = torch.zeros((batch_size,seqlen, self.embedding_size), device=x0.device)
        v_values = torch.zeros((batch_size,seqlen, self.embedding_size), device=x0.device)
       #print(x_values.shape)
        # Compute spring constant k
        k_batch = self.calculate_k(t0) # Shape: [batch_size]
        m_batch = torch.ones_like(k_batch) #* self.mass_coeff
        # Time values
        t_values = torch.linspace(1e-5, self.dt *self.steps, self.embedding_size, device=x0.device)
        
        # Set initial conditions
        x_values[:,:, 0] = x0
        v_values[:,:, 0] = v0
        # Time stepping loop using Euler's method
        for i in range(1, self.embedding_size): 
            a = -(k_batch / m_batch) * x_values[:,:, i - 1]  # Acceleration (Hooke's law)
            v_values[:,:, i] =  a * self.dt  + v_values[:,:, i - 1]# Update velocity
            x_values[:,:, i] = x_values[:,:, i - 1] + v_values[:, :,i] * self.dt  # Update position
        #t_values =torch.nn.functional.interpolate(x_values, size=200, mode="linear", align_corners=False)
        x_values =  torch.nn.functional.interpolate(x_values, size=self.embedding_size, mode="linear", align_corners=False)
        return x_values

    def calculate_k(self, t):
        """Compute the spring constant k given time periods t."""
        n = t/ (2 * torch.pi *self.Tmax)
        #print(n.shape)
        n = torch.clamp(n, self.min_k_clip, self.max_k_clip)  # Clip values
        return 1 / n


 
 
class SpringEncoderANA(nn.Module):
    def __init__(self, dt=1e-5, steps=1000, mass=1.0, embedding_size=128,**kwargs):
        super(SpringEncoderANA, self).__init__()
        self.dt = dt
        self.steps = steps
        #self.mass = mass  # Scalar mass (assumed the same for all objects)
        #self.Tmax = 1000.0
        #self.register_parameter('Tmax', nn.Parameter(torch.tensor(Tmax)))
        self.register_parameter('Tmax',nn.Parameter(torch.tensor(1000.0)))
        #self.register_parameter('dt', nn.Parameter(torch.tensor(dt)))
        #self.register_parameter('steps', nn.Parameter(torch.tensor(steps)))
        self.register_parameter('mass', nn.Parameter(torch.tensor(mass)))
        self.embedding_size = embedding_size
        self.min_k_clip = 1e-5
        self.max_k_clip = 10

    def forward(self, x0, t0,mask):
        """
        Simulates the motion of a spring-mass system over time.
        Arguments:
            x0: Initial positions (Tensor of shape [batch_size])
            t0: Periods (Tensor of shape [batch_size])
            v0: Initial velocities (Tensor of shape [batch_size])
        Returns:
            t_values: Time values (Tensor of shape [steps])
            x_values: Simulated positions (Tensor of shape [batch_size, steps])
        """
        batch_size ,seqlen, input_size= x0.shape  # Batch size 
        t = torch.linspace(0,self.dt*self.steps,self.steps,device = t0.device).repeat(1,seqlen,1) 
        k = self.calculate_k(t0)
        out = x0*torch.cos(k*t/self.mass + t0.repeat(1,1,self.steps) )
        return torch.nn.functional.interpolate(out, size=self.embedding_size, mode="linear", align_corners=False)

    def calculate_k(self, t):
        """Compute the spring constant k given time periods t."""
        n = t/ (2 * torch.pi * self.Tmax)
        #print(n.shape)
        n = torch.clamp(n, self.min_k_clip, self.max_k_clip)  # Clip values
        return 1 / n