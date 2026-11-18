import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
import numpy as np
from   torch.autograd import grad
import random
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
#-------------source----------------
class Source():
    def __init__(self,device ='cpu'):
        self.device = device
        self.f = self.compute_f
        self.y_star = self.compute_y_star
        self.zd = self.compute_zd
        self.yd = self.compute_zd
        self.nu = 100
        self.u_star = self.compute_u_star
        self.analystic = True
    def compute_f(self,x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        cond = (x1 <= 0.5)
        val1 = 200 * (2 * x1 * (x1 - 0.5)**2 - x2 * (6 * x1 - 2) * (1 - x2))
        val2 = 200 * (0.5 - x1)
        return torch.where(cond, val1, val2)
        
    def compute_zd(self,x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        cond = (x1 <= 0.5)
        val1 = 200 * x1 * x2 * (x1 - 0.5)**2 * (1 - x2)
        val2 = 200 * x2 * (x1 - 1) * (x1 - 0.5)**2 * (1 - x2)
        return torch.where(cond, val1, val2)
    
    def compute_y_star(self,x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        cond = (x1 <= 0.5)
        val1 = 200 * x1 * x2 * (x1 - 0.5)**2 * (1 - x2)
        val2 = torch.zeros_like(x1)
        return torch.where(cond, val1, val2)
    
    def compute_u_star(self, x):
        """Computes u* = 0"""
        return torch.zeros_like(x[:, 0:1])
    

class Source_:
    def __init__(self, device='cpu', nu=1.0):
        """
        Source terms constructed from the specified boundary conditions and domain properties
        
        Parameters:
            device: Computation device
            nu: Control penalty coefficient, default 1.0
        """
        self.device = device
        self.nu = nu  # Control penalty coefficient
        self.f = self.compute_f
        self.y_star = self.compute_y_star
        self.u_star = self.compute_u_star
        self.xi_star = self.compute_xi_star
        self.zd = self.compute_zd
        self.yd =self.compute_yd
        self.analystic = True
        
    def compute_zd(self,x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        cond = (x1 <= 0.5)
        val1 = 200 * x1 * x2 * (x1 - 0.5)**2 * (1 - x2)
        val2 = 200 * x2 * (x1 - 1) * (x1 - 0.5)**2 * (1 - x2)
        return torch.where(cond, val1, val2)
    
    def laplacian_zd(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        mask = (x1 <= 0.5)
        # Analytical Laplacian Δzd = ∂²zd/∂x1² + ∂²zd/∂x2²
        term1 = 200 * x2 * (1 - x2) * (6*x1 - 2) 
        term2 = -400 * x1 * (x1 - 0.5)**2
        delta_zd = term1 + term2
        return torch.where(mask, delta_zd, torch.zeros_like(x1))
        
    def compute_y_star(self, x):
        """
        Compute the exact solution y*(x1,x2)
        
        y*(x1,x2) = {
            zd(x1,x2)       if 0 < x1 ≤ 0.5
            0               if 0.5 < x1 ≤ 1
        }
        """
        x1 = x[:, 0:1]
        
        mask_region1 = (x1 > 0) & (x1 <= 0.5)
        
        # In region 1, y* = zd
        zd_val = self.compute_zd(x)
        
        result = torch.zeros_like(x1)
        result = torch.where(mask_region1, zd_val, result)
        
        return result
    
    def compute_u_star(self, x):
        """
        Compute the control variable u*(x1,x2)
        
        u*(x1,x2) = {
            1*zd(x1,x2)   if 0 < x1 ≤ 0.5
            0               if 0.5 < x1 ≤ 1
        }
        """
        x1 = x[:, 0:1]
        
        mask_region1 = (x1 > 0) & (x1 <= 0.5)
        
        # In region 1, u* = zd
        zd_val = self.compute_zd(x)
        u_region1 = zd_val
        
        result = torch.zeros_like(x1)
        result = torch.where(mask_region1, u_region1, result)
        
        return result
    
    def compute_xi_star(self, x):
        """
        Compute the multiplier ξ(x1,x2)
        
        ξ(x1,x2) = {
            0                   if 0 < x1 ≤ 0.5
            200(x1-0.5)         if 0.5 < x1 ≤ 1
        }
        """
        x1 = x[:, 0:1]
        
        mask_region2 = (x1 > 0.5) 
        # In region 2, ξ = C(x1-0.5)
        xi_region2 = 200*(x1 - 0.5)
        
        result = torch.zeros_like(x1)
        result = torch.where(mask_region2, xi_region2, result)
        
        return result
    
    def compute_f(self, x):
        """
        Compute the forcing term f(x1,x2)
        
        f(x1,x2) = {
            -Δzd-zd         if 0 < x1 ≤ 0.5
            -200(x1-0.5)       if 0.5 < x1 ≤ 1
        }
        """
        x1 = x[:, 0:1]
        
        mask_region1 = (x1 > 0) & (x1 <= 0.5)
        mask_region2 = (x1 > 0.5) & (x1 <= 1)
        
        zd_val = self.compute_zd(x)
        laplacian_zd_val = self.laplacian_zd(x)
        
        # Region 1: f = -Δzd - 1zd
        f_region1 = -laplacian_zd_val -  zd_val
        
        # Region 2: f = -200(x1-0.5)
        f_region2 = -200* (x1 - 0.5)
        
        result = torch.zeros_like(x1)
        result = torch.where(mask_region1, f_region1, result)
        result = torch.where(mask_region2, f_region2, result)
        
        return result
    
    def compute_yd(self, x):
        """
        Compute the desired state yd(x1,x2)
        
        yd(x1,x2) = {
            zd-Δzd      if 0 < x1 ≤ 0.5
            200(x1-0.5)        if 0.5 < x1 ≤ 1
        }
        where nu=1
        """
        x1 = x[:, 0:1]
        
        mask_region1 = (x1 > 0) & (x1 <= 0.5)
        mask_region2 = (x1 > 0.5) & (x1 <= 1)
        
        zd_val = self.compute_zd(x)
        laplacian_zd_val = self.laplacian_zd(x)
        
        # Region 1: yd = zd + 100·0.1Δzd (note nu=100)
        yd_region1 = zd_val - laplacian_zd_val
        
        # Region 2: yd = 200(x1-0.5)
        yd_region2 = 200 * (x1 - 0.5)
        
        result = torch.zeros_like(x1)
        result = torch.where(mask_region1, yd_region1, result)
        result = torch.where(mask_region2, yd_region2, result)
        
        return result
    
class Source2:
    def __init__(self, device='cpu'):
        """
        Source terms and exact solutions for the optimal control obstacle problem from Meyer and Thoma, 2013.
        
        Parameters:
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        
        # Define rotation matrix Q (30 degrees counter-clockwise)
        theta = np.pi/6
        self.Q = torch.tensor([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]], device=device).float()
        
        # Midpoint of Ω1
        self.x_hat = torch.tensor([0.8, 0.9], device=device)
        self.analystic = True

    # Inverse transform from the omega region to the original coordinates
    def transform_coords(self, x):
        """Transform coordinates for Ω1 region using rotation Q"""
        return torch.matmul(self.Q.T, (x - self.x_hat).T).T + self.x_hat
    
    def z1(self, x1):
        """Polynomial function z1"""
        return (-4096 * x1**6 + 6144 * x1**5 - 3072 * x1**4 + 512 * x1**3)
    
    def z2(self, x2):
        """Polynomial function z2"""
        return (-244.140625 * x2**6 + 585.9375 * x2**5 - 
                468.75 * x2**4 + 125 * x2**3)
    
    def z1_double_prime(self, x1):
        """Second derivative of z1"""
        return (-122880 * x1**4 + 122880 * x1**3 - 
                36864 * x1**2 + 3072 * x1)
    
    def z2_double_prime(self, x2):
        """Second derivative of z2"""
        return (-7324.21875 * x2**4 + 11718.75 * x2**3 - 
                5625 * x2**2 + 750 * x2)
    
    def q1(self, y1):
        """Quadratic function q1"""
        return -0.5 * (y1 - 0.8)**2 + 0.5
    
    def q2(self, y2):
        """Quadratic function q2"""
        return -0.5 * (y2 - 0.9)**2 + 0.5
    
    
    def p1(self, y):
        """Function p1"""
        return self.q1(y[:, 0:1]) * self.q2(y[:, 1:2])
    
    def in_omega1(self, x):
        """Check if point x is in the rotated square Ω1"""
        # Transform coordinates to the rotated frame
        x_trans = self.transform_coords(x)
        # Check if within the original square before rotation
        return ((torch.abs(x_trans[:, 0] - 0.8) <= 0.05) & 
                (torch.abs(x_trans[:, 1] - 0.9) <= 0.05)).float().unsqueeze(1)
    
    def in_omega2(self, x):
        """Check if point x is in Ω2 = (0.0,0.5)×(0.0,0.8)"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ((x1 > 0.0) & (x1 <= 0.5) & (x2 > 0.0) & (x2 <= 0.8)).float().unsqueeze(1)
    
    def in_omega3(self, x):
        """Check if point x is in Ω3 = (0.5,1.0)×(0.0,0.8)"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ((x1 > 0.5) & (x1 <= 1.0) & (x2 > 0.0) & (x2 <= 0.8)).float().unsqueeze(1)
    
    def in_biactive_set(self, x):
        """Check if point x is in the bi-active set B = (0.0,1.0)×(0.8,1.0)"""
        x2 = x[:, 1]
        return (x2 > 0.8).float().unsqueeze(1)
    
    def compute_yd(self, x):
        """Desired state y_d from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω1 case
        mask1 = self.in_omega1(x)
        y_trans = self.transform_coords(x)
        q1_val = self.q1(y_trans[:, 0:1])
        q2_val = self.q2(y_trans[:, 1:2])
        omega1_val = -1* (q1_val + q2_val)
        result = torch.where(mask1.bool(), omega1_val, result)
        
        # Ω2 case
        mask2 = self.in_omega2(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        omega2_val = self.z1(x1) * self.z2(x2)
        result = torch.where(mask2.bool(), omega2_val, result)
        
        return result
    
    def compute_ud(self, x):
        """Desired control u_d from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω1 case
        mask1 = self.in_omega1(x)
        y_trans = self.transform_coords(x)
        omega1_val = self.p1(y_trans)
        result = torch.where(mask1.bool(), omega1_val, result)
        
        # Ω2 case
        mask2 = self.in_omega2(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        omega2_val = -self.z1_double_prime(x1) - self.z2_double_prime(x2)
        result = torch.where(mask2.bool(), omega2_val, result)
        
        # Ω3 case
        mask3 = self.in_omega3(x)
        x1_shifted = x[:, 0:1] - 0.5
        x2 = x[:, 1:2]
        omega3_val = -self.z1(x1_shifted) * self.z2(x2)
        result = torch.where(mask3.bool(), omega3_val, result)
        
        return result
    
    def compute_y_star(self, x):
        """Exact state solution y* from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω2 case
        mask2 = self.in_omega2(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        omega2_val = self.z1(x1) * self.z2(x2)
        result = torch.where(mask2.bool(), omega2_val, result)
        
        return result
    
    def compute_u_star(self, x):
        """Exact control solution u* from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω2 case
        mask2 = self.in_omega2(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        omega2_val = -self.z1_double_prime(x1) - self.z2_double_prime(x2)
        result = torch.where(mask2.bool(), omega2_val, result)
        
        # Ω3 case
        mask3 = self.in_omega3(x)
        x1_shifted = x[:, 0:1] - 0.5
        x2 = x[:, 1:2]
        omega3_val = -self.z1(x1_shifted) * self.z2(x2)
        result = torch.where(mask3.bool(), omega3_val, result)
        
        return result
    
    def compute_xi_star(self, x):
        """Exact slack variable ξ* from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω3 case
        mask3 = self.in_omega3(x)
        x1_shifted = x[:, 0:1] - 0.5
        x2 = x[:, 1:2]
        omega3_val = self.z1(x1_shifted) * self.z2(x2)
        result = torch.where(mask3.bool(), omega3_val, result)
        
        return result
    
    def compute_p_star(self, x):
        """Exact adjoint state p* from the problem description"""
        result = torch.zeros_like(x[:, 0:1])
        
        # Ω1 case
        mask1 = self.in_omega1(x)
        y_trans = self.transform_coords(x)
        omega1_val = self.p1(y_trans)
        result = torch.where(mask1.bool(), omega1_val, result)
        
        return result


import torch

class Source3:
    def __init__(self, device='cpu'):
        """
        Source terms for the obstacle problem with specified parameters
        
        Parameters:
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        self.alpha = 1.0  # Regularization parameter
        self.analystic = False
    def compute_yd(self, x):
        """
        Compute desired state y_d(x) = {
            -1       if ||x|| ≥ 0.1
            1-100x₁²-50x₂²  else
        }
        """
        norm_x = torch.norm(x, dim=1, keepdim=True)
        mask = norm_x >= 0.1
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        return torch.where(mask, 
                          torch.full_like(x1, -1.0),
                          1 - 100*x1**2 - 50*x2**2)
    
    def compute_f(self, x):
        """Compute source term f(x) = 0.5 + 0.5(x₁ - x₂)"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 0.5 + 0.5*(x1 - x2)
    

class Source3:
    def __init__(self, device='cpu'):
        """
        Source terms for the obstacle problem with specified parameters
        
        Parameters:
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        self.alpha = 1.0  # Regularization parameter
        self.analystic = False
    def compute_yd(self, x):
        """
        Compute desired state y_d(x) = {
            -1       if ||x|| ≥ 0.1
            1-100x₁²-50x₂²  else
        }
        """
        norm_x = torch.norm(x, dim=1, keepdim=True)
        mask = norm_x >= 0.1
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        return torch.where(mask, 
                          torch.full_like(x1, -1.0),
                          1 - 100*x1**2 - 50*x2**2)
    
    def compute_f(self, x):
        """Compute source term f(x) = 0.5 + 0.5(x₁ - x₂)"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 0.5 + 0.5*(x1 - x2)
    

class Source4:
    def __init__(self, device='cpu', nu=0.01):
        """
        Source terms for the degenerate solution example (Example 6.2)
        
        Parameters:
            device: Computation device ('cpu' or 'cuda')
            nu: Regularization parameter (default 0.01 as specified)
        """
        self.device = device
        self.nu = nu
        self.analystic = False

    def compute_yd(self, x):
        """
        Compute desired state y_d(x1,x2) = -|x1x2 - 0.5| + 0.25
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return -torch.abs(x1 * x2 - 0.5) + 0.25

    def compute_f(self, x):
        """
        Source term f(x1,x2) = y_d(x1,x2) = -|x1x2 - 0.5| + 0.25
        (Same as desired state in this example)
        """
        return self.compute_yd(x)

class Source5:
    def __init__(self, device='cpu',scale=1.0):
        """
        Source terms for the test problem with specified y_dagger and xi_dagger
        
        Parameters:
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        self.nu = 1.0  # Control weight (using standard value)
        self.analystic = True  # This example has analytical solutions
        self.scale = scale
    def compute_y_dagger(self, x):
        """
        Compute y_dagger function:
        y_dagger(x1,x2) = {
            160(x1^3-x1^2+0.25x1)(x2^3-x2^2+0.25x2)  in (0,0.5)×(0,0.5)
            0                                         else
        }
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # Create mask for (0,0.5)×(0,0.5) region
        mask = (x1 > 0) & (x1 < 0.5) & (x2 > 0) & (x2 < 0.5)
        
        p1 = 160 * (x1**3 - x1**2 + 0.25*x1)
        p2 = (x2**3 - x2**2 + 0.25*x2)
        y_value = p1 * p2
        
        return self.scale*torch.where(mask, y_value, torch.zeros_like(x1))
    
    def compute_xi_dagger(self, x):
        """
        Compute xi_dagger function:
        xi_dagger(x1,x2) = max(0, -2|x1-0.8| - 2|x1*x2-0.3| + 0.5)
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        term1 = -2 * torch.abs(x1 - 0.8)
        term2 = -2 * torch.abs(x1 * x2 - 0.3)
        
        xi_value = term1 + term2 + 0.5
        
        return torch.relu(xi_value)
    
    def laplacian_y_dagger(self, x):
        """
        Compute Laplacian of y_dagger:
        Δy_dagger = ∂²y_dagger/∂x1² + ∂²y_dagger/∂x2²
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # Create mask for (0,0.5)×(0,0.5) region
        mask = (x1 > 0) & (x1 < 0.5) & (x2 > 0) & (x2 < 0.5)
        
        # d²p1/dx1² = 160 * (6*x1 - 2)
        d2p1_dx1 = 160 * (6*x1 - 2)
        # d²p2/dx2² = 6*x2 - 2
        d2p2_dx2 = 6*x2 - 2
        
        p1 = 160*(x1**3 - x1**2 + 0.25*x1)
        p2 = (x2**3 - x2**2 + 0.25*x2)
        
        # Laplacian = d²p1/dx1² * p2 + p1 * d²p2/dx2²
        laplacian = d2p1_dx1 * p2 + p1 * d2p2_dx2
        
        return self.scale*torch.where(mask, laplacian, torch.zeros_like(x1))
    
    def compute_f(self, x):
        """
        Compute source term f:
        f = -Δy_dagger - y_dagger - xi_dagger
        """
        y_dagger = self.compute_y_dagger(x)
        xi_dagger = self.compute_xi_dagger(x)
        laplacian_y = self.laplacian_y_dagger(x)
        
        return -laplacian_y - y_dagger - xi_dagger
    
    def compute_yd(self, x):
        """
        Compute desired state y_d:
        y_d = y_dagger + xi_dagger - v * Δy_dagger
        where v = 1
        """
        y_dagger = self.compute_y_dagger(x)
        xi_dagger = self.compute_xi_dagger(x)
        laplacian_y = self.laplacian_y_dagger(x)
        
        return y_dagger + xi_dagger - self.nu * laplacian_y
    
    def compute_y_star(self, x):
        """
        Compute exact solution for state y:
        y_star = y_dagger
        """
        return self.compute_y_dagger(x)
    
    def compute_u_star(self, x):
        """
        Compute exact solution for control u:
        u_star = y_dagger
        """
        return self.compute_y_dagger(x)
    
    def compute_xi_star(self, x):
        """
        Compute exact solution for multiplier xi:
        xi_star = xi_dagger
        """
        return self.compute_xi_dagger(x)
    
class Source6:
    def __init__(self, device='cpu'):
        """
        First obstacle problem example (Section 6.1)
        
        Domain: Ω = (0,1)², with homogeneous Dirichlet boundary conditions
        Obstacle: ψ ≡ -0.25
        Body force: f = -2π²sin(πx)sin(πy)
        Desired state: u_d = -sin(πx)sin(πy) in Ω₁=(0.375,0.625)²
        Regularization parameter: α = 1
        
        Optimal control: q ≡ 0
        Optimal value: J(q,u) = (5π²+16π√2+128√2-224)/(512π²)
        """
        self.device = device
        self.alpha = 1.0  # Regularization parameter
        self.analystic = True  # Analytical solution available
        self.nu = self.alpha  # Compatibility with other code
        self.psi = -0.25  # Obstacle value
        
    def in_omega1(self, x):
        """
        Check whether point x lies in the subdomain Ω₁ = (0.375,0.625)²
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return ((x1 > 0.375) & (x1 < 0.625) & 
                (x2 > 0.375) & (x2 < 0.625)).float()
        
    def compute_f(self, x):
        """
        Compute the body force f = -2π²sin(πx)sin(πy)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -2 * (torch.pi**2) * torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
    
    def compute_ud_omega1(self, x):
        """
        Compute the desired state u_d = -sin(πx)sin(πy) in Ω₁
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
    
    def compute_yd(self, x):
        """
        Compute the desired state over the whole domain (0 outside Ω₁)
        """
        omega1_mask = self.in_omega1(x)
        ud_omega1 = self.compute_ud_omega1(x)
        return ud_omega1 * omega1_mask
    
    def compute_y_star(self, x):
        """
        Compute the optimal state
        
        The obstacle value ψ = -0.25 in Ω₁
        Outside Ω₁, solve the PDE: -Δu = f, with homogeneous Dirichlet boundary conditions
        """
        omega1_mask = self.in_omega1(x)
        
        # In Ω₁, the optimal state equals the obstacle value
        u_omega1 = torch.ones_like(x[:, 0:1]) * self.psi
        
        # Outside Ω, the optimal state is determined by the PDE
        # The solution is sin(πx)sin(πy)
        x1, x2 = x[:, 0:1], x[:, 1:2]
        u_outside = -torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)
        
        return torch.relu(-torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)+0.25)-0.25
    
    def compute_u_star(self, x):
        """
        Compute the optimal control, which is identically zero
        """
        return torch.zeros_like(x[:, 0:1])
    
    def compute_xi_star(self, x):
        """
        Compute the multiplier ξ
        
        ξ>0 on the active set (u=ψ), and ξ=0 elsewhere
        """
        omega1_mask = self.in_omega1(x)
        # In Ω₁, compute ξ as the PDE residual
        x1, x2 = x[:, 0:1], x[:, 1:2]
        f_val = self.compute_f(x)
        xi_val = -f_val - self.compute_ud_omega1(x)
        return torch.where(omega1_mask.bool(), xi_val, torch.zeros_like(xi_val))
    
    def compute_optimal_value(self):
        """
        Compute the optimal value J(q,u) = (5π²+16π√2+128√2-224)/(512π²)
        """
        pi = torch.pi
        return (5*pi**2 + 16*pi*torch.sqrt(torch.tensor(2.0)) + 
                128*torch.sqrt(torch.tensor(2.0)) - 224) / (512 * pi**2)
#--------------net----------------

class net_y(nn.Module):
    def __init__(self, source, hidden=64, layers=3, init=True, activation_flag=True, hard_ = None):
        super(net_y, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fcs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 2)])
        self.fc3 = nn.Linear(hidden, 1)
        self.hard = hard_
        self.activation_flag = activation_flag
        if activation_flag:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        if self.hard == None:
            hard = x1 * (1 - x1) * x2 * (1 - x2)
        else:
            hard = self.hard(x1,x2)
        x = self.activation(self.fc1(x))
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.fc3(x)
        x = x * hard
        x = torch.relu(x)
        
        return x


class net_xi(nn.Module):
    def __init__(self, source, hidden=64, layers=3, init=True, activation_flag=True):
        super(net_xi, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fcs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 2)])
        self.fc3 = nn.Linear(hidden, 1)
        
        self.activation_flag = activation_flag
        if activation_flag:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x = self.activation(self.fc1(x))
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.fc3(x)
        x = torch.relu(x)
        return x
    
class net_v(nn.Module):
    def __init__(self, source, hidden=64, layers=3, init=True, activation_flag=True):
        super(net_v, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fcs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 2)])
        self.fc3 = nn.Linear(hidden, 1)
        
        self.activation_flag = activation_flag
        if activation_flag:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.fc3(x)
        return x

class SquaredReLU(nn.Module):
    """
    Custom activation function: (max(0, x))² = (ReLU(x))²
    """
    def __init__(self):
        super(SquaredReLU, self).__init__()
        
    def forward(self, x):
        return torch.relu(x) ** 2  # (max(0, x))²
    
    def extra_repr(self):
        return "SquaredReLU"
    
class CubicReLU(nn.Module):
    """
    Custom activation function: (max(0, x))^3 = (ReLU(x))^3
    """
    def __init__(self):
        super(CubicReLU, self).__init__()
        
    def forward(self, x):
        return torch.relu(x) ** 3  # (max(0, x))^3
    
    def extra_repr(self):
        return "CubicReLU"

class ResidualBlock(nn.Module):
    """
    Residual Block implementation:
    ResidualBlock(x) = activation(linear2(activation(linear1(x))) + x)
    """
    def __init__(self, dim, activation=SquaredReLU()):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + identity
        out = self.activation(out)
        return out



class net_y_res(nn.Module):
    def __init__(self, source, hidden=64, layers=3, init=True, activation='Tanh', hard_=None, obstacle=None):
        super(net_y_res, self).__init__()
        
        self.activation = activation
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Identity':
            self.activation = nn.Identity()
        elif activation == 'Square':
            self.activation = SquaredReLU()
        elif activation == 'Cubic':
            self.activation = CubicReLU()
        elif activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'Swish':
            self.activation = nn.SiLU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        
        
        # Define padding size and apply zero-padding to inputs
        self.padding_size = hidden - 2  # Padding size is hidden - 2 based on your code
        
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden, self.activation) for _ in range(layers - 1)]
        )
        
        self.output_layer = nn.Linear(hidden, 1)
        
        self.hard = hard_
            
        self.obstacle = obstacle  
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):        
        padded_x = torch.cat([x, torch.zeros(x.size(0), self.padding_size).to(x.device)], dim=1)
        
        # Apply hard constraint (if provided)
        x1 = padded_x[:, 0:1]
        x2 = padded_x[:, 1:2]
        
        if self.hard is None:
            hard = x1 * (1 - x1) * x2 * (1 - x2)
        else:
            hard = self.hard(x1, x2)
        
        out = padded_x
        for res_block in self.res_blocks:
            out = res_block(out)
        
        out = self.output_layer(out)
        
        # Apply hard constraint and ensure obstacle constarint
        out = out**2
        
        out = out * hard

        if self.obstacle is not None:
            out = out + self.obstacle(x1,x2)

        return out



class net_v_res(nn.Module):
    def __init__(self, source, hidden=64, layers=3, init=True, activation='Tanh', hard_=None,constraint=None):
        
        super(net_v_res, self).__init__()

        self.activation = activation
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Identity':
            self.activation = nn.Identity()
        elif activation == 'Square':
            self.activation = SquaredReLU()
        elif activation == 'Cubic':
            self.activation = CubicReLU()
        elif activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'Swish':
            self.activation = nn.SiLU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()

        # Pad the 2-dimensional input x with hidden - 2 zeros to match the hidden dimension
        self.padding_size = hidden - 2
        self.hard = hard_
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden, activation=self.activation) for _ in range(layers - 1)]
        )

        self.output_layer = nn.Linear(hidden, 1)
        self.constraint = constraint

        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        padded_x = torch.cat([x, torch.zeros(x.size(0), self.padding_size).to(x.device)], dim=1)

        out = padded_x
        for res_block in self.res_blocks:
            out = res_block(out)

        out = self.output_layer(out)
        if self.hard :
            x1 = padded_x[:, 0:1]
            x2 = padded_x[:, 1:2]
            hard = self.hard(x1, x2)
            out = out * hard
        if self.constraint:
            out = torch.clamp(out, min=self.constraint[0], max=self.constraint[1])
        return out
#--------------visualization----------------

def visual_y(source, net_y, file_name=None):
    device = next(net_y.parameters()).device
    # Sample over [0,1]
    x = torch.linspace(0, 1, 200, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    xy = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)), dim=1)

    y_star = source.compute_y_star(xy).reshape(200, 200).detach().cpu()

    y_app = net_y(xy).reshape(200, 200).detach().cpu()

    diff = y_star - y_app

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    Xc, Yc = X.cpu(), Y.cpu()
    im0 = axs[0].contourf(Xc, Yc, y_star, levels=50, cmap="jet")
    im1 = axs[1].contourf(Xc, Yc, y_app, levels=50, cmap="jet")
    im2 = axs[2].contourf(Xc, Yc, diff, levels=50, cmap="jet")

    axs[0].set_title("y_star")
    axs[1].set_title("y (Approx)")
    axs[2].set_title("Difference (y_star - y)")

    plt.colorbar(im0, ax=axs[0])
    plt.colorbar(im1, ax=axs[1])
    plt.colorbar(im2, ax=axs[2])

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches='tight')


def visual_y_3d(source, net_y, file_name=None):
    device = next(net_y.parameters()).device
    x = torch.linspace(0, 1, 200, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    xy = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)), dim=1)

    y_app = net_y(xy).reshape(200, 200).detach().cpu()
    
    Xc = X.cpu().numpy()
    Yc = Y.cpu().numpy()
    y_app_np = y_app.numpy()
    
    has_analytic = hasattr(source, 'analystic') and source.analystic

    if has_analytic:
        fig = plt.figure(figsize=(20, 5))
        
        y_star = source.compute_y_star(xy).reshape(200, 200).detach().cpu()
        diff = y_star - y_app
        y_star_np = y_star.numpy()
        diff_np = diff.numpy()

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(Xc, Yc, y_star_np, cmap='jet', rstride=2, cstride=2)
        ax1.set_title("y_star")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(Xc, Yc, y_app_np, cmap='jet', rstride=2, cstride=2)
        ax2.set_title("y")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        surf3 = ax3.plot_surface(Xc, Yc, diff_np, cmap='jet', rstride=2, cstride=2)
        ax3.set_title("Difference (y_star - y)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)

        for ax in [ax1, ax2, ax3]:
            ax.invert_xaxis()  # Reverse the x-axis to move x=0 from left to right
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))
            ax.zaxis.set_major_formatter(formatter)
            
    else:
        fig = plt.figure(figsize=(10, 8))
        
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xc, Yc, y_app_np, cmap='jet', rstride=2, cstride=2)
        ax.set_title("Neural Network Solution y")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        
        ax.invert_xaxis()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))
        ax.zaxis.set_major_formatter(formatter)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")




def visual_v_3d(source, net_v, file_name=None):
    """
    Sample (x, y) on a grid over [0,1]×[0,1] and plot 3D surfaces of the numerical
    control solution and, if an analytical solution is available, the exact solution and their difference.
    """
    device = next(net_v.parameters()).device
    n_points = 198
    # Generate a grid over (0,1)
    x = torch.linspace(0, 1, 200, device=device)
    x = x[1:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    xy = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)

    v_app = net_v(xy).reshape(n_points, n_points).detach().cpu()
    
    Xc = X.cpu().numpy()
    Yc = Y.cpu().numpy()
    v_app_np = v_app.numpy()
    
    has_analytic = hasattr(source, 'analystic') and source.analystic

    if has_analytic:
        fig = plt.figure(figsize=(20, 5))
        
        v_star = source.compute_u_star(xy).reshape(n_points, n_points).detach().cpu()
        diff = v_star - v_app
        v_star_np = v_star.numpy()
        diff_np = diff.numpy()

        # v_star
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(Xc, Yc, v_star_np, cmap='jet', rstride=2, cstride=2)
        ax1.set_title("u_star")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

        # v_app
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(Xc, Yc, v_app_np, cmap='jet', rstride=2, cstride=2)
        ax2.set_title("u_app")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        surf3 = ax3.plot_surface(Xc, Yc, diff_np, cmap='jet', rstride=2, cstride=2)
        ax3.set_title("Difference (u_star - u)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
        
        for ax in [ax1, ax2, ax3]:
            ax.invert_xaxis()  # Reverse the x-axis at x=0
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))
            ax.zaxis.set_major_formatter(formatter)
            
    else:
        fig = plt.figure(figsize=(10, 8))
        
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xc, Yc, v_app_np, cmap='jet', rstride=2, cstride=2)
        ax.set_title("Neural Network Solution u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        
        ax.invert_xaxis()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))
        ax.zaxis.set_major_formatter(formatter)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")

def visual_xi_3d(source, y_net, v_net, resolution=200, device='cpu', file_name=None, get_xi=None):
    """
    Plot the multiplier xi in 3D, computed using the supplied get_xi function or the default method
    
    Args:
        source: Source object with a method for computing f
        y_net: Neural network for the state variable y
        v_net: Neural network for the control variable v
        resolution: Grid resolution
        device: Computation device
        file_name: File name for saving the figure
        get_xi: Optional custom function for computing xi, taking (source, y_net, v_net, x_input)
    """
    y_net.eval()
    v_net.eval()
    device = next(y_net.parameters()).device
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    grid_points = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T, 
                              dtype=torch.float32, device=device)
    
    grid_points = grid_points.clone().detach().requires_grad_(True)
    x1 = grid_points[:, 0:1]
    x2 = grid_points[:, 1:2]
    x_input = torch.cat([x1, x2], 1)
    
    if get_xi is not None:
        xi_val = get_xi(source, y_net, v_net, x_input)

    else:
        y_pred = y_net(x_input)
        v_pred = v_net(x_input)
        
        # Compute f(x), trying both supported method names
        try:
            f_val = source.f(x_input)
        except AttributeError:
            try:
                f_val = source.compute_f(x_input)
            except AttributeError:
                raise AttributeError("Source has neither an 'f' method nor a 'compute_f' method")
        
        # Compute -Δy
        y_x1 = grad(y_pred.sum(), x1, create_graph=True)[0]
        y_x2 = grad(y_pred.sum(), x2, create_graph=True)[0]
        
        d2y_dx1 = grad(y_x1.sum(), x1, create_graph=True)[0]
        d2y_dx2 = grad(y_x2.sum(), x2, create_graph=True)[0]
        
        neg_lap_y = -(d2y_dx1 + d2y_dx2)
        
        xi_val = neg_lap_y - v_pred - f_val
    
    xi_np = xi_val.detach().cpu().numpy().reshape(resolution, resolution)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, xi_np, cmap='jet', rstride=2, cstride=2)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_title('Multiplier $\\xi(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\xi$')
    
    ax.invert_xaxis()  # Reverse the x-axis at x=0
    # To adjust the view, use ax.view_init(elev=30, azim=50)
    formatter = ScalarFormatter(useMathText=True) 
    formatter.set_powerlimits((-1, 1))
    ax.zaxis.set_major_formatter(formatter)
    
    if file_name:
        plt.savefig(file_name)
    
    plt.show()
    
    return fig





def compute_energy(source, net_y=None, net_u=None, grid_size=50, device='cpu', requires_grad=False):
    """
    Compute the upper- and lower-level energies of a model or the exact solution using exact gradients
    
    Parameters:
        source: Source object providing the required computation functions
        net_y: Neural network model for the state variable y (use the exact solution if None)
        net_u: Neural network model for the control variable u (use the exact solution if None)
        grid_size: Grid resolution
        device: Computation device ('cpu' or 'cuda')
        requires_grad: Whether to retain gradient computation (default False)
    
    Returns:
        lower_energy: Lower-level energy J(w, u) = 1/2*a(w,w) - <u+f,w>
        upper_energy: Upper-level energy (1/2)||y-y_d||² + (ν/2)||u||²
    """
    x_grid = torch.linspace(0, 1, grid_size, device=device)
    y_grid = torch.linspace(0, 1, grid_size, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    points = points.clone().detach().requires_grad_(True)
    
    try:
        f_val = source.compute_f(points)
    except AttributeError:
        try:
            f_val = source.f(points)
        except AttributeError:
            raise AttributeError("Source has neither a 'compute_f' method nor an 'f' method")
    
    try:
        y_d = source.compute_yd(points)
    except AttributeError:
        try:
            y_d = source.yd(points)
        except AttributeError:
            try:
                y_d = source.compute_zd(points)
            except AttributeError:
                raise AttributeError("Source has no available desired-state method")
    
    if net_u is None:
        try:
            u_val = source.compute_u_star(points)
        except AttributeError:
            try:
                u_val = source.u_star(points)
            except AttributeError:
                raise AttributeError("Source has no available u_star method")
    else:
        u_val = net_u(points)
    
    if net_y is None:
        # The exact solution needs special handling when gradients are required
        try:
            if hasattr(source, 'y_star_with_grad'):
                y_val, grad_y = source.y_star_with_grad(points)
            else:
                # Otherwise, ensure the source function supports gradient propagation
                y_val = source.compute_y_star(points) if hasattr(source, 'compute_y_star') else source.y_star(points)
                
                if requires_grad and not y_val.requires_grad:
                    raise RuntimeError("Source.y_star does not support automatic differentiation; provide an implementation that supports gradients")
                
                if requires_grad or 'grad_y' not in locals():
                    grad_y = torch.autograd.grad(
                        outputs=y_val.sum(),
                        inputs=points,
                        create_graph=requires_grad,
                        retain_graph=True
                    )[0]
        except Exception as e:
            raise RuntimeError(f"Error computing the gradient of the exact solution: {e}")
    else:
        net_y.eval()
        y_val = net_y(points)
        
        # Compute the gradient ∇y
        grad_y = torch.autograd.grad(
            outputs=y_val.sum(),
            inputs=points,
            create_graph=requires_grad,
            retain_graph=True
        )[0]
    
    # Upper-level energy (original objective)
    # Tracking error: (1/2)||y-y_d||²
    tracking_error = 0.5 * torch.mean((y_val - y_d)**2)
    
    # Control penalty: (ν/2)||u||²
    nu = source.nu
    control_penalty = 0.5 * nu * torch.mean(u_val**2)
    
    upper_energy = tracking_error + control_penalty
    
    # Lower-level energy (energy functional)
    # Dirichlet energy: (1/2)||∇y||²
    dirichlet_energy = 0.5 * torch.mean(grad_y[:, 0]**2 + grad_y[:, 1]**2)
    
    # Forcing term: -∫(f+u)y dx
    external_work = -torch.mean((f_val + u_val) * y_val)
    
    lower_energy = dirichlet_energy + external_work
    

    return lower_energy.item(), upper_energy.item()


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





#---------pretrain-------------



from functools import partial
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(current_step,warmup,cycle_factor,cycle_length):
    if current_step < warmup:
        return float(current_step) / float(max(1, warmup))
    else:
        cycle_multiplier = cycle_factor ** ((current_step - warmup) // cycle_length)
        return cycle_multiplier



def pretrain_y_net(net_y, source, dataloader, device, num_epochs=2000, lr=1e-3, 
                   warmup=500, cycle_factor=0.8, cycle_length=1000, save_dir=None):
    """
    Pretrain the state network y_net using the desired state yd and return the pretrained model
    
    Args:
        net_y: State network model
        source: Source2 instance providing the compute_yd method
        dataloader: Data loader
        device: Computation device
        num_epochs: Total number of pretraining epochs
        lr: Initial learning rate
        warmup: Number of warmup steps
        cycle_factor: Learning-rate cycle factor
        cycle_length: Learning-rate cycle length
        save_dir: Directory for saving the model
        
    Returns:
        net_y: Pretrained state network
    """
    print(f"Starting pretraining of the state network y_net to approximate yd...")
    
    optimizer_y = torch.optim.Adam(net_y.parameters(), lr=lr)
    scheduler_y = LambdaLR(optimizer_y, partial(lr_lambda, warmup=warmup, 
                                               cycle_factor=cycle_factor, 
                                               cycle_length=cycle_length))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        loss_file = os.path.join(save_dir, 'pretrain_y_loss.csv')
        with open(loss_file, 'w') as f:
            f.write("epoch,loss\n")
    
    pbar = tqdm(range(1, num_epochs + 1), ascii=True, dynamic_ncols=True)
    for epoch in pbar:
        net_y.train()
        
        total_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0].requires_grad_(True).to(device)
            
            y_pred = net_y(x)
            y_target = torch.relu(source.compute_yd(x))
            
            loss = torch.mean((y_pred - y_target) ** 2)
            
            optimizer_y.zero_grad()
            loss.backward()
            optimizer_y.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        scheduler_y.step()
        
        avg_loss = total_loss / max(batch_count, 1)
        pbar.set_postfix(y_pretrain_loss=f"{avg_loss:.6f}", lr=f"{optimizer_y.param_groups[0]['lr']:.6f}")
        
        if save_dir and epoch % 10 == 0:
            with open(loss_file, 'a') as f:
                f.write(f"{epoch},{avg_loss}\n")
        
        # Periodically clear GPU memory
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if save_dir:
        torch.save(net_y.state_dict(), os.path.join(save_dir, 'pretrain_y_final.pth'))
    
    return net_y


def pretrain_u_net(net_u, source, dataloader, device, num_epochs=2000, lr=1e-3, 
                   warmup=500, cycle_factor=0.8, cycle_length=1000, save_dir=None):
    """
    Pretrain the control network u_net using the desired control ud and return the pretrained model
    
    Args:
        net_u: Control network model
        source: Source2 instance providing the compute_ud method
        dataloader: Data loader
        device: Computation device
        num_epochs: Total number of pretraining epochs
        lr: Initial learning rate
        warmup: Number of warmup steps
        cycle_factor: Learning-rate cycle factor
        cycle_length: Learning-rate cycle length
        save_dir: Directory for saving the model
        
    Returns:
        net_u: Pretrained control network
    """
    print(f"Starting pretraining of the control network u_net to approximate ud...")
    
    optimizer_u = torch.optim.Adam(net_u.parameters(), lr=lr)
    scheduler_u = LambdaLR(optimizer_u, partial(lr_lambda, warmup=warmup, 
                                               cycle_factor=cycle_factor, 
                                               cycle_length=cycle_length))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        loss_file = os.path.join(save_dir, 'pretrain_u_loss.csv')
        with open(loss_file, 'w') as f:
            f.write("epoch,loss\n")
    
    pbar = tqdm(range(1, num_epochs + 1), ascii=True, dynamic_ncols=True)
    for epoch in pbar:
        net_u.train()
        
        total_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0].requires_grad_(True).to(device)
            
            u_pred = net_u(x)
            u_target = 10
            
            loss = torch.mean((u_pred - u_target) ** 2)
            
            optimizer_u.zero_grad()
            loss.backward()
            optimizer_u.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        scheduler_u.step()
        
        avg_loss = total_loss / max(batch_count, 1)
        pbar.set_postfix(u_pretrain_loss=f"{avg_loss:.6f}", lr=f"{optimizer_u.param_groups[0]['lr']:.6f}")
        
        if save_dir and epoch % 10 == 0:
            with open(loss_file, 'a') as f:
                f.write(f"{epoch},{avg_loss}\n")
        
        # Periodically clear GPU memory
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if save_dir:
        torch.save(net_u.state_dict(), os.path.join(save_dir, 'pretrain_u_final.pth'))
    
    return net_u

def J_l(source, net_y,x):
    """
    The lower-level state y minimizes the potential energy:
    J(w, u) = 1/2*a(w,w) - <f,w>
    Use MSELoss to approximate the integral.
    """
    x = x.clone().detach().requires_grad_(True)
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x = torch.cat([x1, x2], 1)
    f_val = source.compute_f(x)
    y_pred  = net_y(x)
    y_x1 = grad(y_pred.sum(), x1, create_graph=True)[0]
    y_x2 = grad(y_pred.sum(), x2, create_graph=True)[0]

    # Compute a(w,w)
    a_val = y_x1**2 + y_x2**2
    linear_term = (f_val)*y_pred

    return torch.mean(0.5*a_val-linear_term)

def pretrain_y_net_with_energy(net_y, source, dataloader, device, num_epochs=2000, lr=1e-3, 
                   warmup=500, cycle_factor=0.8, cycle_length=1000, save_dir=None):
    """
    Pretrain the state network y_net by minimizing the energy functional
    
    Args:
        net_y: State network model
        source: Source instance providing the compute_f method
        dataloader: Data loader
        device: Computation device
        num_epochs: Total number of pretraining epochs
        lr: Initial learning rate
        warmup: Number of warmup steps
        cycle_factor: Learning-rate cycle factor
        cycle_length: Learning-rate cycle length
        save_dir: Directory for saving the model
        
    Returns:
        net_y: Pretrained state network
    """
    print(f"Starting pretraining of the state network y_net to minimize the energy functional...")
    
    optimizer_y = torch.optim.Adam(net_y.parameters(), lr=lr)
    scheduler_y = LambdaLR(optimizer_y, partial(lr_lambda, warmup=warmup, 
                                               cycle_factor=cycle_factor, 
                                               cycle_length=cycle_length))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        loss_file = os.path.join(save_dir, 'pretrain_y_energy_loss.csv')
        with open(loss_file, 'w') as f:
            f.write("epoch,energy\n")
    
    pbar = tqdm(range(1, num_epochs + 1), ascii=True, dynamic_ncols=True)
    for epoch in pbar:
        net_y.train()
        
        total_energy = 0.0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0].requires_grad_(True).to(device)
            
            energy = J_l(source, net_y, x)
            
            optimizer_y.zero_grad()
            energy.backward()
            optimizer_y.step()
            
            total_energy += energy.item()
            batch_count += 1
        
        scheduler_y.step()
        
        avg_energy = total_energy / max(batch_count, 1)
        pbar.set_postfix(energy=f"{avg_energy:.6f}", lr=f"{optimizer_y.param_groups[0]['lr']:.6f}")
        
        if save_dir and epoch % 10 == 0:
            with open(loss_file, 'a') as f:
                f.write(f"{epoch},{avg_energy}\n")
        
        # Periodically clear GPU memory
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if save_dir:
        torch.save(net_y.state_dict(), os.path.join(save_dir, 'pretrain_y_energy_final.pth'))
    
    print(f"Energy-functional pretraining complete; final energy: {avg_energy:.6f}")
    return net_y



#-------------hard constraint----------------
def hard_constraint_L(x1, x2):
    """
    Hard constraint function using PyTorch operations for L-shaped domain.
    
    Args:
        x1: First coordinate tensor
        x2: Second coordinate tensor
        
    Returns:
        Tensor with hard constraint values
    """
    term1 = torch.minimum(1 + x1, 1 - x2)
    term2 = torch.maximum(x2, -x1)
    term3 = torch.maximum(x1, x2 + 1)
    term4 = torch.maximum(-x2, (1 - x1) / 2)
    
    result = term1 * term2 * term3 * term4 * 10
    
    return result


def hard_constraint_2d(x1, x2):
    return x1 * (1 - x1) * x2 * (1 - x2)


def rel_l2_error_yu(net_y, net_v, source, device='cpu', resolution=101):
    """
    Compute the relative L2 errors of y and v over [0,1]×[0,1]:
       rel_l2_y = ||y_app - y_ref|| / ||y_ref||
       rel_l2_v = ||v_app - v_ref|| / ||v_ref||
    """
    # Generate sample points over [0,1]
    x = torch.linspace(0, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    # Reshape to [N,2]
    xy = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)), dim=1)
    X, Y = torch.meshgrid(x[1:-1], x[1:-1], indexing='ij')
    with torch.no_grad():
        y_star = source.compute_y_star(xy)
        v_star = source.compute_u_star(xy)
    
    y_app = net_y(xy)  
    v_app = net_v(xy)  

    num_y = torch.sum((y_app - y_star) ** 2)
    den_y = torch.sum(y_star ** 2)
    num_v = torch.sum((v_app - v_star) ** 2)
    den_v = torch.sum(v_star ** 2)

    # Avoid division by 0; handle a zero reference solution as needed
    rel_l2_y = torch.sqrt(num_y / (den_y + 1e-30))
    rel_l2_v = torch.sqrt(num_v / (den_v + 1e-30))

    print(f"Relative L2 error for y: {rel_l2_y.item():.4e}")
    print(f"Relative L2 error for v: {rel_l2_v.item():.4e}")

    return rel_l2_y, rel_l2_v