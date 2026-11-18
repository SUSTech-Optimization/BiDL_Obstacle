import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


# ============================================================
# Seed & LR schedule (same logic as utils.py)
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_lambda(current_step, warmup, cycle_factor, cycle_length):
    if current_step < warmup:
        return float(current_step) / float(max(1, warmup))
    else:
        cycle_multiplier = cycle_factor ** ((current_step - warmup) // cycle_length)
        return cycle_multiplier


# ============================================================
# Boundary embedding m(x)
# ============================================================

def compute_m(x, d, mode="prod", eps=1e-8):
    """
    m(x) = 4^d * prod_{i=1}^d x_i*(1-x_i)

    mode="prod": direct product (default, numerically stable for moderate d)
    mode="log":  log-sum-exp trick, avoids underflow for large d
    """
    t = x * (1.0 - x)  # [B, d]
    if mode == "prod":
        return (4.0 ** d) * torch.prod(t, dim=1, keepdim=True)
    elif mode == "log":
        log_m = d * math.log(4.0) + torch.sum(torch.log(t + eps), dim=1, keepdim=True)
        return torch.exp(log_m)
    else:
        raise ValueError(f"Unknown m.mode: {mode!r}. Choose 'prod' or 'log'.")


# ============================================================
# SourceHighDim10  (sin product, keeps original implementation)
# ============================================================

class SourceHighDim10:
    """10-dimensional obstacle optimal control problem with analytic solution.
    P(x) = prod_{i=1}^d sin(pi * x_i)
    """

    def __init__(self, device='cpu', d=10, sigma=1.0, c_thr=0.01):
        self.device = device
        self.d = d
        self.sigma = sigma
        self.c_thr = c_thr
        self.analytic = True

    # ---------- core helpers ----------

    def compute_P(self, x):
        """P(x) = prod_{i=1}^d sin(pi * x_i),  shape [B, 1]"""
        sin_vals = torch.sin(math.pi * x)  # [B, d]
        P = torch.prod(sin_vals, dim=1, keepdim=True)  # [B, 1]
        return P

    def compute_y_star(self, x):
        """y*(x) = (max(0, P(x) - c))^2"""
        P = self.compute_P(x)
        return torch.clamp(P - self.c_thr, min=0) ** 2

    def compute_xi_star(self, x):
        """xi*(x) = (max(0, c - P(x)))^2"""
        P = self.compute_P(x)
        return torch.clamp(self.c_thr - P, min=0) ** 2

    def compute_u_star(self, x):
        """u*(x) = y*(x)"""
        return self.compute_y_star(x)

    def compute_lap_y_star(self, x):
        """
        Laplacian of y*(x).
        If P(x) > c:
          lap = 2*pi^2 * P * [ P * sum_i cot^2(pi*x_i) - d*(P - c) ]
        else:
          lap = 0
        Uses mask indexing to avoid cot singularity.
        """
        d = self.d
        c = self.c_thr
        pi = math.pi

        sin_vals = torch.sin(pi * x)  # [B, d]
        P = torch.prod(sin_vals, dim=1, keepdim=True)  # [B, 1]

        B = x.shape[0]
        lap = torch.zeros(B, 1, device=x.device, dtype=x.dtype)

        mask = (P.squeeze(1) > c)  # [B]
        if mask.any():
            x_m = x[mask]          # [M, d]
            sin_m = torch.sin(pi * x_m)
            cos_m = torch.cos(pi * x_m)
            cot2_sum = torch.sum((cos_m / sin_m) ** 2, dim=1, keepdim=True)  # [M, 1]
            Pm = P[mask]           # [M, 1]
            lap[mask] = 2.0 * pi ** 2 * Pm * (Pm * cot2_sum - d * (Pm - c))

        return lap

    def compute_yd(self, x):
        """y_d(x) = y*(x) - lap_y*(x),  detached, no_grad."""
        with torch.no_grad():
            y_star = self.compute_y_star(x)
            lap = self.compute_lap_y_star(x)
            return (y_star - lap).detach()

    def compute_f(self, x):
        """f(x) = -lap_y*(x) - y*(x) - xi*(x),  detached, no_grad."""
        with torch.no_grad():
            lap = self.compute_lap_y_star(x)
            y_star = self.compute_y_star(x)
            xi_star = self.compute_xi_star(x)
            return (-lap - y_star - xi_star).detach()


# ============================================================
# Activation helper
# ============================================================

class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class CubicReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 3


def get_activation(name):
    name = name.lower()
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'swish' or name == 'silu':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'selu':
        return nn.SELU()
    elif name in ('square', 'squaredrelu'):
        return SquaredReLU()
    elif name in ('cubic', 'cubicrelu'):
        return CubicReLU()
    else:
        raise ValueError(f"Unknown activation: {name}")


# ============================================================
# SpinnScalarNet — Separable PINN scalar network
# ============================================================

class SpinnScalarNet(nn.Module):
    """
    SPINN: d independent 1D subnetworks, elementwise product, rank aggregation.

    Forward:
      F_i = subnet_i(x[:, i:i+1])   # [B, rank]
      T   = F_1 * F_2 * ... * F_d   # [B, rank]  elementwise product
      raw = T.sum(dim=1, keepdim=True)          if head="sum"
          = T @ w + b                           if head="linear"
    """

    def __init__(
        self,
        d=10,
        rank=4,
        hidden_1d=32,
        layers_1d=3,
        activation='swish',
        head='sum',
        learnable_head=True,
        init_last_bias=1.0,
        init_last_weight_std=1e-2,
        init_w='ones_over_r',
        init_b=0.0,
    ):
        super().__init__()
        self.d = d
        self.rank = rank
        self.head = head

        # Build d independent 1D subnetworks
        subnets = []
        for _ in range(d):
            subnet_layers = [nn.Linear(1, hidden_1d), get_activation(activation)]
            for _ in range(layers_1d - 1):
                subnet_layers.append(nn.Linear(hidden_1d, hidden_1d))
                subnet_layers.append(get_activation(activation))
            out_layer = nn.Linear(hidden_1d, rank)
            nn.init.normal_(out_layer.weight, std=init_last_weight_std)
            nn.init.constant_(out_layer.bias, init_last_bias)
            subnet_layers.append(out_layer)
            subnets.append(nn.Sequential(*subnet_layers))
        self.subnets = nn.ModuleList(subnets)

        # Head parameters (only used when head="linear")
        if head == 'linear':
            if init_w == 'ones_over_r':
                w_data = torch.ones(rank, 1) / rank
            elif init_w == 'ones':
                w_data = torch.ones(rank, 1)
            else:  # 'small_rand'
                w_data = torch.randn(rank, 1) * 0.01
            b_data = torch.full((1,), float(init_b))
            if learnable_head:
                self.w = nn.Parameter(w_data)
                self.b = nn.Parameter(b_data)
            else:
                self.register_buffer('w', w_data)
                self.register_buffer('b', b_data)
        else:
            self.w = None
            self.b = None

    def forward(self, x):
        # x: [B, d]
        T = None
        for i, subnet in enumerate(self.subnets):
            Fi = subnet(x[:, i:i+1])  # [B, rank]
            T = Fi if T is None else T * Fi
        # T: [B, rank]

        if self.head == 'sum':
            return T.sum(dim=1, keepdim=True)  # [B, 1]
        else:  # 'linear'
            return T @ self.w + self.b          # [B, 1]


# ============================================================
# net_y_spinn_hd — state network with boundary embedding (spinn)
# ============================================================

class net_y_spinn_hd(nn.Module):
    """
    State network for HD obstacle problem (SPINN).
    Output: y_hat = m(x) * embed(raw)
      embed="square": (raw)^2
      embed="relu":   relu(raw)  (recommended for SPINN)
    """

    def __init__(
        self,
        d=10,
        rank=4,
        hidden_1d=32,
        layers_1d=3,
        activation_1d='swish',
        head='sum',
        learnable_head=True,
        init_last_bias=1.0,
        init_last_weight_std=1e-2,
        init_w='ones_over_r',
        init_b=0.0,
        m_mode='prod',
        m_eps=1e-8,
        y_embed='square',
    ):
        super().__init__()
        self.d = d
        self.m_mode = m_mode
        self.m_eps = m_eps
        self.y_embed = y_embed
        self.spinn = SpinnScalarNet(
            d=d, rank=rank, hidden_1d=hidden_1d, layers_1d=layers_1d,
            activation=activation_1d, head=head, learnable_head=learnable_head,
            init_last_bias=init_last_bias, init_last_weight_std=init_last_weight_std,
            init_w=init_w, init_b=init_b,
        )

    def forward(self, x):
        m   = compute_m(x, self.d, self.m_mode, self.m_eps)  # [B, 1]
        raw = self.spinn(x)                                    # [B, 1]
        if self.y_embed == 'relu':
            return F.relu(raw) * m
        else:  # 'square'
            return raw ** 2 * m


# ============================================================
# net_u_spinn_hd — control network with boundary embedding (spinn)
# ============================================================

class net_u_spinn_hd(nn.Module):
    """
    Control network for HD obstacle problem (SPINN).
    Output: u_hat = m(x) * raw  (no non-negativity constraint)
    """

    def __init__(
        self,
        d=10,
        rank=4,
        hidden_1d=32,
        layers_1d=3,
        activation_1d='swish',
        head='sum',
        learnable_head=True,
        init_last_bias=1.0,
        init_last_weight_std=1e-2,
        init_w='ones_over_r',
        init_b=0.0,
        m_mode='prod',
        m_eps=1e-8,
    ):
        super().__init__()
        self.d = d
        self.m_mode = m_mode
        self.m_eps = m_eps
        self.spinn = SpinnScalarNet(
            d=d, rank=rank, hidden_1d=hidden_1d, layers_1d=layers_1d,
            activation=activation_1d, head=head, learnable_head=learnable_head,
            init_last_bias=init_last_bias, init_last_weight_std=init_last_weight_std,
            init_w=init_w, init_b=init_b,
        )

    def forward(self, x):
        m   = compute_m(x, self.d, self.m_mode, self.m_eps)  # [B, 1]
        raw = self.spinn(x)                                    # [B, 1]
        return raw * m


# ============================================================
# SobolDataLoader
# ============================================================

class SobolDataLoader:
    """Sobol QMC sampler on [0,1]^d."""

    def __init__(self, dim, batch_size, steps_per_epoch, device, dual_sampling=False):
        self.dim = dim
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.dual_sampling = dual_sampling
        self.engine = torch.quasirandom.SobolEngine(dim, scramble=True)
        if dual_sampling:
            self.engine2 = torch.quasirandom.SobolEngine(dim, scramble=True)
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        # Reinitialize each epoch with a fresh scramble.
        # Sobol's low-discrepancy benefit is within a single draw sequence;
        # with scramble=True each restart is independently randomized anyway.
        self.engine = torch.quasirandom.SobolEngine(self.dim, scramble=True)
        if self.dual_sampling:
            self.engine2 = torch.quasirandom.SobolEngine(self.dim, scramble=True)
        return self

    def __next__(self):
        if self.current_step >= self.steps_per_epoch:
            self.current_step = 0
            raise StopIteration
        self.current_step += 1

        x = self.engine.draw(self.batch_size).to(self.device)
        if self.dual_sampling:
            x2 = self.engine2.draw(self.batch_size).to(self.device)
            return x, x2
        return (x,)

    def __len__(self):
        return self.steps_per_epoch


# ============================================================
# Visualization: 2D slice at fixed x3..x10 = fixed_val
# ============================================================

def visual_slice_2d(source, net_y, net_u, d, fixed_val=0.5, resolution=200,
                    save_dir=None, epoch=0):
    """
    Fix x3..xd = fixed_val, sweep x1,x2 in [0,1].
    Plot y_pred vs y_star, u_pred vs u_star, and errors.

    Returns:
        (fig_y, fig_u): matplotlib figure objects for SwanLab logging.
    """
    device = next(net_y.parameters()).device
    x_lin = torch.linspace(0, 1, resolution, device=device)
    X1, X2 = torch.meshgrid(x_lin, x_lin, indexing='ij')
    x12 = torch.stack([X1.flatten(), X2.flatten()], dim=1)  # [N, 2]

    # Build [N, d] tensor
    N = x12.shape[0]
    fixed = torch.full((N, d - 2), fixed_val, device=device)
    x_full = torch.cat([x12, fixed], dim=1)  # [N, d]

    with torch.no_grad():
        y_pred = net_y(x_full).cpu().reshape(resolution, resolution).numpy()
        u_pred = net_u(x_full).cpu().reshape(resolution, resolution).numpy()
        y_star = source.compute_y_star(x_full).cpu().reshape(resolution, resolution).numpy()
        u_star = source.compute_u_star(x_full).cpu().reshape(resolution, resolution).numpy()

    X1c = X1.cpu().numpy()
    X2c = X2.cpu().numpy()

    # --- y comparison ---
    fig_y = plt.figure(figsize=(18, 5))
    ax1 = fig_y.add_subplot(131, projection='3d')
    ax1.plot_surface(X1c, X2c, y_star, cmap='coolwarm', alpha=0.8)
    ax1.set_title('y* (exact)')

    ax2 = fig_y.add_subplot(132, projection='3d')
    ax2.plot_surface(X1c, X2c, y_pred, cmap='coolwarm', alpha=0.8)
    ax2.set_title('y_net (pred)')

    ax3 = fig_y.add_subplot(133, projection='3d')
    ax3.plot_surface(X1c, X2c, np.abs(y_pred - y_star), cmap='hot', alpha=0.8)
    ax3.set_title('|y_net - y*|')

    fig_y.suptitle(f'State y — epoch {epoch} (x3..x{d}={fixed_val})')
    fig_y.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_y.savefig(os.path.join(save_dir, f'y_slice_epoch{epoch}.png'), dpi=100)

    # --- u comparison ---
    fig_u = plt.figure(figsize=(18, 5))
    ax1 = fig_u.add_subplot(131, projection='3d')
    ax1.plot_surface(X1c, X2c, u_star, cmap='coolwarm', alpha=0.8)
    ax1.set_title('u* (exact)')

    ax2 = fig_u.add_subplot(132, projection='3d')
    ax2.plot_surface(X1c, X2c, u_pred, cmap='coolwarm', alpha=0.8)
    ax2.set_title('u_net (pred)')

    ax3 = fig_u.add_subplot(133, projection='3d')
    ax3.plot_surface(X1c, X2c, np.abs(u_pred - u_star), cmap='hot', alpha=0.8)
    ax3.set_title('|u_net - u*|')

    fig_u.suptitle(f'Control u — epoch {epoch} (x3..x{d}={fixed_val})')
    fig_u.tight_layout()
    if save_dir:
        fig_u.savefig(os.path.join(save_dir, f'u_slice_epoch{epoch}.png'), dpi=100)

    return fig_y, fig_u
