"""
2-Stage HD Obstacle Optimal Control Training Script.

Stage 1: Bilevel MEHA training (y, u, theta) — records J_u & J_l
Stage 2: Fix u, refine y with J_l only

Config: from run_20260303_094937 (D=5, SPINN, sin source, beta1=0.9)
Plotting: follows draw.ipynb publication style (PDF output)

Usage:
    python example3_high_dimensional_space.py
"""

import os
from pathlib import Path
import sys
import time
import copy
import math

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from datetime import datetime

from utils_hd import (
    SourceHighDim10,
    net_y_spinn_hd,
    net_u_spinn_hd,
    SobolDataLoader,
    set_seed,
    lr_lambda,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


# ============================================================
# Loss functions
# ============================================================

def J_u_hd(source, net_y, net_u, x):
    """Upper-level objective: 0.5*||y - y_d||^2 + 0.5*sigma*||u||^2"""
    yd = source.compute_yd(x)
    y_pred = net_y(x)
    u_pred = net_u(x)
    term1 = 0.5 * torch.mean((y_pred - yd) ** 2)
    term2 = 0.5 * source.sigma * torch.mean(u_pred ** 2)
    return term1 + term2


def J_l_hd(source, net_y, net_u, x):
    """Lower-level energy: 0.5*||grad y||^2 - <f+u, y>"""
    f_val = source.compute_f(x)
    x = x.clone().detach().requires_grad_(True)
    y_pred = net_y(x)
    u_pred = net_u(x)
    grad_y = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0]
    dirichlet = 0.5 * torch.mean(torch.sum(grad_y ** 2, dim=1, keepdim=True))
    work = -torch.mean((f_val + u_pred) * y_pred)
    return dirichlet + work


# ============================================================
# Stage 1: Bilevel MEHA Trainer (simplified logging)
# ============================================================

class TrainerHD:
    """Bilevel MEHA trainer — only records J_u and J_l to loss.csv."""

    def __init__(
        self, y_net, u_net, theta_net, source, train_loader,
        opt_y, opt_u, opt_theta, sched_y, sched_u, sched_theta,
        gamma, c0, p, device, num_epochs, save_dir,
        clip_u=1.0, log_interval=10,
    ):
        self.y_net = y_net.to(device)
        self.u_net = u_net.to(device)
        self.theta_net = theta_net.to(device)
        self.source = source
        self.train_loader = train_loader
        self.opt_y = opt_y
        self.opt_u = opt_u
        self.opt_theta = opt_theta
        self.sched_y = sched_y
        self.sched_u = sched_u
        self.sched_theta = sched_theta
        self.gamma = gamma
        self.c0 = c0
        self.p = p
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.clip_u = clip_u
        self.log_interval = log_interval

        os.makedirs(save_dir, exist_ok=True)
        self.loss_csv = os.path.join(save_dir, 'loss.csv')
        with open(self.loss_csv, 'w') as f:
            f.write("epoch,time,J_u_loss,J_l_loss\n")

    # ---- helpers ----

    @staticmethod
    def _zero_grads(net):
        for p in net.parameters():
            p.grad = None

    @staticmethod
    def _accum_grad(net, grads, scale):
        for param, g in zip(net.parameters(), grads):
            scaled = g * scale
            if param.grad is None:
                param.grad = scaled.detach().clone()
            else:
                param.grad += scaled.detach()

    # ---- core training ----

    def train_one_epoch(self, epoch):
        self.y_net.train()
        self.u_net.train()
        self.theta_net.train()

        c_k = self.c0 * (epoch + 1) ** self.p
        inv_ck = 1.0 / c_k

        y_params = list(self.y_net.parameters())
        u_params = list(self.u_net.parameters())
        theta_params = list(self.theta_net.parameters())

        last_j = 0.0
        last_e = 0.0

        for batch in self.train_loader:
            if len(batch) == 2:
                x_yz, x_u = batch
            else:
                x_yz = batch[0]
                x_u = batch[0]

            self._zero_grads(self.y_net)
            self._zero_grads(self.u_net)
            self._zero_grads(self.theta_net)

            x_yz = x_yz.detach().requires_grad_(True)
            x_u = x_u.detach().requires_grad_(True)

            # ─── z-step (theta_net) ───
            e_theta = J_l_hd(self.source, self.theta_net, self.u_net, x_yz)
            g_theta = torch.autograd.grad(e_theta, theta_params, retain_graph=True)
            prox = [(pt.detach() - py.detach()) / self.gamma
                    for pt, py in zip(theta_params, y_params)]
            d_z = [g + px for g, px in zip(g_theta, prox)]
            self._accum_grad(self.theta_net, d_z, 1.0)

            # ─── y-step (y_net) ───
            j_val = J_u_hd(self.source, self.y_net, self.u_net, x_u)
            e_y = J_l_hd(self.source, self.y_net, self.u_net, x_yz)
            g_j_y = torch.autograd.grad(j_val, y_params, retain_graph=True)
            g_e_y = torch.autograd.grad(e_y, y_params, retain_graph=True)
            moreau = [(pt.detach() - py.detach()) / self.gamma
                      for pt, py in zip(theta_params, y_params)]
            d_y = [inv_ck * gj + ge + m
                   for gj, ge, m in zip(g_j_y, g_e_y, moreau)]
            self._accum_grad(self.y_net, d_y, 1.0)

            # ─── u-step (u_net) ───
            g_j_u = torch.autograd.grad(j_val, u_params, retain_graph=False)
            g_e_u = torch.autograd.grad(e_y, u_params, retain_graph=False)
            g_e_theta_u = torch.autograd.grad(e_theta, u_params, retain_graph=False)
            d_u = [inv_ck * gju + geu - geut
                   for gju, geu, geut in zip(g_j_u, g_e_u, g_e_theta_u)]
            self._accum_grad(self.u_net, d_u, 1.0)

            # gradient clipping on u
            if self.clip_u > 0:
                torch.nn.utils.clip_grad_norm_(self.u_net.parameters(), self.clip_u)

            self.opt_theta.step()
            self.opt_y.step()
            self.opt_u.step()

            last_j = j_val.item()
            last_e = e_y.item()

        self.sched_y.step()
        self.sched_u.step()
        self.sched_theta.step()

        return last_j, last_e

    def fit(self):
        pbar = tqdm(range(self.num_epochs), ascii=True, dynamic_ncols=True)
        t0 = time.time()

        for epoch in pbar:
            j_val, e_y = self.train_one_epoch(epoch)
            pbar.set_postfix(J_u=f"{j_val:.3e}", J_l=f"{e_y:.3e}")

            if epoch % self.log_interval == 0:
                elapsed = time.time() - t0
                with open(self.loss_csv, 'a') as f:
                    f.write(f"{epoch},{elapsed:.1f},{j_val:.6e},{e_y:.6e}\n")

            if epoch % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================
# Stage 2: Fix u, refine y with J_l only
# ============================================================

def train_fixed_u(trainer, epochs=10000, lr_factor=0.1):
    """
    Stage 2: Clone networks from Stage 1, freeze u_net,
    and train y_net with J_l (energy functional) only.
    Mimics example1.py clone_trainer_and_train_fixed_v.
    """
    y_net = copy.deepcopy(trainer.y_net)
    u_net = copy.deepcopy(trainer.u_net)
    source = trainer.source
    device = trainer.device

    # Freeze u_net
    u_net.eval()
    for p in u_net.parameters():
        p.requires_grad = False

    # New optimizer: base_lr * lr_factor
    lr = trainer.opt_y.param_groups[0]['lr'] * lr_factor
    optimizer = torch.optim.Adam(y_net.parameters(), lr=lr)
    scheduler = LambdaLR(
        optimizer,
        partial(lr_lambda, warmup=1000, cycle_factor=0.8, cycle_length=1000),
    )

    # Loss CSV
    stage2_csv = os.path.join(trainer.save_dir, 'loss_stage2.csv')
    with open(stage2_csv, 'w') as f:
        f.write("epoch,time,J_l_loss\n")

    print(f"\n{'='*60}")
    print(f"Stage 2: Fix u, train y for {epochs} epochs (lr={lr:.2e})")
    print(f"{'='*60}")

    pbar = tqdm(range(1, epochs + 1), ascii=True, dynamic_ncols=True)
    t0 = time.time()

    for epoch in pbar:
        y_net.train()
        total_loss = 0.0
        count = 0

        for batch in trainer.train_loader:
            x = batch[0].detach().requires_grad_(True).to(device)
            loss = J_l_hd(source, y_net, u_net, x)

            optimizer.zero_grad()
            grads = torch.autograd.grad(loss, y_net.parameters())
            for param, g in zip(y_net.parameters(), grads):
                param.grad = g
            optimizer.step()

            total_loss += loss.item()
            count += 1

        scheduler.step()
        avg = total_loss / max(count, 1)
        pbar.set_postfix(J_l=f"{avg:.3e}")

        if epoch % 10 == 0:
            with open(stage2_csv, 'a') as f:
                f.write(f"{epoch},{time.time()-t0:.1f},{avg:.6e}\n")

        if epoch % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Stage 2 done. Final J_l = {avg:.6e}")
    return y_net, u_net


# ============================================================
# Pretrain y_net
# ============================================================

def pretrain_y(net_y, source, device, epochs=100, lr=1e-4, batch_size=16384):
    """Pretrain y_net against relu(y_d)."""
    print("Pretraining y_net against y_d ...")
    loader = SobolDataLoader(
        dim=source.d, batch_size=batch_size,
        steps_per_epoch=1, device=device,
    )
    optimizer = torch.optim.Adam(net_y.parameters(), lr=lr)
    scheduler = LambdaLR(
        optimizer,
        partial(lr_lambda, warmup=0, cycle_factor=0.9, cycle_length=epochs),
    )

    pbar = tqdm(range(1, epochs + 1), ascii=True, dynamic_ncols=True)
    for ep in pbar:
        net_y.train()
        for batch in loader:
            x = batch[0].to(device)
            loss = torch.mean((net_y(x) - torch.relu(source.compute_yd(x))) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.6e}")
        if ep % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Pretrain done. loss={loss.item():.6e}")
    return net_y


# ============================================================
# Publication plotting — strictly follows draw_.ipynb style
# ============================================================

def _stylize_3d(ax):
    """3D axis formatting from draw_.ipynb"""
    ax.tick_params(axis='both', which='major', labelsize=14)


def _stylize_heatmap():
    """Heatmap axis formatting from draw_.ipynb"""
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def _add_colorbar(mappable, shrink=1.0):
    """Colorbar formatting from draw_.ipynb"""
    cbar = plt.colorbar(mappable, pad=0.1, fraction=0.046, shrink=shrink)
    cbar.ax.tick_params(labelsize=14)
    return cbar


def plot_publication_figures(
    net_y, net_u, source, d, save_dir,
    prefix='example_hd', fixed_val=0.5, resolution=200,
):
    """
    Generate 6 publication-quality PDF figures (3 y + 3 u).
    Fix x3..xd = fixed_val, sweep (x1, x2) in [0,1]^2.
    Style strictly follows draw_.ipynb.
    """
    device = next(net_y.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    # Build 2D slice grid
    x_lin = torch.linspace(0, 1, resolution, device=device)
    X1, X2 = torch.meshgrid(x_lin, x_lin, indexing='ij')
    x12 = torch.stack([X1.flatten(), X2.flatten()], dim=1)  # [N, 2]
    N = x12.shape[0]

    if d > 2:
        fixed = torch.full((N, d - 2), fixed_val, device=device)
        x_full = torch.cat([x12, fixed], dim=1)  # [N, d]
    else:
        x_full = x12

    with torch.no_grad():
        y_pred = net_y(x_full).cpu().numpy().reshape(resolution, resolution)
        u_pred = net_u(x_full).cpu().numpy().reshape(resolution, resolution)
        y_exact = source.compute_y_star(x_full).cpu().numpy().reshape(resolution, resolution)
        u_exact = source.compute_u_star(x_full).cpu().numpy().reshape(resolution, resolution)

    X = X1.cpu().numpy()
    Y = X2.cpu().numpy()
    error_y = np.abs(y_pred - y_exact)
    error_u = np.abs(u_pred - u_exact)

    # ---- y numerical (3D surface) ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, y_pred, cmap='coolwarm')
    _stylize_3d(ax)
    _add_colorbar(surf, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_y_numerical.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- y exact (3D surface) ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, y_exact, cmap='coolwarm')
    _stylize_3d(ax)
    _add_colorbar(surf, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_y_exact.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- y error (heatmap) ----
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(error_y, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    _stylize_heatmap()
    _add_colorbar(im, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_y_error.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- u numerical (3D surface) ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u_pred, cmap='coolwarm')
    _stylize_3d(ax)
    _add_colorbar(surf, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_v_numerical.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- u exact (3D surface) ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u_exact, cmap='coolwarm')
    _stylize_3d(ax)
    _add_colorbar(surf, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_v_exact.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- u error (heatmap) ----
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(error_u, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    _stylize_heatmap()
    _add_colorbar(im, shrink=1.0)
    plt.savefig(os.path.join(save_dir, f'{prefix}_v_error.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"6 publication figures (3y + 3u) saved to {save_dir}")


def plot_loss_curves(loss_csv, save_dir, prefix='example_hd'):
    """
    Plot 2 loss PDF figures (upper & lower) in draw_.ipynb style.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(loss_csv)
    epochs = df['epoch'].values

    # ---- Upper-level loss ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['J_u_loss'].values, lw=2, label='Upper-level loss $L$')
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_Upper.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ---- Lower-level loss ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['J_l_loss'].values, lw=2,
            label='Lower-level loss $\\ell$', color='#ff7f0e')
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_lower.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"2 loss curves saved to {save_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ── Config from run_20260303_094937 ──────────────────────
    D             = 5
    SEED          = 123
    EPOCHS_S1     = 40000
    EPOCHS_S2     = 10000
    BATCH         = 16384
    STEPS_EPOCH   = 4
    GAMMA         = 50.0
    C0            = 5.0
    P_EXP         = 0.3
    LR            = 1e-3
    BETA1, BETA2  = 0.9, 0.99
    CLIP_U        = 1.0
    WARMUP        = 500
    CYC_FACTOR    = 0.9
    CYC_LEN       = 1000
    LOG_INTV      = 10

    # ── Setup ────────────────────────────────────────────────
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    source = SourceHighDim10(device=device, d=D, sigma=1.0, c_thr=0.01)

    # SPINN networks
    spinn_kw = dict(
        d=D, rank=2, hidden_1d=32, layers_1d=3,
        activation_1d='swish', head='sum', m_mode='prod',
    )
    y_net     = net_y_spinn_hd(**spinn_kw, y_embed='relu').to(device)
    u_net     = net_u_spinn_hd(**spinn_kw).to(device)
    theta_net = net_y_spinn_hd(**spinn_kw, y_embed='relu').to(device)

    # DataLoader (dual_sampling=True matches effective behavior of the original run)
    loader = SobolDataLoader(
        dim=D, batch_size=BATCH, steps_per_epoch=STEPS_EPOCH,
        device=device, dual_sampling=True,
    )

    # ── Pretrain ─────────────────────────────────────────────
    y_net = pretrain_y(y_net, source, device, epochs=100, lr=1e-4, batch_size=BATCH)
    theta_net.load_state_dict(y_net.state_dict())

    # ── Optimizers & schedulers ──────────────────────────────
    betas = (BETA1, BETA2)
    opt_y = torch.optim.Adam(y_net.parameters(), lr=LR, betas=betas)
    opt_u = torch.optim.Adam(u_net.parameters(), lr=LR, betas=betas)
    opt_z = torch.optim.Adam(theta_net.parameters(), lr=LR, betas=betas)

    sfn = partial(lr_lambda, warmup=WARMUP, cycle_factor=CYC_FACTOR, cycle_length=CYC_LEN)
    sch_y = LambdaLR(opt_y, sfn)
    sch_u = LambdaLR(opt_u, sfn)
    sch_z = LambdaLR(opt_z, sfn)

    save_dir = os.path.join(
        PACKAGE_ROOT, "outputs", "example3",
        f"run_2stage_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # ── Stage 1: Bilevel MEHA ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 1: Bilevel MEHA training for {EPOCHS_S1} epochs")
    print(f"{'='*60}")

    trainer = TrainerHD(
        y_net=y_net, u_net=u_net, theta_net=theta_net,
        source=source, train_loader=loader,
        opt_y=opt_y, opt_u=opt_u, opt_theta=opt_z,
        sched_y=sch_y, sched_u=sch_u, sched_theta=sch_z,
        gamma=GAMMA, c0=C0, p=P_EXP, device=device,
        num_epochs=EPOCHS_S1, save_dir=save_dir,
        clip_u=CLIP_U, log_interval=LOG_INTV,
    )
    trainer.fit()

    # Save Stage 1 checkpoints
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(y_net.state_dict(), os.path.join(ckpt_dir, 'y_net_stage1.pth'))
    torch.save(u_net.state_dict(), os.path.join(ckpt_dir, 'u_net_stage1.pth'))
    torch.save(theta_net.state_dict(), os.path.join(ckpt_dir, 'theta_net_stage1.pth'))
    print(f"Stage 1 checkpoints saved to {ckpt_dir}")

    # ── Stage 2: Fix u, refine y ─────────────────────────────
    y_net_s2, u_net_s2 = train_fixed_u(trainer, epochs=EPOCHS_S2)
    torch.save(y_net_s2.state_dict(), os.path.join(ckpt_dir, 'y_net_stage2.pth'))
    print(f"Stage 2 checkpoint saved to {ckpt_dir}")

    # ── Publication figures ──────────────────────────────────
    figs_dir = os.path.join(save_dir, 'figs')

    # 3 y figures (stage 2 y) + 3 u figures (stage 1 u, fixed in stage 2)
    plot_publication_figures(y_net_s2, u_net_s2, source, D, figs_dir)

    # 2 loss curves (stage 1 upper & lower)
    plot_loss_curves(trainer.loss_csv, figs_dir)

    print(f"\nAll outputs saved to {save_dir}")
    print("  - loss.csv            : Stage 1 loss (J_u, J_l)")
    print("  - loss_stage2.csv     : Stage 2 loss (J_l)")
    print("  - checkpoints/        : model weights")
    print("  - figs/               : 8 publication PDFs")
    print("Done!")
