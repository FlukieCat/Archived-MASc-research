#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:43:25 2018

@author: h6you
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy import interpolate
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import concurrent.futures
import timeit
import time
import dill
from tqdm import tqdm
import os
from surface import Surface
from math import log
import matplotlib
matplotlib.style.use('ggplot')

# =============================================================================
# import signal
# from signal import signal as signal1
# signal1(signal.SIGPIPE,signal.SIG_DFL)
# =============================================================================
GasSpecies = ['CO', 'CO2', 'H2', 'H2O']
De = {'CO': 8.10e-7,
      'CO2': 6.46e-7,
      'H2': 3.03e-6,
      'H2O': 1.01e-6}  # m2/s
D_AB = {'CO': 2.44e-4,
        'CO2': 1.93e-4,
        'H2': 9.25e-4,
        'H2O': 3.05e-4}  # m2/s
BulkConcentrations = {'CO': 0.15,
                      'CO2': 0.00,
                      'H2': 0.05,
                      'H2O': 0.00}  # partial pressure atm 'CO' p = 0.3 * 1 atm
site_distance = 2.087e-10
CT = 1 / (site_distance * site_distance)  # site density [site/m2] or [1/m2]
NA = 6.02e23  # Avogadro constant [molecule/mol] or [1/mol]
T = 1223      # temperature [K]
por = 0.36
tor = 4         # tortuosity, normally 2-6
Sh = 2          # Sherwood Number, Sh = K*(2R)/D_AB
Rp = 1.0e-8
L = 1.0e-4  # pore lenghth
dt = 0.01       # dt, time interval of PDE
Sp = 4 * 3.1415926 * L ** 2
N = 1
V = 1e-6

Cin = 2  # inlet concentration mM

kg = {}
for i in D_AB.keys():
    kg[i] = (Sh * D_AB[i]) / (2 * L)  # [m/s]
h = {}
for i in kg.keys():
    h[i] = - (kg[i] * L) / De[i]
a = {}
for i in De.keys():
    a[i] = (dt * De[i]) / Rp ** 2
b = {}
for i in De.keys():
    b[i] = (dt * De[i]) / L ** 2
k = {}
for i in kg.keys():
    k[i] = (N * Sp * kg[i]) / V  # [1/s]

Nt, Nx, Ny = 101, 31, 6
Nkmc = 10  # number of kMC tiles
gap_nodes = int(Nx / Nkmc)
dx, dy = 1. / (Nx - 1), 1. / (Ny - 1)
x_arr = np.linspace(0, 1, num=Nx)
y_arr = np.linspace(0, 1, num=Ny)
t_arr = np.linspace(0, (Nt - 1) * dt, num=Nt)
R = np.array([y_arr, ] * Nx).transpose()
R[0, :] = 0.1  # does nothing but silences the warning msg.
R_inv = np.reciprocal(R)
mx_shape = (Ny * len(GasSpecies), Nx)  # shape of the spatial domain

# store concentration profile inside the particle
CO = np.zeros((Nt, Ny, Nx))
CO2 = np.zeros((Nt, Ny, Nx))
H2 = np.zeros((Nt, Ny, Nx))
H2O = np.zeros((Nt, Ny, Nx))

# store bulk concentration profile
CO_b = np.zeros(Nt)
H2_b = np.zeros(Nt)
CO2_b = np.zeros(Nt)
H2O_b = np.zeros(Nt)
CO_b[0] = BulkConcentrations['CO']
H2_b[0] = BulkConcentrations['H2']
CO2_b[0] = BulkConcentrations['CO2']
H2O_b[0] = BulkConcentrations['H2O']

# store concentration of each species at the point closest to the inlet
CO_p = np.zeros(Nt)
H2_p = np.zeros(Nt)
CO2_p = np.zeros(Nt)
H2O_p = np.zeros(Nt)

Flux_CO_init = np.zeros((1, Nx))
Flux_CO2_init = np.zeros((1, Nx))
Flux_H2_init = np.zeros((1, Nx))
Flux_H2O_init = np.zeros((1, Nx))
Flux = {'CO': Flux_CO_init,
        'CO2': Flux_CO2_init,
        'H2': Flux_H2_init,
        'H2O': Flux_H2O_init}
Flux_CO = np.zeros((1, Nkmc))
Flux_CO2 = np.zeros((1, Nkmc))
Flux_H2 = np.zeros((1, Nkmc))
Flux_H2O = np.zeros((1, Nkmc))
COV_CO = []
COV_H2 = []
COV_H =  []
length = 32

surface_stash = []
for i in range(Nkmc):
    surface_stash.append(Surface(i, dt, length, 0, 0))

def ode_Cbulk(y, t, k, C_pore):
    yCO, yCO2, yH2, yH2O = y
    dydt = [k['CO'] * (C_pore['CO'] - yCO),
            k['CO2'] * (C_pore['CO2'] - yCO2),
            k['H2'] * (C_pore['H2'] - yH2),
            k['H2O'] * (C_pore['H2O'] - yH2O)]
    return dydt

def c_bulk(y0, t, k, C_pore):
    sol = odeint(ode_Cbulk, y0, t, args=(k, C_pore))
    BulkConcentrations['CO'] = sol[-1, 0]
    BulkConcentrations['CO2'] = sol[-1, 1]
    BulkConcentrations['H2'] = sol[-1, 2]
    BulkConcentrations['H2O'] = sol[-1, 3]
    return BulkConcentrations

def pde_bk(m, u, GasSpecies, Flux, BulkConcentrations):
    u = u.reshape(mx_shape)
    m = m.reshape(mx_shape)
    f = np.zeros(mx_shape)
    for i in range(len(GasSpecies)):
        # Symmetry
        f[i*Ny, 1:-1] = m[i*Ny, 1:-1] - u[i*Ny, 1:-1] - dt*(
                4 * a[GasSpecies[i]] * (m[i*Ny+1, 1:-1] - m[i*Ny, 1:-1]) / dy**2
                         + b[GasSpecies[i]] * (
                         m[i*Ny, 2:] - 2 * m[i*Ny, 1:-1] + m[i*Ny, :-2]) / dx**2)
        # Interior
        f[i*Ny+1:(i+1)*Ny-1, 1:-1] = m[i*Ny+1:(i+1)*Ny-1, 1:-1] - u[i*Ny+1:(i+1)*Ny-1, 1:-1] - dt*(
            a[GasSpecies[i]]*(R_inv[1:-1, 1:-1]*((m[i*Ny+2:(i+1)*Ny,1:-1] - m[i*Ny:(i+1)*Ny-2,1:-1])/(2*dy))
            + (m[i*Ny+2:(i+1)*Ny,1:-1] - 2*m[i*Ny+1:(i+1)*Ny-1,1:-1] + m[i*Ny:(i+1)*Ny-2,1:-1]) / dy**2)
            + b[GasSpecies[i]]*(
                    (m[i*Ny+1:(i+1)*Ny-1, 2:] - 2*m[i*Ny+1:(i+1)*Ny-1, 1:-1] + m[i*Ny+1:(i+1)*Ny-1, 0:-2]) / dx**2))

        # --------------------- Center --------------------- #
        f[i*Ny+1:(i+1)*Ny-1, 0] = m[i*Ny+1:(i+1)*Ny-1,0] - u[i*Ny+1:(i+1)*Ny-1,0] - dt*(
                            a[GasSpecies[i]]*(R_inv[1:-1,0]*((m[i*Ny+2:(i+1)*Ny,0]-m[i*Ny:(i+1)*Ny-2,0])/(2*dy))
                            +(m[i*Ny+2:(i+1)*Ny,0]-2*m[i*Ny+1:(i+1)*Ny-1,0]+m[i*Ny:(i+1)*Ny-2,0])/dy**2) 
                            + 2*b[GasSpecies[i]]*((m[i*Ny+1:(i+1)*Ny-1,1]-m[i*Ny+1:(i+1)*Ny-1,0])/dx**2))
        # (0,0)
        f[i*Ny,0] = m[i*Ny, 0] - u[i*Ny,0] - dt*(4*a[GasSpecies[i]]*(m[i*Ny+1,0]-m[i*Ny,0])/dy**2 
                     + 2*b[GasSpecies[i]]*(m[i*Ny,1]-m[i*Ny,0])/dx**2)
        # (-1,0)
        dmdy = (Rp/(De[GasSpecies[i]]*Cin)) * Flux[GasSpecies[i]][0, 0]  
        f[(i+1)*Ny-1,0] = m[(i+1)*Ny-1,0] - u[(i+1)*Ny-1,0] - dt*(
                          a[GasSpecies[i]]*((2/dy**2)*(dy*(dmdy)-m[(i+1)*Ny-1,0]+m[(i+1)*Ny-2,0]) + dmdy) 
                          + b[GasSpecies[i]]*(2/dx**2)*(m[(i+1)*Ny-1,1] - m[(i+1)*Ny-1,0]))  

        # --------------------- Inlet --------------------- #
        f[i*Ny+1:(i+1)*Ny-1, -1] = m[i*Ny+1:(i+1)*Ny-1,-1] - u[i*Ny+1:(i+1)*Ny-1,-1] - dt*(
                    a[GasSpecies[i]]*(R_inv[1:-1,0]*((m[i*Ny+2:(i+1)*Ny,-1] - m[i*Ny:(i+1)*Ny-2,-1])/(2*dy))
                    + (m[i*Ny+2:(i+1)*Ny,-1] - 2*m[i*Ny+1:(i+1)*Ny-1,-1] + m[i*Ny:(i+1)*Ny-2,-1])/dy**2)
                    +  b[GasSpecies[i]]*(2*(h[GasSpecies[i]]*dx*(m[i*Ny+1:(i+1)*Ny-1,-1] - BulkConcentrations[GasSpecies[i]])-m[i*Ny+1:(i+1)*Ny-1,-1]+m[i*Ny+1:(i+1)*Ny-1,-2])/dx**2))
        # (0,1)
        f[i*Ny,-1] = m[i*Ny,-1] - u[i*Ny,-1] - dt*( 
                  4*a[GasSpecies[i]]*(m[i*Ny+1,-1] - m[i*Ny,-1])/dy**2 
                + (2*b[GasSpecies[i]]/dx**2)*(h[GasSpecies[i]]*dx*(m[i*Ny,-1]- BulkConcentrations[GasSpecies[i]])-m[i*Ny,-1]+m[i*Ny,-2]))

        # (-1,1)
        dmdy = (Rp/(De[GasSpecies[i]]*Cin)) * Flux[GasSpecies[i]][0, -1]  
        f[(i+1)*Ny-1,-1] = m[(i+1)*Ny-1,-1] - u[(i+1)*Ny-1,-1] - dt*(
                  a[GasSpecies[i]]*(2*(dy*(dmdy) - m[(i+1)*Ny-1,-1] + m[(i+1)*Ny-2,-1])/dy**2 + dmdy)
                  + (2*b[GasSpecies[i]]/dx**2)*(h[GasSpecies[i]]*dx*(m[(i+1)*Ny-1,-1]-BulkConcentrations[GasSpecies[i]])-m[(i+1)*Ny-1,-1]+m[(i+1)*Ny-1,-2]))

        #----------------------------------Surface---------------------------------#  
        dmdy = (Rp / (De[GasSpecies[i]] * Cin)) * Flux[GasSpecies[i]][0, 1:-1] 
        f[(i+1)*Ny-1,1:-1] = m[(i+1)*Ny-1, 1:-1] - u[(i+1)*Ny-1, 1:-1] - dt*(
                a[GasSpecies[i]]*((2/dy**2)*(dy * dmdy - m[(i+1)*Ny-1,1:-1] + m[(i+1)*Ny-2,1:-1]) + dmdy)  + (b[GasSpecies[i]]/dx**2)*(m[(i+1)*Ny-1,2:] - 2*m[(i+1)*Ny-1,1:-1] + m[(i+1)*Ny-1,:-2]))

    f = f.ravel()
    return f

def plot_C_profile(x_arr, y_arr, CO, CO2, H2, H2O):
    R_mesh, Z_mesh = np.meshgrid(x_arr, y_arr, indexing='xy')
    fig1 = plt.figure(figsize=(20, 12))
    ax = fig1.add_subplot(2, 2, 1, projection='3d')
    ax.grid(b='off')
    ax.plot_surface(
        R_mesh, Z_mesh, CO[-1], rstride=1, cstride=1,
        cmap=cm.summer, linewidth=0, antialiased=False)
    ax.set_xlabel(r'Pore Length ($10^{-4}$ m)')
    ax.set_ylabel(r'Pore Radius ($10^{-8}$ m)')
    ax.set_zlabel(r'Concentration ($\mathrm{mol/m}^3$)')
    ax.set_title('Concentration profile of $\mathrm{CO}$ inside the pore')
    ax.view_init(azim=-60)
    ax = fig1.add_subplot(2, 2, 2, projection='3d')
    ax.grid(b='off')
    ax.plot_surface(
        R_mesh, Z_mesh, CO2[-1], rstride=1, cstride=1,
        cmap=cm.winter, linewidth=0, antialiased=False)
    ax.set_xlabel(r'Pore Length ($10^{-4}$ m)')
    ax.set_ylabel(r'Pore Radius ($10^{-8}$ m)')
    ax.set_zlabel(r'Concentration ($\mathrm{mol/m}^3$)')
    ax.set_title(r'Concentration profile of $\mathrm{CO_2}$ inside the pore')
    ax.view_init(azim=-60)
    ax = fig1.add_subplot(2, 2, 3, projection='3d')
    ax.grid(b='off')
    ax.plot_surface(
        R_mesh, Z_mesh, H2[-1], rstride=1, cstride=1,
        cmap=cm.winter, linewidth=0, antialiased=False)
    ax.set_xlabel(r'Pore Length ($10^{-4}$ m)')
    ax.set_ylabel(r'Pore Radius ($10^{-8}$ m)')
    ax.set_zlabel(r'Concentration ($\mathrm{mol/m}^3$)')
    ax.set_title(r'Concentration profile of $\mathrm{H_2}$ inside the pore')
    ax.view_init(azim=-60)
    ax = fig1.add_subplot(2, 2, 4, projection='3d')
    ax.grid(b='off')
    ax.plot_surface(
        R_mesh, Z_mesh, H2O[-1], rstride=1, cstride=1,
        cmap=cm.winter, linewidth=0, antialiased=False)
    ax.set_xlabel(r'Pore Length ($10^{-4}$ m)')
    ax.set_ylabel(r'Pore Radius ($10^{-8}$ m)')
    ax.set_zlabel(r'Concentration ($\mathrm{mol/m}^3$)')
    ax.set_title(r'Concentration profile of $\mathrm{H_2O}$ inside the pore')
    ax.view_init(azim=-60)
    fig1.tight_layout()
    plt.savefig("C_profile.png")

def plot_coverage(Nkmc, COV_CO, COV_H, COV_H2):
    colors_array = cm.rainbow(np.linspace(0, 1, len(COV_CO)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    x_arr = np.linspace(0, 1, num=Nkmc)
    fig, axes = plt.subplots()
    for i in range(len(COV_CO)):
        axes.plot(x_arr, COV_CO[i], color=rainbow[i], label=f'{i+1}th sample')
    axes.set_xlabel(r'Pore Length ($10^{-4}m$)')
    axes.set_ylabel(r'CO Coverage')
    axes.legend(loc='best')
    plt.savefig("Coverage_CO.png")

    fig, axes = plt.subplots()
    for i in range(len(COV_H)):
        axes.plot(x_arr, COV_H[i], color=rainbow[i], label=f'{i+1}th sample')
    axes.set_xlabel(r'Pore Length ($10^{-4}m$)')
    axes.set_ylabel(r'H Coverage')
    axes.legend(loc='best')
    plt.savefig("Coverage_H.png")

    fig, axes = plt.subplots()
    for i in range(len(COV_H2)):
        axes.plot(x_arr, COV_H2[i], color=rainbow[i], label=f'{i+1}th sample')
    axes.set_xlabel(r'Pore Length ($10^{-4}m$)')
    axes.set_ylabel(r'H2 Coverage')
    axes.legend(loc='best')
    plt.savefig("Coverage_H2.png")

def plot_bulk(t_arr, CO2_b, H2O_b, CO_b, H2_b):
    fig1, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(t_arr, CO2_b, 'r.', label=r'$\mathrm{CO_2}$ vol %')
    axes[0].set_xlabel(r'Time (s)')
    axes[0].set_ylabel(r'vol %')
    axes[0].legend()
    axes[1].plot(t_arr, H2O_b, 'b.', label=r'$\mathrm{H_2O}$ vol %')
    axes[1].set_xlabel(r'Time (s)')
    axes[1].set_ylabel(r'vol %')
    axes[1].legend()
    fig1.tight_layout()
    plt.savefig("Bulk_out_profile.png")

    fig2, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(t_arr, CO_b, 'r.', label=r'$\mathrm{CO}$ vol %')
    axes[0].set_xlabel(r'Time (s)')
    axes[0].set_ylabel(r'vol %')
    axes[0].legend()
    axes[1].plot(t_arr, H2_b, 'b.', label=r'$\mathrm{H_2}$ vol %')
    axes[1].set_xlabel(r'Time (s)')
    axes[1].set_ylabel(r'vol %')
    axes[1].legend()
    fig2.tight_layout()
    plt.savefig("Bulk_in_profile.png")


def get_param_list(surface_stash, Cco, Ch2):
    param_list = []
    for i in range(Nkmc):
        param_list.append([surface_stash[i],
                           Cin * Cco[(i) * gap_nodes],
                           Cin * Ch2[(i) * gap_nodes],
                           Cco[(i) * gap_nodes],
                           Ch2[(i) * gap_nodes]])
    return param_list


def get_flux(params):
    surface, C_CO, C_H2, p_CO, p_H2 = params
    t = 0
    step = 0
    surface.update(C_CO, C_H2, p_CO, p_H2)
    while t < surface.t_final:
        r = np.random.uniform()
        while r == 0.0:
            r = np.random.uniform()
        if surface.W_tot == 0:
            return [surface, 0, 0, 0, 0]
        dt = - (1 / surface.W_tot) * log(r)
        surface.execute_event(C_CO, C_H2, p_CO, p_H2)
        step += 1
        t += dt
    CO_flux = - (surface.num_CO_consumed / surface.NA) / (
        t * surface.Area)
    CO2_flux = (surface.num_CO2_produced / surface.NA) / (
        t * surface.Area)
    H2_flux = - (surface.num_H2_consumed / surface.NA) / (
        t * surface.Area)
    H2O_flux = (surface.num_H2O_produced / surface.NA) / (
        t * surface.Area)
    # print(surface.index, surface.num_CO_ad, surface.num_H2_ad)
    return [surface, CO_flux, CO2_flux, H2_flux, H2O_flux]

if __name__ == "__main__":
    u = np.zeros(mx_shape)
    m = u.copy()
    t = 0.0
    # initialize concentration in the pore
    sol = fsolve(pde_bk, m, args=(
                 u, GasSpecies,
                 Flux, BulkConcentrations)).reshape(mx_shape)
    t = t + dt
    u = sol.copy()
    C = {'co': u[(GasSpecies.index('CO') + 1) * Ny - 1, :],
         'h2': u[(GasSpecies.index('H2') + 1) * Ny - 1, :]}
    CO[1, :, :] = u[:Ny, :].reshape(Ny, Nx)
    CO2[1, :, :] = u[Ny:2 * Ny, :].reshape(Ny, Nx)
    H2[1, :, :] = u[2 * Ny:3 * Ny, :].reshape(Ny, Nx)
    H2O[1, :, :] = u[3 * Ny:4 * Ny, :].reshape(Ny, Nx)
    C_pore = {'CO': u[0, -1],
              'CO2': u[Ny, -1],
              'H2': u[2 * Ny, -1],
              'H2O': u[3 * Ny, -1]
              }
    CO_p[1] = C_pore['CO']
    H2_p[1] = C_pore['H2']
    CO2_p[1] = C_pore['CO2']
    H2O_p[1] = C_pore['H2O']
    Cb0 = [
        BulkConcentrations['CO'],
        BulkConcentrations['CO2'],
        BulkConcentrations['H2'],
        BulkConcentrations['H2O']]
    BulkConcentrations = c_bulk(Cb0, t_arr[:2], k, C_pore)
    CO_b[1] = BulkConcentrations['CO']
    H2_b[1] = BulkConcentrations['H2']
    CO2_b[1] = BulkConcentrations['CO2']
    H2O_b[1] = BulkConcentrations['H2O']
    results = []
    param_list = get_param_list(surface_stash, C['co'], C['h2'])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_generator = executor.map(get_flux, param_list)
    for result in result_generator:
        results.append(result)
    del(result_generator)
    for i in range(Nkmc):
        surface_stash[i], Flux_CO[0, i], Flux_CO2[0, i], Flux_H2[0, i],\
            Flux_H2O[0, i] = results[i]
    x = np.linspace(0, 1, num=Nkmc)
    fCO = interpolate.interp1d(x, Flux_CO)
    fCO2 = interpolate.interp1d(x, Flux_CO2)
    fH2 = interpolate.interp1d(x, Flux_H2)
    fH2O = interpolate.interp1d(x, Flux_H2O)

    F_CO = fCO(x_arr)
    F_CO2 = fCO2(x_arr)
    F_H2 = fH2(x_arr)
    F_H2O = fH2O(x_arr)
    Flux = {'CO': F_CO,
            'CO2': F_CO2,
            'H2': F_H2,
            'H2O': F_H2O}

    tic = timeit.default_timer()
    for t in tqdm(range(2, Nt)):
        macro_sol = fsolve(pde_bk, m, args=(
            u, GasSpecies,
            Flux, BulkConcentrations)).reshape(mx_shape)
        u = macro_sol.copy()
        C = {'co': u[(GasSpecies.index('CO') + 1) * Ny - 1, :],
             'h2': u[(GasSpecies.index('H2') + 1) * Ny - 1, :]}
        for i in range(Nx):
            if C['co'][i] < 0:
                C['co'][i] = 0
            if C['h2'][i] < 0:
                C['h2'][i] = 0
        results = []
        param_list = get_param_list(surface_stash, C['co'], C['h2'])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result_generator = executor.map(get_flux, param_list)
        for result in result_generator:
            results.append(result)
        del(result_generator)
        for i in range(Nkmc):
            surface_stash[i], Flux_CO[0, i], Flux_CO2[0, i], Flux_H2[0, i],\
                Flux_H2O[0, i] = results[i]

        if t % int(0.1 * Nt) == 0:
            COV_CO.append([
                surface_stash[i].num_CO_ad / length ** 2 for i in range(Nkmc)])
            COV_H2.append([
                surface_stash[i].num_H2_ad / length ** 2 for i in range(Nkmc)])
            COV_H.append([
                surface_stash[i].num_H_ad / length ** 2 for i in range(Nkmc)])

        # if t in list(map(one_tenth, range(0, Nt, int(0.1 * Nt)))):
        #     COV_CO[int(t / int(0.1 * Nt))] = np.array(
        #         [surface_stash[i].num_CO_ad / length ** 2 for i in range(
        #             Nkmc)])
        #     COV_H2[int(t / int(0.1 * Nt))] = np.array(
        #         [surface_stash[i].num_H2_ad / length ** 2 for i in range(
        #             Nkmc)])
        #     COV_H[int(t / int(0.1 * Nt))] = np.array(
        #         [surface_stash[i].num_H_ad / length ** 2 for i in range(Nkmc)])

        x = np.linspace(0, 1, num=Nkmc)
        fCO = interpolate.interp1d(x, Flux_CO)
        fCO2 = interpolate.interp1d(x, Flux_CO2)
        fH2 = interpolate.interp1d(x, Flux_H2)
        fH2O = interpolate.interp1d(x, Flux_H2O)
        F_CO = fCO(x_arr)
        F_CO2 = fCO2(x_arr)
        F_H2 = fH2(x_arr)
        F_H2O = fH2O(x_arr)
        Flux = {'CO': F_CO,
                'CO2': F_CO2,
                'H2': F_H2,
                'H2O': F_H2O}
        CO[t, :, :] = macro_sol[:Ny, :].reshape(Ny, Nx)
        CO2[t, :, :] = macro_sol[Ny:2 * Ny, :].reshape(Ny, Nx)
        H2[t, :, :] = macro_sol[2 * Ny:3 * Ny, :].reshape(Ny, Nx)
        H2O[t, :, :] = macro_sol[3 * Ny:4 * Ny, :].reshape(Ny, Nx)
        C_pore = {'CO': u[0, -1],
                  'CO2': u[Ny, -1],
                  'H2': u[2 * Ny, -1],
                  'H2O': u[3 * Ny, -1]
                  }
        CO_p[t] = C_pore['CO']
        H2_p[t] = C_pore['H2']
        CO2_p[t] = C_pore['CO2']
        H2O_p[t] = C_pore['H2O']
        Cb0 = [
            BulkConcentrations['CO'],
            BulkConcentrations['CO2'],
            BulkConcentrations['H2'],
            BulkConcentrations['H2O']]
        if t == Nt - 1:
            t_in = t_arr[t:]
        else:
            t_in = t_arr[t:t + 2]
        BulkConcentrations = c_bulk(Cb0, t_in, k, C_pore)
        CO_b[t] = BulkConcentrations['CO']
        H2_b[t] = BulkConcentrations['H2']
        CO2_b[t] = BulkConcentrations['CO2']
        H2O_b[t] = BulkConcentrations['H2O']
    toc = timeit.default_timer()
    print('simulation time:', (toc - tic))

    plot_bulk(t_arr, CO2_b, H2O_b, CO_b, H2_b)
    plot_coverage(Nkmc, COV_CO, COV_H, COV_H2)
    plot_C_profile(x_arr, y_arr, CO, CO2, H2, H2O)

    # ctime = time.localtime()
    # filename = os.path.abspath(
    #     os.path.dirname(
    #         __file__)) + f'\\{time.strftime("%Hh%Mm%Ss-%Y-%m-%d", ctime)}.out'
    filename = 'results'
    with open(f'{filename}.dill', 'wb') as f:
        dill.dump_session(f)

    exit()
