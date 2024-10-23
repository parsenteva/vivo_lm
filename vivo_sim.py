#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:35:29 2023

@author: polina
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from vivo_lm import fit_lm_mm, fit_lm_ot, calculate_bootstrap_ci
mpl.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

def vivo_simulation(beta_0, beta_1, nb_groups=2, nb_anim=100, rho=5, 
                    gaussian=True, nb_sim=1, scale_mu=10, delta_mu=2, sigma_sq_x=1, 
                    prop_x=0.5, mu_to_sigma=None, return_var_beta=False,
                    plot_distrib=False, plot_box=False, random_gen=None):
    """
    Simulate the predictor x and the predicted variable y according
    to the linear model in the case when they cannot be observed simultaneously.
    
    """
    if random_gen:
        random_gen = random_gen
    else:
        random_gen = np.random
    groups = np.arange(nb_groups)
    nb_anim_obs_x = int(nb_anim * prop_x)
    mu_x = delta_mu * groups + scale_mu
    if gaussian:
        x_sim = (random_gen.normal(mu_x, np.sqrt(sigma_sq_x),
                                    (nb_anim, nb_sim, len(groups)))
                   .transpose((0, 2, 1)).squeeze())
        if plot_distrib:
            x_plot = np.linspace(x_sim.min(), x_sim.max(), 100)
            for i, mu in enumerate(mu_x):
                sgm = sigma_sq_x[i] if hasattr(sigma_sq_x, "__len__") else sigma_sq_x
                plt.plot(x_plot, stats.norm.pdf(x=x_plot, loc=mu, scale=sgm))
            fig_num = 1
            flg = True
            while flg:
                if os.path.exists(f'distrib_{fig_num}.pdf'):
                    fig_num += 1
                else:
                    plt.savefig(f'distrib_{fig_num}.pdf')
                    flg = False
            plt.show()
    else:
        delta_mu_u = np.sqrt(12 * sigma_sq_x)
        x_sim = np.zeros((nb_anim, len(groups), nb_sim)).squeeze()
        for d in range(len(groups)):
            x_sim[:, d] = (random_gen.uniform(mu_x[d] - 0.5 * delta_mu_u, 
                                               mu_x[d] + 0.5 * delta_mu_u, 
                                               (nb_anim, nb_sim)).squeeze())
    x_sim_obs = x_sim[:nb_anim_obs_x, :]
    if beta_1 != 0:
        sigma_sq_eps = beta_1**2 * np.array(sigma_sq_x) / (rho**2 - 1)
    else:
        sigma_sq_eps = sigma_sq_x
    epsilon = (random_gen.normal(0, np.sqrt(sigma_sq_eps),
                                 (nb_anim, nb_sim, len(groups)))
               .transpose((0, 2, 1)).squeeze())
    y_sim = beta_0 + beta_1 * x_sim + epsilon
    y_sim_obs = y_sim[nb_anim_obs_x:, :]
    if plot_box:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(data=x_sim_obs, ax=axs[0])
        sns.boxplot(data=y_sim_obs, ax=axs[1])
        fig_num = 1
        flg = True
        while flg:
            if os.path.exists(f'box_{fig_num}.pdf'):
                fig_num += 1
            else:
                plt.savefig(f'box_{fig_num}.pdf')
                flg = False
        plt.show()
    return x_sim_obs, y_sim_obs

def plot_ci_heatmaps(filename, increasing=True, title=None, savefilename=None):
    """
    Produce heatmap plots summarizing results of a simulation study.
    
    """
    xls_file = pd.ExcelFile(filename)
    sheets_dict = pd.read_excel(xls_file, None)
    conds_lvl_1 = sheets_dict.keys()
    all_cond_values = []
    for c in list(conds_lvl_1):
        cond_df = sheets_dict[c]
        cond_values = cond_df.values[:, 1:].astype(float)
        all_cond_values.append(cond_values)
    all_cond_values = np.array(all_cond_values)
    global_min = all_cond_values.min()
    global_max = all_cond_values.max()
    half_spectrum = global_min + (global_max - global_min) / 2
    if increasing is True:
        cmap = 'Greens'
    else:
        cmap = 'Greens_r'
    fig, axes = plt.subplots(1, len(conds_lvl_1), figsize=(8 * len(conds_lvl_1), 6))
    for i1, c1 in enumerate(conds_lvl_1):
        pos = axes[i1].imshow(all_cond_values[i1], aspect='auto', cmap=cmap,
                             vmin=global_min, vmax=global_max)
        rect = plt.Rectangle((0, 0), 1, 1, color='lightgrey', linewidth=3,
                             transform=axes[i1].transAxes, zorder=-1)
        fig.patches.append(rect)
        axes[i1].axis('off')
        # add the values
        for (i, j), value in np.ndenumerate(all_cond_values[i1],):
            axes[i1].text(j, i, f'{value:#.2f}', va='center', ha='center', fontsize=27,
                    color=('black' if (((value < half_spectrum) and increasing)
                                       or ((value > half_spectrum) and not increasing))
                           else 'white'))
        conds_lvl_2 = sheets_dict[c1].columns[1:]
        for i2, c2 in enumerate(conds_lvl_2):
            y_coord = len(sheets_dict[c1].values[:, 0]) - 0.3
            axes[i1].text(i2, y_coord, c2.replace(', ', '\n'),
                          va='center', ha='center', fontsize=12)
        axes[i1].set_title(c1, fontsize=20)
    methods = sheets_dict[c1].values[:, 0]
    for i3, m in enumerate(methods):
        axes[0].text(-1.1, i3+0.1, methods[i3], va='center', ha='center', fontsize=17)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar_ax.axis('off')
    fig.colorbar(pos, ax=cbar_ax)
    if title is not None:
        fig.suptitle(title, y = 1, fontsize=30)
    if savefilename is not None:
        plt.savefig(f"{savefilename}.pdf")
    plt.show()
    
    
methods = ['mm (asymp)', 'mm (boot)', 'ot (boot)', 'mm (student)']
beta_0 = 1
beta_1 = 2
delta_mu = 1
nb_anim_lvls = [10, 30]
nb_groups_lvls = [4, 10]
sigma_sq_x_lvls = [0.75, 2]
rho_lvls = [1.01, 1.1]
nb_sim = 100
nb_boot = 500
rand_seed = 0
all_cond_mean_ampl = np.zeros((4, 4, len(methods) - 1))
all_cond_include_beta_1 = np.zeros((4, 4, len(methods) - 1))
all_cond_power = np.zeros((4, 4, len(methods) - 1))
all_cond_beta_estim_lr = np.zeros((4, 4, nb_sim))
all_cond_beta_std_lr = np.zeros((4, 4, nb_sim))
all_cond_ci_delta_lr = np.zeros((4, 4, nb_sim))

### Simulation study:
print('Simulation parameters:')
for c1, sigma_sq_x in enumerate(sigma_sq_x_lvls):
    print('sigma_sq_x =', sigma_sq_x)
    for c2, rho in enumerate(rho_lvls):
        print('rho =', rho)
        for c3, nb_anim in enumerate(nb_anim_lvls):
            print('nb_anim =', nb_anim)
            for c4, nb_groups in enumerate(nb_groups_lvls):
                print('nb_groups =', nb_groups)
                betas_estim = np.zeros((2, nb_sim))
                sgm_beta_1_hat = np.zeros((nb_sim))
                all_boot_betas = np.zeros((nb_sim, 2, nb_boot))
                all_ci = np.zeros((nb_sim, 3, 2))
                random_gen = np.random.RandomState(rand_seed)
                for i in range(nb_sim):
                    if i%100 == 0 and i != 0:
                        print(f'Simulation {i+1}')
                    x_sim_obs, y_sim_obs = vivo_simulation(beta_0, beta_1, nb_groups=nb_groups, 
                                                           nb_anim=nb_anim * 2, delta_mu=delta_mu, 
                                                           plot_box=False, 
                                                           plot_distrib=False,
                                                           sigma_sq_x=sigma_sq_x, 
                                                           random_gen=random_gen, rho=rho)
                    # Estimating regression coefficients:
                    (beta_0_init, 
                     betas_estim[0, i],
                     sgm_beta_1_hat[i]) = fit_lm_mm(x_sim_obs, y_sim_obs)
                    
                    d = stats.norm(scale=np.sqrt(sgm_beta_1_hat[i] / nb_anim)).ppf(0.975)
                    all_ci[i, 0] = [betas_estim[0, i] - d, betas_estim[0, i] + d]
                    
                    betas_estim[1, i] = fit_lm_ot(x_sim_obs, y_sim_obs,
                                                  beta_1_init=np.random.uniform(-5, 5),
                                                  beta_0_init=np.random.uniform(-5, 5))[1]
                    # Calculating bootstrap confidence intervals:
                    all_boot_betas[i], all_ci[i, 1:] = calculate_bootstrap_ci(x_sim_obs, 
                                                                          y_sim_obs, 
                                                                          nb_boot=nb_boot)
                    # Linear regression on means per group (naive method):
                    x_sim_means = x_sim_obs.mean(axis=0)
                    y_sim_means = y_sim_obs.mean(axis=0)
                    lin_reg_means = stats.linregress(x_sim_means, y_sim_means)
                    # Two-sided inverse Students t-distribution
                    ts = abs(stats.t.ppf(0.05 / 2, len(x_sim_means)-2))
                    all_cond_beta_estim_lr[2 * c3 + c4, 
                                           2 * c1 + c2, i] = lin_reg_means.slope
                    all_cond_beta_std_lr[2 * c3 + c4, 
                                         2 * c1 + c2, i] = lin_reg_means.stderr
                    all_cond_ci_delta_lr[2 * c3 + c4, 
                                         2 * c1 + c2, i] = ts * lin_reg_means.stderr
                    
                all_cond_mean_ampl[2 * c3 + c4,
                                   2 * c1 + c2] = (all_ci[:, :, 1] 
                                                 - all_ci[:, :, 0]).mean(axis=0)
                all_cond_include_beta_1[2 * c3 + c4, 
                                        2 * c1 + c2] = ((all_ci[:, :, 0] <= beta_1) 
                                                        & (all_ci[:, :, 1] >= beta_1)).mean(axis=0)
                all_cond_power[2 * c3 + c4, 
                               2 * c1 + c2] = ((all_ci[:, :, 0] >= 0) 
                                               | (all_ci[:, :, 1] <= 0)).mean(axis=0)

x_ticks = np.arange(4)
x_ticks_d = 0.075
### Save results:
folder_name = "Results_sim_data"
scnd_dim_labels = [rf'sigma2x={sigma_sq_x_lvls[0]}, rho={rho_lvls[0]}', 
                   rf'sigma2x={sigma_sq_x_lvls[0]}, rho={rho_lvls[1]}', 
                   rf'sigma2x={sigma_sq_x_lvls[1]}, rho={rho_lvls[0]}', 
                   rf'sigma2x={sigma_sq_x_lvls[1]}, rho={rho_lvls[1]}']
lr_ci_ampl = 2 * all_cond_ci_delta_lr.mean(axis=2)
lr_cov_rate = ((all_cond_beta_estim_lr - all_cond_ci_delta_lr <= beta_1) 
               & (all_cond_beta_estim_lr + all_cond_ci_delta_lr >= beta_1)).mean(axis=2)
lr_power = ((all_cond_beta_estim_lr - all_cond_ci_delta_lr >= 0) 
            | (all_cond_beta_estim_lr + all_cond_ci_delta_lr <= 0)).mean(axis=2)

# Coverage rate:
coverage_rate_dict = {'Nb. anim.=10, Nb. groups.=4' : {}, 
                      'Nb. anim.=10, Nb. groups.=10' : {}, 
                      'Nb. anim.=30, Nb. groups.=4' : {}, 
                      'Nb. anim.=30, Nb. groups.=10' : {}}
writer = pd.ExcelWriter(folder_name + "/sim_coverage_rate.xlsx",
                        engine="xlsxwriter")
for i, key in enumerate(coverage_rate_dict.keys()):
    for j in range(4):
        all_cond_ij = np.append(all_cond_include_beta_1[i, j], lr_cov_rate[i, j])
        coverage_rate_dict[key][scnd_dim_labels[j]] = all_cond_ij
    coverage_rate_df = pd.DataFrame.from_dict(coverage_rate_dict[key])
    coverage_rate_df.index = methods
    coverage_rate_df.to_excel(writer, sheet_name=key)
writer.close()

# Amplitude:
ci_amplitude_dict = {'Nb. anim.=10, Nb. groups.=4' : {}, 
                     'Nb. anim.=10, Nb. groups.=10' : {}, 
                     'Nb. anim.=30, Nb. groups.=4' : {}, 
                     'Nb. anim.=30, Nb. groups.=10' : {}}
writer = pd.ExcelWriter(folder_name + "/sim_ci_amplitude.xlsx",
                        engine="xlsxwriter")
for i, key in enumerate(ci_amplitude_dict.keys()):
    for j in range(4):
        all_cond_ij = np.append(all_cond_mean_ampl[i, j], lr_ci_ampl[i, j])
        ci_amplitude_dict[key][scnd_dim_labels[j]] = all_cond_ij
    ci_amplitude_df = pd.DataFrame.from_dict(ci_amplitude_dict[key])
    ci_amplitude_df.index = methods
    ci_amplitude_df.to_excel(writer, sheet_name=key)
writer.close()

# Power:
ci_power_dict = {'Nb. anim.=10, Nb. groups.=4' : {}, 
                 'Nb. anim.=10, Nb. groups.=10' : {}, 
                 'Nb. anim.=30, Nb. groups.=4' : {}, 
                 'Nb. anim.=30, Nb. groups.=10' : {}}
writer = pd.ExcelWriter(folder_name + "/sim_ci_power.xlsx", engine="xlsxwriter")
for i, key in enumerate(ci_power_dict.keys()):
    for j in range(4):
        all_cond_ij = np.append(all_cond_power[i, j], lr_power[i, j])
        ci_power_dict[key][scnd_dim_labels[j]] = all_cond_ij
    ci_power_df = pd.DataFrame.from_dict(ci_power_dict[key])
    ci_power_df.index = methods
    ci_power_df.to_excel(writer, sheet_name=key)
writer.close()

### Plot heatmaps:
plot_ci_heatmaps(folder_name + "/sim_coverage_rate_tmp.xlsx", title="Coverage rate")
plot_ci_heatmaps(folder_name + "/sim_ci_amplitude_tmp.xlsx", increasing=False, title="Amplitude")
plot_ci_heatmaps(folder_name + "/sim_ci_power_tmp.xlsx", title="Power")
