#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:59:18 2023

@author: apollinaria45
"""
import time
import os
from itertools import product
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from vivo_lm import fit_lm_mm, fit_lm_ot, calculate_bootstrap_ci
import warnings
warnings.filterwarnings("ignore")
mpl.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'
np.random.seed(0)

def preprocess_data(table_with_dummies, x_names, y_names, select=None, disp=1):
    """
    Process data with dummies for predictor, predicted and grouping 
    variables, namely by extracting data for given dummy levels if select 
    is not None, and identifying group levels present in for all predictors
    and predicted variables.
    
    """
    if select is not None:
        select_table_dummies = table_with_dummies[table_with_dummies.loc[:, select]
                                                  .prod(axis=1).astype(bool)]
    else:
        select_table_dummies = table_with_dummies.copy()
    common_groups = select_table_dummies['Group'][select_table_dummies[y_names[0]].notna()].unique()
    for vn in y_names[1:] + x_names:
        groups = select_table_dummies['Group'][select_table_dummies[vn].notna()].unique()
        common_groups = np.intersect1d(common_groups, groups)
    select_table_dummies.reset_index(drop=True, inplace=True)
    select_table_dummies = select_table_dummies[select_table_dummies['Group']
                                                .isin(common_groups)]
    if select_table_dummies.empty:
        if disp:
            print("No common groups for the given variable selection.")
        return
    return select_table_dummies, common_groups

def extract_observations(select_table_dummies, x_names, y_names):
    """
    Extract arrays with observations for selected variables x and y in a form
    of 2-dimensional arrays with the second dimension corresponding to groups.
    
    """
    subdatasets_y = []
    for yn in y_names:
        data = select_table_dummies[select_table_dummies[yn].notna()]
        subdatasets_y.append(data.pivot(columns='Group', values=yn).values)
    subdatasets_x = []
    for xn in x_names:
        data = select_table_dummies[select_table_dummies[xn].notna()]
        subdatasets_x.append(data.pivot(columns='Group', values=xn).values)
    return subdatasets_x, subdatasets_y

def scale_data(subdatasets_x, subdatasets_y, loc_x=None, scale_x=None, 
               loc_y=None, scale_y=None):
    """
    Scale data subsets for comparability.

    """
    scaled_subdatasets_x = []
    scaled_subdatasets_y = []
    for x in subdatasets_x:
        if loc_x is not None:
            scaled_x = x - loc_x
        else:
            scaled_x = x - np.nanmean(x)
        if scale_x is not None:
            scaled_x /=  scale_x
        else:
            scaled_x /=  np.sqrt(np.nanvar(x))
        scaled_subdatasets_x.append(scaled_x)
    for y in subdatasets_y:
        if loc_y is not None:
            scaled_y = y - loc_y
        else:
            scaled_y = y - np.nanmean(y)
        if scale_y is not None:
            scaled_y /=  scale_y
        else:
            scaled_y /=  np.sqrt(np.nanvar(y))
        scaled_subdatasets_y.append(scaled_y)
    return scaled_subdatasets_x, scaled_subdatasets_y

def regress_data(subdatasets_x, subdatasets_y, ci_level=0.95, nb_boot=100):
    """
    Estimate linear regression coefficients and confidence intervals for all
    combinations of variables x and y in subdatasets, with all considered 
    methods: method of moments (bootstrap), optimal transport (bootstrap), and
    the naive method.

    """
    nb_x = len(subdatasets_x)
    nb_y = len(subdatasets_y)
    betas = np.zeros((nb_x, nb_y, 3, 2))
    beta_1_stats = np.zeros((nb_x, nb_y, 3, 3))
    for i, x_obs in enumerate(subdatasets_x):
        for j, y_obs in enumerate(subdatasets_y):
            # Get rid of (most of) NaNs:
            nb_groups = x_obs.shape[1]
            x_nonnanind = ~np.isnan(x_obs)
            x_nobs = (x_nonnanind).sum(axis=0)
            x_obs_clean = np.empty((x_nobs.max(), nb_groups))
            x_obs_clean[:] = np.nan
            y_nonnanind = ~np.isnan(y_obs)
            y_nobs = (y_nonnanind).sum(axis=0)
            y_obs_clean = np.empty((y_nobs.max(), nb_groups))
            y_obs_clean[:] = np.nan
            for c in range(nb_groups):
                x_obs_clean[: x_nobs[c], c] = x_obs[:, c][x_nonnanind[:, c]]
                y_obs_clean[: y_nobs[c], c] = y_obs[:, c][y_nonnanind[:, c]]
            ## Analysis:
            all_boot_betas = np.zeros((2, nb_boot))
            all_ci = np.zeros((2, 2))
            betas[i, j, 0] = fit_lm_mm(x_obs_clean, y_obs_clean)[:-1]
            betas[i, j, 1] = fit_lm_ot(x_obs_clean, y_obs_clean,
                                       beta_1_init=np.random.uniform(-5, 5),
                                       beta_0_init=np.random.uniform(-5, 5))[:-1]
            # Simple LR on means:
            x_means = np.nanmean(x_obs_clean, axis=0)
            y_means = np.nanmean(y_obs_clean, axis=0)
            lin_reg_means = stats.linregress(x_means, y_means)
            betas[i, j, 2, 0] = lin_reg_means.intercept
            betas[i, j, 2, 1] = lin_reg_means.slope
            # Two-sided inverse Students t-distribution
            ts = abs(stats.t.ppf(0.05 / 2, len(x_means)-2))
            beta_1_stats[i, j, 2, 0] = lin_reg_means.stderr
            ci_delta_lr = ts * lin_reg_means.stderr
            beta_1_stats[i, j, 2, 1] = lin_reg_means.slope - ci_delta_lr
            beta_1_stats[i, j, 2, 2] = lin_reg_means.slope + ci_delta_lr
            # Bootstrap:
            all_boot_betas, all_ci = calculate_bootstrap_ci(x_obs_clean, y_obs_clean, 
                                                            nb_boot=nb_boot,
                                                            conf_lvl=ci_level)
            
            beta_1_stats[i, j, :-1, 0] = all_boot_betas.var(axis=1, ddof=1)
            beta_1_stats[i, j, :-1, 1 :] = all_ci.copy()
    return betas, beta_1_stats

def data_boxplot(data, xn, yn, plot_title=None, decode_groups=None, 
                 group_order=None, save_plots=False, 
                 save_as='my_boxplot.png'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    xtkz = [t.get_text()  for t in axs[0].get_xticklabels()]
    sns.boxplot(data=data, x='Group', 
                y=xn, ax=axs[0], order=group_order)
    
    axs[0].set_title(xn)
    if decode_groups is not None:
        xtkz = [t.get_text()  for t in axs[0].get_xticklabels()]
        axs[0].set_xticklabels([decode_groups[c] for c in xtkz])
    sns.boxplot(data=data, x='Group', 
                y=yn, ax=axs[1], order=group_order)
    axs[1].set_title(yn)
    if decode_groups is not None:
        ytkz = [t.get_text()  for t in axs[1].get_xticklabels()]
        axs[1].set_xticklabels([decode_groups[c] for c in ytkz])
    plt.suptitle(plot_title)
    if save_plots:
        plt.savefig(save_as)
    plt.show()

def run_model_on_subsets(table_with_dummies, x_names, y_names, 
                         var_combinations=None, loc_x=None, scale_x=None, 
                         loc_y=None, scale_y=None, disp=0, disp_plot=False, 
                         decode_groups=None,  ordered_groups=None, 
                         save_plots=False, ci_level=0.95, nb_boot=100):
    """
    Run the entire pipeline on raw data: preprocessing, susbets identification,
    and regression estimation.

    """
    results_dict = {}
    method_labels = ['mm', 'ot', 'SimpleLR']
    if save_plots:
        folder_name = str(time.time()).replace('.', '')
        os.makedirs(folder_name)    
    for c in var_combinations:
        key_c = c[0] 
        for ci in c[1:-1]:
            key_c.join('_' + ci)
        key_c += '_' + c[-1]
        if disp:
            print(key_c)
        results_dict[key_c] = {}
        try:
            (select_table_dummies, 
             common_groups) = preprocess_data(table_with_dummies, x_names, 
                                              y_names, select=c, disp=disp)
            (subdatasets_x, subdatasets_y) = extract_observations(select_table_dummies, 
                                                                  x_names, y_names)
            (scaled_subdatasets_x,
             scaled_subdatasets_y) = scale_data(subdatasets_x, subdatasets_y, 
                                                loc_x=loc_x, scale_x=scale_x, 
                                                loc_y=loc_y, scale_y=scale_y)
            (betas, beta_1_stats) = regress_data(scaled_subdatasets_x,
                                                 scaled_subdatasets_y, 
                                                 ci_level=ci_level, 
                                                 nb_boot=nb_boot)
            for i, xn in enumerate(x_names):
                for j, yn in enumerate(y_names):
                    key_n = xn + '_' + yn
                    signif = np.sign(beta_1_stats[i, j, :, 1]) == np.sign(beta_1_stats[i, j, :, 2])
                    results_dict[key_c][key_n] = {'beta_0' : dict(zip(method_labels, 
                                                                      betas[i, j, :, 0])),
                                                  'beta_1' : dict(zip(method_labels, 
                                                                      betas[i, j, :, 1])),
                                                  'var_beta_1_hat' : dict(zip(method_labels, 
                                                                              beta_1_stats[i, j, :, 0])),
                                                  'CI_l_bound' : dict(zip(method_labels, 
                                                                          beta_1_stats[i, j, :,  1])),
                                                  'CI_u_bound' : dict(zip(method_labels, 
                                                                          beta_1_stats[i, j, :, 2])),
                                                  'Significant' : dict(zip(method_labels, 
                                                                           signif))}
                    if disp_plot:
                        group_order = np.array(ordered_groups)[np.in1d(ordered_groups,
                                                                       common_groups)]
                        save_as=(folder_name + f'/{key_c}_{i}{j}.png' if save_plots 
                                 else None)
                        data_boxplot(select_table_dummies, xn, yn, 
                                     plot_title=key_c, decode_groups=decode_groups, 
                                     group_order=group_order, 
                                     save_plots=save_plots, 
                                     save_as=save_as)
        except TypeError:
            pass
    return results_dict

### Data import and preprocessing (SBRT):
histo_data_SBRT = pd.read_excel("Data/Table_Histo_SBRT.xlsx")
gene_data_SBRT = pd.read_excel("Data/Table_Genes_SBRT.xlsx")
gene1 = 'IL1b'
gene2 = 'IL6'
gene3 = 'TNF'
histo_data_SBRT.replace({'1 mois' : '1m',
                    '3 mois' : '3m', 
                    '6 mois' : '6m', 
                    '12 mois' : '12m', 
                    '45 jours' : 'J45', 
                    '2 mois' : '2m',
                    '21 jours' : 'J21'}, inplace=True)
gene_data_SBRT.replace({'PA' : 'PATCH',
                    'PI' : 'IPSI',
                    '1mm' : 1,
                    '3mm' : 3,
                    '7mm' : 7,
                    'CT' : 0}, inplace=True)
gene_data_SBRT.rename({'Vol' : 'Volume'},  axis=1, inplace=True)
gene_data_SBRT.loc[:, 'dCT'] *= -1 
gene_data_SBRT_fmt = gene_data_SBRT.join(gene_data_SBRT.pivot(columns='Gene', 
                                                              values='dCT'))
gene_data_SBRT_fmt.drop(columns=['Gene', 'dCT'], inplace=True)
big_table_SBRT = pd.concat([histo_data_SBRT, gene_data_SBRT_fmt])


len(big_table_SBRT[(big_table_SBRT['Volume']==0)&(big_table_SBRT['Localisation']=='PD')
          &(big_table_SBRT['Time']=='1m')&(big_table_SBRT['Septa'].notna())])

big_table_SBRT_dummies = pd.get_dummies(data=big_table_SBRT, 
                                        columns=['Volume', 'Localisation', 'Time'],
                                        dtype=int)

for i, c in enumerate(['Volume_0', 'Volume_1', 'Volume_3','Volume_7', 'Volume_10']):
    big_table_SBRT_dummies.loc[:, c] *= i + 1
    
for i, c in enumerate(['Localisation_IPSI', 'Localisation_PATCH',
                       'Localisation_PD', 'Localisation_PG']):
    big_table_SBRT_dummies.loc[:, c] *= i + 1
    
for i, c in enumerate(['Time_12m', 'Time_1m', 'Time_2m', 'Time_3m', 
                       'Time_6m', 'Time_J21', 'Time_J45']):
    big_table_SBRT_dummies.loc[:, c] *= i + 1
    
big_table_SBRT_dummies['Group'] = (big_table_SBRT_dummies.iloc[:, 5:10].sum(axis=1).astype('str') + 
                                   big_table_SBRT_dummies.iloc[:, 10:14].sum(axis=1).astype('str') +
                                   big_table_SBRT_dummies.iloc[:, 14:].sum(axis=1).astype('str'))

loc_vol_combinations = list(product(big_table_SBRT_dummies.columns[10:14], 
                                    big_table_SBRT_dummies.columns[5:10]))

### Data import and preprocessing (WTI):
histo_data_WTI = pd.read_excel("Data/Table_Histo_WTI.xlsx")
gene_data_WTI = pd.read_excel("Data/Table_Genes_WTI.xlsx")

histo_data_WTI['ID'] = histo_data_WTI.index
gene_data_WTI.rename({'Assay' : 'ID',},  axis=1, inplace=True)
histo_data_WTI.rename({'Time' : 'Group'},  axis=1, inplace=True)
gene_data_WTI.replace({'NI' : 0, '1S' : np.nan, '2S' : 2, '13S' : 13, '23S' : 23},
                      inplace=True)
gene_data_WTI.dropna(axis=0, inplace=True)
gene_data_WTI.iloc[:, -3:] *= -1
big_table_WTI = pd.concat([histo_data_WTI, gene_data_WTI])

gene1bis = 'IL1a'

# Compute quantities for data scaling:
all_genes_values = np.concatenate([gene_data_SBRT['dCT'].values, 
                                  gene_data_WTI.iloc[:, -3:].values.flatten()])
all_genes_mean = all_genes_values.mean()
all_genes_std = all_genes_values.std()
all_histo_values = np.concatenate([histo_data_SBRT['Septa'].values, 
                                  histo_data_WTI['Septa'].values])
all_histo_mean = np.nanmean(all_histo_values)
all_histo_std = np.nanstd(all_histo_values)


##### ##### ##### ##### SBRT ##### ##### ##### #####
# Creating group decode dictionary with respect to time for readable boxplots:
time_cats = big_table_SBRT_dummies.columns[14 : -1]
time_decode = {}
for group_code in big_table_SBRT_dummies['Group'].unique():
    time_ind = int(group_code[2]) - 1
    time_decode[group_code] = time_cats[time_ind][5 :]
# Reordering groups with respect to time for readable boxplots:
ordered_time_cats = time_cats[np.array([-2, 1, -1, 2, 3, 4, 0])]
ordered_time_group_codes = []
for time_group in ordered_time_cats:
    filter_func = lambda k: time_decode[k] == time_group[5 :]
    ordered_time_group_codes.extend(list(filter(filter_func, time_decode)))

# Analysis:
sbrt_results_dict = run_model_on_subsets(big_table_SBRT_dummies,
                                         [gene1, gene2, gene3], ['Septa'],
                                         loc_vol_combinations, disp=1,
                                         loc_x=all_genes_mean,
                                         loc_y=all_histo_mean,
                                         scale_x=all_genes_std,
                                         scale_y=all_histo_std,
                                         decode_groups=time_decode,
                                         ordered_groups=ordered_time_group_codes,
                                         ci_level=0.95, nb_boot=1000)
sbrt_results_dict  = {k: v for k, v in sbrt_results_dict.items() if v}
sbrt_results_df = pd.DataFrame.from_dict({(i,j,k): sbrt_results_dict[i][j][k] 
                                          for i in sbrt_results_dict.keys() 
                                          for j in sbrt_results_dict[i].keys()
                                          for k in sbrt_results_dict[i][j].keys()},
                                         orient='index')

##### ##### ##### ##### WTI ##### ##### ##### #####
(select_table_dummies_WTI, 
 common_groups_WTI) = preprocess_data(big_table_WTI, [gene1bis, gene2, gene3], ['Septa'])
(subdatasets_x_WTI, subdatasets_y_WTI) = extract_observations(select_table_dummies_WTI, 
                                                      [gene1bis, gene2, gene3],
                                                      ['Septa'])
(scaled_subdatasets_x_WTI,
 scaled_subdatasets_y_WTI) = scale_data(subdatasets_x_WTI, subdatasets_y_WTI,
                                        loc_x=all_genes_mean,
                                        loc_y=all_histo_mean,
                                        scale_x=all_genes_std,
                                        scale_y=all_histo_std,)
(betas_WTI, beta_1_stats_WTI) = regress_data(scaled_subdatasets_x_WTI,
                                             scaled_subdatasets_y_WTI, 
                                             ci_level=0.95)
wti_results_dict = {}
method_labels = ['mm', 'ot', 'SimpleLR']
for i, g_name in enumerate([gene1bis, gene2, gene3]):
    signif_WTI = np.sign(beta_1_stats_WTI[i, 0, :, 1]) == np.sign(beta_1_stats_WTI[i, 0, :, 2])
    wti_results_dict[g_name] = {'beta_0' : dict(zip(method_labels, 
                                                    betas_WTI[i, 0, :, 0])),
                                'beta_1' : dict(zip(method_labels, 
                                                    betas_WTI[i, 0, :, 1])),
                                'var_beta_1_hat' : dict(zip(method_labels, 
                                                            beta_1_stats_WTI[i, 0, :, 0])),
                                'CI_l_bound' : dict(zip(method_labels, 
                                                        beta_1_stats_WTI[i, 0, :, 1])),
                                'CI_u_bound' : dict(zip(method_labels, 
                                                        beta_1_stats_WTI[i, 0, :, 2])),
                                'Significant' : dict(zip(method_labels, 
                                                         signif_WTI))}
wti_results_nested_dict = {'WTI' : wti_results_dict}

wti_results_df = pd.DataFrame.from_dict({(i,j,k): wti_results_nested_dict[i][j][k]
                                          for i in wti_results_nested_dict.keys() 
                                          for j in wti_results_nested_dict[i].keys()
                                          for k in wti_results_nested_dict[i][j].keys()},
                                         orient='index')


### Combining and saving results:
vol_effect_results_df = pd.concat([sbrt_results_df, wti_results_df])

ind_groups, ind_genes, ind_stats = zip(*vol_effect_results_df.index)
loc_n_vol = pd.Series(ind_groups).str.rsplit('_', expand=True)
loc_n_vol.loc[144:, 1] = loc_n_vol.loc[144:, 0].values
loc_n_vol.loc[144:, 3] = loc_n_vol.loc[144:, 0].values
vol_effect_results_df.loc[:, 'Localisation'] = loc_n_vol[1].values
vol_effect_results_df.loc[:, 'Volume'] = loc_n_vol[3].values

vol_effect_results_df.round(2).to_excel('Results_real_data/Volume_effect_results.xlsx')


### Plot examples:
# Example of a data boxplot:
(select_table_dummies_1, 
 common_groups_1) = preprocess_data(big_table_SBRT_dummies, 
                                    [gene1, gene2, gene3], ['Septa'],
                                    select=['Volume_3', 'Localisation_PATCH'])
group_order_1 = np.array(ordered_time_group_codes)[np.in1d(ordered_time_group_codes,
                                               common_groups_1)]
data_boxplot(select_table_dummies_1, 'IL6', 'Septa', plot_title='Vol. 3 mm, PATCH', 
             decode_groups=time_decode, group_order=group_order_1)

# Plot model prediction:
x_vals = np.linspace(-19, -14, 100)
dct_1 = sbrt_results_dict['Localisation_PATCH_Volume_1']['IL6_Septa']
dct_2 = sbrt_results_dict['Localisation_PATCH_Volume_3']['IL6_Septa']
dct_3 = wti_results_dict['IL6']
renormed_reg = lambda x, beta, means, stds: (stds[1] * (beta[0] 
                                                        + (means[1] - beta[1] 
                                                           * means[0]) / stds[0]),
                                             beta[1] * stds[1] / stds[0])
coefs_1_mm = renormed_reg(x_vals, (dct_1['beta_0']['mm'], dct_1['beta_1']['mm']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))
coefs_1_ot = renormed_reg(x_vals, (dct_1['beta_0']['ot'], dct_1['beta_1']['ot']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))
coefs_2_mm = renormed_reg(x_vals, (dct_2['beta_0']['mm'], dct_2['beta_1']['mm']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))
coefs_2_ot = renormed_reg(x_vals, (dct_2['beta_0']['ot'], dct_2['beta_1']['ot']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))
coefs_3_mm = renormed_reg(x_vals, (dct_3['beta_0']['mm'], dct_3['beta_1']['mm']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))
coefs_3_ot = renormed_reg(x_vals, (dct_3['beta_0']['ot'], dct_3['beta_1']['ot']), 
                          (all_genes_mean, all_histo_mean), (all_genes_std, all_histo_std))

fig = plt.figure(1, (10, 5))
sns.set_context('notebook')
plt.plot(x_vals, coefs_1_mm[0] + coefs_1_mm[1] * x_vals, '-b', label='Patch 1mm (mm)')
plt.plot(x_vals, coefs_1_ot[0] + coefs_1_ot[1] * x_vals, '--b', label='Patch 1mm (ot)')
plt.plot(x_vals, coefs_2_mm[0] + coefs_2_mm[1] * x_vals, '-g',  label='Patch 3mm (mm)')
plt.plot(x_vals, coefs_2_ot[0] + coefs_2_ot[1] * x_vals, '--g', label='Patch 3mm (ot)')
plt.plot(x_vals, coefs_3_mm[0] + coefs_3_mm[1] * x_vals, '-r', label='WTI (mm)')
plt.plot(x_vals, coefs_3_ot[0] + coefs_3_ot[1] * x_vals, '--r', label='WTI (ot)')
plt.xlabel('-dCT (IL6)')
plt.ylabel('Septa thickness')
plt.legend()
plt.show()
