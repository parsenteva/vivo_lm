#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:35:29 2023

@author: Polina Arsenteva
"""
import numpy as np
from scipy import optimize

def fit_lm_mm(x, y, pi_k = None, ddof=1):
    """
    Calculate the linear regression coefficients estimators with the
    moments method in the case when the predictor x and the predicted variable
    y cannot be observed simultaneously.
    
    """
    nb_groups = x.shape[1]
    if pi_k is None: # assuming weights are equally distributed
        pi_k = np.ones(nb_groups) / nb_groups
    mu_k_hat_x = np.nanmean(x, axis=0)
    mu_hat_x = np.average(mu_k_hat_x, weights=pi_k, axis=0)
    mu_k_hat_y = np.nanmean(y, axis=0)
    mu_hat_y = np.average(mu_k_hat_y, weights=pi_k, axis=0)
    if len(x.shape) > 2:
        pi_k_rep = np.repeat(pi_k, x.shape[-1]).reshape((nb_groups, x.shape[-1]))
    else:
        pi_k_rep = pi_k.copy()
    mu_diff = mu_k_hat_x - mu_hat_x
    q1 = (pi_k_rep * mu_diff * (mu_k_hat_y - mu_hat_y)).sum(axis=0)
    q2 = (pi_k_rep * mu_diff ** 2).sum(axis=0)
    beta_1_hat = q1 / q2
    beta_0_hat = mu_hat_y - beta_1_hat * mu_hat_x
    # Calculating the asymptotic variance of the slope estimator:
    sigma_sq_k_hat_x = np.nanvar(x, axis=0, ddof=ddof)
    sigma_sq_k_hat_y = np.nanvar(y, axis=0, ddof=ddof)
    sgm_beta_1_hat = (mu_diff ** 2 * (beta_1_hat  ** 2 * sigma_sq_k_hat_x
                                  + sigma_sq_k_hat_y) * (pi_k ** 2)).sum()
    sgm_beta_1_hat /= ((mu_diff ** 2 * pi_k).sum()) ** 2
    return beta_0_hat, beta_1_hat, sgm_beta_1_hat

def fit_lm_ot(x, y, pi_k = None, beta_1_init=5, ddof=1, beta_0_init=0, 
              sigma_sq_eps_init=0.1, disp=False, lower_bnd_sigma=0):
    """
    Calculate the linear regression coefficients estimators with the
    optimal transport method in the case when the predictor x and the predicted
    variable y cannot be observed simultaneously.
    
    """
    nb_groups = x.shape[1]
    if pi_k is None: # assuming weights are equally distributed
        pi_k = np.ones(nb_groups) / nb_groups
    sigma_sq_k_hat_x = np.nanvar(x, axis=0, ddof=ddof)
    sigma_sq_k_hat_y = np.nanvar(y, axis=0, ddof=ddof)
    mu_k_hat_x = np.nanmean(x, axis=0)
    mu_k_hat_y = np.nanmean(y, axis=0)
    term_1 = lambda a1, a0: (mu_k_hat_y - a0 - a1 * mu_k_hat_x)
    term_inside_sqrt = lambda a1, sigma_sq: (a1**2 * sigma_sq_k_hat_x + sigma_sq)
    term_sqrt = lambda a1, sigma_sq: np.sqrt(term_inside_sqrt(a1, sigma_sq))
    term_2 = lambda a1, sigma_sq: (a1 * sigma_sq_k_hat_x * (np.sqrt(sigma_sq_k_hat_y) 
                                                      - term_sqrt(a1, sigma_sq)) 
                                  / (term_sqrt(a1, sigma_sq)))
    f_der_a1 = lambda u: (pi_k * (mu_k_hat_x * term_1(u[0], u[1]) 
                                  + term_2(u[0], u[2]))).sum() * (-2)
    f_der_a0 = lambda u: (-2) * (pi_k * term_1(u[0], u[1])).sum()
    f_der_sigma = lambda u: (pi_k * (1 - np.sqrt(sigma_sq_k_hat_y)
                                     / term_sqrt(u[0], u[2]))).sum()
    f_obj = lambda u: (pi_k * (term_1(u[0], u[1])**2 
                               + (np.sqrt(sigma_sq_k_hat_y) 
                                  - term_sqrt(u[0], u[2]))**2)).sum()
    f_der = lambda u: (f_der_a1(u), f_der_a0(u), f_der_sigma(u))
    bnds = ((None, None), (None, None), (lower_bnd_sigma, None))
    (beta_1_hat, 
     beta_0_hat, 
     sigma_sq_hat_eps) = optimize.minimize(f_obj, jac=f_der, bounds=bnds,
                                           x0=(beta_1_init, beta_0_init, 
                                               sigma_sq_eps_init),
                                           options={'disp': disp}).x
    return beta_0_hat, beta_1_hat, sigma_sq_hat_eps
    

def calculate_bootstrap_ci(x, y, nb_boot=100, conf_lvl=0.95, size_boot_x=None, 
                           size_boot_y=None):
    """
    Calculate bootstrap confidence interval estimators for the
    linear regression coefficients with the moments and optimal transport 
    methods in the case when the predictor x and the predicted
    variable y cannot be observed simultaneously.
    
    """
    boot_betas = np.zeros((2, nb_boot))
    ci = np.zeros((2, 2))
    if size_boot_x is None:
        size_boot_x = x.shape[0]
    if size_boot_y is None:
        size_boot_y = y.shape[0]
    for i in range(nb_boot):
        x_toomanynans = True
        while x_toomanynans:
            boot_i_x_ind = np.random.randint(x.shape[0], size=(size_boot_x, x.shape[1]))
            boot_i_x = np.stack([x[boot_i_x_ind[:, j], j] for j in range(x.shape[1])]).T
            x_toomanynans = ((~np.isnan(boot_i_x)).sum(axis=0) < 3).any()
        y_toomanynans = True
        while y_toomanynans:
            boot_i_y_ind = np.random.randint(y.shape[0], size=(size_boot_y, y.shape[1]))
            boot_i_y = np.stack([y[boot_i_y_ind[:, j], j] for j in range(y.shape[1])]).T
            y_toomanynans = ((~np.isnan(boot_i_y)).sum(axis=0) < 3).any()
        (beta_0_init, 
         boot_betas[0, i], _) = fit_lm_mm(boot_i_x, boot_i_y)
        boot_betas[1, i] = fit_lm_ot(boot_i_x, boot_i_y,
                                     beta_1_init=np.random.uniform(-5, 5),
                                     beta_0_init=np.random.uniform(-5, 5))[1]
    ci_l_bound = (1 - conf_lvl) / 2
    ci_u_bound = conf_lvl + ci_l_bound 
    ci[0] = np.quantile(boot_betas[0], (ci_l_bound, ci_u_bound))
    ci[1] = np.quantile(boot_betas[1], (ci_l_bound, ci_u_bound))
    return boot_betas, ci
