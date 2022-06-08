# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:50:47 2021

@author: leyuan
"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
sns.set(style="darkgrid")

import os
import copy
import random
import pickle


import numpy as np
import pandas as pd


# =============================================================================
dir = './synthetic_results/plots_lunar'

paths = [ 'dqn.csv', 'ddqn.csv', 'qrdqn.csv', '4_methods.csv']
paths = [os.path.join(dir, path) for path in paths]
ckpts = list(range(20000, 60000, 10000))
labels0 = ['DQN', 'DDQN', 'QR-DQN', 'REM', 'BCQ', 'BEAR']
labels1 = ['SEAL-DQN', 'SEAL-DDQN', 'SEAL-QR-DQN']

fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True, sharex=True)
fig.suptitle('LunarLander-v2')
for i in range(3):       
    dat = pd.read_csv(paths[i])

    dqn_r = dat.iloc[:4, :5]
    peal_r = dat.iloc[4:, :5]

    
    dqn_r_mean = dqn_r.mean(axis=1)
    dqn_r_sd = dqn_r.std(axis=1)/np.sqrt(5)
    

    
    peal_r_mean = peal_r.mean(axis=1)
    peal_r_sd = peal_r.std(axis=1)/np.sqrt(5)
    
    
    axes[i].errorbar(ckpts, dqn_r_mean, dqn_r_sd, color='b', label=labels0[i], marker='o')
    axes[i].errorbar(ckpts, peal_r_mean, peal_r_sd, color='r', label=labels1[i], marker='^')
    axes[i].set_xticks(ckpts)
    # axes[i].set_ylim(50, 600)
    axes[i].legend(loc='upper left')

dat = pd.read_csv(paths[-1])
rem_r = dat.iloc[:4, :5]
discrete_bcq_r = dat.iloc[4:8, :5]
kl_r = dat.iloc[8:12, :5]
pear_r = dat.iloc[12:, :5]

rem_r_mean = rem_r.mean(axis=1)
rem_r_sd = rem_r.std(axis=1)/np.sqrt(5)


discrete_bcq_r_mean = discrete_bcq_r.mean(axis=1)
discrete_bcq_r_sd = discrete_bcq_r.std(axis=1)/np.sqrt(5)


kl_r_mean = kl_r.mean(axis=1)
kl_r_sd = kl_r.std(axis=1)/np.sqrt(5)

    
axes[3].errorbar(ckpts, rem_r_mean, rem_r_sd, color='b', label=labels0[3], marker='o')
axes[3].errorbar(ckpts, discrete_bcq_r_mean, discrete_bcq_r_sd, color='y', label=labels0[4], marker='o')
axes[3].errorbar(ckpts, kl_r_mean,kl_r_sd, color='g', label=labels0[5], marker='o')
axes[3].errorbar(ckpts, peal_r_mean, peal_r_sd, color='r', label=labels1[2], marker='^')
axes[3].set_xticks(ckpts)
# axes[i][3].set_ylim(50, 850)
axes[3].legend(loc='upper left', fontsize=10)
# axes[3].legend(loc='lower right')

axes[0].set_ylabel('value')
# plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig('lunar_lander.png', dpi=400)
# ========================================================================================



# =============================================================================
dir = './synthetic_results/plots_qbert'

paths = [ 'dqn.csv', 'ddqn.csv', 'qrdqn.csv', '4_methods.csv']
paths = [os.path.join(dir, path) for path in paths]
ckpts = list(range(20000, 60000, 10000))
labels0 = ['DQN', 'DDQN', 'QR-DQN', 'REM', 'BCQ', 'BEAR']
labels1 = ['SEAL-DQN', 'SEAL-DDQN', 'SEAL-QR-DQN']

fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True, sharex=True)
fig.suptitle('Qbert-ram-v0')
for i in range(3):       
    dat = pd.read_csv(paths[i])

    dqn_r = dat.iloc[:4, :5]
    peal_r = dat.iloc[4:, :5]

    
    dqn_r_mean = dqn_r.mean(axis=1)
    dqn_r_sd = dqn_r.std(axis=1)/np.sqrt(5)
    

    
    peal_r_mean = peal_r.mean(axis=1)
    peal_r_sd = peal_r.std(axis=1)/np.sqrt(5)
    
    
    axes[i].errorbar(ckpts, dqn_r_mean, dqn_r_sd, color='b', label=labels0[i], marker='o')
    axes[i].errorbar(ckpts, peal_r_mean, peal_r_sd, color='r', label=labels1[i], marker='^')
    axes[i].set_xticks(ckpts)
    # axes[i].set_ylim(50, 600)
    axes[i].legend(loc='upper left')

dat = pd.read_csv(paths[-1])
rem_r = dat.iloc[:4, :5]
discrete_bcq_r = dat.iloc[4:8, :5]
kl_r = dat.iloc[8:12, :5]
pear_r = dat.iloc[12:, :5]

rem_r_mean = rem_r.mean(axis=1)
rem_r_sd = rem_r.std(axis=1)/np.sqrt(5)


discrete_bcq_r_mean = discrete_bcq_r.mean(axis=1)
discrete_bcq_r_sd = discrete_bcq_r.std(axis=1)/np.sqrt(5)


kl_r_mean = kl_r.mean(axis=1)
kl_r_sd = kl_r.std(axis=1)/np.sqrt(5)

    
axes[3].errorbar(ckpts, rem_r_mean, rem_r_sd, color='b', label=labels0[3], marker='o')
axes[3].errorbar(ckpts, discrete_bcq_r_mean, discrete_bcq_r_sd, color='y', label=labels0[4], marker='o')
axes[3].errorbar(ckpts, kl_r_mean,kl_r_sd, color='g', label=labels0[5], marker='o')
axes[3].errorbar(ckpts, peal_r_mean, peal_r_sd, color='r', label=labels1[2], marker='^')
axes[3].set_xticks(ckpts)
# axes[i][3].set_ylim(50, 850)
axes[3].legend(loc='upper left', fontsize=10)
# axes[3].legend(loc='lower right')

axes[0].set_ylabel('value')
# plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig('qbert.png', dpi=400)
# ========================================================================================




# ========================================================================================
dir = './real_data_results/'
paths = ['dqn.csv', 'ddqn.csv', 'qrdqn.csv', '4_methods.csv']
paths = [os.path.join(dir, path) for path in paths]
ckpts = list(range(1000, 11000, 1000))
labels0 = ['DQN', 'DDQN', 'QR-DQN', 'REM', 'BCQ', 'BEAR']
labels1 = ['SEAL-DQN', 'SEAL-DDQN', 'SEAL-QR-DQN']

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)

for i, path in enumerate(paths[:-1]):
    dat = pd.read_csv(path)

    dqn = dat.iloc[:10, ]
    peal = dat.iloc[10:, ]
    
    mean1 = dqn.mean(axis=1)
    sd1 = dqn.std(axis=1)/np.sqrt(dat.shape[1])
    
    mean2 = peal.mean(axis=1)
    sd2 = peal.std(axis=1)/np.sqrt(dat.shape[1])

    axes[i].errorbar(ckpts[1::2], mean1[1::2], sd1[1::2], color='b', label=labels0[i], marker='o')
    axes[i].errorbar(ckpts[1::2], mean2[1::2], sd2[1::2], color='r', label=labels1[i], marker='^')
    axes[i].set_ylim(-57.5, -53)
    axes[i].legend(loc='upper left')

dat = pd.read_csv(paths[-1])
rem = dat.iloc[:10, ]
discrete_bcq = dat.iloc[10:20, ]
kl = dat.iloc[20:30, ]
peal = dat.iloc[30:, ]

mean1 = rem.mean(axis=1)
sd1 = rem.std(axis=1)/np.sqrt(dat.shape[1])

mean2 = discrete_bcq.mean(axis=1)
sd2 = discrete_bcq.std(axis=1)/np.sqrt(dat.shape[1])

mean3 = kl.mean(axis=1)
sd3 = kl.std(axis=1)/np.sqrt(dat.shape[1])

mean4 = peal.mean(axis=1)
sd4 = peal.std(axis=1)/np.sqrt(dat.shape[1])

axes[3].errorbar(ckpts[1::2], mean1[1::2], sd1[1::2], color='b', label=labels0[3], marker='o')
axes[3].errorbar(ckpts[1::2], mean2[[16, 15, 14, 13, 17]], sd2[[16, 15, 14, 13, 17]], color='y', label=labels0[4], marker='o')
axes[3].errorbar(ckpts[1::2], mean3[2:7], sd3[2:7], color='g', label=labels0[5], marker='o')
axes[3].errorbar(ckpts[1::2], mean4[1::2], sd4[1::2], color='r', label=labels1[2], marker='^')
axes[i].set_ylim(-57.5, -53)
axes[3].legend(loc='upper left')

axes[0].set_ylabel('value')

plt.savefig('seal_mh.png', dpi=400, bbox_inches='tight')
# ========================================================================================