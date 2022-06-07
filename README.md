# SEAL

# Author Contributions Checklist Form

# Part 1: Data

## Abstract

The case study considered in this paper is implemented using the OhioT1DM dataset for blood glucose level prediction. The data contains continuous measurements for six patients with type 1 diabetes over eight weeks. The objective is to learn an optimal policy that maps patients’ time-varying covariates into the amount of insulin injected at each time to maximize patients’ health status.

## Availability

The OhioT1DM dataset is available from [http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).   However, a Data Use Agreement (DUA) is required to protect the data and ensure that it is used only for research purposes. 

## Description

In our experiment, we divide each day of follow-up into one hour intervals and a treatment decision is made every
hour. We consider three important time-varying state variables, including the average blood glucose levels $G_{t}$ during the one hour interval $(t-1, t]$, the carbohydrate estimate for the meal $C_{t}$ during $(t-1, t]$ and $\text{Ex}_{t}$ which measures exercise intensity during $(t-1, t]$. At time $t$, we define the action $A_t$ by discretizing the amount of insulin $\text{In}\_t$ injected. The reward $R\_t$ is chosen according to the Index of Glycemic Control that is a deterministic function $G\_{t+1}$. Detailed definitions of $A\_t$ and $R\_t$ are given as follows,

$$A_t  = \begin{cases}
0, & \text{In}_t = 0; \\
m, & 4m-4 < \text{In}_t \le 4m \,~~(m=1, 2, 3); \\
4, & \text{In}_t > 12.
\end{cases} ~~~~~~~~~~~~R_t = \begin{cases}
-\frac{1}{30}(80-G_{t+1})^2, & G_{t+1} < 80; \\
0, & 80 \le G_{t+1} \le 140; \\
-\frac{1}{30}(G_{t+1} - 140)^{1.35}, & 140 \le G_{t+1}.
\end{cases}$$

Let $X\_t=(G\_t,C\_t,\hbox{Ex}\_t)$. We define the state $S\_t$ by concatenating measurements over the last four decision points, i.e., $S\_t = (X\_{t-3}, A\_{t-3}, \cdots, X\_t)^\top$. This ensures the Markov assumption is satisfied. The number of decision points for each patient in the OhioT1DM dataset ranges from 1119 to 1288. Transitions across different days are treated as independent trajectories. 

The data used in the paper is called the `trajs_mh.pkl`.  However, for confidentiality considerations, we add some noises to the data and provide the `trajs_mh_sim.pkl` in the `data` folder.  The code to generate these two data is placed in the `generate_trajs_mh.py`  in the `data` folder. Once you have downloaded the raw data, put it in the same folder as the code, and then run the code to get the data used in this paper.

## Synthetic data

Two OpenAI Gym environments,` LunarLander-v2` and `Qbert-ram-v0` are used to generate the synthetic data. You can run  the `qr_dqn_online.ipynb` in the `data` folder and change the `env_name` and `num_actions` to get the `trajs_qr_dqn_lunar.pkl` and the `trajs_qr_dqn_qbert.pkl`  respectively. These two files are zipped in the `data/simulation_data.rar`. 

# Part 2: Code

## Abstract

The `seal` folder contains the core code to implement the proposed method and various utility functions.

## Description

### seal package Overview

- `models`: network structures.
- `agents`: DQN, QR-DQN, MultiHeadDQN (REM),  DiscreteBCQ, BEAR(MMD replaced by KL_control) agents.
- `replay_buffers`: basic and prioritized replay buffers.
- `algos`: behavior cloning, density estimator, advantage learner, fitted Q evaluation, etc.
- `utils`: utility functions.

### Computational Complexity

we assume that the forward and backward of network complexity is `S`.

- step 2: training `L` DQN agents, batch size `B_1`, training steps `I_1`, total `O(L * I_1 * B_1 * S)`
- step 3: training `L` density estimators, batch size `B_2`, training steps `I_2`, total `O(L * I_2 * B_2^4 * S)`
- step 4: pseudo Q computations, batch size `B_3`, total `O(B_3 * N * T * A * S)`, where `N` number of trajs, `T` average length of trajs, `A` number of actions.
- step 5: training tau, batch size `B_4`, training steps `I_4`, total `O(I_4 * B_4 * S)`

## Optional Information

All experiments run on a single computer instance with 40 Intel(R) Xeon(R) 2.20GHz CPUs.

- Python version: Python 3.6.8 :: Anaconda custom (64-bit)

### Main packages for the proposed estimator

- numpy == 1.18.1
- pandas == 1.0.3
- sklearn == 0.22.1
- tensorflow == 2.1.0
- tendorflow-probability == 0.9

# Part 3: Reproducibility workflow

## Synthetic data results

- copy the `data/trajs_qr_dqn.pkl` under the `data/online` folder  to the `lunarlander-v2/dqn_2_200/random/` folder, and switch to the `lunarlander-v2` and run `python batch_peal_vs_dqn.py &` (around 20 hours without GPU support). This will generate DQN V.S. PEAL offline training results. Similarly, we can obtain DDQN, QR-DQN, REM, Discrete-BCQ and Discrete-BEAR results. Same procedures to take with `Qbert-ram-v0`.
- 

## Real data results

- run the `DQN_mh.ipynb` under the `realdata` after putting the `trajs_mhm.pkl` which is in the `data` folder into the `realdata/data/mh/dqn` folder. This will generate DQN v.s. PEAL training results. Similarly, we can obtain DDQN, QR-DQN, REM, Discrete-BCQ and Discrete-BEAR results. 
- 

## Figures

