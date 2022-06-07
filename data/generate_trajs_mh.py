# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:12:09 2021

@author: leyuan
"""

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('Data_Ohio.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)
cols = data.columns

def get_reward(g): # cost
    if g < 80:
        return -1./30*(80-g)**2
    elif g >= 140:
        return -1./30*(g-140)**1.35
    else:
        return 0
    
data['reward'] = data['glucose'].apply(get_reward).shift(-1)

data[['glucose','meal']] = (data[['glucose','meal']] - 
                            data[['glucose','meal']].mean()) / data[['glucose','meal']].std()

datas = pd.concat([data[cols].shift(i).rename({col: col+'_{}'.format(i) for col in cols}, axis=1) 
                   for i in range(4)], axis=1)

datas['reward'] = data['reward']

states_col = [col+'_{}'.format(i) for i in range(4) for col in cols]
states_col.remove('A_0')
action_col = 'A_0'
reward_col = 'reward'

    
## trajs per day per person ###    
removes = set([0, 1, 2]) # drop first three transitions every day
persons = []
i, trajs, traj = 1, [], []
for index, row in datas.iterrows():
    if index < 1100*i-1:
        if index % 24 in removes: # drop first three transitions
            continue
        s = row[states_col].values
        a = int(row[action_col])
        r = row[reward_col]
        s_ = datas.loc[index+1][states_col].values
        traj.append([s, a, r, s_, False])
        if index % 24 == 23:
            traj[-1][-1] = True
            trajs.append(traj)
            traj = []
            
    if index == 1100*i-1:
        if len(traj) > 0:
            trajs.append(traj)
        persons.append(trajs)
        trajs, traj = [], []
        i += 1
        
    
trajs = [traj for person in persons for traj in person]


with open('./trajs_mh_sim.pkl', 'wb') as f:
    pickle.dump(trajs, f)


##=========================== add noise ======================================
# path = './trajs_mh.pkl'
# with open(path, 'rb') as f:
#     trajs = pickle.load(f)
    
rm = []
for i, traj in enumerate(trajs):
    if len(traj) != 21:
        rm.append(i)

keep = set(range(len(trajs))) - set(rm)

trajs = np.array(trajs)

trajs = trajs[list(keep)]


states = np.array([item[0] for traj in trajs for item in traj])
actions = np.array([item[1] for traj in trajs for item in traj])
rewards = np.array([item[2] for traj in trajs for item in traj])
next_states = np.array([item[3] for traj in trajs for item in traj])
dones = np.array([item[4] for traj in trajs for item in traj])


# add noises to rewards,states, next_states
rewards = rewards + np.random.randn(5649) * np.sqrt(2)
states = states + np.random.randn(5649, 15)* np.sqrt(2)
next_states = next_states + np.random.randn(5649, 15)* np.sqrt(2)


# regenerate the trajs 
new_trajs = []
traj = []

for i in range(5649):
    state = states[i]
    action = actions[i]
    reward = rewards[i]
    next_state = next_states[i]
    done = dones[i]
    traj.append([state, action, reward, next_state, done])
    if (i+1) % 21 == 0:
        new_trajs.append(traj)
        traj = []
    
        
with open('./trajs_mh_sim.pkl', 'wb') as f:
    pickle.dump(new_trajs, f)














