import numpy as np
import tensorflow as tf
import gym
import random
import copy

import os
import sys

from seal.utils.epsilon_decay import linearly_decaying_epsilon
from seal.models.box2d_models import DiscreteBCQNetwork
from seal.replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from seal.agents.default_config import DEFAULT_CONFIG as config

class DiscreteBCQAgent():
    def __init__(self, name='LunarLander-v2',
                 num_actions=4,
                 network=DiscreteBCQNetwork,
                 config=config):
        self.name = name
        self.env = gym.make(name)
        self.env.spec.max_episode_steps = config['max_episode_steps']
        self.num_actions = num_actions
        self.network = network
        self.config = config
        
        self.threshold = config['bcq_threshold']
        self.gamma = config['gamma']
        # model & target model
        self.model = self.network(num_actions, config['hiddens'], config['activation'])
        self.target_model = self.network(num_actions, config['hiddens'], config['activation'])
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(config['lr'], 
                                                                     decay_steps=config['decay_steps'], 
                                                                     decay_rate=1)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        # loss
        self.loss = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
        # replay buffer
        if config['prioritized_replay']:
            self.replay_buffer = PrioritizedReplayBuffer(size=config['buffer_size'], 
                                         alpha=config['prioritized_replay_alpha'], 
                                         beta=config['prioritized_replay_beta'],
                                         online=config['online'],
                                         persistent_directory=config['persistent_directory'],
                                         episode_counts_to_save=config['episode_counts_to_save'], 
                                         sample_steps_to_refresh=config['sample_steps_to_refresh'])
        else:
            self.replay_buffer = ReplayBuffer(size=config['buffer_size'],
                                         online=config['online'], 
                                         persistent_directory=config['persistent_directory'],
                                         episode_counts_to_save=config['episode_counts_to_save'], 
                                         sample_steps_to_refresh=config['sample_steps_to_refresh'])
        # training_steps
        self.training_steps = 0
        # evaluation scores
        self.eval_episode_rewards = []
        self.eval_episode_steps = []

    def learn(self):
        self._learn_online() if self.config['online'] else self._learn_offline()
        
    def _learn_online(self):
        config = self.config
        
        state = self.env.reset()
        episode_id = 0
        for step_id in range(config['max_training_steps']):
            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            self.replay_buffer.add(state, action, reward, next_state, done, episode_id)
            
            if len(self.replay_buffer) > config['min_replay_history']:
                if self.training_steps % config['update_period'] == 0:
                    meanq = self._train()
                if self.training_steps % config['target_update_period'] == 0:
                    if len(self.model.weights) != len(self.target_model.weights): # training not started yet
                        self.target_model(state[None])
                    self._update_target_weights()
                    
            state = next_state
            
            if done:
                state = self.env.reset()
                episode_id += 1
                
            self.training_steps += 1
            
            if self.training_steps % config['training_steps_to_checkpoint'] == 0:
                path = config['checkpoint_path'] + 'discretebcq_{}.ckpt'.format(self.training_steps)
                self.save(path)
                print('saving model weights at {}'.format(path))
            
            if self.training_steps % config['training_steps_to_eval'] == 0:
                self._eval(5)
                # reset env
                # state = self.env.reset()
                episode_id += 1
                ### log progress
                mean_episode_reward = np.mean(self.eval_episode_rewards[-10:])
                mean_episode_step = np.mean(self.eval_episode_steps[-10:])
                max_episode_reward = np.max(self.eval_episode_rewards[-10:])
                max_episode_step  = np.max(self.eval_episode_steps[-10:])
                print("------------------------------------------------")
                print("episodes %d" % episode_id)
                print("timestep %d" % self.training_steps)
                print("exploration %f" % config['epsilon_fn'](self.training_steps, 
                                                              config['epsilon_start'],
                                                              config['epsilon_decay_period'],
                                                              config['epsilon_end'],
                                                              config['min_replay_history']))
                print("learning_rate %f" % self.optimizer.lr(self.training_steps))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("max reward (100 episodes) %f" % max_episode_reward)
                print("mean step (100 episodes) %f" % mean_episode_step)
                print("max step (100 episodes) %f" % max_episode_step)
                if len(self.replay_buffer) > config['min_replay_history']:
                    print("mean q values %f" % meanq)
                sys.stdout.flush()

                if mean_episode_reward > config['target_mean_episode_reward']:
                    break

        if len(self.replay_buffer._trajectory_storage) > 0:
            self.replay_buffer.save()
            
    def _learn_offline(self):
        config = self.config
        
        for step_id in range(config['max_training_steps']):
            if self.training_steps % config['update_period'] == 0:
                meanq = self._train()
            if self.training_steps % config['target_update_period'] == 0:
                self._update_target_weights()
                
            self.training_steps += 1
            
            if self.training_steps % config['training_steps_to_checkpoint'] == 0:
                if config['double'] == False:
                    path = config['checkpoint_path'] + 'offline_discretebcq_{}.ckpt'.format(self.training_steps)
                else:
                    path = config['checkpoint_path'] + 'offline_discretebcq_{}.ckpt'.format(self.training_steps)
                self.save(path)
                print('saving model weights at {}'.format(path))
                
            if self.training_steps % config['training_steps_to_eval'] == 0:
                self._eval(5)
                ### log progress
                mean_episode_reward = np.mean(self.eval_episode_rewards[-10:])
                mean_episode_step = np.mean(self.eval_episode_steps[-10:])
                max_episode_reward = np.max(self.eval_episode_rewards[-10:])
                max_episode_step = np.max(self.eval_episode_steps[-10:])
                print("------------------------------------------------")
                print("timestep %d" % self.training_steps)
                print("learning_rate %f" % self.optimizer.lr(self.training_steps))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("max reward (100 episodes) %f" % max_episode_reward)
                print("mean step (100 episodes) %f" % mean_episode_step)
                print("max step (100 episodes) %f" % max_episode_step)
                print("mean q values %f" % meanq)             
                sys.stdout.flush()
                
                if mean_episode_reward > config['target_mean_episode_reward']:
                    break
                    
    def _select_action(self, state):
        config = self.config
        
        if config['eval_mode']:
            epsilon = config['epsilon_eval']
        else:
            epsilon = config['epsilon_fn'](
                self.training_steps,
                config['epsilon_start'],
                config['epsilon_decay_period'],
                config['epsilon_end'],
                config['min_replay_history'])
            
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q, imt, i = tf.stop_gradient(self.model(state[None]))
#             imt = tf.exp(imt)
            imt = tf.cast(imt / tf.reduce_max(imt, axis=-1, keepdims=True) > self.threshold, dtype=tf.float32)
            # use large negative number to mask actions from argmax
        
        return int(tf.argmax(imt * q + (1. - imt) * (-1e8), axis=-1))
        
    def _update_target_weights(self):
        config = self.config
        
        weights = self.model.get_weights()
        tgt_weights = self.target_model.get_weights()
        for idx in range(len(weights)):
            tgt_weights[idx] = config['tau'] * tgt_weights[idx] + (1 - config['tau']) * weights[idx]
        self.target_model.set_weights(tgt_weights)

    def _eval(self, n_episodes=5):
#         config = self.config
        env = gym.make(self.name)
        self.config['eval_mode'] = True
        for i in range(n_episodes):
            rewards, steps = 0, 0
            state = env.reset()
            for t in range(config['max_episode_steps']):
                action = self._select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                rewards += reward
                steps += 1
                if done: break       
            self.eval_episode_rewards.append(rewards)
            self.eval_episode_steps.append(steps)
        env.close()
        self.config['eval_mode'] = False
        
    def save(self, path):
        self.model.save_weights(path)
        
    def load(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        
    def policy(self, states, actions):
        q_values = self.model(states).q_values
        q_actions = np.argmax(q_values, axis=1)
        return tf.cast(actions == q_actions, tf.float32)
    
    def greedy_actions(self, states):
        return np.argmax(self.model(states).q_values, axis=1)
    
    def _train(self):
        config = self.config
        
        transitions = self.replay_buffer.sample(config['batch_size'])
        
        states, actions, rewards = transitions[0], transitions[1], transitions[2]
        next_states, dones = transitions[3], transitions[4]
        is_non_terminal = 1. - tf.cast(dones, tf.float32)
        
        
        
        with tf.GradientTape() as tape:
            # compute the target Q value
            q, imt, i = self.model(next_states)
#             imt = tf.exp(imt)
            imt = tf.cast(imt / tf.reduce_max(imt, axis=-1, keepdims=True) > self.threshold, dtype=tf.float32)
            
            # use large negative number to mask actions from argmax
            next_actions = tf.argmax(imt * q + (1. - imt) * (-1e8), axis=-1)

            indices = tf.stack([tf.range(next_actions.shape[0], dtype=tf.int64), 
                                        next_actions], axis=-1)
            
            q, imt, i = tf.stop_gradient(self.target_model(next_states))
            targets = rewards + self.gamma * tf.gather_nd(q, indices=indices) * is_non_terminal
        
            
            # Get current Q estimate
            current_q, imt, i = self.model(states)
            indices = tf.stack([tf.range(actions.shape[0], dtype=tf.int64), actions], axis=-1)
            chosen_q_vals = tf.gather_nd(current_q, indices=indices)
            
            
            # compute Q loss
            q_loss = tf.reduce_mean(self.loss(chosen_q_vals, targets))
            i_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(actions, depth=self.num_actions), imt))
            
            final_loss = q_loss + i_loss + 1e-2 * tf.reduce_mean(tf.pow(i, 2))
        
        # minimize loss
        grads = tape.gradient(final_loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -config['grad_clip'], config['grad_clip']) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return tf.math.reduce_mean(chosen_q_vals)