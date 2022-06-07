import numpy as np
import tensorflow as tf
import gym
import random

import os
import sys

from peal.utils.epsilon_decay import linearly_decaying_epsilon
from peal.models.box2d_models import MultiHeadQNetwork
from peal.replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from peal.agents.default_config import DEFAULT_CONFIG as config

class MultiHeadDQNAgent:
    def __init__(self, name='LunarLander-v2',
                 num_actions=4,
                 network=MultiHeadQNetwork,
                 config=config):
        self.name = name
        self.env = gym.make(name)
        self.env.spec.max_episode_steps = config['max_episode_steps']
        self.num_actions = num_actions
        self.network = network
        self.config = config
        # model & target model
        self.model = self.network(num_actions, config['hiddens'], config['activation'],
                                  config['num_heads'], config['num_convex_combinations'])
        self.target_model = self.network(num_actions, config['hiddens'], config['activation'],
                                  config['num_heads'], config['num_convex_combinations'])
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
                    self._train()
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
                path = config['checkpoint_path'] + 'dqn_{}.ckpt'.format(self.training_steps)
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
                sys.stdout.flush()

                if mean_episode_reward > config['target_mean_episode_reward']:
                    break

        if len(self.replay_buffer._trajectory_storage) > 0:
            self.replay_buffer.save()
            
    def _learn_offline(self):
        config = self.config
        
        for step_id in range(config['max_training_steps']):
            if self.training_steps % config['update_period'] == 0:
                self._train()
            if self.training_steps % config['target_update_period'] == 0:
                self._update_target_weights()
                
            self.training_steps += 1

            if self.training_steps % config['training_steps_to_checkpoint'] == 0:
                path = config['checkpoint_path'] + 'offline_rem_{}.ckpt'.format(self.training_steps)
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
                sys.stdout.flush()
                
                if mean_episode_reward > config['target_mean_episode_reward']:
                    break
                    
    def _select_action(self, state):
        config = self.config
        
        if config['eval_mode']:
            epsilon = config['epsilon_eval']
        else:
            epsilon = config['epsilon_fn'](self.training_steps,
                config['epsilon_start'],
                config['epsilon_decay_period'],
                config['epsilon_end'],
                config['min_replay_history'])
            
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.model(state[None]).q_values[0])
        
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

        with tf.GradientTape() as tape:
            # current q heads, shape: batch_size x num_actions x num_heads
            q_heads = self.model(states).q_heads 
            indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
            chosen_q_heads = tf.gather_nd(q_heads, indices=indices)
            
            if config['double']:
                next_q_values = self.model(next_states).q_values
                next_q_values_argmax = np.argmax(next_q_values, axis=1)
                indices = tf.stack([tf.range(next_q_values_argmax.shape[0]),
                                    next_q_values_argmax], axis=-1)
                next_qt_heads = self.target_model(next_states).q_heads
                next_vals = tf.gather_nd(next_qt_heads, indices=indices)
            else:
                next_vals = tf.math.reduce_max(self.target_model(next_states).q_heads, axis=1)
                
            is_non_terminal = 1. - tf.cast(dones, tf.float32)
            rewards = tf.clip_by_value(tf.cast(rewards, tf.float32), 
                                       -config['reward_clip'], config['reward_clip'])
            targets = tf.stop_gradient(rewards[:,None] + config['gamma'] * next_vals * is_non_terminal[:,None])
            # huber loss
            loss = self.loss(chosen_q_heads, targets)
            loss = tf.math.reduce_mean(loss, axis=-1)
            
            if config['prioritized_replay']: # update priorities
                weights, idxes, update = transitions[5], transitions[6], transitions[7]
                if update:
                    priorities = tf.clip_by_value(loss, 1e-3, 10)
                    self.replay_buffer.update_priorities(idxes, priorities)
                    
                loss = weights * loss
            
            final_loss = tf.math.reduce_mean(loss)
        
        grads = tape.gradient(final_loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -config['grad_clip'], config['grad_clip']) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))