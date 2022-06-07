from seal.agents.dqn import DQNAgent
from seal.agents.qr_dqn import QuantileAgent
from seal.agents.multi_head_dqn import MultiHeadDQNAgent
from seal.agents.default_config import DEFAULT_CONFIG

__all__ = ['DQNAgent',
           'QuantileAgent', 
           'MultiHeadDQNAgent', 
           'DEFAULT_CONFIG']