"""
    This is a wrapper for the environment and parallel session to output a gym environment
"""


from typing import List, Dict, Tuple
import numpy as np

from .environment import HanabiParallelEnvironment
from hanabi_multiagent_framework.utils import make_hanabi_env_config
# import hanabi_multiagent_framework as hmf
from .parallel_session import HanabiParallelSession


import dm_env
from dm_env import specs
import gym
from gym import spaces


_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class GymFromDMEnv(gym.Env):
    def __init__ (
            self,
            env: dm_env.Environment    
        ):
        
        self._env = env  # type: dm_env.Environment
        self._last_observation = None  # type: Optional[np.ndarray]
        self.game_over = False  # Needed for Dopamine agents.
        
    
    def step (
            self,
            action: int) -> _GymTimestep:
        pass
    
    def reset(self) -> np.ndarray:
        pass
    
    @property
    def action_space(self):
        pass
    
    
    @property 
    def observation_spcae(self):
        pass
    
    @property
    def reward_range(self):
        pass
    
    def __getattr__(self, attr):
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, attr)

def space2spec(
        space: gym.Space, 
        name: Optional[str] = None):
    """
        Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
        Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
        specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
        Dict spaces are recursively converted to tuples and dictionaries of specs.
        
        Args:
            space: The Gym space to convert.
            name: Optional name to apply to all return spec(s).
        
        Returns:
            A dm_env spec or nested structure of specs, corresponding to the input
            space.
    """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                minimum=space.low, maximum=space.high, name=name)

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                                maximum=1.0, name=name)

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                minimum=np.zeros(space.shape),
                                maximum=space.nvec, name=name)

    elif isinstance(space, spaces.Tuple):
        return tuple(space2spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {key: space2spec(value, name) for key, value in space.spaces.items()}

    else:
        raise ValueError('Unexpected gym space: {}'.format(space))


class DMEnvFromGym(dm_env.Environment):
    """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

    def __init__(
        self, 
        gym_env: gym.Env
        ):
        
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observations')
        self._action_spec = space2spec(self.gym_env.action_space, name='actions')
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation = self.gym_env.reset()
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        if self._reset_next_step:
            return self.reset()

        # Convert the gym step result to a dm_env TimeStep.
        observation, reward, done, info = self.gym_env.step(action)
        self._reset_next_step = done

        if done:
            is_truncated = info.get('TimeLimit.truncated', False)
            if is_truncated:
                return dm_env.truncation(reward, observation)
            else:
                return dm_env.termination(reward, observation)
        else:
            return dm_env.transition(reward, observation)

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
