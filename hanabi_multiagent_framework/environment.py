"""
This file implements a wrapper for hanabi_learning_environment.HanabiParallelEnv.
"""

from typing import Tuple, Dict, List
import numpy as np
from dm_env import specs as dm_specs
from dm_env import TimeStep, StepType
from hanabi_learning_environment import pyhanabi

class HanabiParallelEnvironment:
    """Hanabi parallel environment wrapper for use with HanabiParallelSession.
    """
    def __init__(self, env_config: Dict[str, str], n_parallel: int):
        self._parallel_env = pyhanabi.HanabiParallelEnv(env_config, n_parallel)
        self.n_players = self._parallel_env.parent_game.num_players()
        self.step_types = np.full((n_parallel,), StepType.FIRST)

    @property
    def last_observation(self):
        """Last observation"""
        return self._parallel_env.last_observation

    def step(self, action_ids: List[int], agent_id: int) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                                  np.ndarray,
                                                                  np.ndarray,
                                                                  None]:
        """Take one step in all game states.
        Args:
            action_ids -- list with ids of the actions for each state.
            agent_id -- id of the agent taking the actions.
        Return:
            a tuple consisting of:
              - observation tuple (vectorized observation, legal moves)
              - reward array
              - done array
              - info (always None)
        """

        last_score = self.last_observation.scores.copy()

        # Detect any illegal moves
        illegal_move_ids = \
                self.last_observation.legal_moves[range(self.num_states), action_ids] == 0

        # Replace illegal moves with legal ones. This is done to avoid exceptions in the underlying
        # hanabi learning environment which assumes that the client handles the exception and always
        # supplies the legal ones.
        # Illegal moves are considered as loosing the game and are punished as such.
        # The corresponding states are marked as terminal and should be restarted.
        action_ids[illegal_move_ids] = \
                np.argmax(self.last_observation.legal_moves[illegal_move_ids], axis=1)

        self._parallel_env.apply_batch_move(action_ids, agent_id)

        # Observe next agent
        self._parallel_env.observe_agent((agent_id + 1) % self.n_players)

        # Reward is the score differential. May be large and negative at game end.
        reward = self.last_observation.scores - last_score
        # illegal moves are punished as loosing the game
        reward[illegal_move_ids] = -last_score[illegal_move_ids]

        terminal = np.logical_or(illegal_move_ids,
                                 self.last_observation.done == 1)

        self.step_types = np.full((self.num_states,), StepType.MID)
        self.step_types[terminal] = StepType.LAST

        return ((self.last_observation.batch_observation,
                 self.last_observation.legal_moves),
                reward,
                self.step_types)

    @property
    def game_config(self):
        """Gather some game configuration details.
        It includes
          -- hand size,
          -- number of cards,
          -- number of colors,
          -- number of ranks,
          -- number of like tokens,
          -- number of information token.
        """
        hand_size = self._parallel_env.parent_game.hand_size()
        n_cards = self._parallel_env.parent_game.num_colors() \
                * self._parallel_env.parent_game.cards_per_color()
        n_colors = self._parallel_env.parent_game.num_colors()
        n_ranks = self._parallel_env.parent_game.num_ranks()
        n_life_tokens = self._parallel_env.parent_game.max_life_tokens()
        n_info_tokens = self._parallel_env.parent_game.max_information_tokens()
        return {"hand_size" : hand_size,
                "n_cards" : n_cards,
                "n_colors" : n_colors,
                "n_ranks" : n_ranks,
                "n_life" : n_life_tokens,
                "n_info" : n_info_tokens,
                }

    def reset_states(self,
                     states: np.ndarray,
                     current_agent_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resets the terminal states and returns upated observations.

        Args:
            states           -- states which should be reset
            current_agent_id -- id of an agent to perform next move.

        Returns:
            observation: (vectorized observation, legal moves)
        """
        self._parallel_env.reset_states(states, current_agent_id)
        self._parallel_env.observe_agent(current_agent_id)
        self.step_types[states] = StepType.FIRST
        return ((self.last_observation.batch_observation.copy(),
                 self.last_observation.legal_moves.copy()),
                self.step_types)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resets the environment for a new game. Should be called once after
        the instatiation of this env to retrieve initial observations.

        Returns:
            observation: (vectorized observation, legal moves)
        """
        self._parallel_env.reset()
        self.step_types = np.full((self.num_states,), StepType.FIRST)
        return (self.last_observation.batch_observation.copy(),
                self.last_observation.legal_moves.copy())

    @property
    def max_moves(self):
        """Total number of possible moves"""
        return self._parallel_env.parent_game.max_moves()

    @property
    def observation_len(self):
        """length of the vectorized observation of a single state"""
        return self._parallel_env.observation_len()

    @property
    def num_states(self):
        """Number of parallel states"""
        return self._parallel_env.num_states()

    @property
    def num_players(self):
        """Number of parallel states"""
        return self._parallel_env.parent_game.num_players()

    def observation_spec_vec(self) -> Tuple[dm_specs.BoundedArray, dm_specs.BoundedArray]:
        """Returns the vectorized observation spec.
        Observation is a tuple containing observation and legal moves.
        """
        return (dm_specs.BoundedArray(shape=(self.num_states, self.observation_len),
                                      dtype=np.int8,
                                      name="agent_observations",
                                      minimum=0, maximum=1),
                dm_specs.BoundedArray(shape=(self.num_states, self.max_moves),
                                      dtype=np.int8,
                                      name="legal_moves",
                                      minimum=0, maximum=1))

    def observation_spec(self) -> Tuple[dm_specs.BoundedArray, dm_specs.BoundedArray]:
        """Returns the observation spec.
        Observation is a tuple containing observation and legal moves.
        """
        return (dm_specs.BoundedArray(shape=(self.observation_len,),
                                      dtype=np.float16,
                                      name="agent_observation",
                                      minimum=0, maximum=1),
                dm_specs.BoundedArray(shape=(self.max_moves,),
                                      dtype=np.float16,
                                      name="legal_moves",
                                      minimum=0, maximum=1))

    def action_spec_vec(self) -> dm_specs.BoundedArray:
        """Returns the vectorized action spec."""
        return dm_specs.BoundedArray(shape=(self.num_states,),
                                     dtype=np.int,
                                     name="actions",
                                     minimum=0, maximum=self.max_moves)

    def action_spec(self) -> dm_specs.DiscreteArray:
        """Returns the action spec."""
        return dm_specs.DiscreteArray(self.max_moves,
                                      dtype=np.int,
                                      name="action")

    def reward_spec_vec(self) -> dm_specs.Array:
        """Returns the vectorized reward spec."""
        return dm_specs.Array(shape=(self.num_states,), dtype=float, name="reward")

    def reward_spec(self) -> dm_specs.Array:
        """Returns the reward spec."""
        return dm_specs.Array(shape=(), dtype=float, name="reward")
