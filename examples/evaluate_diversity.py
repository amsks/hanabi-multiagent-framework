import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, AgentType
from hanabi_learning_environment.pyhanabi_pybind import RewardShapingParams as ShapingParams
from hanabi_learning_environment.pyhanabi_pybind import RewardShaper
import logging
import time

from matplotlib import pyplot as plt

import dill as pickle
import sklearn.metrics as metrics


@gin.configurable
def RewardShapingParams(
    shaper: bool = True,
    min_play_probability: float = 0.8,
    w_play_penalty: float = 0,
    m_play_penalty: float = 0,
    w_play_reward: float = 0,
    m_play_reward: float = 0,
    penalty_last_of_kind: float = 0
):

    if shaper:

        return ShapingParams(
            shaper,
            min_play_probability,
            w_play_penalty,
            m_play_penalty,
            w_play_reward,
            m_play_reward,
            penalty_last_of_kind
        )

    else:
        return None


def load_agent(env):
    """[summary]

    Args:
        env ([]): [description]

    Returns:
        [type]: [description]
    """

    # load reward shaping infos
    params = RewardShapingParams()
    reward_shaper = RewardShaper(params=params) if params is not None else None
    # load agent based on type
    agent_type = AgentType()

    if agent_type.type == 'rainbow':
        agent_params = RlaxRainbowParams()
        return DQNAgent(env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        agent_params,
                        reward_shaper)

    # elif agent_type.type == 'rulebased':
    #     agent_params = RulebasedParams()
    #     return RulebasedAgent(agent_params.ruleset)

def load_obs(path):

    if not os.path.isdir(path):
        raise Exception('{} is no valid path!'.format(path))

    obs_db = []

    for file in os.listdir(path):
        loaded = []
        with open(str(path) + "/" + str(file), "rb") as f:
            loaded = pickle.load(f)

        obs_db.append(loaded)

    return np.array(obs_db)

def no_agent_in_path(path):
    '''
        Gathers information on the number of agents in the specified path and also separates between target 
        and online parameters, as well as non-weights-files
    '''
    if not os.path.isdir(path):
        raise Exception('{} is no valid path!'.format(path))

    paths_agents = []

    for file in os.listdir(path):
        '''Supposes that agent weights have been stored with "target" as some part of their name'''
        if 'target' in str(file) or 'rainbow' in str(file):
            paths_agents.append(file)

    return len(paths_agents), paths_agents

def get_name(path):
    '''
        Gathers information on the number of agents in the specified path and also separates between target 
        and online parameters, as well as non-weights-files
    '''
    if not os.path.isdir(path):
        raise Exception('{} is no valid path!'.format(path))

    paths_agents = []

    for file in os.listdir(path):
        '''Supposes that agent weights have been stored with "target" as some part of their name'''
        if 'target' in str(file):
            paths_agents.append(file)

    return paths_agents

def simple_match(actions_agent_a, actions_agent_b, no_obs):
    '''Compares both action vectors and calculates the number of matching actions'''
    return (1 - np.sum(actions_agent_a == actions_agent_b)/no_obs)

def preprocess_obs_for_agent(obs, agent, stack, env):

    if agent.requires_vectorized_observation():
        vobs = np.array(env._parallel_env.encoded_observations)
        if stack is not None:
            stack.add_observation(vobs)
            vobs = stack.get_current_obs()
        vlms = np.array(env._parallel_env.encoded_legal_moves)
        return (obs, (vobs, vlms))

    return obs


def calculate_diversity(
        self_play_agent,
        path_to_agents,
        observations,
        div_parallel_eval):  # sourcery skip: for-index-replacement

    # Get the names and numbers of partners from the pool
    n_agents, names = no_agent_in_path(path_to_agents)
    population = [[self_play_agent for _ in range(2)] for _ in range(n_agents)]

    print('Evaluating {} agents from the following given files {}'.format(n_agents, names))

    # Restore the weight for each partner in the pool
    for i in range(n_agents):
        for agent in population[i]:
            location = os.path.join(path_to_agents, names[i])
            agent.restore_weights(location, location)

    
    action_matrix = []
    for agent in population:
        actions = [agent[0].exploit(o) for o in observations]
        action_matrix.append(actions)

    action_matrix = np.array(action_matrix)

    # Flatten the actions to get a single list for each index in the
    # matrix
    action_mat = [action.flatten() for action in action_matrix]

    # Calculate the diveristy between each pair of agents 
    diversity = []
    
    for i in range(len(action_mat)):   
        div = []    
        for j in range(len(action_mat)):
            no_obs_test = len(observations) * 2 * div_parallel_eval
            div.append(
                simple_match(
                    action_mat[i],
                    action_mat[j],
                    no_obs_test
                ))
        diversity.append(np.array(div))

    # Return the mean diversity of the agent with all the partners in the pool
    return np.array(diversity)


@gin.configurable(denylist=['output_dir'])
def session(
        hanabi_game_type="Hanabi-Small",
        n_players: int = 2,
        max_life_tokens: int = None,
        n_parallel: int = 32,
        n_parallel_eval: int = 1_000,
        n_train_steps: int = 4,
        n_sim_steps: int = 2,
        epochs: int = 1_000_000,
        epoch_offset=0,
        eval_freq: int = 500,
        self_play: bool = True,
        output_dir: str = "./output",
        start_with_weights: bool = None,
        n_backup: int = 500,
        restore_weights: bool = None,
        path_to_agents: str = "./Agents/Database",
        path_to_obs: str = "./Agents/Observations",
        div_parallel_eval: int = 100,
        div_factor: float = 1.0):  # sourcery no-metrics


    # Define environment configuration based on game parameters
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)

    # If the maximum tokens are not none, load them up
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)


    # create training and evaluation parallel environment
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    
    with gin.config_scope('agent_0'):
        self_play_agent = load_agent(eval_env)
    
    agents = [self_play_agent for _ in range(n_players)]
    
    # start parallel session for training and evaluation
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()

    print("Game config", parallel_session.parallel_env.game_config)

    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel)

    ## Initializing diversity locations
    # div_agents = [self_play_agent for _ in range(2)]
    div_obs = load_obs(path_to_obs)[0]
    
    # Crea te an observation stacker
    stacker = [self_play_agent.create_stacker(
                                eval_env.observation_len,
                                eval_env.num_states
                                )
                ]

    # The observations that will be used to evaluate diversity
    obs = [ preprocess_obs_for_agent(
                        o, 
                        self_play_agent,
                        stacker[0], 
                        eval_env)  for o in div_obs ]

    # start time
    start_time = time.time()


    # Calculate Diversity
    diversity = calculate_diversity(
        self_play_agent=self_play_agent,
        observations=obs,
        path_to_agents=path_to_agents,
        div_parallel_eval=n_parallel_eval
    )

    np.fill_diagonal(diversity, 1)
    
    #printout diversity matrix in heatmap
    plt.imshow(diversity)
    plt.colorbar()
    plt.savefig(output_dir + "/diversity.png")


def main(args):

    if args.agent_config_path is not None:
        gin.parse_config_file(args.agent_config_path)

    del args.agent_config_path
    
    session(**vars(args))


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Train a dm-rlax based rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Destination for storing weights and statistics")


    parser.add_argument(
        "--path_to_agents", type=str, default="../Agents/Database",
        help="Path to load the weights of partner from a Database"
    )

    parser.add_argument(
        "--path_to_obs", type=str, default="../Agents/Observations",
        help="Path to the observation database"
    )
    
    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    args = parser.parse_args()

    #main(**vars(args))
    main(args)
    
    
