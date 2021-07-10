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
    """
    Load the agent based on the type of agent and the paramters associated with it

    Args:
        env :  Object to the environment

    Returns:
        Agent instantiated
    """
    
    # load reward shaping infos
    params = RewardShapingParams()
    reward_shaper = RewardShaper(params = params) if params is not None else None
    # load agent based on type
    agent_type = AgentType()

    if agent_type.type == 'rainbow':
        agent_params = RlaxRainbowParams()
        return DQNAgent(
                env.observation_spec_vec_batch()[0],
                env.action_spec_vec(),
                agent_params,
                reward_shaper   )

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
        if 'target' in str(file):
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
    agents,  
    path_to_partner, 
    observations,
    diversity_env, 
    div_parallel_eval):
    
    # Get the names and numbers of partners from the pool
    n_partners, names = no_agent_in_path(path_to_partner)
    partners = [[self_play_agent for _ in range(2)] for _ in range(n_partners) ]

    print('Evaluating {} agents from the following given files {}'.format(
        n_partners, names))

    # Restore the weight for each partner in the pool
    for i in range(n_partners):
        for agent in partners[i]:
            location = os.path.join(path_to_partner, names[i])
            agent.restore_weights(location, location)

    # Create a population so that our agent is at index 0
    population = [agents]
    for partner in partners:
        population.append(partner)

    # Get the actions by each member of the population
    action_matrix = []
    for agent in population: 
        actions = [agent[0].exploit(o) for o in observations]
        action_matrix.append(actions)

    action_matrix = np.array(action_matrix)
    
    # Flatten the actions to get a single list for each index in the
    # matrix
    action_mat = [action.flatten() for action in action_matrix]

    # Calculate the diveristy between the current agent and each partner 
    # based on the actions
    diversity = []
    for i in range(1,len(action_mat)):
        no_obs_test = len(observations) * 2 * div_parallel_eval
        diversity.append(
            simple_match(
                action_mat[0], 
                action_mat[i], 
                no_obs_test
        ))
        
    print(f"Diversity with each partner : {diversity}")
    print(f"Average Diversity           : {np.mean(diversity)}")
    
    # Return the mean diversity of the agent with all the partners in the pool
    return np.mean(diversity)


@gin.configurable(denylist=['output_dir', 'self_play'])
def session(
            #agent_config_path=None,
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 32,
            n_parallel_eval:int = 1_000,
            n_train_steps: int = 4,
            n_sim_steps: int = 2,
            epochs: int = 1_000_000,
            epoch_offset = 0,
            eval_freq: int = 500,
            self_play: bool = True,
            output_dir: str = "./output",
            start_with_weights: bool =None,
            n_backup: int = 500, 
            restore_weights: bool = None,
            log_observation: bool=False, 
            path_to_partner: str = "./Agents/Database",
            path_to_obs: str = "./Agents/Observations", 
            div_parallel_eval: int = 100,
            div_factor: float = 1.0,
            log_diversity: bool = False,
            invert_diversity: bool = False

    ):  # sourcery no-metrics
    
    print(epochs, n_parallel, n_parallel_eval)

    # create folder structure
    os.makedirs(os.path.join(output_dir, "weights"))
    os.makedirs(os.path.join(output_dir, "stats"))
    for i in range(n_players):
        os.makedirs(os.path.join(output_dir, "weights", "agent_" + str(i)))

    #logger
    logger = logging.getLogger('Training_Log')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'debug.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create hanabi environment configuration
    # env_conf = make_hanabi_env_config('Hanabi-Small-CardKnowledge', n_players)
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)

    # If the maximum tokens are not none, load them up
    if max_life_tokens is not None:
            env_conf["max_life_tokens"] = str(max_life_tokens)
    logger.info('Game Config\n' + str(env_conf))

    # create training and evaluation parallel environment
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    # Another parallel environment for calculating diversity
    diversity_env = hmf.HanabiParallelEnvironment(env_conf, div_parallel_eval * 2)
    with gin.config_scope('agent_0'):
        div_agent = load_agent(diversity_env)

    # get agent and reward shaping configurations
    if self_play:

        with gin.config_scope('agent_0'):

            self_play_agent = load_agent(env)
            if restore_weights is not None:
                self_play_agent.restore_weights(restore_weights, restore_weights)
            agents = [self_play_agent for _ in range(n_players)]
            logger.info("self play")
            logger.info("Agent Config\n" + str(self_play_agent))
            logger.info("Reward Shaper Config\n" +
                        str(self_play_agent.reward_shaper))

    else:

        agents = []
        logger.info("multi play")

        for i in range(n_players):
            with gin.config_scope('agent_'+str(i)): 
                agent = load_agent(env)
                logger.info("Agent Config " + str(i) + " \n" + str(agent))
                logger.info("Reward Shaper Config\n" + str(agent.reward_shaper))
                agents.append(agent)

    # load previous weights            
    if start_with_weights is not None:
        print(start_with_weights)
        for aid, agent in enumerate(agents):
            if "agent_" + str(aid) in start_with_weights:
                agent.restore_weights(*(start_with_weights["agent_" + str(aid)]))

    # start parallel session for training and evaluation          
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)



    print("Game config", parallel_session.parallel_env.game_config)

    # evaluate the performance before training
    mean_reward_prev = parallel_eval_session.run_eval().mean()

    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel)

    ## Initializing diversity locations
    # div_agents = [self_play_agent for _ in range(2)]
    div_obs = load_obs(path_to_obs)[0]
    stacker = [self_play_agent.create_stacker(
        diversity_env.observation_len, diversity_env.num_states)]

    obs = [
        preprocess_obs_for_agent(o, self_play_agent, stacker[0], diversity_env)
        for o in div_obs
    ]

    # start time
    start_time = time.time()

    diversity_tracker = []
    # start training - epochs control the total training steps 
    for epoch in range(epoch_offset, epochs + epoch_offset):

        # Calculate Diversity
        diversity = calculate_diversity(
            diversity_env= diversity_env, 
            agents=parallel_session.agents.agents,
            self_play_agent=div_agent, 
            observations= obs, 
            path_to_partner=path_to_partner, 
            div_parallel_eval=div_parallel_eval
        )

        diversity_tracker.append(diversity)

        if log_diversity:
            np.save(os.path.join(   output_dir, 
                                    "stats",
                                    ) + "/diversity.npy",
                                diversity_tracker)

        # Train
        if invert_diversity:
            parallel_session.train( 
                n_iter=eval_freq,
                n_sim_steps=n_sim_steps,
                n_train_steps=n_train_steps,
                n_warmup=n_warmup,
                diversity= 1. / diversity, 
                factor=div_factor
            )
            
        else:
            parallel_session.train(
                n_iter=eval_freq,
                n_sim_steps=n_sim_steps,
                n_train_steps=n_train_steps,
                n_warmup=n_warmup,
                diversity=diversity,
                factor=div_factor
            )


        # np.save(os.path.join(output_dir, "stats", str(epoch)) + "_training_rewards.npy", rewards)


        # no warmup after epoch 0
        n_warmup = 0

        # print number of train steps
        if self_play:
            print("step", (epoch + 1) * eval_freq * n_train_steps * n_players)
        else:
            print("step", (epoch + 1) * eval_freq * n_train_steps)

        # evaluate
        mean_reward = parallel_eval_session.run_eval(
            dest=os.path.join(output_dir, "stats", str(epoch)),
            store_moves=True,
            store_steps = True, 
            log_observation=log_observation
            ).mean()

        # compare to previous iteration and store checkpoints
        if (epoch + 1) % n_backup == 0:

            print('save weights', epoch)

            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), 
                    "ckpt_" + str(agents[0].train_step))

            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), 
                        "ckpt_" + str(agents[0].train_step))

        # store the best network
        if mean_reward_prev < mean_reward:

            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), "best") 

            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), "best") 

            mean_reward_prev = mean_reward

        # logging
        logger.info("epoch {}: duration={}s    reward={}".format(epoch, time.time()-start_time, mean_reward))
        start_time = time.time()
        
        
def linear_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        increase = (val_end - val_start) / n_steps
        return min(val_end, val_start + step * increase)
    
    return schedule


def ramp_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        return val_start if step < n_steps else val_end
    
    return schedule


@gin.configurable
def schedule_beta_is(value_start, value_end, steps):
    return linear_schedule(value_start, value_end, steps)


@gin.configurable
def schedule_risk_penalty(value_start, value_end, steps):
    return ramp_schedule(value_start, value_end, steps)

        
def main(args):
    
    # load configuration from gin file
    if args.agent_config_path is not None:
        gin.parse_config_file(args.agent_config_path)
    
    del args.agent_config_path
    session(**vars(args))

            
if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Train a dm-rlax based rainbow agent.")

    parser.add_argument(
        "--self_play", default=False, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")

    parser.add_argument(
        "--restore_weights", type=str, default=None,
        help="Path to weights of pretrained agent.")
    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Destination for storing weights and statistics")

    parser.add_argument(
        "--log_observation", default=False, action='store_true',
        help="Set true to log observation objects"
    )  
    
    parser.add_argument(
        "--start_with_weights", type=json.loads, default=None,
        help="Initialize the agents with the specified weights before training. Syntax: {\"agent_0\" : [\"path/to/weights/1\", ...], ...}") 

    parser.add_argument(
        "--path_to_partner", type=str, default="../Agents/Database",
        help= "Path to load the weights of partner from a Database"
    )

    parser.add_argument(
        "--path_to_obs", type=str, default="../Agents/Observations", 
        help= "Path to the observation database"
    )

    parser.add_argument(
        "--div_factor", type=float, default=1.,
        help="Importance factor for diversity"
    )

    parser.add_argument(
        "--log_diversity", default=False, action='store_true',
        help = "Store Diveristy over the epochs"
    )

    parser.add_argument(
        "--invert_diversity", default=False, action='store_true',
        help="Use Inversion of diversity for minimization"
    )

    args = parser.parse_args()

    #main(**vars(args))  
    main(args)         
            
            
