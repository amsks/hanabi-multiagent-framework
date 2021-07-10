import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, AgentType, PBTParams
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper
from hanabi_multiagent_framework.utils import eval_pretty_print
#from hanabi_agents.rule_based import RulebasedParams, RulebasedAgent
#from hanabi_agents.rule_based.predefined_rules import piers_rules, piers_rules_adjusted
import logging
import time
import random
import math

import haiku as hk
from haiku._src.data_structures import to_mutable_dict, FlatMapping
import pickle
import jax.numpy as jnp

from pympler.tracker import SummaryTracker


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
    reward_shaper = RewardShaper(params=params) if params is not None else None
    # load agent based on type
    agent_type = AgentType()

    if agent_type.type == 'rainbow':
        agent_params = RlaxRainbowParams()
        return DQNAgent(
            env.observation_spec_vec_batch()[0],
            env.action_spec_vec(),
            agent_params,
            reward_shaper)
    else:
        return None


def split_evaluation(total_reward, n_network, prev_rew):
    '''
    Assigns the total rewards from the different parallel states to the respective atomic agent
    '''
    # calculate the states per agent
    states_per_agent = int(len(total_reward) / n_network)

    print('Splitting evaluations for {} states and {} agents!'.format(
        len(total_reward), n_network))

    mean_reward = np.zeros(n_network)

    for i in range(n_network):
        mean_score = total_reward[
            i*states_per_agent: (i + 1)*states_per_agent
        ].mean()

        mean_reward[i] = mean_score
        print('Average score achieved by AGENT_{} = {:.2f} & reward over past runs = {}'.format(
            i, mean_score, np.average(prev_rew, axis=1)[i]))
    return mean_reward


def add_reward(x, y):
    '''Add reward to reward matrix by pushing prior rewards back'''
    print(x)
    print(y)
    x = np.roll(x, -1)
    x[:, -1] = y
    return x


def define_couples(index_loser, index_survivor):
    a = np.arange(len(index_survivor)).reshape(-1, 1)
    b = np.random.choice(len(index_survivor), len(
        index_survivor), replace=True).reshape(-1, 1)
    return np.hstack([a, b])


def perform_pbt(agent, hyperparams, couples, pbt_params, index_loser, index_survivor):
    # sourcery no-metrics
    weights = agent.trg_params
    opt_states = agent.opt_state[0]

    weights_dict = to_mutable_dict(weights)
    #     opt_states_dict = to_mutable_dict(opt_states)
    objective_weights = {}

    # print(weights_dict['noisy_mlp/~/noisy_linear_0']['b_mu'])
    for key in weights_dict.keys():
        sub_dict = weights_dict[key]
        intermediate_dict = {}
        for sub_key in sub_dict.keys():

            root_array = np.array(sub_dict[sub_key])
            # print('original array is', type(root_array), root_array)
            for pair in couples:
                # print('changing these two')
                # print(root_array[index_survivor[pair[1]]], '>>>>>>>>>>>>')
                root_array[index_loser[pair[0]]
                           ] = root_array[index_survivor[pair[1]]]
            device_array = jnp.array(root_array)

            # print('new device array is', device_array)
            # time.sleep(10)
            # objective[key][sub_key] = device_array
            intermediate_dict[sub_key] = device_array
        objective_weights[key] = FlatMapping(intermediate_dict)

    # print(objective_weights['noisy_mlp/~/noisy_linear_0']['b_mu'])

    for field in opt_states._fields:
        # print('>>>>>>>>.', opt_states)
        # print(field)
        # print(type(getattr(opt_states, field)))
        if str(type(getattr(opt_states, field))) == "<class 'haiku._src.data_structures.FlatMapping'>":
            # print('FlatMapping here we are')
            #             print(getattr(opt_states, field))
            obj = to_mutable_dict(getattr(opt_states, field))

            for key in obj.keys():
                intermediate_dict = {}
                for sub_key in obj[key].keys():
                    array = np.array(obj[key][sub_key])
                    for pair in couples:
                        array[index_loser[pair[0]]
                              ] = array[index_survivor[pair[1]]]
                    device_array = jnp.array(array)
                    # objective[key][sub_key] = device_array
                    intermediate_dict[sub_key] = device_array
                obj[key] = FlatMapping(intermediate_dict)
            # print(str(field))
            opt_states = opt_states._replace(**{field: FlatMapping(obj)})
    #             setattr(opt_states, field, FlatMapping(obj))
            # print('<<<<<<<<<<<<<,,,,,', opt_states)

    for pair in couples:
        if 'lr' in hyperparams:
            lrs = agent.lr
            # print('lrs before >>>>>>>', lrs)
            choices = [lrs[index_survivor[pair[1]]] * pbt_params.lr_factor,
                       lrs[index_survivor[pair[1]]], lrs[index_survivor[pair[1]]]/pbt_params.lr_factor]
            lrs[index_loser[pair[0]]] = random.choice(choices)
            # print('lrs after >>>>>>>>', lrs)
            agent.lr = lrs

        if 'buffersize' in hyperparams:
            buffersizes = agent.buffersize
            # print('buffer before >>>>>', buffersizes)
            choices = [buffersizes[index_survivor[pair[1]]] * pbt_params.buffersize_factor,
                       buffersizes[index_survivor[pair[1]]], buffersizes[index_survivor[pair[1]]]/pbt_params.buffersize_factor]
            choice = int(random.choice(choices))
            if choice <= 512:
                choice = 512
            elif choice >= 2**20:
                choice = 2**20
            buffersizes[index_loser[pair[0]]] = choice
            agent.buffersize = buffersizes
            agent.buffer[index_loser[pair[0]]].change_size(
                buffersizes[index_survivor[pair[1]]])
            # print('buffer after >>>>>>', buffersizes)

        if 'alpha' in hyperparams:
            alphas = agent.alpha
            # print('alphas before >>>>>', alphas)
            choices = [alphas[index_survivor[pair[1]]] * pbt_params.factor_alpha,
                       alphas[index_survivor[pair[1]]], alphas[index_survivor[pair[1]]]/pbt_params.factor_alpha]
            alphas[pair[0]] = random.choice(choices)
            agent.alpha = alphas
            agent.buffer[index_loser[pair[0]]
                         ].alpha = alphas[index_survivor[pair[1]]]
            # print('alphas after >>>>>', alphas)

    return FlatMapping(objective_weights), [opt_states]


@gin.configurable(denylist=['output_dir', 'self_play'])
def session(
    #agent_config_path=None,
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
    output_dir="/output",
    start_with_weights=None,
    n_backup=500,
    restore_weights=None
):  # sourcery no-metrics

    max_score = 25 if hanabi_game_type == 'Hanabi-Small' else 15

    with gin.config_scope('agent_0'):
        pbt_params = PBTParams()
        agent_params = RlaxRainbowParams()

    print(epochs, n_parallel, n_parallel_eval)
    #tracker = SummaryTracker()

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

    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    logger.info('Game Config\n' + str(env_conf))

    # create training and evaluation parallel environment
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    # get agent and reward shaping configurations
    if self_play:

        with gin.config_scope('agent_0'):

            agent = load_agent(env)
            if restore_weights is not None:
                agent.restore_weights(restore_weights, restore_weights)
            agents = [agent for _ in range(n_players)]
            logger.info("self play")
            logger.info("Agent Config\n" + str(agent))
            logger.info("Reward Shaper Config\n" +
                        str(agent.reward_shaper))

    else:

        agents = []
        logger.info("multi play")

        for i in range(n_players):
            with gin.config_scope('agent_'+str(i)):
                agent = load_agent(env)
                logger.info("Agent Config " + str(i) + " \n" + str(agent))
                logger.info("Reward Shaper Config\n" +
                            str(agent.reward_shaper))
                agents.append(agent)

    # load previous weights
    if start_with_weights is not None:
        print(start_with_weights)
        for aid, agent in enumerate(agents):
            if "agent_" + str(aid) in start_with_weights:
                agent.restore_weights(
                    *(start_with_weights["agent_" + str(aid)]))

    # start parallel session for training and evaluation
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    print("Game config", parallel_session.parallel_env.game_config)

    population_size = int(agent_params.n_network)
    epochs_alive = np.ones(population_size) + pbt_params.life_span

    # evaluate the performance before training
    mean_reward_prev = np.zeros(
        (agent_params.n_network, pbt_params.n_mean))
    print(mean_reward_prev)
    total_reward = parallel_eval_session.run_eval()
    mean_reward = split_evaluation(
        total_reward, agent_params.n_network, mean_reward_prev)
    # mean_reward_prev = parallel_eval_session.run_eval().mean()

    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel) + n_players

    # start time
    start_time = time.time()

    # activate store_td
    for a in agents:
        a.store_td = epoch_offset < 50
    print('store TD', agents[0].store_td)

    # start training
    for epoch in range(epoch_offset+4, epochs + epoch_offset, 5):

        # Train
        parallel_session.train( n_iter=eval_freq,
                                n_sim_steps=n_sim_steps,
                                n_train_steps=n_train_steps,
                                n_warmup=n_warmup   )

        # no warmup after epoch 0
        n_warmup = 0

        # print number of train steps
        print("step", agents[0].train_step)
        #if self_play:
        #    print("step", (epoch + 1) * eval_freq * n_train_steps * n_players)
        #else:
        #    print("step", (epoch + 1) * eval_freq * n_train_steps)

        # evaluate
        print(mean_reward_prev, mean_reward)
        mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
        output_path = os.path.join(output_dir, "stats", str(epoch))
        total_reward = parallel_eval_session.run_eval(
            dest=output_path,
            store_steps=False,
            store_moves=False
        )
        
        mean_reward = split_evaluation(
            total_reward, agent_params.n_network, mean_reward_prev)

        stochasticity = agents[0].get_stochasticity()
        # np.save(output_path + "_stochasticity.npy", stochasticity)

        #drawn_td = agents[0].get_drawn_tds(deactivate=False)
        #np.save(output_path + "_drawn_tds.npy", drawn_td)

        epochs_alive += 5
        print(epochs_alive)

        if (epoch + 1) % 50 == 0:
            buffer_td = agents[0].get_buffer_tds()
            np.save(output_path + "_buffer_tds.npy", buffer_td)

        if epoch < 49:
            drawn_td = agents[0].get_drawn_tds(deactivate=False)
            np.save(output_path + "_drawn_tds.npy", drawn_td)
        elif epoch == 49:
            drawn_td = agents[0].get_drawn_tds(deactivate=True)
            np.save(output_path + "_drawn_tds.npy", drawn_td)

        # compare to previous iteration and store checkpoints
        if (epoch + 1) % n_backup == 0:

            print('save weights', epoch)
            # True if (epoch + 1) < epochs + epoch_offset else False
            only_weights = False

            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"),
                    "ckpt_" + str(agents[0].train_step),
                    only_weights=only_weights)

            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights",
                                        "agent_" + str(aid)),
                        "ckpt_" + str(agent.train_step),
                        only_weights=only_weights)

        # store the best network
        if np.max(mean_reward_prev) < np.max(mean_reward):

            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), "best")

            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), "best")

            # mean_reward_prev = mean_reward

        # logging
        logger.info("epoch {}: duration={}s    reward={}".format(
            epoch, time.time()-start_time, mean_reward))
        start_time = time.time()

        #tracker.print_diff()


def linear_schedule(val_start, val_end, n_steps):

    def schedule(step):
        increase = (val_end - val_start) / n_steps
        if val_end > val_start:
            return min(val_end, val_start + step * increase)
        else:
            return max(val_end, val_start + step * increase)

    return schedule


def exponential_schedule(val_start, val_end, decrease):

    def schedule(step):
        return max(val_start * (decrease**step), val_end)

    return schedule


def ramp_schedule(val_start, val_end, n_steps):

    def schedule(step):
        return val_start if step < n_steps else val_end

    return schedule


@gin.configurable
def schedule_beta_is(value_start, value_end, steps):
    return linear_schedule(value_start, value_end, steps)


@gin.configurable
def schedule_epsilon(value_start=1, value_end=0, steps=50*2000):
    return linear_schedule(value_start, value_end, steps)


@gin.configurable
def schedule_tau(value_start=1, value_end=0.0001, decrease=0.99995):
    return exponential_schedule(value_start, value_end, decrease)


def main(args):

    # load configuration from gin file
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
        "--start_with_weights", type=json.loads, default=None,
        help="Initialize the agents with the specified weights before training. Syntax: {\"agent_0\" : [\"path/to/weights/1\", ...], ...}")

    args = parser.parse_args()

    #main(**vars(args))
    main(args)
