import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.ppo import PPO_Agent, PPOParams, AgentType
from hanabi_learning_environment.pyhanabi_pybind import RewardShapingParams as ShapingParams
from hanabi_learning_environment.pyhanabi_pybind import RewardShaper
import logging
import time


@gin.configurable
def RewardShapingParams(
    shaper: bool = True,
    min_play_probability: float = 0.8,
    w_play_penalty: float = 0,
    m_play_penalty: float = 0,
    w_play_reward: float = 0,
    m_play_reward: float = 0,
    penalty_last_of_kind: float = 0 ):

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

    # load reward shaping infos
    params = RewardShapingParams()
    reward_shaper = RewardShaper(params=params) if params is not None else None
    
    # load agent based on type
    agent_type = AgentType()

    
    agent_params = PPOParams()
    return PPO_Agent(
        env.observation_spec_vec_batch()[0],
        env.action_spec_vec(),
        agent_params,
        reward_shaper   
    )

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
    epoch_offset: int =0,
    eval_freq: int = 500,
    self_play: bool = True,
    output_dir="./output",
    start_with_weights=None,
    n_backup=500,
    restore_weights=None    ):  # sourcery no-metrics

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
            # TODO -> Handle this in the agent file
            if restore_weights is not None:
                agent.restore_weights(restore_weights, restore_weights)
            agents = [agent for _ in range(n_players)]
            logger.info("self play")
            logger.info("Agent Config\n" + str(agent))
            # logger.info("Reward Shaper Config\n" + str(agent.reward_shaper))

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
                # TODO handle this in agent file 
                agent.restore_weights(
                    *(start_with_weights["agent_" + str(aid)]))

    # start parallel session for training and evaluation
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    print("Game config", parallel_session.parallel_env.game_config)

    # evaluate the performance before training
    mean_reward_prev = parallel_eval_session.run_eval().mean()

    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel)

    # start time
    start_time = time.time()

    # start training
    for epoch in range(epoch_offset, epochs + epoch_offset):

        # train
        parallel_session.train( n_iter=eval_freq,
                                n_sim_steps=n_sim_steps,
                                n_train_steps=n_train_steps,
                                n_warmup=n_warmup)

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
            store_steps=True
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
                        os.path.join(   output_dir, "weights",
                                        "agent_" + str(aid)),
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
        logger.info("epoch {}: duration={}s    reward={}".format(
            epoch, time.time()-start_time, mean_reward))
        start_time = time.time()

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
