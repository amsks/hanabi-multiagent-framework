import os
import numpy as np
import gin
import logging
import time
import shutil
import ray

import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams 
from hanabi_agents.rlax_dqn import RewardShapingParams
from hanabi_agents.pbt import AgentDQNPopulation
from hanabi_multiagent_framework.utils import eval_pretty_print


ray.init(num_gpus =1)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

"""
This is an example on how to run the PBT approach for training on DQN/Rainbow agents --> One agent interoperating with
environment and distributing/merging obtained observations/actions to the actual agents.
"""

def load_agent(env):
    
    reward_shaping_params = RewardShapingParams()
    # reward_shaper = RewardShaper(reward_shaping_params)
    population_params = PBTParams()

    agent_params = RlaxRainbowParams()
    print(agent_params)
    return AgentDQNPopulation(
                    env.num_states,
                    env.observation_spec_vec_batch()[0],
                    env.action_spec_vec(),
                    population_params,
                    agent_params,
                    reward_shaping_params)



@gin.configurable(blacklist=['output_dir', 'self_play'])
def session(
            #agent_config_path=None,
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 256,
            n_parallel_eval:int = 2048,
            n_train_steps: int = 1,
            n_sim_steps: int = 1,
            epochs: int = 1_000,
            eval_freq: int = 500,
            self_play: bool = True,
            output_dir = "/output",
    ):

    # TODO: differentiate between agents with/without weights to save (rule vs reinforce)? --> pass
    '''################################################################################################################
    1. Make directory-structure to save checkpoints and stats.
    '''
    population_params = PBTParams()
    population_size = population_params.population_size
    discard_perc = population_params.discard_percent
    lifespan = population_params.life_span
    ########### debugging##########

    shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    ###############################
    os.makedirs(os.path.join(output_dir, "weights"))
    os.makedirs(os.path.join(output_dir, "stats"))
    for i in range(n_players):
        os.makedirs(os.path.join(output_dir, "weights", "pos_" + str(i)))
        for j in range(population_size):
            os.makedirs(os.path.join(output_dir, "weights","pos_" + str(i), "agent_" + str(j)))


    #assert n_parallel and n_parallel_eval are multiples of popsize
    assert n_parallel % population_size == 0, 'n_parallel has to be multiple of pop_size'
    assert n_parallel_eval % population_size == 0, 'n_parallel_eval has to be multiple of pop_size'


    '''################################################################################################################
    2. Helper functions
    '''
    def create_exp_decay_scheduler(val_start, val_min, inflection1, inflection2):
        def scheduler(step):
            if step <= inflection1:
                return val_start
            elif step <= inflection2:
                return val_start / 2
            else:
                return max(val_min, min(val_start / (step - inflection2) * 10000, val_start / 2))
        return scheduler

    def create_linear_scheduler(val_start, val_end, interscept):
        def scheduler(step):
            return min(val_end, val_start + step * interscept)
        return scheduler

    def choose_fittest(mean_reward, discard_perc, agent):
        """Chosses the fittest agents after evaluation run and overwrites all the other agents with weights + permutation of lr + buffersize"""
        no_fittest = mean_reward.shape[0] - int(mean_reward.shape[0] * discard_perc)
        index_loser = np.argpartition(mean_reward, no_fittest)[:no_fittest]
        index_survivor = np.argpartition(-mean_reward, no_fittest)[:no_fittest]
        agent.survival_fittest(index_survivor, index_loser)

    def split_evaluation(total_reward, n_parallel, no_pbt_agents):
        '''Assigns the total rewards from the different parallel states to the respective atomic agent'''
        states_per_agent = int(n_parallel / no_pbt_agents)
        mean_reward = np.zeros(no_pbt_agents)
        for i in range(no_pbt_agents):
            mean_score = total_reward[i * states_per_agent: (i + 1) * states_per_agent].mean()
            mean_reward[i] = mean_score
            print('Average score achieved by AGENT_{} = '.format(i), mean_score)
        return mean_reward
    
    def generation_scheduler(epochs, val_start = 50, val_end = 200):
        '''Determines the cycle with with the population is evaluated'''
        def scheduler(step):
            
            return scheduler

    '''################################################################################################################
    3. Initialize environments to play with
    '''
    eps_schedule = create_exp_decay_scheduler(0.5, 0.01, 200000, 2000000)
    beta_is_schedule = create_linear_scheduler(0.0, 1.0, 25e-7 / 4)


    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)

    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    '''################################################################################################################
    4. Initialize managing-agents containing atomic sub-agents with parallel sessions.
    '''
    if self_play:
        
        with gin.config_scope('agent_0'):

            self_play_agent = load_agent(env)
            agents = [self_play_agent for _ in range(n_players)]
    # TODO: --later-- non-self-play
    else:

        agent_1 = AgentDQNPopulation()
        agent_X = None
        ...
        agents = [agent_1]

    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
    print("Game config", parallel_session.parallel_env.game_config)

    '''################################################################################################################
    5. Start Training/Evaluation
    '''
    # eval before
    mean_reward_prev = np.zeros(population_size)
    total_reward = parallel_eval_session.run_eval()
    mean_reward = split_evaluation(total_reward, n_parallel, population_size)

    start_time = time.time()
    # train
    parallel_session.train(
        n_iter=eval_freq,
        n_sim_steps=n_sim_steps,
        n_train_steps=n_train_steps,
        n_warmup=int(256 * 5 * n_players / n_sim_steps))

    print("step", 1 * eval_freq * n_train_steps)
    # eval
    mean_reward_prev = mean_reward
    total_reward = parallel_eval_session.run_eval(dest=os.path.join(output_dir, "stats_", "0"))
    mean_reward= split_evaluation(total_reward, n_parallel, population_size)

    if self_play:
        agents[0].save_weights(
            os.path.join(output_dir, "weights","pos_0"), mean_reward)
    else:
        for aid, agent in enumerate(agents):
            agent.save_weights(
                os.path.join(output_dir, "weights","pos_" + str(aid)), mean_reward)

    for epoch in range(epochs):

        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0)
        print("step", (epoch + 2) * eval_freq * n_train_steps)
        
        # eval after
        mean_reward_prev = mean_reward
        total_reward = parallel_eval_session.run_eval(
            dest=os.path.join(
                output_dir,
                "stats", str(epoch + 1))
            )
        mean_reward = split_evaluation(total_reward, n_parallel, population_size)

        if self_play:
            agents[0].save_weights(
                os.path.join(output_dir, "weights", "pos_0"), mean_reward)
            if epoch > 0 && epoch % lifespan == 0:
                choose_fittest(mean_reward, discard_perc, agents[0])
        else:
            for aid, agent in enumerate(agents):
                agent.save_weights(
                    os.path.join(output_dir, "weights", "pos_" + str(aid)), mean_reward)
                #TODO: Questionable for non-selfplay --> just one agent?
                if epoch > 0 && epoch % lifespan == 0:
                    choose_fittest(mean_reward, discard_perc, agent)

        logger.info("epoch {}: duration={}s    reward={}".format(epoch, time.time()-start_time, mean_reward))
        start_time = time.time()

        # if epoch % (100000 // eval_freq) == 0:
        #     if self_play:
        #         agents[0].save_weights(
        #             os.path.join(output_dir, "weights", "agent_0"), "ckpt_" + str(agents[0].train_step))
        #     else:
        #         for aid, agent in enumerate(agents):
        #             agent.save_weights(
        #                 os.path.join(output_dir, "weights", "agent_" + str(aid)), "ckpt_" + str(agents[0].train_step))
        # if mean_reward_prev < mean_reward:
        #     if self_play:
        #         agents[0].save_weights(
        #             os.path.join(output_dir, "weights", "agent_0"), "best")
        #     else:
        #         for aid, agent in enumerate(agents):
        #             agent.save_weights(
        #                 os.path.join(output_dir, "weights", "agent_" + str(aid)), "best")

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

#     parser.add_argument(
#         "--hanabi_game_type", type=str, default="Hanabi-Small-Oracle",
#         help='Can be "Hanabi-{VerySmall,Small,Full}-{Oracle,CardKnowledge}"')
#     parser.add_argument("--n_players", type=int, default=2, help="Number of players.")
#     parser.add_argument(
#         "--max_life_tokens", type=int, default=None,
#         help="Set a different number of life tokens.")
# #     parser.add_argument(
# #         "--n_parallel", type=int, default=32,
# #         help="Number of games run in parallel during training.")
    parser.add_argument(
        "--self_play", default=True, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")
#     parser.add_argument(
#         "--n_train_steps", type=int, default=4,
#         help="Number of training steps made in each iteration. One iteration consists of n_sim_steps followed by n_train_steps.")
#     parser.add_argument(
#         "--n_sim_steps", type=int, default=2,
#         help="Number of environment steps made in each iteration.")
#     parser.add_argument(
#         "--epochs", type=int, default=1_000_000,
#         help="Total number of rotations = epochs * eval_freq.")
# #     parser.add_argument(
# #         "--eval_n_parallel", type=int, default=1_000,
# #         help="Number of parallel games to use for evaluation.")
#     parser.add_argument(
#         "--eval_freq", type=int, default=500,
#         help="Number of iterations to perform between evaluations.")
    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Destination for storing weights and statistics")


    args = parser.parse_args()

    #main(**vars(args))  
    main(args)         
            