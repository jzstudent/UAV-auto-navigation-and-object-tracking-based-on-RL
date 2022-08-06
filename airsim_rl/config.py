import argparse


def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='AirSim_RL')

    # env
    parser.add_argument("--env_name", type=str, default='AirSimEnv-v42')  # AirSimEnv-v42  CartPole-v0

    # prepare
    parser.add_argument("--algorithm_name", type=str, default='ppo-mlp')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=1)#this must be 1 in airsim env
    parser.add_argument("--num_env_steps", type=int, default=3e5, help='number of environment steps to train (default: 10e6)')

    # lstm
    parser.add_argument("--recurrent_policy", action='store_false', default=False, help='use a recurrent policy')
    parser.add_argument("--data_chunk_length", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--recurrent_input_size", type=int, default=512)# the feature dims of visual extractor output
    parser.add_argument("--recurrent_hidden_size", type=int, default=512)

    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=8, help='number of ppo epochs (default: 4)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.15, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 32)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1.0, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--lr", type=float, default=5e-4, help='learning rate (default: 7e-4)')
    parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--max-grad-norm", type=float, default=5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use-gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae-lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=False)
    parser.add_argument("--huber_delta", type=float, default=10.0)


    # replay buffer
    parser.add_argument("--episode_length", type=int, default=512, help='number of forward steps in A2C (default: 5)')

    # run
    parser.add_argument("--use-linear-lr-decay", action='store_false', default=False, help='use a linear schedule on the learning rate')
    
    # save
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--continue_last", default=False)

    # log
    parser.add_argument("--log_interval", type=int, default=1)

    #eval
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--ifi", type=float, default=0.333333)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default='results/AirSimEnv-v42/ppo-lstm3/run5/models')

    args = parser.parse_args()

    return args
