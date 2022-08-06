#!/usr/bin/env python
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from gym_airsim.envs.AirGym import AirSimEnv
from algorithm.ppo import PPO
from algorithm.model import Policy
import shutil
from config import get_config
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
import cv2
import baselines
import time
import numpy as np
import os
import collections
from tqdm import trange

def main():

    args = get_config()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    torch.set_num_threads(torch.get_num_threads())

    # path
    model_dir = Path('./results') / args.env_name / args.algorithm_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)


    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir))

    # episode steps should be the same with settings
    #args.episode_length = settings.nb_max_episodes_steps
    shutil.copy("./config.py", str(run_dir / 'config.py'))
    shutil.copy("./settings_folder/settings.py", str(run_dir / 'settings.py'))

    ##You need first start Unreal Editor, then the initialization can be completed
    env = AirSimEnv(need_render=False)
    env.seed(args.seed)

    #Policy network
    if args.continue_last:
        actor_critic=torch.load(args.model_dir + "/agent_model" + ".pt")['model']
    else:
        actor_critic = Policy(env.observation_space.shape,
                              env.action_space,
                              base_kwargs={'recurrent': args.recurrent_policy,
                                           'recurrent_input_size': args.recurrent_input_size,
                                           'recurrent_hidden_size': args.recurrent_hidden_size,
                                           'hidden_size':args.hidden_size
                                           }
                              )

    actor_critic.to(device)

    agent = PPO(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.data_chunk_length,
                args.value_loss_coef,
                args.entropy_coef,
                logger=logger,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=args.use_clipped_value_loss,
                use_huber_loss=args.use_huber_loss,
                huber_delta=args.huber_delta,
                device=device)

    # replay buffer
    rollout = RolloutStorage(args.episode_length,
                             args.n_rollout_threads,
                             env.observation_space.shape,
                             env.action_space,
                             actor_critic.recurrent_hidden_state_size)
    # reset env
    obs= env.reset()

    # rollout
    if len(env.observation_space.shape) == 1:
        rollout.obs[0].copy_(torch.tensor([obs]))
        rollout.recurrent_hidden_states.zero_()
    elif len(env.observation_space.shape) == 3:
        ob = np.array([obs[0]])
        inform=np.array([obs[1]])
        rollout.obs[0].copy_(torch.tensor(ob))
        rollout.inform[0].copy_(torch.tensor(inform))
        rollout.recurrent_hidden_states.zero_()
    else:
        raise NotImplementedError
    rollout.to(device)

    # run
    episodes = int(args.num_env_steps / args.episode_length / args.n_rollout_threads)
    start=time.time()

    num_epi=0
    total_rew=0
    total_step=0
    rews_deque=collections.deque(maxlen=100)
    steps_deque=collections.deque(maxlen=100)
    success_deque=collections.deque(maxlen=100)

    for episode in trange(episodes):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer,
                                   episode,
                                   episodes,
                                   args.lr)



        for step in range(args.episode_length):
            # Sample actions

            with torch.no_grad():
                if len(env.observation_space.shape) == 1:
                    value, action, action_log_prob, recurrent_hidden_states= \
                        actor_critic.act(rollout.obs[step],
                                         rollout.recurrent_hidden_states[step],
                                         rollout.masks[step])
                elif len(env.observation_space.shape) == 3:
                    value, action, action_log_prob, recurrent_hidden_states= \
                        actor_critic.act([rollout.obs[step], rollout.inform[step]],
                                         rollout.recurrent_hidden_states[step],
                                         rollout.masks[step])

            # rearrange action
            actions_env = []
            for i in range(args.n_rollout_threads):
                if env.action_space.__class__.__name__ == 'Discrete':
                    one_hot_action = action.clone().detach().cpu().numpy()[i][0]#dis action
                else:
                    one_hot_action = action.clone().detach().cpu().numpy()[i]  #
                actions_env.append(one_hot_action)

            # Obser reward and next obs
            obs, reward, done, infos = env.step(np.array(actions_env))
            total_rew+=reward
            total_step+=1
            if done:
                num_epi+=1
                if env.success:
                    success_deque.append(1)
                else:
                    success_deque.append(0)
                obs= env.reset()
                rews_deque.append(total_rew)
                steps_deque.append(total_step)
                logger.add_scalars('mean_episode_reward',
                                   {'mean_episode_reward': sum(rews_deque)/len(rews_deque)},
                                   num_epi)
                logger.add_scalars('mean_episode_length',
                                   {'mean_episode_reward': sum(steps_deque) / len(steps_deque)},
                                   num_epi)
                logger.add_scalars('success_rate',
                                   {'success_rate': sum(success_deque) / len(success_deque)},
                                   num_epi)
                total_rew=0
                total_step=0

            img = obs[0]
            img = np.hstack(img)
            img = np.array(img, dtype=np.uint8)
            #img = np.array(obs[0][0] , dtype=np.uint8)
            cv2.imshow("0", img)
            cv2.waitKey(1)

            # If done then clean the history of observations.
            # insert data in buffer
            mask = []
            bad_mask = []

            #for i in range(len(done)):
            if done:
                mask.append([0.0])
                bad_mask.append([0.0])
            else:
                mask.append([1.0])
                bad_mask.append([1.0])

            if len(env.observation_space.shape) == 1:
                ob = np.array([obs])
                reward=torch.tensor([[reward]])
                rollout.insert(
                    torch.tensor(ob),
                    torch.tensor([env.state()]),
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    torch.tensor(reward),
                    torch.tensor(mask),
                    torch.tensor(bad_mask))

            elif len(env.observation_space.shape) == 3:
                ob = np.array([obs[0]])
                inform = np.array([obs[1]])
                rollout.insert(
                    torch.tensor(ob),
                    torch.tensor(inform),
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    torch.tensor([[reward]]),
                    torch.tensor(mask),
                    torch.tensor(bad_mask))

        with torch.no_grad():
            if len(env.observation_space.shape) == 1:
                next_value = actor_critic.get_value(
                    rollout.obs[-1],
                    rollout.recurrent_hidden_states[-1],
                    rollout.masks[-1])
            elif len(env.observation_space.shape) == 3:
                next_value = actor_critic.get_value(
                    [rollout.obs[-1], rollout.inform[-1]],
                    rollout.recurrent_hidden_states[-1],
                    rollout.masks[-1])

        rollout.compute_returns(next_value,
                                args.use_gae,
                                args.gamma,
                                args.gae_lambda,
                                )

        # update the network
        #env.airgym.client.simPause(True)
        agent.update(rollout)
        #env.airgym.client.simPause(False)

        # clean the buffer and reset
        if len(env.observation_space.shape) == 1:
            rollout.obs[0]=rollout.obs[-1]
            rollout.recurrent_hidden_states[0]=rollout.recurrent_hidden_states[-1]
        elif len(env.observation_space.shape) == 3:
            rollout.obs[0].copy_(rollout.obs[-1])
            rollout.inform[0].copy_(rollout.inform[-1])
            rollout.recurrent_hidden_states[0].copy_(rollout.recurrent_hidden_states[-1])
        else:
            raise NotImplementedError

        #rollout.to(device)

        # save for every interval-th episode or for the last epoch
        if (episode % args.save_interval == 0 or episode == episodes - 1):
            torch.save({
                'model': actor_critic
            },
                str(save_dir) + "/agent_model_{}".format(episode) + ".pt")


    logger.close()

if __name__ == "__main__":
    main()
