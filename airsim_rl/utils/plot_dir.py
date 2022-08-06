#!/usr/bin/env python

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import copy
'''
from matplotlib import rc
sns.set(font_scale=1.2)
rc('text', usetex=True)
rc('font', **{'family': 'serif',
              'serif': ['Computer Modern Roman'],
              'monospace': ['Computer Modern Typewriter'],
              })
'''


if __name__ == '__main__':
    # get the configuration
    parser = argparse.ArgumentParser(description="Plot results from a dir")
    parser.add_argument(
        '-i',
        "--input",
        type=str,
        required=True,
        help="The directory of the summary file"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="exp name"
    )
    parser.add_argument(
        "-s",
        "--number_steps",
        type=int,
        required=True
    )
    parser.add_argument(
        "-t",
        "--timesteps_per_batch",
        type=int,
        required=False,
        help="The directory of the summary file",
        default=1
    )
    parser.add_argument(
        "-a",
        "--average_across_timesteps",
        type=int,
        required=False,
        help="The directory of the summary file",
        default=3
    )
    args = parser.parse_args()

    # the parameters dict
    data = {'Reward': [], 'Iteration': [], 'Type': [], 'Unit': [],
            'Action': []}
    exp_legend_list = []

    # fetch the data information
    file_path_list = glob.glob(args.input + '*.npy')
    file_path_list = file_path_list[::-1]
    agent_id = 0
    for file_path in file_path_list:
        agent_id += 1
        agent_name = 'Agent-' + str(agent_id) + '-' + \
            (file_path.split('-')[1]).split('.')[0]
        exp_data = np.load(file_path).item()

        reward_data = exp_data['reward'][: args.number_steps]
        action_data = exp_data['action'][: args.number_steps]

        for sliding_window in range(args.average_across_timesteps):
            # slide the value
            data_length = len(action_data)

            timesteps_per_batch = args.timesteps_per_batch
            timesteps_per_batch *= \
                float(data_length) / (data_length - args.average_across_timesteps)

            data['Reward'].extend([i for i in reward_data])
            data['Action'].extend(action_data)
            data['Iteration'].extend(
                timesteps_per_batch * (
                    np.array(range(data_length)) + sliding_window -
                    args.average_across_timesteps
                )
            )
            # the name of the model
            data['Type'].extend([agent_name] * data_length)
            data['Unit'].extend([sliding_window] * data_length)

    # Total reward
    backup_data = copy.deepcopy(data)

    # plot the figures
    figure_path = os.path.join('./', 'action' + '.pdf')
    ax = plt.figure()
    data = pd.DataFrame(data)
    sns.tsplot(data=data, time='Iteration', value='Action', unit='Unit',
               condition='Type', ci=100)
    plt.autoscale()

    plt.savefig(figure_path)
    figure_path = os.path.join('./', 'reward' + '.pdf')
    ax = plt.figure()
    del backup_data['Action']
    num_data = len(backup_data['Reward']) / 2
    backup_data['Type'].extend(['Total'] * num_data)
    backup_data['Unit'].extend(backup_data['Unit'][:num_data])
    backup_data['Iteration'].extend(backup_data['Iteration'][:num_data])
    backup_data['Reward'].extend(
        [backup_data['Reward'][i] + backup_data['Reward'][i + num_data]
         for i in range(num_data)]
    )

    data = pd.DataFrame(backup_data)
    sns.tsplot(data=data, time='Iteration', value='Reward', unit='Unit',
               condition='Type', ci=100)
    plt.autoscale()

    plt.savefig(figure_path)
