import os
# import numba
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from SuttonBartoRL import *
from gym_bandits.bandit import *


def incremental_epsilon_greedy_bandit(env, Q, eps):
    r = np.random.uniform()
    if r > eps:
        return np.argmax(Q)
    else:
        return env.action_space.sample()


# @numba.jit(nopython=True)
def learn(env, n=10, debug=False):
    """

    :param env: A bandit environment from the gym_bandits package
    :param n: the number of individual bandit problems to run
    :return:
    """
    k = env.action_space.n  # how many arms
    all_qs = np.zeros((n, k))  # place to store learnt action values
    for i in range(n):
        eps = 1
        Q = np.zeros(k)
        N = np.zeros(k)  # time step counter
        env.reset()
        for t in range(2000):
            action = incremental_epsilon_greedy_bandit(env, Q, eps=eps)
            # observation always [0] and done always True so can be ignored
            # info always {} so can also be ignored
            _, reward, _, _ = env.step(action)

            # update N for incremental average
            N[action] += 1

            # update rule, derived from average update rule        return [seed]
            Q[action] = Q[action] + (1 / N[action]) * (reward - Q[action])
            # incrementally dereasing epsilon
            eps *= 0.999

            if debug:
                print(f"game {i} step {t} current eps {eps}")
                print(f"action: {action}")
                print(f"reward: {reward}")
                print(f"N[action]: {N[action]}")
                print(f"Q[action]: {Q[action]}\n")

        all_qs[i, :] = Q

    env.close()
    return all_qs


def plot(Q, burnin=1000, filename=None):
    fig = plt.figure()
    sns.violinplot(data=Q[burnin:, :])
    sns.despine(fig=fig, top=True, right=True)
    plt.title('Results of 10-armed bandit problem (burnin={} iterations'.format(burnin))
    plt.xlabel('Action (k)')
    plt.ylabel('Estimated action value')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    return fig


if __name__ == '__main__':
    env = gym.make("BanditTenArmedGaussian-v0")

    N = 1000
    estimated_action_values = learn(env, N, debug=False)
    print(estimated_action_values)

    print(env.p_dist)
    print(env.r_dist)

    fname = os.path.join(BANDITS_DIRECTORY, '10ArmedBanditResults.png')
    plot(estimated_action_values, filename=fname)
