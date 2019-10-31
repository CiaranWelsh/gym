import os
import numba

from gym_bandits.bandit import *


def incremental_epsilon_greedy_bandit(env, Q, eps):
    r = np.random.uniform()
    if r > eps:
        return np.argmax(Q)
    else:
        return env.action_space.sample()


@numba.jit(nopython=True)
def learn(env, n=1000):
    """

    :param env: A bandit environment from the gym_bandits package
    :param n: the number of individual bandit problems to run
    :return:
    """
    k = env.action_space.n  # how many arms
    all_qs = np.zeros((n, k))  # place to store learnt action values
    for i in range(n):
        Q = np.zeros(k)
        N = np.zeros(k)  # time step counter
        env.reset()
        for t in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            # update N for incremental average
            N[action] += 1

            # update rule, derived from average update rule
            Q[action] = Q[action] + 1 / N[action] * (reward - Q[action])

        all_qs[i, :] = Q

    env.close()
    return all_qs


if __name__ == '__main__':
    env = gym.make("BanditTenArmedUniformDistributedReward-v0")

    action_values = learn(env)
    print(action_values)
