import gym
import SuttonBartoRL
import gym_minigrid



def solve_gridworld(env):

    print(env.step())







if __name__ == '__main__':
    env = gym.make('MiniGrid-Empty-5x5-v0')

    print(env.observation_space)
    print(env.action_space)

    print(env.step(1))


























