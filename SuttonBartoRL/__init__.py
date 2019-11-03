import os

WORKING_DIRECTORY = os.path.dirname(__file__)
BANDITS_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Bandits')
GRIDWORLD_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Gridworld')
GRIDWORLD_ENVIRONMENT_DIRECTORY = os.path.join(os.path.dirname(WORKING_DIRECTORY), 'gym-minigrid')


import site
site.addsitedir(GRIDWORLD_ENVIRONMENT_DIRECTORY)






