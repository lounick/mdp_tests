__author__ = 'nick'
"""
Testing of mdp communications policy for AUVs with MILP value functions
"""


from sim_auv import AuvExecutor
from sim_auv_planner import AuvPlanner
from msgs import *
from target import Target
from pipe import Pipe

from value_iteration import vi

import numpy as np

"""
Must create a play game function. Where I will create all the stuff needed to run. Also will have to create targets in
an incremental fashion. In that way I can calculate the transmission policy.
"""

def play_a_game():
    # Randomly generate targets
    # Whenever a new observation is made we calculate the communication policy and decide to communicate or not.
    # The game runs until the max time is reached.
    pass


def main():
    num_games = 1000
    rewards = np.zeros(num_games)
    for i in range(num_games):
        # Play a game
        # Collect reward in array
        pass
    print(rewards.mean())

if __name__ == '__main__':
    main()