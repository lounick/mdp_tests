__author__ = 'nick'


import numpy as np
import mdptoolbox as mdp
from value_iteration import vi

from scipy.spatial.distance import cityblock

DEBUG = False


def calc_manhattan_dist(x1, x2, y1, y2):
    return abs(x1-x2)+abs(y1-y2)

def generate_table(dimX, dimY, tile_colours, trap_probs):
    # Generate colours list for colours (Red, Purple, Green, Yellow)
    red = 0
    purple = 1
    green = 2
    yellow = 3

    tiles = []
    for i in range(len(tile_colours)):
        tiles.extend([i]*tile_colours[i])

    board = np.zeros((dimX, dimY))
    traps = np.zeros((dimX, dimY))
    for i in range(dimX):
        for j in range(dimY):
            tile = np.random.randint(0, len(tiles))
            board[i, j] = tiles[tile]
            prob = np.random.rand()
            if prob < trap_probs[tiles[tile]]:
                traps[i, j] = 1
            tiles.pop(tile)

    # Can't have trap at the start
    traps[0, 0] = 0

    # Generate random ending position
    endX = np.random.randint(0, 4)
    if endX == 0:
        # If the end position is on the first row we can't end at the start
        endY = np.random.randint(1, 4)
    else:
        endY = np.random.randint(0, 4)

    # Can't have trap at the end
    traps[endX, endY] = 0

    if DEBUG:
        print("Start: (0, 0)")
        print("End: ({0}, {1})".format(endX, endY))
        print(board)
        print(traps)

    return (endX, endY), board, traps


def play_a_game(goal, board, traps, trap_distr, chips):
    # Play a game of coloured trails

    # Initialisation
    # Always start from (0, 0) and with no information
    pos = [0,0]
    always_communicate = False
    smart_communicate = False
    comms_cost = 35
    traps_known = False
    # print(goal)
    goali = goal[0]
    goalj = goal[1]

    # print(board)
    # print(traps)

    # Calculate a policy for movement given the current trap distribution knowledge
    P = np.zeros((16, 4, 16))
    for i in range(4):  # For each action
        for j in range(4):
            for k in range(4):
                if i == 0:
                    if j == 0:
                        nextj = j
                    else:
                        nextj = j - 1
                    nextk = k
                if i == 1:
                    if j == 3:
                        nextj = j
                    else:
                        nextj = j + 1
                    nextk = k
                if i == 2:
                    if k == 3:
                        nextk = k
                    else:
                        nextk = k + 1
                    nextj = j
                if i == 3:
                    if k == 0:
                        nextk = k
                    else:
                        nextk = k - 1
                    nextj = j
                P[j*4+k,i,nextj*4+nextk] = 1

    R = np.zeros((16, 4, 16))
    for i in range(4):  # For each action
        for j in range(4):
            for k in range(4):
                if i == 0:
                    if j == 0:
                        nextj = j
                    else:
                        nextj = j - 1
                    nextk = k
                if i == 1:
                    if j == 3:
                        nextj = j
                    else:
                        nextj = j + 1
                    nextk = k
                if i == 2:
                    if k == 3:
                        nextk = k
                    else:
                        nextk = k + 1
                    nextj = j
                if i == 3:
                    if k == 0:
                        nextk = k
                    else:
                        nextk = k - 1
                    nextj = j
                if nextj == goal[0] and nextk == goal[1]:
                    R[j*4+k, i, nextj*4+nextk] = 100
                else:
                    color = board[nextj, nextk]
                    R[j*4+k, i, nextj*4+nextk] = 0#-trap_distr[int(color)]#*10*calc_manhattan_dist(goali, nextj, goalj, nextk)

    # print(R)

    uninformed_policy = vi(P, R)
    # print(uninformed_policy)

    RI = np.zeros((16, 4, 16))
    for i in range(4):  # For each action
        for j in range(4):
            for k in range(4):
                if i == 0:
                    if j == 0:
                        nextj = j
                    else:
                        nextj = j - 1
                    nextk = k
                if i == 1:
                    if j == 3:
                        nextj = j
                    else:
                        nextj = j + 1
                    nextk = k
                if i == 2:
                    if k == 3:
                        nextk = k
                    else:
                        nextk = k + 1
                    nextj = j
                if i == 3:
                    if k == 0:
                        nextk = k
                    else:
                        nextk = k - 1
                    nextj = j
                if nextj == goal[0] and nextk == goal[1]:
                    RI[j*4+k, i, nextj*4+nextk] = 100
                elif traps[nextj, nextk] == 1:
                    RI[j*4+k, i, nextj*4+nextk] = -10*calc_manhattan_dist(goali, nextj, goalj, nextk)
                else:
                    RI[j*4+k, i, nextj*4+nextk] = 0
    informed_policy = vi(P, RI)
    # print(informed_policy)

    if smart_communicate:
        # We need to calculate the communication policy
        # We have two actions inform and not inform
        P_Comm = np.zeros((32, 2, 32)) # 32 states (16 for each policy)
        for i in range(2):
            for j in range(16):
                if i == 0: # not inform
                    P_Comm[j, i, j] = 1
                if i == 1:
                    P_Comm[j, i, j+16] = 1

        R_Comm = np.zeros((32, 2, 32))
        for i in range(2):
            for j in range(16):
                if i == 0:
                    R_Comm[j, i, j] = 0
                if i == 1:
                    # Calculate the value of the current policy
                    # Get all the actions gor this policy.
                    vcurr = 0
                    actions = uninformed_policy[j]
                    max_prob = np.amax(actions)
                    idx = np.argwhere(actions == max_prob).flatten().tolist()
                    for k in range(len(idx)):  # For each possible action
                        action = idx[k]
                        if action == 0:
                            next_state = j - 4
                            if next_state < 0:
                                next_state = j
                        if action == 1:
                            next_state = j + 4
                            if next_state > 15:
                                next_state = j
                        if action == 2:
                            if j%4 == 3:
                                next_state = j
                            else:
                                next_state = j + 1
                        if action == 3:
                            if j%4 == 0:
                                next_state = j
                            else:
                                next_state = j - 1
                        nexti = next_state/4
                        nextj = next_state%4
                        if(traps[nexti,nextj] == 0):
                            vcurr += max_prob*R[j,action,next_state]
                        else:
                            vcurr += max_prob*(-10*calc_manhattan_dist(goali, nexti, goalj, nextj))

                    vnext = 0
                    actions = informed_policy[j]
                    max_prob = np.amax(actions)
                    idx = np.argwhere(actions == max_prob).flatten().tolist()
                    for k in range(len(idx)):  # For each possible action
                        action = idx[k]
                        if action == 0:
                            next_state = j - 4
                            if next_state < 0:
                                next_state = j
                        if action == 1:
                            next_state = j + 4
                            if next_state > 15:
                                next_state = j
                        if action == 2:
                            if j%4 == 3:
                                next_state = j
                            else:
                                next_state = j + 1
                        if action == 3:
                            if j%4 == 0:
                                next_state = j
                            else:
                                next_state = j - 1
                        vnext += max_prob*RI[j,action,next_state]

                    R_Comm[j, i, j+16] = abs(vnext - vcurr) - comms_cost
                    # print(R_Comm[j, i, j+16])

        # print("P_Comm")
        # print(P_Comm)
        # print("R_Comm")
        # print(R_Comm)
        comms_policy = vi(P_Comm, R_Comm)
        # print(comms_policy[0])

    # Play the game!
    curri = 0
    currj = 0
    reward = 0
    while True:
        # Communication phase
        if not traps_known:
            # What is the communication policy?
            if always_communicate:
                # We always communicate calculate the new policy
                traps_known = True
            elif smart_communicate:
                # We have not communicated yet so decide if you have to communicate
                communicate = False
                curr_state = curri * 4 + currj
                if comms_policy[curr_state,1] == 1:
                    communicate = True
                if communicate:
                    # If we decide to communicate we need to recalculate the movement policy
                    traps_known = True
            else:
                # No communication
                pass


        # Movement phase
        if traps_known:
            # if you know the traps find the shortest path that doesn't include traps
            # Chose your next action based on a path till the end
            # Calculate your state
            curr_state = curri * 4 + currj
            # Get action based on state
            actions = informed_policy[curr_state]
            max_prob = np.amax(actions)
            idx = np.argwhere(actions == max_prob).flatten().tolist()
            if len(idx) == 1:
                action = idx[0]
            else:
                rand_num = np.random.rand()
                for i in range(len(idx)):
                    if (i+1)*max_prob > rand_num:
                        action = idx[i]
            # Find your next state
            if action == 0:
                if curri == 0:
                    nexti = curri
                else:
                    nexti = curri - 1
                nextj = currj
            elif action == 1:
                if curri == 3:
                    nexti = curri
                else:
                    nexti = curri + 1
                nextj = currj
            elif action == 2:
                if nextj == 3:
                    nextj = currj
                else:
                    nextj = currj + 1
                nexti = curri
            elif action == 3:
                if nextj == 0:
                    nextj = currj
                else:
                    nextj = currj - 1
                nexti = curri

            next_state = nexti * 4 + nextj
            # Pay the token
            color = int(board[nexti, nextj])
            chips[color] -= 1
            # Move
            previ = curri
            prevj = currj
            curri = nexti
            currj = nextj

            # Find out if anything happened
            if curri == goali and currj == goalj:  # Got to the goal!
                reward += 100 - comms_cost
                # for i in range(len(chips)):
                #     if chips[i] < 0:
                #         print("ERROR!!!!!")
                #         reward = 0
                break
            elif traps[curri, currj] == 1:  # Hit a trap!
                # return calc_manhattan_dist(curri, goali, currj, goalj)*(-10)
                reward += -10*(np.abs(curri-goali)+np.abs(currj-goalj))-comms_cost
                # if chips[i] < 0:
                #         reward = 0
                #         print("ERROR!!!!!")
                break
            else:
                reward += RI[previ*4+prevj, action, curri*4+currj]
        else:
            # Chose your next action based on a path till the end
            # Calculate your state
            curr_state = curri * 4 + currj
            # Get action based on state
            actions = uninformed_policy[curr_state]
            max_prob = np.amax(actions)
            idx = np.argwhere(actions == max_prob).flatten().tolist()
            if len(idx) == 1:
                action = idx[0]
            else:
                rand_num = np.random.rand()
                for i in range(len(idx)):
                    if (i+1)*max_prob > rand_num:
                        action = idx[i]
            # Find your next state
            if action == 0:
                if curri == 0:
                    nexti = curri
                else:
                    nexti = curri - 1
                nextj = currj
            elif action == 1:
                if curri == 3:
                    nexti = curri
                else:
                    nexti = curri + 1
                nextj = currj
            elif action == 2:
                if nextj == 3:
                    nextj = currj
                else:
                    nextj = currj + 1
                nexti = curri
            elif action == 3:
                if nextj == 0:
                    nextj = currj
                else:
                    nextj = currj - 1
                nexti = curri

            next_state = nexti * 4 + nextj
            # Pay the token
            color = int(board[nexti, nextj])
            chips[color] -= 1
            # Move
            previ = curri
            prevj = currj
            curri = nexti
            currj = nextj

            # Find out if anything happened
            if curri == goali and currj == goalj:  # Got to the goal!
                reward += 100
                # for i in range(len(chips)):
                #     if chips[i] < 0:
                #         print("ERROR!!!!!")
                #         reward = 0
                break
            elif traps[curri, currj] == 1:  # Hit a trap!
                # return calc_manhattan_dist(curri, goali, currj, goalj)*(-10)
                reward += -10*(np.abs(curri-goali)+np.abs(currj-goalj))
                # if chips[i] < 0:
                #         reward = 0
                #         print("ERROR!!!!!")
                break
            else:
                reward += R[previ*4+prevj, action, curri*4+currj]
    return reward



def main():
    tile_colours = [2, 4, 5, 5]
    trap_distr = [0.15, 0.15, 0.15, 0.15] # RPGY
    # trap_distr = [0.15, 0, 0.15, 0] # RPGY
    chips = [2, 3, 3, 3]
    rewards = np.zeros(1000)
    for i in range(1000):
        reward = 0
        chips = [2, 3, 3, 3]
        end, board, traps = generate_table(4, 4, tile_colours, trap_distr)
        # print("End: ({0}, {1})".format(end[0], end[1]))
        # print(board)
        # print(traps)
        # end = (2,3)
        reward = play_a_game(end, board, traps, trap_distr, chips)
        # if reward == 0:
        #     i -= 1
        #     continue
        # print(reward)
        # print(i,reward)
        rewards[i] = reward
    print(rewards.mean())

if __name__ == "__main__":
    main()
