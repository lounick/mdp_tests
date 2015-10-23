__author__ = 'nick'

# Function that performs value iteration. Input is an action transition probability matrix and a reward matrix

import numpy as np

DEBUG = False


def vi(P, R, gamma=None, theta=None):
    # Initialisation
    if gamma is None:
        gamma = 0.9
    if theta is None:
        theta = 0.0001

    num_states = P.shape[0]
    num_actions = P.shape[1]

    # Calculate the values
    V = np.zeros(num_states)
    while True:
        delta = 0
        oldV = V
        for i in range(num_states):
            v = V[i]
            V[i] = 0
            for j in range(num_actions):
                next_states = np.argwhere(P[i,j,:]>0).flatten().tolist()
                tmp_val = 0
                for state in next_states:
                    tmp_val += P[i, j, state] * (R[i, j, state] + gamma * oldV[state])
                if tmp_val > V[i]:
                    V[i] = tmp_val
            if delta < abs(v - V[i]):
                delta = abs(v - V[i])
        if delta < theta:
            break

    if DEBUG:
        print(V)

    # Calculate the policy
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        values = np.zeros(num_actions)
        for j in range(num_actions):
            next_states = np.argwhere(P[i,j,:]>0).flatten().tolist()
            for state in next_states:
                values[j] += P[i, j, state] * (R[i, j, state] + gamma * V[state])
        max_val = np.amax(values)
        idx = np.argwhere(values == max_val).flatten().tolist()
        prob = 1.0/len(idx)
        for k in range(len(idx)):
            policy[i, idx[k]] = prob

    if DEBUG:
        print(policy)

    return policy


def main():
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

    goal = (3, 3)

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
                    R[j*4+k, i, nextj*4+nextk] = -1

    print(P)
    print(R)
    policy = vi(P, R)
    print(policy[0])
    max_prob = np.amax(policy[0])
    idx = np.argwhere(policy[14] == max_prob).flatten().tolist()
    if len(idx) == 1:
        print(idx[0])
    else:
        print("MORE THAN ONE INDEXES")


if __name__ == "__main__":
    main()