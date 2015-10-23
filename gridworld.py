from samba.dcerpc.misc import policy_handle

__author__ = 'nick'

# Value iteration algorithm for a small 4*4 gridworld with 4 move actions. Goal state is state (4,4)

import mdptoolbox as mdp
import numpy as np

# We have four actions. Move north, move south, move east and move west

P = np.zeros((4, 4, 4))


# Populate the transition probability per action
for i in range(4):
    for j in range(4):
        for k in range(4):  # For each action
            if k == 0:  # Action move north
                if i == 0:
                    P[i, j, k] = 1
                else:
                    P[i-1, j, k] = 1
            elif k == 1:  # Action move south
                if i == 3:
                    P[i, j, k] = 1
                else:
                    P[i+1, j, k] = 1
            elif k == 2:  # Action move east
                if j == 3:
                    P[i, j, k] = 1
                else:
                    P[i, j+1, k] = 1
            elif k == 3:  # Action move west
                if j == 0:
                    P[i, j, k] = 1
                else:
                    P[i, j-1, k] = 1

# print ("Action move north:")
# print(P[:, :, 0])
# print ("Action move south:")
# print(P[:, :, 1])
# print ("Action move east:")
# print(P[:, :, 2])
# print ("Action move west:")
# print(P[:, :, 3])

# Populate the reward function

R = np.zeros((4, 4, 4))

goali = 3
goalj = 3

for i in range(4):
    for j in range(4):
        for k in range(4):  # For each action of this state
            if k == 0:  # Action move north
                if i == 0:
                    nexti = 0
                else:
                    nexti = i - 1

                nextj = j

                if nexti == goali and nextj == goalj:
                    R[i, j, k] = 100
                else:
                    R[i, j, k] = -1
                    if nexti == 2 and nextj == 2:
                        R[i, j, k] = -1000

            if k == 1:  # Action move south
                if i == 3:
                    nexti = i
                else:
                    nexti = i + 1

                nextj = j

                if nexti == goali and nextj == goalj:
                    R[i, j, k] = 100
                else:
                    R[i, j, k] = -1
                    if nexti == 2 and nextj == 2:
                        R[i, j, k] = -1000

            if k == 2:  # Action move east
                if j == 3:
                    nextj = j
                else:
                    nextj = j + 1

                nexti = i

                if nexti == goali and nextj == goalj:
                    R[i, j, k] = 100
                else:
                    R[i, j, k] = -1
                    if nexti == 2 and nextj == 2:
                        R[i, j, k] = -1000

            if k == 3:  # Action move west
                if j == 0:
                    nextj = j
                else:
                    nextj = j - 1

                nexti = i

                if nexti == goali and nextj == goalj:
                    R[i, j, k] = 100
                else:
                    R[i, j, k] = -1
                    if nexti == 2 and nextj == 2:
                        R[i, j, k] = -1000

# print(R[:,:,0])
# print(R[:,:,1])
# print(R[:,:,2])
# print(R[:,:,3])

# Perform value iteration

# Initialisation
theta = 0.00001
gamma = 0.9
V = np.zeros((4,4))

while True:
    delta = 0
    oldV = V
    for i in range(4):
        for j in range(4):
            v = V[i, j]
            V[i, j] = 0
            for k in range(4):
                if k == 0:  # Action move north
                    if i == 0:
                        nexti = 0
                    else:
                        nexti = i - 1
                nextj = j
                if k == 1:  # Action move south
                    if i == 3:
                        nexti = i
                    else:
                        nexti = i + 1
                    nextj = j
                if k == 2:  # Action move east
                    if j == 3:
                        nextj = j
                    else:
                        nextj = j + 1
                    nexti = i
                if k == 3:  # Action move west
                    if j == 0:
                        nextj = j
                    else:
                        nextj = j - 1
                    nexti = i

                tmpVal = R[i, j, k] + gamma*V[nexti,nextj]
                if tmpVal > V[i,j]:
                    V[i,j] = tmpVal
            if delta < abs(v - V[i, j]):
                delta = abs(v - V[i, j])
    if delta < theta:
        break

print(V)

# Calculate the policy
policy = np.zeros((4,4,4))
for i in range(4):
    for j in range(4):
        values = np.zeros(4)
        for k in range(4):
            if k == 0:  # Action move north
                if i == 0:
                    nexti = 0
                else:
                    nexti = i - 1
            nextj = j
            if k == 1:  # Action move south
                if i == 3:
                    nexti = i
                else:
                    nexti = i + 1
                nextj = j
            if k == 2:  # Action move east
                if j == 3:
                    nextj = j
                else:
                    nextj = j + 1
                nexti = i
            if k == 3:  # Action move west
                if j == 0:
                    nextj = j
                else:
                    nextj = j - 1
                nexti = i

            values[k] = R[i,j,k] + gamma*V[nexti,nextj]
        max = np.amax(values)
        idx = np.argwhere(values == max).flatten().tolist()
        prob = 1.0/len(idx)
        for k in range(len(idx)):
            policy[i, j, idx[k]] = prob

print(policy[:,:,0])
print(policy[:,:,1])
print(policy[:,:,2])
print(policy[:,:,3])

P = np.zeros((4,16,16))
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
            P[i,j*4+k,nextj*4+nextk] = 1
print(P[0,:,:])

R = np.zeros((4, 16))
for i in range(4):
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
            if nextj == goali and nextk == goalj:
                R[i,j*4+k] = 100
            else:
                R[i,j*4+k] = -1
                if nextj == 2 and nextk == 2:
                    R[i,j*4+k] = -1000

R = R.transpose()

print(R)

vi = mdp.mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
vi.run()
print(vi.V)
print(vi.policy) # result is (0, 0, 0)
print(vi.max_iter)