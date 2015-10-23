import mdptoolbox as mdp
import mdptoolbox.example as mdpex
import numpy as np

P, R = mdp.example.forest()

print(isinstance(P, np.ndarray))

print(P)
print(P.shape)
print(R.transpose())

print(R.shape)

vi = mdp.mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
vi.run()
print(vi.policy) # result is (0, 0, 0)
print(vi.max_iter)