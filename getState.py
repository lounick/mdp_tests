__author__ = 'nick'
# input: site variable representing the current configuration of the J sites
# output: corresponding state index for MDP Toolbox
# note: we use a ternary base.

import numpy as np

def getState(site):
    J = site.shape[0]
    baseTern = (3*np.ones(J))**range(J)
    state = sum(np.multiply(site, baseTern))
    return state

if __name__ == "__main__":
    print(getState(np.array([1,2,3])))