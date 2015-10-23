__author__ = 'nick'
# input: stateid number
# output: site variable corresponding to the configuration of the J sites
# note: we use a ternary base.

import numpy as np

def getSite(stateid, J):
    baseTern = (3*np.ones(J))**range(J)
    site = np.zeros(J)
    for i in range(J-1, -1, -1):
        if stateid - 2 * baseTern[i] >= 0:
            site[i] = 2
            stateid = stateid - 2 * baseTern[i]
        elif stateid - baseTern[i] >= 0:
            site[i] = 1
            stateid = stateid - baseTern[i]
    return site

if __name__ == "__main__":
    print(getSite(100, 10))
    print(getSite(-1,5))