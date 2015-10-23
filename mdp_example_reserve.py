__author__ = 'nick'
import numpy as np
import  getSite
import getState
import math

def mdp_example_reserve(M = None, pj = None):
    # mdp_example_reserve   Generate a Markov Decision Process example based on
    #                       a simple reserve design problem
    #                       (see the related documentation for more detail)
    # Arguments -------------------------------------------------------------
    #   M(JxI) = species distribution across sites
    #   J = number of sites (> 0), optional (default 5)
    #   I = number of species (>0), optional (default 7)
    #   pj = probability of development occurence, in ]0, 1[, optional (default 0.1)
    # Evaluation -------------------------------------------------------------
    #   P(SxSxA) = transition probability matrix
    #   R(SxA) = reward matrix
    #   M(JxI) = random species distribution across sites

    if M != None and (M.shape[0] == 0 or M.shape[1] == 0):
        print("MDP Toolbox ERROR: M is a JxI matrix with Number of sites J and species I must be greater than 0")
    elif pj != None and (pj < 0 or pj > 1):
        print("MDP Toolbox ERROR: Probability pj must be in [0; 1]")
    else:
        # Initialise optional items
        J = 0
        I = 0
        if pj == None:
            pj = 0.1
        if M == None:
            J = 5
            I = 7
            M = np.round(np.random.rand(J,I))
        J, I = M.shape

        # Definition of states
        S = 3**J # for each site, 0 is available; 1 is reserved; 2 is developed
        A = J # action space

        # There are J actions corresponding to the selection of a site for
        # reservation. A site can only be reserved if it is available.
        # By convention we will use a ternary base where state #0 is the state
        # that corresponds to [0,0, ..,0] all sites are available. State #1 is
        # [1,0,0,...,0]; state 2 is [2,0,0 ..,0] and state 3 is [0,1,0, .. 0] and so forth.
        # for example
        # site = [0,0,1,2] means the first 2 sites are available (site(1:2)=0), site 3 is
        # reserved (site(3)=1) and site 4 is developped (site(4)=2).
        #
        # Build P(AxSxS)
        # complexity is in SxAx2^navail; with 2^navail<=S;

        P = np.zeros(S, S, A)
        for s1 in range(S):  # for all states
            site1 = getSite(s1-1, J)  # state n becomes n-1
            for a in range(A):  # for all actions
                site2 = site1  # site2 represents the site after action a is performed
                if site1[a] == 0:  # if a is an available site
                    site2[a] = 1  # site a is reserved
                availSite = np.where(site2 == 0)[0]  # where are the potential available sites
                if not np.all(availSite == 0):  # some sites might be candidate for development
                    navail = availSite.shape[0]  # how many?
                    siten = np.ones((2**navail, 1))*site2  # siten is the set of successor state, number of successors = 2 ^ navail sites
                    aux = np.zeros(navail)  # trick to build the set of successor states
                    for k in range(2**navail):  # there are exactly 2^navail successors
                        siten[k, availSite] = aux * 2  # init to aux *.2 because developed site are #2
                        ndev = sum(abs(site2[availSite] - aux))  # how many are developed
                        s2 = getState(siten[k,:]) + 1  # corresponding state
                        P[s1, s2, a] = (pj**ndev) * ((1 - pj)**(navail - ndev))  # corresponding prob of transition
                        aux =
