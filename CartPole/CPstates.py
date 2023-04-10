import numpy as np
import math

# define a grid class to hold the edges of the observation space and the number of bins in each dimension
class Grid:
    def __init__(self,n_bins,a,b,c,d):   
        self.obs_interval = []
        self.obs_interval.append((-a, a))
        self.obs_interval.append((-b, b))
        self.obs_interval.append((-c, c))
        self.obs_interval.append((-d, d))
        # same number of bins in each dimention, could be changed
        self.bins = (n_bins,n_bins,n_bins,n_bins)

# takes an observation and transforms it to grid coordinates 
def observation_to_grid_state(obs,grid):
    """_
    Args:
        obs (four element array): observation from CartPole problem  
        grid (Grid): Grid object that holds definitions for grided state space 
    Returns:
        state: a tuple with four elements that can be used as a dictionary key for Q-values
    """
    state = []
    for i in range(4): # for each of the four observation values
        bin_min = grid.obs_interval[i][0]
        bin_max = grid.obs_interval[i][1]
        nbins = grid.bins[i]
        if obs[i] <= bin_min: # goes to -infty
            state.append(0)
        elif obs[i] >= bin_max: # goes to +infty
            state.append(nbins-1)
        else: # somewhere in between bin_min and bin_max
            interval_width = bin_max-bin_min
            interval_incr = interval_width/(nbins-2)
            bin = math.floor((obs[i]-bin_min)/interval_incr)
            state.append(bin+1)
    # return state as a tuple so that it can be used as a dictionary key
    return (state[0],state[1],state[2],state[3]) 
