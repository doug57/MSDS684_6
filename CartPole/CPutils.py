import numpy as np
from CartPole.CPlearn import best_action

# returns a list of states that are around a given state
def state_neighborhood(state,size,Q):
    neighborhood = []
    for state_prime in Q:
        distance = np.sum(np.absolute(np.subtract(state_prime,state)))
        if distance <= size:
            neighborhood.append(state_prime)
    return neighborhood

# given list of terminal states, returns unique terminal states
def unique_states(terminal_states):
    terminal_states_set = set(terminal_states)
    states = list(terminal_states_set)
    return states

# returns a policy, a function of each state that returns an action
def Q_to_policy(grid,Q):
    pi = {}
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i,j,k,m)
                    pi[state] = best_action(state,Q)
    return pi

#returns the value, a function of each state that returns its value
def Q_to_values(grid,Q):
    values = {}
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i,j,k,m)
                    action = best_action(state,Q)
                    values[state] = Q[state][action]
    return values
