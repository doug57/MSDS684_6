from math import sqrt
import numpy as np
from CartPole.CPstates import observation_to_grid_state

# plays one episode
# play (learn) the game once, updating Q-values on the trajectory
def learn_Q(
    env, grid, gamma, epsilon, alpha, Q, visited_states, terminal_states):

    # initialize/reset cartpole game to initial state
    obs,_ = env.reset()
    state = observation_to_grid_state(obs, grid)  # obs -> grid
    # add state, or increment count, to dictionary of visited states
    # visited_states maps state tuple to number of visits
    if state not in visited_states:
        visited_states[state] = 1
    else:
        visited_states[state] = 1 + visited_states[state]

    game_length = 0
    inf_norm = 0
    l2_norm = 0

    # Q-learning loop
    converged = False
    while not converged:
        game_length += 1  # increment game length counter
        # do epsilon greedy decision of action at this state
        action = best_action(state, Q)  # determine best action
        if random_action(epsilon) == 0:  # epsilon greedy exploration
            action = random_action(0.5)  # sometime choose action random
        # take a step
        obs, reward, done, _, _ = env.step(action)  # take a step using action
        state_prime = observation_to_grid_state(obs, grid)  # obs -> grid

        # add state, or increment count, to dictionary of visited states
        if state_prime not in visited_states:  # dictionary of visited states
            visited_states[state_prime] = 1
        else:
            visited_states[state_prime] += 1

        if not done:
            # determine value of destination state
            action_prime = best_action(state_prime, Q)
            value_prime = Q[state_prime][action_prime]

            # perform Q update
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * value_prime - Q[state][action]
            )

            # find the largest change in the Q values, for this episode,
            # the infinity norm
            abs_change = abs((reward + gamma * value_prime - Q[state][action]))
            if abs_change >= inf_norm:
                inf_norm = abs_change
            # compute the l2 norm
            l2_norm += abs_change * abs_change

        if done:
            # Q update if state_prime is a terminal state
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
            # record terminal states
            if state not in terminal_states:
                terminal_states[state] = 1
            else:
                terminal_states[state] = 1 + terminal_states[state]
            converged = True
        # get ready for next move
        state = state_prime

    l2_norm = (1 / 2) * sqrt(l2_norm) / game_length

    return game_length, inf_norm, l2_norm

# play game once with policy pi
def play_game(env, grid, pi):
    trajectory = []
    obs, _ = env.reset()
    state = observation_to_grid_state(obs, grid)
    trajectory.append(state)
    sum_reward = 0
    converged = False
    while not converged:
        action = pi[state]
        obs, reward, done, maxstep, _ = env.step(action)

        if maxstep > env._max_episode_steps:
            converged = True
        if done:
            converged = True
        if not done:
            sum_reward += reward
            state = observation_to_grid_state(obs, grid)
            trajectory.append(state)

    return sum_reward, trajectory

# play game once with policy pi
def render_play_game(env, grid, pi):
    trajectory = []
    obs, _ = env.reset()
    state = observation_to_grid_state(obs, grid)
    trajectory.append(state)
    sum_reward = 0
    converged = False
    while not converged:
        action = pi[state]
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            converged = True
        if not done:
            sum_reward += reward
            state = observation_to_grid_state(obs, grid)
            trajectory.append(state)

    return sum_reward, trajectory

# Initialize Q table, a Python dictionary that maps states
def init_Q(grid):
    Q = {}
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i, j, k, m)
                    Q[state] = {}
                    actions = [0, 1]
                    for a in actions:
                        Q[state][a] = 0
    return Q

def randomize_Q(grid, Q, low, high):
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i, j, k, m)
                    actions = [0, 1]
                    for a in actions:
                        Q[state][a] = random_in_interval(low, high)

def init_V(grid):
    V = {}
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i, j, k, m)
                    V[state] = 0.0
    return V

# returns action with largest value at a state
def best_action(state, Q):
    if Q[state][0] > Q[state][1]:
        return 0
    else:
        return 1

# returns a random action based on chosen critical value
def random_action(critical_value):
    p = np.random.random_sample()  # on [0,1)
    if p < critical_value:
        return 0
    else:
        return 1

# returns random number in an interval
def random_in_interval(low, high):
    return np.random.random_sample() * (high - low) + low

# this Python dictionary that maps state/action pairs to destination
# states and counts how many times that transition occurs
# this can be used to see if state transitions are stochastic or
# determine whether the obs -> grid is sufficiently dense
def init_state_transition_count(grid):
    state_transitions = {}
    for i in range(grid.bins[0]):
        for j in range(grid.bins[1]):
            for k in range(grid.bins[2]):
                for m in range(grid.bins[3]):
                    state = (i, j, k, m)
                    state_transitions[state] = {}
                    actions = [0, 1]
                    for a in actions:
                        state_transitions[state][a] = {}
    return state_transitions
