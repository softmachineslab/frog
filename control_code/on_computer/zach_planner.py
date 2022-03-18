

import numpy as np

# This function will take the current state of the robot and, comparing it to stored data captured from DER, create an action plan with the ultimate goal of following a path. The output is the next action.
def planner(initial_state, memory, path):
    # Assuming state is 1D vector: [x, y, theta, V_x, V_y, V_theta]
    # Assuming memory is (for now) 2D numpy array of 3,375 data points captured from DER with form:
    # [action V_x, V_y, V_theta, del_x, del_y, del_theta, del_v_x, del_v_y, del_v_theta]

    velocity_mem = memory[:,1:4]

    transitions = memory[4:]

    weighted_transitions, euclidean_dist_1 = compute_weighted_transitions(initial_state, velocity_mem, transitions)

    1st_layer_states = initial_state + weighted_transitions


    min_cost = 10^20

    for idx,state in enumerate(1st_layer_states):

        weighted_transitions, euclidean_dist_2 = compute_weighted_transitions(state, velocity_mem, transitions)

        2nd_layer_states = initial_state + weighted_transitions

        # not sure if we'll need a for loop here or not:

        costs = compute_cost(2nd_layer_states, path, euclidean_dist_1[idx], euclidean_dist_2)

        this_min_cost = np.min(costs)

        if this_min_cost < min_cost:
            min_cost = this_min_cost
            min_idx = idx

    action = memory[min_idx,0]
    return action




def compute_weighted_transitions(state, velocity_mem, transitions):
    euclidean_dist = np.linalg.norm(velocity_mem - state[3:], axis = 0)

    weights = 1/euclidean_dist/np.linalg.norm(1/euclidean_dist)

    weighted_transitions = np.multiply(transitions, weights)

    return weighted_transitions, euclidean_dist
