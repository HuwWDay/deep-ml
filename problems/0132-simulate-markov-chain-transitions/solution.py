import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    number_of_states = np.shape(transition_matrix)[0]
    new_state = initial_state
    states = np.array(new_state)
    for step in range(num_steps):
        probs = transition_matrix[new_state]
        new_state = np.random.choice(number_of_states, p=probs)
        states = np.append(states, new_state)
    return states