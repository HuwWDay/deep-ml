import numpy as np

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    """
    ucb_values = np.zeros_like(values, dtype=float)

    for a in range(len(counts)):
        if counts[a] == 0:
            ucb_values[a] = float("inf")  # ensure every action is tried once
        else:
            ucb_values[a] = values[a] + c * np.sqrt(np.log(t) / counts[a])

    return int(np.argmax(ucb_values))
