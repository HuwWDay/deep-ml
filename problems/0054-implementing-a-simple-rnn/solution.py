import numpy as np
import math

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
    # new_hidden_state = tanh (Wx * input_sequence + Wh * initial_hidden_state)
    input_sequence = np.array(input_sequence)
    initial_hidden_state = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    
    for x in input_sequence:
        up1 = Wx.dot(x)
        up2 = Wh.dot(initial_hidden_state)
        up3 = up1 + up2 + b
        initial_hidden_state = [math.tanh(up) for up in up3]
    final_hidden_state = initial_hidden_state
	return final_hidden_state