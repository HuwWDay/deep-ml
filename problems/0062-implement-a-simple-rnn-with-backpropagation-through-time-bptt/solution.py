import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def _prepare_sequence(self, seq):
        seq = np.array(seq)

        if isinstance(seq, list):
            return seq

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        return [row.reshape(-1, 1) for row in seq]

    def forward(self, x):
        """
        Returns ONLY the output predictions stacked as:
        [[y1],
         [y2],
         [y3],
         [y4]]
        """
        x = self._prepare_sequence(x)

        T = len(x)
        h = np.zeros((self.hidden_size, 1))
        ys = []

 