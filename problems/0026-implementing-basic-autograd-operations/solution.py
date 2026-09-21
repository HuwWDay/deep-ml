import math

class Value:
    """
    Stores a single scalar value and its gradient. Implements automatic
    differentiation (autograd) for basic arithmetic operations and ReLU.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initializes a Value object.

        Args:
            data (float or int): The scalar value.
            _children (tuple): Children nodes (Value objects) that produced this node.
            _op (str): The operation that produced this node.
            label (str): A label for debugging/visualization.
        """
        self.data = data
        self.grad = 0.0  # Gradient starts at 0
        
        # _backward is a function that computes the local gradient and
        # propagates it to the children. It's set by the operation
        # that creates this Value object.
        self._backward = lambda: None
        
        # _prev stores the set of children nodes
        self._prev = set(_ch