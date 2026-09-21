def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    data_x = [(a, b) for (a, b) in data if a == x]
    # Filter X = x
    total = len(data_x)
    if total == 0:
        return 0.0
    count_y = len([(a, b) for (a, b) in data_x if b == y])
    # Countif Y=y, divide by all ys

    return round(count_y/total, 4)