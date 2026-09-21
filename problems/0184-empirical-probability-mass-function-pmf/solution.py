def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    support = list(set(samples))
    size = len(samples)
    if size == 0:
        return []
    dis = []
    for value in support:
        dis.append((value, samples.count(value) / size))
    return dis



