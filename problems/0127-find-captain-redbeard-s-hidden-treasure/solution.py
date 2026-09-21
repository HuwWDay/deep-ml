def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 − 3x^3 + 2 is minimized.
    Uses analytic critical-point evaluation to guarantee global minimum.
    """
    import math

    # Define the function
    def f(x):
        return x**4 - 3*x**3 + 2

    # Critical points: solutions to f'(x) = x^2 (4x − 9) = 0
    critical_points = [0, 9/4]

    # Pick the point that yields the smallest f(x)
    return min(critical_points, key=f)
