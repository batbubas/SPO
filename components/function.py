
def rosenbrock(X):
    """INPUTS
    X: arguments of the function Rosenbrock
    OUTPUTS
    f : evaluation of the Rosenbrock function given the inputs

    DOMAIN         : Xi is within [-5,10] although can be [-2.048,2.048]
    DIMENSIONS     : any
    GLOBAL MINIMUM : f(x)=0 x=[1,...,1]
    """
    f = sum(100.0 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2 for i in range(0, len(X) - 1))
    return f


def two_parameter_rosenbrock(x,y, b=100):
    f = lambda x, y: (x - 1) ** 2 + b * (y - x ** 2) ** 2
    return f(x,y)