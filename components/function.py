import itertools
import numpy as np


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


def two_parameter_rosenbrock(x, y, b=100):
    f = lambda x, y: (x - 1) ** 2 + b * (y - x ** 2) ** 2
    return f(x, y)


def f2(x):
    '''INPUTS
#     X: arguments of the f2 Function
#     OUTPUTS
#     f : evaluation of the f2 function given the inputs

#     DOMAIN         : [-100,100]
    DIMENSION       : 20,30
#     GLOBAL MINIMUM : f(x)=
#     '''

    f = sum((x[i] - i) ** 2 for i in range(len(x)))
    return f


def Griewank(x):
    '''
    INPUTS
    x : arguments of the function Ackley
    Output
    f : evaluation of the Ackley function given the inputs

    DOMAIN           : [-600,600]
    DIMENSION       : 20,30
    GLOBAL MINIMUM   : f(x)=0 x=
    '''
    d = len(x)
    a = 1 / 4000
    b = 1
    sum1 = sum(x[i] ** 2 for i in range(d))
    sum2 = itertools.product(np.cos(x[i] / np.sqrt(i)) for i in range(d))
    return a * sum1 - sum2 + b


def Ackley(x):
    '''
    INPUTS
    x : arguments of the function Ackley
    Output
    f : evaluation of the Ackley function given the inputs

    DOMAIN           : [-32,32]
    DIMENSION       : 20,30
    GLOBAL MINIMUM   : f(x)=0 x=[0...0]
    '''
    d = len(x)
    a = 20
    b = 0.2
    c = np.pi * 2
    sum1 = sum(x[i] ** 2 for i in range(d))
    sum1 = (-a) * np.exp(((-b) * np.sqrt(sum1 / d)))
    sum2 = sum(np.cos(c * x[i]) for i in range(d))
    sum2 = np.exp((sum2 / d))
    return sum1 - sum2 + a + np.exp(1)


def Brown(x):
    '''
    INPUTS
    x : arguments of the function Ackley
    Output
    f : evaluation of the Ackley function given the inputs

    DOMAIN           : [-1,4]
    DIMENSION       : 20,30
    GLOBAL MINIMUM   :
    '''
    d = len(x)
    a = (x[i] ** (2 * ((x[i + 1]) ** 2) + 1) for i in range(d))
    b = (x[i + 1] ** (2 * ((x[i]) ** 2) + 1) for i in range(d))
    sum1 = sum(a + b)
    return sum1


def Schaffersf6(x):
    '''
    INPUTS
    x : arguments of the function Ackley
    Output
    f : evaluation of the Ackley function given the inputs

    DOMAIN           : [2]
    DIMENSION       : 2
    GLOBAL MINIMUM   : f(x)=0 x=[0...0]
    '''
    a = 0.5
    para = x * 10
    para = x[0:2]
    num = (np.sin(np.sqrt((para[0] * para[0]) + (para[1] * para[1])))) * (
        np.sin(np.sqrt((para[0] * para[0]) + (para[1] * para[1])))) - a
    denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1]))) ** 2
    sum1 = a + (num / denom)
    return sum1
