import itertools
import numpy as np
from numba import njit


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


# NIE DZIALA W LINICJE RETURN
# def Griewank(x):
#     '''
#     INPUTS
#     x : arguments of the function Ackley
#     Output
#     f : evaluation of the Ackley function given the inputs
#
#     DOMAIN           : [-600,600]
#     DIMENSION       : 20,30
#     GLOBAL MINIMUM   : f(x)=0 x=
#     '''
#     d = len(x)
#     a = 1 / 4000
#     b = 1
#     sum1 = sum(x[i] ** 2 for i in range(d))
#     sum2 = itertools.product(np.cos(x[i] / np.sqrt(i)) for i in range(d))
#     return a * sum1 - sum2 + b

def griewank(xs):
    sum = 0
    for x in xs:
        sum += x * x
    product = 1
    for i in range(len(xs)):
        product *= np.cos(xs[i] / np.sqrt(i + 1))
    return 1 + sum / 4000 - product



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


def ackley(X):
    # returns the point value of the given coordinate
    part_1 = -0.2 * np.sqrt(0.5 * (X[0] * X[0] + X[1] * X[1]))
    part_2 = 0.5 * (np.cos(2 * np.pi * X[0]) + np.cos(2 * np.pi * X[1]))
    value = np.exp(1) + 20 - 20 * np.exp(part_1) - np.exp(part_2)
    # returning the value
    return value


# NIE DZIALA NA SUMIE
# def Brown(x):
#     '''
#     INPUTS
#     x : arguments of the function Ackley
#     Output
#     f : evaluation of the Ackley function given the inputs
#
#     DOMAIN           : [-1,4]
#     DIMENSION       : 20,30
#     GLOBAL MINIMUM   :
#     '''
#     d = len(x)
#     a = (x[i] ** (2 * ((x[i + 1]) ** 2) + 1) for i in range(d))
#     b = (x[i + 1] ** (2 * ((x[i]) ** 2) + 1) for i in range(d))
#     sum1 = sum(a + b)
#     return sum1
def brown(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 ** 2) ** (x2 ** 2 + 1) + (x2 ** 2) ** (x1 ** 2 + 1)


# CHYBA BRAK KWADRATU W NOMINATOR ZA SIN
# def Schaffersf6(x):
#     '''
#     INPUTS
#     x : arguments of the function Ackley
#     Output
#     f : evaluation of the Ackley function given the inputs
#
#     DOMAIN           : [2]
#     DIMENSION       : 2
#     GLOBAL MINIMUM   : f(x)=0 x=[0...0]
#     '''
#     a = 0.5
#     para = x * 10
#     para = x[0:2]
#     num = (np.sin(np.sqrt((para[0] * para[0]) + (para[1] * para[1])))) * (
#         np.sin(np.sqrt((para[0] * para[0]) + (para[1] * para[1])))) - a
#     denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1]))) ** 2
#     sum1 = a + (num / denom)
#     return sum1

def schaffer(x):
    x1 = x[0]
    x2 = x[1]
    nom = np.sin(np.sqrt(x1 ** 2 + x2 ** 2)) ** 2 - 0.5
    denom = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return 0.5 + (nom / denom)
