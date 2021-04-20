import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from components.function import rosenbrock

varbound = np.array([[-3, 3]] * 2)

model = ga(function=rosenbrock, dimension=2, variable_type='real', variable_boundaries=varbound)

model.run()
