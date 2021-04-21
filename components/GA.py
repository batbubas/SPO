import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from components.function import rosenbrock, brown

varbound = np.array([[-1, 4]] * 2)

model = ga(function=brown, dimension=2, variable_type='real', variable_boundaries=varbound)

model.run()
