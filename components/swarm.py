from dataclasses import dataclass
import numpy as np
from components.particle import Particle
from typing import List, Optional, Callable


@dataclass
class Swarm:
    function: Optional[Callable]
    b_low = -3
    b_high = 3
    particles: List[Particle]
    best_position: Optional[np.array] = None
    # parameters
    omega: float = 1.27
    phi_p: float = 1.33
    phi_g: float = 1.45
    # learning rate
    lr: float = 1.

    def init_swarm_best(self):
        x = self.function
        # print(self.function(self.particles[0].best_position))
        initial_best_with_positions = ([(self.function(particle.best_position), particle.best_position) for particle in self.particles])
        best_candidate = min(initial_best_with_positions)
        self.best_position = best_candidate[1]

    def init_the_swarm(self, func):
        self.function = func
        for particle in self.particles:
            particle.init_position(self.b_low, self.b_high)
            particle.init_best_position()
            self.init_swarm_best()

            particle.init_velocity(self.b_low, self.b_high)

    def iterate_the_swarm(self):
        for particle in self.particles:
            particle.update_velocity(omega=self.omega,phi_p=self.phi_p,phi_g=self.phi_g,swarm_best=self.best_position)
            particle.update_position(self.lr, self.b_low, self.b_high)
            particle.update_best_position(self.function)

            if (updated_best := particle.should_update_global(self.function, self.best_position) )is not None:
                self.best_position = updated_best


# if __name__ == "__main__":
    # swarm = Swarm(particles=[Particle() for i in range(10)], function=rosenbrock)
    # # print(swarm)
    # swarm.init_the_swarm(rosenbrock)
    # # print(swarm)
    # list_of_positions = np.array([particle.position for particle in swarm.particles])
    #
    #
    #
    # print(list_of_positions)
    # for i in range(1):
    #     swarm.iterate_the_swarm()
    #     print(swarm.best_position)
