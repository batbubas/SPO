from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass()
class Particle:
    velocity: np.array = np.array([0, 0])
    position: np.array = np.array([0, 0])
    best_position: np.array = np.array([0, 0])

    def init_position(self, start, stop):
        self.position = np.random.uniform(low=start, high=stop, size=2)

    def init_velocity(self, start, stop):
        self.velocity = np.random.uniform(low=-np.abs(stop - start), high=np.abs(stop - start), size=2)

    def update_position(self, lr):
        self.position = self.position * lr + self.velocity

    def update_velocity(self, omega, phi_p, phi_g, swarm_best):
        # I CANT DECIDE IF THIS SHOULD BE PICKED RANDOM FOR ALL DIM AT ONCE OR SEPARATELY
        rp, rg = self.get_random_coeffs()
        # vx update
        m = self.velocity
        omega_v = (omega * self.velocity[0])
        phi_p_v = (phi_p * rp * (self.best_position[0] - self.position[0]))
        phi_g_v = (phi_g * rg * (swarm_best[0] - self.position[0]))
        self.velocity[0] = omega_v + phi_p_v + phi_g_v
        # vy update
        rp, rg = self.get_random_coeffs()

        omega_v = (omega * self.velocity[1])
        phi_p_v = (phi_p * rp * (self.best_position[1] - self.position[1]))
        phi_g_v = (phi_g * rg * (swarm_best[1] - self.position[1]))
        self.velocity[1] = omega_v + phi_p_v + phi_g_v

    @staticmethod
    def get_random_coeffs():
        random_coeffs = np.random.uniform(size=2)
        rp, rg = random_coeffs[0], random_coeffs[1]
        return rp, rg

    def init_best_position(self):
        self.best_position = self.position

    def update_best_position(self, function):
        if function(self.position) < function(self.best_position):
            self.best_position = self.position
            return self.best_position
        else:
            return None

    def should_update_global(self, function, global_best):
        if function(self.best_position) < function(global_best):
            return self.best_position
        else:
            return None


if __name__ == "__main__":
    p = Particle()
    p.init_velocity(-10, 10)
    p.init_position(-10, 10)
    p.init_best_position()
    print(p)
    p.update_position()
    print(p)
