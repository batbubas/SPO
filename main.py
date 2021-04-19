# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 27 11:34:43 2021
#
# @author: Asus
# """
import sys
from datetime import datetime
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import cm

from components.function import rosenbrock, two_parameter_rosenbrock
from components.particle import Particle
from components.swarm import Swarm
from tqdm import tqdm


def main(num_particles, func, border_low, border_high,
         learning_rate, omega, phi_p, phi_g,
         holdup_limit):
    print("STARTING CALCULATIONS")
    swarm = Swarm(particles=[Particle() for i in range(num_particles)], function=func)
    swarm.b_low = border_low
    swarm.b_high = border_high
    swarm.lr = learning_rate
    swarm.omega = omega
    swarm.phi_p = phi_p
    swarm.phi_g = phi_g
    swarm.init_the_swarm(func)
    print("STARTING BEST", swarm.best_position, "WITH VALUE", rosenbrock(swarm.best_position))

    pbar = tqdm(total=holdup_limit)
    with tqdm(total=holdup_limit) as pbar:
        while swarm.holdup_epoch <= holdup_limit:
            swarm.epoch += 1
            swarm.iterate_the_swarm()
            # print("HOLDUP : ", swarm.holdup_epoch)
            pbar.update(swarm.holdup_epoch - pbar.n)
        else:
            pbar.close()
            print("FINAL BEST: ", swarm.best_position,
                  "WITH VALUE: ", rosenbrock(swarm.best_position),
                  "EPOCH REACHED: ", swarm.epoch)


if __name__ == "__main__":
    params = {
        "num_particles": 20,
        "func": rosenbrock,
        "border_low": -3,
        "border_high": 3,
        "learning_rate": 0.3,
        "omega": 1.27,
        "phi_p": 1.33,
        "phi_g": 1.45,
        "holdup_limit": 10000
    }
    main(**params)

