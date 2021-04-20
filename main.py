# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 27 11:34:43 2021
#
# @author: Asus
# """

from components.function import rosenbrock, Ackley, ackley, schaffer, brown, griewank
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
    #tutaj trzeba by zrobic nowe params dla kazdej funkcji bo zmieniaja sie borders etc
    # wywolac to dla tych funkcji i mamy wyniki
    # ale wczesniej trzeba by wybrac jedna funkcje np rosenbrocka polceam
    # i dla rosenbrocka pozmieniac omega phi_p phi_g etc
    # holdup limit to warunek zatrzymania jak przez np 100 iteracji nie zmieni sie best to zatrzymujemy program
    # mozna tez zamienic w linicje
    params = {
        "num_particles": 20,
        "func": ackley,
        "border_low": -32,
        "border_high": 32,
        "learning_rate": 0.3,
        "omega": 1.67,
        "phi_p": 1.43,
        "phi_g": 1.45,
        "holdup_limit": 10000
    }
    main(**params)
