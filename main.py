# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 27 11:34:43 2021
#
# @author: Asus
# """

from components.function import rosenbrock, Ackley, ackley, schaffer, brown, griewank, f2
from components.particle import Particle
from components.swarm import Swarm
from tqdm import tqdm
import numpy as np
import csv


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
        all_calculations.append((swarm.lr, swarm.omega, swarm.phi_p, swarm.phi_g,
                                 swarm.best_position, rosenbrock(swarm.best_position), swarm.epoch))


if __name__ == "__main__":
    all_calculations = []
    params = {
                        "num_particles": 20,
                        "func": ackley,
                        "border_low": -32,
                        "border_high": 32,
                        "learning_rate": 0.3,
                        "omega": 0.4,
                        "phi_p": 3.0,
                        "phi_g":  1.888,
                        "holdup_limit": 5
                    }
    main(**params)
    # tutaj trzeba by zrobic nowe params dla kazdej funkcji bo zmieniaja sie borders etc
    # wywolac to dla tych funkcji i mamy wyniki
    # ale wczesniej trzeba by wybrac jedna funkcje np rosenbrocka polceam
    # i dla rosenbrocka pozmieniac omega phi_p phi_g etc
    # holdup limit to warunek zatrzymania jak przez np 100 iteracji nie zmieni sie best to zatrzymujemy program
    # mozna tez zamienic w linijce 29 tego pliku zamiast while <= holdup limit na jakas liczbe ustawiona iteracji
    # animacja dziala tylko z rosenbrock chyba ze dorobi się funkcje takie jak two_parameter_rosenbrock dla reszty
    # inaczej ciezko narysowac sama funkcje w animated solution
    # dołożyłem basic GA dziala ok

    # TODO: WLASNIE SOBIE USWIADOMILME ZE WYMIAR PRZESTRZENI TO ILOSC X np x1 , x2 do n omg XD
    # to blyby szybki fix ale trzeba by zmienic zdefiniowane funkcje XD no nic

    # all_calculations = []
    # for i in [0.3, 0.4, 0.7, 0.9]:
    #     for j in np.linspace(0.4, 0.91, 10):
    #         for c1 in np.linspace(1, 3, 10):
    #             for c2 in np.linspace(1, 3, 10):
    #                 params = {
    #                     "num_particles": 20,
    #                     "func": schaffer,
    #                     "border_low": -100,
    #                     "border_high": 100,
    #                     "learning_rate": i,
    #                     "omega": j,
    #                     "phi_p": c1,
    #                     "phi_g": c2,
    #                     "holdup_limit": 5000
    #                 }
    #                 main(**params)
    #
    # with open('optimization_schaffer_precise.csv', 'w') as out:
    #     csv_out = csv.writer(out)
    #     csv_out.writerow(['learning_rate', 'omega', "phi_p", "phi_g", "best", "best_value", "epoch"])
    #     for row in all_calculations:
    #         csv_out.writerow(row)
    #
    # print("schaffer")
