import sys
import importlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from components.anim_iter import create_figure, create_particle_best_plot, start_swarm, plot_function
from components.function import two_parameter_rosenbrock

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
print(config.sections())

if __name__ == "__main__":

    num_particles = int(config["Swarm"]["particles"])
    functions = importlib.import_module("components.function")

    func = getattr(functions, f'{config["Swarm"]["function"]}')
    bl = float(config["Swarm"]["lower_bound"])
    bh = float(config["Swarm"]["upper_bound"])

    holdup_limit = int(config["Params"]["holdup_limit"])

    swarm = start_swarm(num_particles, func, bl, bh)
    swarm.lr = float(config["Params"]["learning_rate"])
    swarm.phi_p = float(config["Params"]["phi_p"])
    swarm.omega = float(config["Params"]["omega"])
    swarm.phi_g = float(config["Params"]["phi_g"])

    print(swarm.phi_g, swarm.phi_p, swarm.omega)

    fig, ax, iteration = create_figure()
    particles, best, list_of_positions = create_particle_best_plot(ax,swarm)

    def animate(i):  # 'i' is the number of frames
        # update the data
        swarm.iterate_the_swarm()
        particles.set_offsets(np.array([particle.position for particle in swarm.particles]))
        best.set_offsets(swarm.best_position)
        iteration.set_text(' frame number = %.1d' % i)
        return particles, best, iteration


    def gen_function() -> object:
        global swarm
        i = 0
        while swarm.holdup_epoch <= holdup_limit:
            i += 1
            print("HOLDUP NUMBER", swarm.holdup_epoch, "epoch", i)
            yield i


    def init():

        plot_function(ax, func, swarm.b_low, swarm.b_high)

        particles.set_offsets(list_of_positions)
        best.set_offsets(swarm.best_position)
        plt.title("PSO")
        plt.xlabel("x")
        plt.ylabel("y")
        iteration.set_text('')
        return particles, best, iteration  # return the variables that will updated in each frame


    ani = animation.FuncAnimation(fig, animate, frames=gen_function,
                                  init_func=init, interval=100, blit=True, save_count=sys.maxsize)
    plt.show()
    print("___BEST STARTING___ : ", swarm.best_position)
    print("___STARTING VALUE ___ : ", func(swarm.best_position))

    now = datetime.now()
    ani.save(f'PSO_{now.date()}_{now.time().strftime("%H_%M")}.gif')  # save command
    print("___BEST OVERALL___ : ", swarm.best_position)
    print("___WITH VALUE ___ : ", func(swarm.best_position))

    fig2, ax2 = plt.subplots()
    plot_function(ax2, two_parameter_rosenbrock, swarm.b_low, swarm.b_high)
    particles_final = ax2.scatter(list_of_positions[:, 0], list_of_positions[:, 1])
    best_final = ax2.scatter(swarm.best_position[0], swarm.best_position[1])

    plt.show()