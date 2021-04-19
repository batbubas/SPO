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

fig, ax = plt.subplots()
iteration = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

swarm = Swarm(particles=[Particle() for i in range(50)], function=rosenbrock)
swarm.b_low = -3
swarm.b_high = 3
swarm.init_the_swarm(rosenbrock)

list_of_positions = np.array([particle.position for particle in swarm.particles])

print(list_of_positions)
for i in range(1):
    swarm.iterate_the_swarm()
    print(swarm.best_position)

particles = ax.scatter(list_of_positions[:, 0], list_of_positions[:, 1])
best = ax.scatter(swarm.best_position[0], swarm.best_position[1])


def init():

    plot_function(ax, two_parameter_rosenbrock, swarm.b_low, swarm.b_high)

    particles.set_offsets(list_of_positions)
    best.set_offsets(swarm.best_position)
    plt.title("PSO")
    plt.xlabel("x")
    plt.ylabel("y")
    iteration.set_text('')
    return particles, best, iteration  # return the variables that will updated in each frame


def plot_function(plot_axes: plt.Axes, fn: Callable, min_val, max_val):
    X = np.arange(min_val,  max_val, 0.15)
    Y = np.arange(min_val, max_val, 0.15)
    X, Y = np.meshgrid(X, Y)
    b = 100
    Z = fn(X, Y)
    plot_axes.contour(X, Y, Z, 200)


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
    while swarm.holdup_epoch <= 300:
        i += 1
        print("HOLDUP NUMBER", swarm.holdup_epoch, "epoch", i)
        yield i

    # stop condition no upgrade after some time i guess
    #generate stop condtion
    # R takiego ze norma roznicy xm (pozycji teraz czastki) i globalnej min -{ r musi byc najwieksza rowne z kazdej czastek }
    # R podzielone przez diameeer (S) swarmu - R dla poczatkowych czastek



ani = animation.FuncAnimation(fig, animate, frames=gen_function,
                              init_func=init, interval=100, blit=True, save_count=sys.maxsize)
plt.show()
print("___BEST STARTING___ : ", swarm.best_position)
print("___STARTING VALUE ___ : ", rosenbrock(swarm.best_position))

now = datetime.now()
ani.save(f'PSO_{now.date()}_{now.time().strftime("%H_%M")}.gif')  # save command
print("___BEST OVERALL___ : ", swarm.best_position)
print("___WITH VALUE ___ : ", rosenbrock(swarm.best_position))

fig2, ax2 = plt.subplots()
plot_function(ax2, two_parameter_rosenbrock, swarm.b_low, swarm.b_high)
particles_final = ax2.scatter(list_of_positions[:, 0], list_of_positions[:, 1])
best_final = ax2.scatter(swarm.best_position[0], swarm.best_position[1])

plt.show()
# # --- IMPORT DEPENDENCIES ------------------------------------------------------+
#
# from random import random
# from random import uniform
#
#
# # --- MAIN ---------------------------------------------------------------------+
#
# def sphere(x):
#     total = 0
#     for i in range(len(x)):
#         total += x[i] ** 2
#     return total
#
#
# if __name__ == "pso.sphere":
#     sphere()
#
#
# class Particle:
#     def __init__(self, x0):
#         self.position_i = []  # particle position
#         self.velocity_i = []  # particle velocity
#         self.pos_best_i = []  # best position individual
#         self.err_best_i = -1  # best error individual
#         self.err_i = -1  # error individual
#
#         for i in range(0, num_dimensions):
#             self.velocity_i.append(uniform(-1, 1))
#             self.position_i.append(x0[i])
#
#     # evaluate current fitness
#     def evaluate(self, costFunc):
#         self.err_i = costFunc(self.position_i)
#
#         # check to see if the current position is an individual best
#         if self.err_i < self.err_best_i or self.err_best_i == -1:
#             self.pos_best_i = self.position_i.copy()
#             self.err_best_i = self.err_i
#
#     # update new particle velocity
#     def update_velocity(self, pos_best_g):
#         w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
#         c1 = 1  # cognative constant
#         c2 = 2  # social constant
#
#         for i in range(0, num_dimensions):
#             r1 = random()
#             r2 = random()
#
#             vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
#             vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
#             self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
#
#     # update the particle position based off new velocity updates
#     def update_position(self, bounds):
#         for i in range(0, num_dimensions):
#             self.position_i[i] = self.position_i[i] + self.velocity_i[i]
#
#             # adjust maximum position if necessary
#             if self.position_i[i] > bounds[i][1]:
#                 self.position_i[i] = bounds[i][1]
#
#             # adjust minimum position if neseccary
#             if self.position_i[i] < bounds[i][0]:
#                 self.position_i[i] = bounds[i][0]
#
#
# def minimize(costFunc, x0, bounds, num_particles, maxiter, verbose=False):
#     global num_dimensions
#
#     num_dimensions = len(x0)
#     err_best_g = -1  # best error for group
#     pos_best_g = []  # best position for group
#
#     # establish the swarm
#     swarm = []
#     for i in range(0, num_particles):
#         swarm.append(Particle(x0))
#
#     # begin optimization loop
#     i = 0
#     while i < maxiter:
#         if verbose: print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')
#
#         # cycle through particles in swarm and evaluate fitness
#         for j in range(0, num_particles):
#             swarm[j].evaluate(costFunc)
#
#             # determine if current particle is the best (globally)
#             if swarm[j].err_i < err_best_g or err_best_g == -1:
#                 pos_best_g = list(swarm[j].position_i)
#                 err_best_g = float(swarm[j].err_i)
#
#         # cycle through swarm and update velocities and position
#         for j in range(0, num_particles):
#             swarm[j].update_velocity(pos_best_g)
#             swarm[j].update_position(bounds)
#         i += 1
#
#     # print final results
#     if verbose:
#         print('\nFINAL SOLUTION:')
#         print(f'   > {pos_best_g}')
#         print(f'   > {err_best_g}\n')
#
#     return err_best_g, pos_best_g
