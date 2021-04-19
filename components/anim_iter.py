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


def create_figure():
    fig, ax = plt.subplots()
    iteration = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    return fig, ax, iteration


def create_particle_best_plot(axes: plt.Axes, swarm: Swarm):
    list_of_positions = np.array([particle.position for particle in swarm.particles])

    particles = axes.scatter(list_of_positions[:, 0], list_of_positions[:, 1])
    best = axes.scatter(swarm.best_position[0], swarm.best_position[1])
    return particles, best, list_of_positions


def plot_function(plot_axes: plt.Axes, fn: Callable, min_val, max_val):
    X = np.arange(min_val, max_val, 0.15)
    Y = np.arange(min_val, max_val, 0.15)
    X, Y = np.meshgrid(X, Y)
    b = 100
    Z = fn(X, Y)
    plot_axes.contour(X, Y, Z, 200)


def start_swarm(num_particles, func: Callable, border_low, border_high):
    swarm = Swarm(particles=[Particle() for i in range(num_particles)], function=func)
    swarm.b_low = border_low
    swarm.b_high = border_high
    swarm.init_the_swarm(func)
    return swarm
