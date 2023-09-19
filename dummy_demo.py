################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
env.play()

# ----------------------------------------

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    #magic numbers
    number_of
    
    for generation in generations:
        fitness= evaluate(population)
        parents = selection #[[]]
        Variation
        population = survivor_selection
        if stop_cond
            break

if __name__ == "__main__":
    main()