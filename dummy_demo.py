################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import sqrt
import random
import matplotlib.pyplot as plt



experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def init_population(iIndividuals, iN_vars, iL_bound, iU_bound):
    population = np.random.uniform(iL_bound, iU_bound, (iIndividuals, iN_vars))
    return population

def init_simulation(iNum_of_neurons):
    global env
    env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(iNum_of_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
    return env

def parent_selection(population):
    
    # Select 5 random parents ids
    num_parents = 5
    random_parents_id = np.random.choice(population.shape[0], num_parents, replace=False)

    # Select the 5 according parents
    random_parents = population[random_parents_id]

    # Calculate fitness scores for each parent
    fitness_scores = np.array([fit_pop(row) for row in random_parents])

    # Find the index of the parent with the highest fitness score
    best_parent_id = np.argmax(fitness_scores)

    # Select the row with the maximum value in the first column
    return random_parents[best_parent_id]


def print_generational_gain(history):
    ''' 
    Purpose: shows a line diagram of the average fitness gain over generations 

    Input: history = matrix generation with average fitness
    
    Print statement: Linediagram 
    '''
    x = history[0] #generation number
    y = history[1] #average fitness

    plt.plot(x,y, "line")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average fitness per generation")

    


def print_generational_gain(history):
    ''' 
    Purpose: shows a line diagram of the average fitness gain over generations 

    Input: history = matrix generation with average fitness
    
    Print statement: Linediagram 
    '''
    x = history[0] #generation number
    y = history[1] #average fitness

    plt.plot(x,y, "line")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average fitness per generation")

    



def main():
    #magic numbers
    iNum_of_individuals = 100
    iNum_of_neurons = 10
    iL_bound = -1
    iU_bound = 1
    iN_generations = 10
    dStop_time = 3000
    mHistory = list()


    
    env = init_simulation(iNum_of_neurons)
    iNum_of_vars = (env.get_num_sensors()+1)*iNum_of_neurons + (iNum_of_neurons+1)*5 #is this general?
    population = init_population(iNum_of_individuals, iNum_of_vars, iL_bound, iU_bound)

    
    for i in range(iN_generations):
        # Start timer (time in seconds)
        timer = time.time() 

        #evaluate current population
        fitness = evaluate(population)
        dAverage_fitness = sum(fitness)/len(fitness)
        mHistory.append([i, dAverage_fitness])


        # parents = parent_selection(fitness)
        # parents = selection #[[]]
        # Variation
        # population = survivor_selection

       

        if timer >= 100:
            # log(results)
            print("Time is up")
            quit()


    print_generational_gain(mHistory)
if __name__ == "__main__":
    main()