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

def parent_selection(pop_fitness, iNum_of_parents= 6):
    
    # Select n random parents ids

    random_parents_id = np.random.choice(pop_fitness.shape[0], iNum_of_parents, replace=False)

    # Select the n according individuals
    random_parents = np.array(pop_fitness[random_parents_id])
    random_parents = np.column_stack((random_parents_id, random_parents))

    # Find the index of the parent with the highest fitness score
    best_parent_id = np.argmax(random_parents[:,1])
    parent_one = random_parents[best_parent_id,0]

    # Create a new matrix without the best parent
    random_parents = np.delete(random_parents, best_parent_id, axis=0)

    # select the second parent
    second_best = np.argmax(random_parents[:,1])
    second_parent = random_parents[second_best,0]

    # Select the row with the maximum value in the first column
    return int(parent_one), int(second_parent)




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


def crossover(pop, fixed_start=True, fixed_end=True, n_vars = None, n_offspring=2, p_left=0.5, p_mutation = 0.2):
    
    total_offspring = np.zeros((0,n_vars))
    
    for p in range(0, pop.shape[0], 2):  # stepsize 2, since you choose 2 parents and otherwise you get 2 times the number of offspring
        parents = np.zeros((2, pop.shape[1]))
        parents[0],parents[1] = parent_selection(pop)
        
        offspring = np.zeros((n_offspring, n_vars))
        
        index_list = np.arange(0, len(parents[0])).tolist() #create a list of index to sample from
    
        if fixed_start:
            index_list.remove(0) #Remove the first index 0 from the list
    
        if fixed_end:
            index_list.remove(len(parents[0])-1) #Remove the last index from the list
        
        #Sample one integers from the index list
        a = random.choice(index_list)
             
        for c in range(len(offspring)):
            if np.random.uniform() <= p_left:  # recombine left part
                offspring[c, a:] = parents[c, a:]
                offspring[c, :a] = (parents[0, :a] + parents[1, :a]) / 2

            else:  # recombine right part
                offspring[c, :a] = parents[c, :a]
                offspring[c, a:] = (parents[c, a:] + parents[1, a:]) / 2
                
            # mutation 
            for i in range(len(offspring[c])):
                if np.random.uniform(0,1) <= p_mutation:
                    offspring[c, i] = np.random.uniform(-1, 1)
            
            total_offspring = np.vstack((total_offspring, offspring[c]))
     
    return total_offspring
        

def print_generational_gain(history):
    ''' 
    Purpose: shows a line diagram of the average fitness gain over generations 

    Input: history = matrix generation with average fitness
    
    Print statement: Linediagram 
    '''
    
    for row in history:
        x = [el[0] for el in history]#generation number
        y = [el[1] for el in history] #average fitness
        plt.plot(x,y)

    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average fitness per generation")
    plt.show()



def main():
    #magic numbers
    iNum_of_individuals = 100
    iNum_of_neurons = 10
    iL_bound = -1
    iU_bound = 1
    iN_generations = 3
    dStop_time = 3000
    mHistory = [] #becomes a list of lists
    np.random.seed(1234)


    
    env = init_simulation(iNum_of_neurons)
    iNum_of_vars = (env.get_num_sensors()+1)*iNum_of_neurons + (iNum_of_neurons+1)*5 #is this general?
    population = init_population(iNum_of_individuals, iNum_of_vars, iL_bound, iU_bound)

    timer = time.time() 
    for i in range(iN_generations):
        # Start timer (time in seconds)
        
        # print(i)
        #evaluate current population
        fitness = evaluate(population)
        dAverage_fitness = sum(fitness)/len(fitness)
        mHistory.append([i, dAverage_fitness])
        population = crossover(population,n_vars=iNum_of_vars)
        # Variation
        # population = survivor_selection

       

        # if timer >= 1000:
        #     # log(results)
        #     print("Time is up")
        #     break


    print_generational_gain(mHistory)
    quit()
if __name__ == "__main__":
    main()