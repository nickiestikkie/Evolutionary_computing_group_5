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



experiment_name = 'freek_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, e


# evaluation
def evaluate(x):
    fitness = np.array(list(map(lambda y: simulation(env,y)[0], x)))
    enemy_life = np.array(list(map(lambda y: simulation(env,y)[1], x)))
    return fitness, enemy_life


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

def parent_selection(pop, iNum_of_parents=6):
    
    random_parents = pop[np.random.choice(pop.shape[0], iNum_of_parents, replace=False)]
    random_parents_fitness = evaluate(random_parents)[0]
    best_parent = np.where(random_parents_fitness == np.max(random_parents_fitness))

    return random_parents[best_parent[0][0]]


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


def crossover(pop, fixed_start=True, fixed_end=True, n_offspring=2, p_left=0.5, p_mutation = 0.2):
    n_vars = pop[0].shape[0]
    total_offspring = np.zeros((0,n_vars))
    
    for p in range(0, pop.shape[0], 2):  # stepsize 2, since you choose 2 parents and otherwise you get 2 times the number of offspring
        parents = np.zeros((2, pop.shape[1]))
        parents[0] = parent_selection(pop)
        parents[1] = parent_selection(pop)
        
        offspring = np.zeros((n_offspring, n_vars))
        
        index_list = np.arange(0, len(parents[0])).tolist() #create a list of index to sample from
    
        if fixed_start:
            index_list.remove(0) #Remove the first index 0 from the list
    
        if fixed_end:
            index_list.remove(len(parents[0])-1) #Remove the last index from the list
        
        #Sample one integers from the index list
        a = np.random.choice(index_list)
             
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
    individuals = 100
    hidden_neurons = 10
    lower_bound = -1
    upper_bound = 1
    generations = 15
    stop_time = 3000
    mHistory = [] # becomes a list of lists
    np.random.seed(1234)

    env = init_simulation(hidden_neurons)
    number_of_weights = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
    population = init_population(individuals, number_of_weights, lower_bound, upper_bound)

    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen enemy_life best mean std')

    timer = time.time() # Start timer (time in seconds)
    for i in range(generations):
        # create new generation
        population = crossover(population)   
         
        # evaluate current population
        fitness = evaluate(population)[0]
        enemy_life = np.min(evaluate(population)[1])
                
        mean_fitness = np.mean(fitness)
        best_fitness = np.argmax(fitness)
        std_fitness = np.std(fitness)
        
        dAverage_fitness = sum(fitness)/len(fitness)
        
        mHistory.append([i, dAverage_fitness])
        
         # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(enemy_life,6))+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6)))
        file_aux.write('\n'+str(i)+' '+str(round(enemy_life,6))+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))   )
        file_aux.close()
           

    print_generational_gain(mHistory)
    quit()
if __name__ == "__main__":
    main()