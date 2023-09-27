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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, p, e, t


# evaluation
def evaluate(x):
    fitness, person_life, enemy_life, time = np.array(list(map(lambda y: simulation(env,y), x)))
    return fitness, enemy_life


def init_population(iIndividuals, iN_vars, iL_bound, iU_bound):
    population = np.random.uniform(iL_bound, iU_bound, (iIndividuals, iN_vars))
    return population

def init_simulation(iNum_of_neurons):
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[1],
                      multiplemode='no',
                      playermode="ai",
                      player_controller=player_controller(iNum_of_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    return env

def parent_selection(population, num_parents=8):
    # Randomly select 'num_parents' individuals from the population
    random_parents = population[np.random.choice(population.shape[0], num_parents, replace=False)]
    
    # Evaluate the fitness of the randomly selected parents
    random_parents_fitness = evaluate(random_parents)[0]
    
    # Find the index of the best parent (individual with the highest fitness)
    best_parent_index = np.argmax(random_parents_fitness)
    
    # Get the best parent based on its index
    best_parent = random_parents[best_parent_index]

    return best_parent

def survivor_selection(current_pop, total_offspring, p_new=0.3):
    # Evaluate fitness
    current_population_fitness = evaluate(current_pop)[0]
    offspring_fitness = evaluate(total_offspring)[0]
    
    # Sort the current population and offspring based on fitness in descending order
    current_population_sorted_indices = np.argsort(current_population_fitness)[::-1]
    offspring_sorted_indices = np.argsort(offspring_fitness)[::-1]
    
    # Determine how many individuals to keep from the current population
    num_to_keep = int(current_pop.shape[0] * (1 - p_new))
    num_to_introduce = int(current_pop.shape[0] * p_new)
    
    # Select the top individuals from the current population and the offspring
    individuals_to_keep = current_pop[current_population_sorted_indices][:num_to_keep]
    new_individuals = total_offspring[offspring_sorted_indices][:num_to_introduce]

    # Combine the selected individuals from both the current population and the offspring
    new_population = np.concatenate((individuals_to_keep, new_individuals), axis=0)
    return new_population

def crossover(pop, fixed_start=True, fixed_end=True, n_offspring=2, p_left=0.5, p_mutation=0.1, mutation_rate=float):
    n_vars = pop[0].shape[0]
    total_offspring = np.zeros((0,n_vars))
    
    for p in range(0, pop.shape[0], 2):  # stepsize 2, since you choose 2 parents and otherwise you get 2 times the number of offspring
        parents = np.zeros((2, pop.shape[1]))
        parents[0] = parent_selection(pop)
        parents[1] = parent_selection(pop)
        
        offspring = np.zeros((n_offspring, n_vars))
        
        index_list = np.arange(0, len(parents[0])).tolist() # create a list of index to sample from
    
        if fixed_start:
            index_list.remove(0) # Remove the first index 0 from the list
    
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
                    offspring[c, i] += np.random.normal(0, mutation_rate)
            
            total_offspring = np.vstack((total_offspring, offspring[c]))

    # return total_offspring
    return survivor_selection(pop, total_offspring)

def print_generational_gain(history):
    ''' 
    Purpose: shows a line diagram of the average fitness gain over generations 

    Input: history = matrix generation with average fitness
    
    Print statement: Linediagram 
    '''
    
    for row in history:
        x = [el[0] for el in history] # generation number
        y = [el[1] for el in history] # average fitness
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
    generations = 30
    stop_time = 3000
    mHistory = [] # becomes a list of lists

    initial_mutation_rate = 5
    final_mutation_rate = 0.001
    mutation_rates = initial_mutation_rate * np.exp(np.linspace(0, np.log(final_mutation_rate / initial_mutation_rate), generations))
    np.random.seed(1234)

    env = init_simulation(hidden_neurons)
    number_of_weights = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
    population = init_population(individuals, number_of_weights, lower_bound, upper_bound)
    
    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')

    timer = time.time() # Start timer (time in seconds)
    for i in range(generations):
        
        # create new gen
        population = crossover(population, mutation_rate=mutation_rates[i])
        
        # evaluate current population
        fitness = evaluate(population)[0]
        enemy_life = np.min(evaluate(population)[1])
        
        mean_fitness = np.mean(fitness)
        best_fitness = np.argmax(fitness)
        std_fitness = np.std(fitness)
        
        mHistory.append([i, mean_fitness])

        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))   )
        file_aux.close()

        print(f'enemy life = {enemy_life}')   

    print_generational_gain(mHistory)


if __name__ == "__main__":
    main()