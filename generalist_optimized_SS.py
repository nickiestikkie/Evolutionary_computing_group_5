################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Group 5              #
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

experiment_name = 'steady_state_e3'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, e


# evaluation
def evaluate(x):
    # Create a matrix to store simulation results
    matrix = np.array(list(map(lambda y: simulation(env,y), x)))

    # Extract fitness values from the matrix (first column)
    fitness = np.array([sublist[0] for sublist in matrix])
    
    # Extract enemy life values from the matrix (second column)
    enemy_life = np.array([sublist[1] for sublist in matrix])
    return fitness, enemy_life


def init_population(individuals, number_of_weights, lower_bound, upper_bound):
    # Each row represents an individual and contains 'number_of_weights' random values
    # These random values are uniformly distributeds between 'lower_bound' and 'upper_bound'
    population = np.random.uniform(lower_bound, upper_bound, (individuals, number_of_weights))
    return population

def init_simulation(iNum_of_neurons):
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[3],
                      multiplemode='no',
                      playermode="ai",
                      player_controller=player_controller(iNum_of_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    return env

def parent_selection(population, fitness, num_parents=10):
    # Randomly select 'num_parents' individuals from the population
    random_parents = np.random.choice(population.shape[0], num_parents, replace=False)

    # Retrieve the fitness scores of the randomly selected parents
    random_parents_fitness = fitness[random_parents]
    
    # Find the index of the best parent (individual with the highest fitness)
    best_parent_index = np.argmax(random_parents_fitness)
    
    # Get the best parent based on its index
    best_parent = population[random_parents[best_parent_index]]

    return best_parent

def survivor_selection(current_pop, total_offspring, fitness, enemy_life, p_new=0.3):
    # Evaluate fitness
    current_population_fitness = fitness
    offspring_fitness, offspring_enemy_life = evaluate(total_offspring)
    
    # Sort the current population and offspring based on fitness in descending order
    current_population_sorted_indices = np.argsort(current_population_fitness)[::-1]
    offspring_sorted_indices = np.argsort(offspring_fitness)[::-1]
    
    # Determine how many individuals to keep from the current population
    num_to_keep = int(current_pop.shape[0] * (1 - p_new))
    num_to_introduce = int(current_pop.shape[0] * p_new)
    
    # Select the top individuals from the current population and the offspring
    individuals_to_keep = current_pop[current_population_sorted_indices][:num_to_keep]
    fitness_individuals_to_keep = current_population_fitness[current_population_sorted_indices][:num_to_keep]
    enemy_life_individuals_to_keep = enemy_life[current_population_sorted_indices][:num_to_keep]

    new_individuals = total_offspring[offspring_sorted_indices][:num_to_introduce]
    fitnes_new_individuals = offspring_fitness[offspring_sorted_indices][:num_to_introduce]
    enemy_life_new_individuals = offspring_enemy_life[offspring_sorted_indices][:num_to_introduce]

    # Combine the selected individuals from both the current population and the offspring
    new_population = np.concatenate((individuals_to_keep, new_individuals), axis=0)
    fitness = np.concatenate((fitness_individuals_to_keep, fitnes_new_individuals), axis=0)
    enemy_life = np.concatenate((enemy_life_individuals_to_keep, enemy_life_new_individuals), axis=0)
    
    return new_population, fitness, enemy_life

def crossover(population, fitness, enemy_life, fixed_start=True, fixed_end=True, n_offspring=2, p_left=0.5, p_mutation=0.15, mutation_rate=float, generational_model=bool):
    # If generational_model = True, the generational model is used
    # If generational_model = False, the steady-state model is used
    n_vars = population[0].shape[0]
    total_offspring = np.zeros((0,n_vars))
    
    for p in range(0, population.shape[0], 2):  # stepsize 2, since you choose 2 parents and otherwise you get 2 times the number of offspring
        parents = np.zeros((2, population.shape[1]))
        parents[0] = parent_selection(population, fitness)
        parents[1] = parent_selection(population, fitness)
        
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

    if generational_model == True:
        fitness, enemy_life = evaluate(total_offspring)
        return total_offspring, fitness, enemy_life
    else:
        return survivor_selection(population, total_offspring, fitness, enemy_life)

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
    # magic numbers
    generational_model = False # if generational_model = False, the steady-state model is applied
    individuals = 100
    hidden_neurons = 10
    lower_bound = -1
    upper_bound = 1
    generations = 30
    stop_time = 3000
    mHistory = [] # becomes a list of lists

    initial_mutation_rate = 10
    final_mutation_rate = 0.01
    mutation_rates = initial_mutation_rate * np.exp(np.linspace(0, np.log(final_mutation_rate / initial_mutation_rate), generations))

    env = init_simulation(hidden_neurons)
    number_of_weights = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
    population = init_population(individuals, number_of_weights, lower_bound, upper_bound)
    fitness, enemy_life = evaluate(population)

    for j in range(10):
    
        env = init_simulation(hidden_neurons)
        number_of_weights = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
        population = init_population(individuals, number_of_weights, lower_bound, upper_bound)
        fitness, enemy_life = evaluate(population)
        
        mean_fitness = np.mean(fitness)
        best_fitness = np.argmax(fitness)
        std_fitness = np.std(fitness)
    
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n'+str(0)+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))   )
        file_aux.close()

        start_time = time.time() # Start timer (time in seconds)
        for i in range(generations):
            # create new gen
            population, fitness, enemy_life = crossover(population, fitness, enemy_life, mutation_rate=mutation_rates[i], generational_model=generational_model) 
        
            mean_fitness = np.mean(fitness)
            best_fitness = np.argmax(fitness)
            std_fitness = np.std(fitness)
        
            # mHistory.append([i, mean_fitness])

            # saves results
            file_aux  = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i+1)+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6)))
            file_aux.write('\n'+str(i+1)+' '+str(round(fitness[best_fitness],6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))   )
            file_aux.close()

            print(f'enemy life = {np.min(enemy_life)}')
        
        # save best individual for this run
        np.savetxt(experiment_name+'/best'+str(j+1)+'.txt', population[best_fitness])
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    # print_generational_gain(mHistory)
    quit()
if __name__ == "__main__":
    main()
    
