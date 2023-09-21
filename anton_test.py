import numpy as np

def fitness(row):
     # Find the index of the row with the maximum value in the first column
    value = row[0]
    # print(value)
    return value


# make a pop_fitness that returns the fitness of the entire pop and the index of every indiviudal

def parent_selection(pop_fitness):
    
    # Select n random parents ids
    num_parents = 6
    random_parents_id = np.random.choice(pop_fitness.shape[0], num_parents, replace=False)

    # Select the n according individuals
    random_parents = np.array(pop_fitness[random_parents_id])
    random_parents = np.column_stack((random_parents_id, random_parents))

    # Find the index of the parent with the highest fitness score
    best_parent_id = np.argmax(random_parents[:,1])
    parent_one = random_parents[best_parent_id,0]

    print(random_parents)

    print("best parent is", best_parent_id)
    print(parent_one)

    # Create a new matrix without the best parent
    random_parents = np.delete(random_parents, best_parent_id, axis=0)

    print(random_parents)

    # select the second parent
    second_best = np.argmax(random_parents[:,1])
    second_parent = random_parents[second_best,0]

    print(second_best)
    print("2nd parent is", second_parent)


    # Select the row with the maximum value in the first column
    return parent_one, second_parent


population =  np.array([
    [6, 5, 8],
    [7, 0, 4],
    [2, 5, 6],
    [0, 4, 11],
    [1, 2, 3],
    [4, 11, 12],
    [5, 1, 1],
    [3, 8, 9],
    [8, 10, 2]
])

parent = parent_selection(population)
print(parent)
# print(parent)

# def parent_selection(population):
    
#     # Select 5 random parents ids
#     num_parents = 5
#     random_parents_id = np.random.choice(population.shape[0], num_parents, replace=False)

#     # Select the 5 according parents
#     # random_parents = population[random_parents_id]
#     # print(random_parents)

#     # Calculate fitness scores for each parent
#     fitness_scores = np.array([fitness(row) for row in random_parents_id])

#     # Find the index of the parent with the highest fitness score
#     best_parent_id = np.argmax(fitness_scores)
#     print(best_parent_id)

#     # Select the row with the maximum value in the first column
#     return random_parents[best_parent_id]
