import numpy as np

def fitness(row):
     # Find the index of the row with the maximum value in the first column
    value = row[1]
    print(value)
    return value

def parent_selection(population):
    
    # Select 5 random parents ids
    num_parents = 5
    random_parents_id = np.random.choice(population.shape[0], num_parents, replace=False)

    # Select the 5 according parents
    random_parents = population[random_parents_id]

    # Calculate fitness scores for each parent
    fitness_scores = np.array([fitness(row) for row in random_parents])

    # Find the index of the parent with the highest fitness score
    best_parent_id = np.argmax(fitness_scores)

    # Select the row with the maximum value in the first column
    return random_parents[best_parent_id]


population =  np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 1, 1]
])

parent = parent_selection(population)
print(parent)

