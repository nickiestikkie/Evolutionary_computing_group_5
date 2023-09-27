
import matplotlib.pyplot as plt
import numpy as np

## line plots

# test variables
mean_1 = np.linspace(1, 75, 30)
std_1 = np.linspace(0, 10, 30)
best_1 = np.linspace(25, 90, 30)
best_std_1 = np.linspace(0, 3, 30)

mean_2 = np.linspace(4, 60, 30)
std_2 = np.linspace(0, 10, 30)
best_2 = np.linspace(25, 60, 30)
best_std_2 =np.linspace(0, 3, 30)

mean = np.vstack([mean_1, mean_1, mean_1, mean_2, mean_2, mean_2])
std = np.vstack([std_1, std_1, std_1, std_2, std_2, std_2])
best = np.vstack([best_1, best_1, best_1, best_2, best_2, best_2])
best_std = np.vstack([best_std_1, best_std_1, best_std_1, best_std_2, best_std_2, best_std_2])

generation= np.linspace(1, 30, 30)

# function
def line_plot(generation, mean, std, best):
    fig = plt.figure(figsize=(12, 10))
    
    method = ['Gen', 'SS']
    enemy = [1, 2, 3]
    
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1,) 
    
        ax.plot(generation, mean[i], 'b', label='mean')
        ax.plot(generation, best[i], 'r', label='best')
    
        ax.fill_between(generation, mean[i]-std[i], mean[i]+std[i], color='blue', alpha=0.2)
        ax.fill_between(generation, best[i]-best_std[i], best[i]+best_std[i], color='red', alpha=0.2)
        
        ax.set_ylim(0, 100)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        
        
        method_idx, enemy_idx = divmod(i, len(enemy))
        ax.set_title(f'{method[method_idx]}, Enemy: {enemy[enemy_idx]}')
                    
        ax.legend()
    
    plt.tight_layout()

# ## boxplots

# generate example gain
gain = np.random.uniform(75, 95, (6, 10))

# boxplot funtion
def boxplot(gain):
    plt.boxplot(gain.T);
    plt.title('Gain distribution per algoritm, per enemy')
    plt.xticks([1, 2, 3, 4, 5, 6], ['Gen 1', 'Gen 2', 'Gen 3', 'SS 1', 'SS 2', 'SS 3'])
    plt.ylabel('Gain')



