#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #                                 
# Author: Group 5			 			                                      		  #  				                              			
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'gain_gen_e1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
iNum_of_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  multiplemode='no',
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(iNum_of_neurons),
                  enemymode="static",
                  speed="fastest",
                  level=2,
                  visuals=False)

# create empty array for gain
gain = np.zeros(10)

# loop for each solution
for i in range(10): 
    # load solution
	sol = np.loadtxt('generational_e1/best'+str(i+1)+'.txt')
	g = 0
	# run each solution 5 times
	for j in range(5):
		print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(i+j)+' \n')
		f, p, e, t = env.play(pcont=sol)
		g += p - e
	gain[i] = g / 5

# save array with the gain for each solution
np.savetxt(experiment_name+'.txt', gain)