# -------------------------
# File: Deep Dueling Q Learning
# Author: Shuo Yang
# Date: 2018.5.31
# -------------------------
import sys
sys.path.append("game/")
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

import wrapped_flappy_bird as game
from Dueling import BrainDQN
import numpy as np

# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playFlappyBird():
	# Step 1: init BrainDQN
	actions = 2
	brain = BrainDQN(actions,dueling = True)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	#brain.get_graph()
	# Step 3.2: run the game
	for i in range(1001000):
		action = brain.getAction()
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,action,reward,terminal)
		#if i%10000==0:
		#	brain.plot_cost()
	#print(brain.cost_his)
	

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()