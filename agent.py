#
# THIS SCRIPT DEFINES THE AGENT CLASS
# Johannes Mayer
# hnsmyr@gmail.com
# Inspired by 'Python Machine Learning, Third Edition' - Sebastian Raschka & Vahid Mirjalili
# 

import numpy as np
from collections import defaultdict

class Agent():
	def __init__(self, env, epsilon = 0.00, epsilon_min = 0.00, epsilon_decay = 1.0, discount_factor = 0.99, learning_rate = 0.99):
		self.env = env	
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay

		## DEFINE Qtable
		## https://stackoverflow.com/questions/71743169/q-table-representation-for-nested-lists-as-states-and-tuples-as-actions
		## this method creates a table entry only for those states where q-value is changed, i.e. for these states that are visited 
		## this is way faster and more memory efficient than creating a numpy array for all states!!	
		self.Qtable = defaultdict(lambda: np.zeros(self.env.nactions))


	## SELECT ACTION (BEFORE NEXT STEP IS MADE)
	def select_action(self, state):
		if np.random.uniform() < self.epsilon:
			## chooses random action if a random number (between 0 and 1) is smaller than defined self.epsilon
			## this represents the epsilon-greedy policy
			action = np.random.choice(self.env.nactions)
		else:
			## selects the action with max q-value
			## if there are multiple actions with max q-value, it selects one of them randomly
			q_vals = self.Qtable[state]
			indices_maxq = np.where(q_vals == np.amax(q_vals))[0]
			action = np.random.choice(indices_maxq)
		return action

	## Q-LEARNING ALGORITHM
	def learn(self, state, action, reward, next_state, done):

		## this is the actual Q-learning update rule
		## according to Sutton & Barto's 'Reinforcement learning, An introduction' - Eq. 6.8, P131
		## if it terminates, respective q-value is updated only with the negative reward of the terminal state (or positive in case of goal state)
		## i.e. for any goal/terminal state, Q table is never updated but set to reward (from https://en.wikipedia.org/wiki/Q-learning)
		if done:
			Qupdate = reward
		else:
			Qupdate = reward + self.discount_factor * np.max(self.Qtable[next_state])
	
		self.Qtable[state][action] += self.learning_rate * (Qupdate - self.Qtable[state][action])

		## DECAY EPSILON 
		if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

	def save_qtable(self):
		## taken from:  https://stackoverflow.com/questions/49963862/what-is-the-best-way-to-save-a-complex-dictionary-to-file
		odict = dict(self.Qtable)
		np.save('Qtable', np.array(dict(odict)))
		return

	def load_qtable(self):
		idict = np.load('Qtable.npy',allow_pickle=True)
		self.Qtable = idict.item()
		return


## ACTUAL ITERATION: this function trains the Q table
def train_qlearning(agent, env, nepisodes=10000, stop_by_10 = False, reward_goal_state = 1.):
	store_rewards = []
	store_last_5routes = [[0]]*5
	first_finding = 0

	## loop over episodes
	for episode in range(nepisodes):
		store_track = []
		total_reward = 0.
		nmoves = 0

		## this resets the agent to the starting coordinates
		state = env.reset()

		## this while-loop represents one episode, i.e. it loops over the grid/states of the enviroment based on the actions chosen. 
		## it selects an action, performs a step, and updates the Q-table accordingly. 
		## the episode ends if the agent ends up in a terminal or goal state
		while True:
			action = agent.select_action(state)
			next_state, reward, done = env.step(action)
			agent.learn(state, action, reward, next_state, done)
			store_track += [state]
			state = next_state
			nmoves += 1
			if done:
				break
			final_reward = reward
			total_reward += reward

		if first_finding == 0 and final_reward == reward_goal_state: first_finding = episode 


		if episode%100 == 0: print(f'Episode {episode:6}: {nmoves:6} moves, total_reward = {total_reward:7.1f}, final_reward = {final_reward:7.1f}', end='\r')

		# this stores only the last 5 tracks 
		store_last_5routes.pop(0)
		store_last_5routes.append(store_track)

		### stop if goal was found in 10 consecutive episodes
		store_rewards += [final_reward]
		if stop_by_10 and episode > 10 and all(np.array(store_rewards[-10:]) == 1.0):
			print('GOAL FOUND IN 10 CONSECUTIVE EPISODES!')
			break

	return store_last_5routes, first_finding


### TEST FUNCTION: this function uses the trained Q table to find the shortest path -> 'evaluation', no rewards or updates of the Q table
def test_qlearning(agent, env):
	agent.epsilon = 0.
	state = env.reset()	
	store_track = []
	for _ in range(2000):
		action = agent.select_action(state)
		next_state, _, done = env.step(action)
		## agent.learn(state, action, reward, next_state, done)
		store_track += [state]
		state = next_state
		if done:
			break
	return store_track



		
