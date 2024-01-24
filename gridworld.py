#
# THIS SCRIPT DEFINES THE ENVIROMENT CLASS
# Johannes Mayer
# hnsmyr@gmail.com
# Inspired by 'Python Machine Learning, Third Edition' - Sebastian Raschka & Vahid Mirjalili
# 

import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
#import multiprocessing as mp

import matplotlib
#matplotlib.rcParams['axes.labelweight'] = 'light'
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['font.style'] = 'normal'
matplotlib.rcParams['font.family'] = 'sans'
#matplotlib.rcParams['font.weight'] = 'light'
matplotlib.rcParams['font.size'] = 12

class Enviroment():
	def __init__(self, save_transition_array=False, load_transition_array=False, start_coords=(126,3), goal_coords=(135,143),
				reward_goal_state = 1., reward_street_state = -1., reward_terminal_state = -999.):

		print(f' -- Initialization')
		print(f' -- Start: {start_coords}')
		print(f' --  Goal: {goal_coords}')

		## THIS LINES READ THE GOOGLE MAPS SCREENSHOT
		img = image.imread('Input-Screenshot.png')
		img = np.flip(img[:,:,0],axis=0).T
		img[img<0.5] = 0
		img[img>=0.5] = 1
		img = 1 - img[:,:]
		self.img = img	

		self.nlons = img.shape[0]
		self.nlats = img.shape[1]
		self.nstates = self.nlats * self.nlons
		print(f' -- Grid world dimensions = {img.shape}') # 246,150
		print(f' -- Number of non-terminal states: {len(np.concatenate(np.where(img == 1)))}')

		## INITIAL STATE COORDINATES
		self.start_coords = start_coords
		self.start_state = self.grid2state(self.start_coords)

		## FINAL STATE COORDINATES
		self.goal_coords = goal_coords
		self.goal_state = self.grid2state(self.goal_coords)

		## DEFINE TERMINAL CELLS BASED ON INPUT IMAGE
		terminal_cells = []
		for i in range(self.nlons):
			for j in range(self.nlats):
				if img[i,j] == 0: terminal_cells += [(i,j)]

		## DEFINE TERMINAL STATE LIST
		terminal_states_only = [self.grid2state((lon,lat)) for (lon,lat) in terminal_cells]
		self.terminal_states = [self.goal_state] + terminal_states_only

		## DEFINE ACTIONS
		go_north = lambda lon, lat: (max(lon - 1, 0), lat)
		go_east  = lambda lon, lat: (lon, min(lat + 1, self.nlats - 1))
		go_south = lambda lon, lat: (min(lon + 1, self.nlons - 1), lat)
		go_west  = lambda lon, lat: (lon, max(lat - 1, 0))

		self.actions = [go_north, go_east, go_south, go_west]
		self.nactions = len(self.actions)
		
		## COMPUTE TRANSITION MATRIX IN ADVANCE (instead of computing [next step, reward, done] every time step in function 'step')
		## this tremendously speeds up the computation!
		## e.g. to perform 10000 episodes it takes about 120 seconds instead of 2040 seconds @ i7-7820X
		if load_transition_array:
			self.Transition = np.load('transition_array.npy')
			print(self.Transition.shape)
		else:
			self.Transition = np.zeros([self.nstates,self.nactions,3])
			for s in range(self.nstates):
				lon, lat = self.state2grid(s)
				for a in range(self.nactions):
					next_state = self.grid2state(self.actions[a](lon,lat))
				
					if self.check_terminal(next_state):
						reward = (reward_goal_state if next_state == self.goal_state else reward_terminal_state)
					else:
						reward = reward_street_state

					if self.check_terminal(s):
						done = True
						next_state = s
					else:
						done = False
					self.Transition[s,a,:] = [next_state, reward, done]

			if save_transition_array: np.save('transition_array', self.Transition)

		self.nvisits_tracker = np.zeros(self.nstates)

		print(' -- End of initialization')


	def grid2state(self, grid):
		return grid[0]*self.nlats + grid[1]

	def state2grid(self, state):
		y = state%self.nlats
		x = int((state - y)/self.nlats)
		return (x,y)

	## STEP FUNCTION: return the next_state and reward based on the action chosen by the policy
	def step(self, action):
		self.nvisits_tracker[self.state] += 1
		next_state, reward, done = self.Transition[self.state,action,:]
		self.state = int(next_state)
		return [int(next_state), reward, bool(done)]

	def reset(self):
		self.state = self.start_state
		return self.start_state

	def check_terminal(self, state):
		return state in self.terminal_states

	## PLOT FUNCTION: this functions plots the stored routes of one simulation
	def plot_final_path(self, stored_routes, save_fig=None):
		lons = np.arange(self.img.shape[0])-0.5 # shift by 0.5 such that the box is centered at integer values
		lats = np.arange(self.img.shape[1])-0.5

		plt.figure(figsize=(9,6))
		ax = plt.subplot()
		
		txt_title = f'Last {len(stored_routes)} routes' if len(stored_routes) > 1 else f'Final route'
		plt.title(txt_title)
		
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')

		cs = ax.pcolormesh(lons,lats,self.img[:,:].T, cmap='binary_r',vmin=0,vmax=1, rasterized=True)  #'Blues_r') #'binary_r')

		for i in range(len(stored_routes)): 
			plt_coord = np.zeros([len(stored_routes[i]),2])			
			for j in range(len(stored_routes[i])):
				plt_coord[j,:] = self.state2grid(stored_routes[i][j])
			plt.plot(plt_coord[:,0], plt_coord[:,1], ls='-', lw=[5,4,3,2,1][i], color=['darkred','red','c','orange','g'][i])

		plt.plot( self.goal_coords[0], self.goal_coords[1],'go',ms=12)
		plt.plot(self.start_coords[0],self.start_coords[1],'ro',ms=12)
		if save_fig == None:
			plt.show()
		else:
			plt.savefig(save_fig,dpi=300,bbox_inches='tight')


	## PLOT FUNCTION: this plots the number of visits of each state as heat map
	def plot_nvisits(self, save_fig=None):
		nvisits_arr = np.zeros([self.nlons,self.nlats])
		for i in range(self.nstates):
			nvisits_arr[self.state2grid(i)] = self.nvisits_tracker[i]

		# set starting coordinates to 0 to avoid unneccessary large color scaling -> nvisits of starting point = nepisodes
		nvisits_arr[self.start_coords] = 0

		# same for goal state
		nvisits_arr[self.goal_coords] = 0
		
		lons = np.arange(self.img.shape[0])-0.5 # shift by 0.5 such that the box is centered at integer values
		lats = np.arange(self.img.shape[1])-0.5

		plt.figure(figsize=(9,6))
		ax = plt.subplot()
		plt.title('Number of visits')
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')		
		cs = ax.pcolormesh(lons,lats,nvisits_arr.T, cmap='hot', rasterized=True) 
		cbar = plt.colorbar(cs,ax=ax,orientation='horizontal', pad=0.12, shrink=0.65,aspect=20, anchor=(0.5,0.0))
		if save_fig == None:
			plt.show()
		else:
			plt.savefig(save_fig,dpi=300,bbox_inches='tight')





