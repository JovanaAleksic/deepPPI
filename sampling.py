'''
Selective Undersampling - SUS
'''

import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from tokenization_proteinBERT import *
import random
import copy


##########################################################################################

class DataProcess:
	def __init__(self, data, cutoff):
		self.data = data

		self.data['F1:F2'] = data['F1_AA'] + data['F2_AA']
		self.data['F1:F2'] = data['F1:F2'].apply(lambda x: np.array(tokenize_seq(x)))

		self.X = self.data.iloc[:, 1:3].values  # feature values
		self.y = self.data.iloc[:, 0].values	# target values

		self.X_train = np.array(np.vstack(self.data.iloc[:, 3].values))  # feature values
		self.y_train = self.y	# target values

		self.indexes_to_undersample = np.where(self.y_train < cutoff)[0]
		self.minor_indexes = np.where(self.y_train >= cutoff)[0]

##########################################################################################

class SUS:
	def __init__(self, data, n_neighbors, blob_threshold, spread_threshold):
		self.X_main = data.X
		self.X = data.X_train
		self.y = data.y_train
		self.n_neighbors = n_neighbors
		self.blob_tr = blob_threshold
		self.spread_tr = spread_threshold
		self.array_of_indexes_major = data.indexes_to_undersample
		self.array_of_indexes_minor = data.minor_indexes


	def sample(self):

		# subset of data to be undersampled
		self.X_major = self.X[self.array_of_indexes_major]
		self.y_major = self.y[self.array_of_indexes_major]
		# indexes of rare data
		self.X_minor = self.X[self.array_of_indexes_minor]
		self.y_minor = self.y[self.array_of_indexes_minor]

		N = len(self.y_major) # data nb
		all_indexes = np.array(range(N))
		grade_array = np.zeros(N)
		visited = np.zeros(N)


		print(f"Started fitting the knn model...")
		# knn model
		knn = NearestNeighbors(n_neighbors=(self.n_neighbors + 1)) # parameter to be set by user
		knn.fit(self.X_major) # only fit model to set of datapoints to be undersampled
		self.knn = knn

		distances, neighbour_indexes = self.knn.kneighbors(self.X_major)
		distances = distances[:, 1:] # remove self
		neighbour_indexes = neighbour_indexes[:, 1:] # remove self
		avg_distances = np.mean(distances, axis=1)

		# 75% blob_tr parameter to be set by the user
		self.blob = np.percentile(avg_distances, self.blob_tr)

		print(f"Average distances computed...")

		close_neighbours_indexes = np.where(distances < self.blob, neighbour_indexes, -1)
		close_neighbours_nb = np.sum(close_neighbours_indexes > -1, axis=1)


		# no close neighbours
		grade_array[close_neighbours_nb == 0] = 2
		visited[close_neighbours_nb == 0] = 1

		process_indexes = np.where(close_neighbours_nb > 0)[0] 


		for i in process_indexes:
			close_neighboursIndx = close_neighbours_indexes[i]
			close_neighboursIndx = close_neighboursIndx[close_neighboursIndx > -1] # keep only indexes that are "close"
			if i % 10000 == 0:
				print(f"Processing instance {i}")

			if visited[i]!=1:
				# close_neighbour_indexes changed in recursion - copy copy to avoid change
				# = operator changes the original array as well
				cluster_indxs = []  # initialize empty return list
				cluster_indxs = SUS.datadecision(copy.copy(close_neighboursIndx), self.y_major, cluster_indxs, self.spread_tr)

				grade_array[cluster_indxs] = 1
				visited[close_neighboursIndx]=1
				visited[i]=1

		undersampled_indexes = all_indexes[(grade_array == 1) | (grade_array == 2)]

		X_sus = np.concatenate((self.X_main[self.array_of_indexes_minor], self.X_main[undersampled_indexes]), axis=0)
		y_sus = np.concatenate((self.y_minor, self.y_major[undersampled_indexes]), axis=0)

		# just for experiments
		self.reduction = undersampled_indexes.size / self.array_of_indexes_major.size
		# print(f"Reduction performed: {self.reduction}")

		return X_sus, y_sus.reshape(-1, 1)	


	@staticmethod
	def datadecision(close_neighboursIndx, y_major, return_list, spread_tr):

		close_neighbour_y = y_major[close_neighboursIndx]

		average_y = np.mean(close_neighbour_y)
		variance_y = np.var(close_neighbour_y)
		y_spread = variance_y / average_y if variance_y != 0 else 0
 
		# 0.5 parameter as well
		if y_spread < spread_tr:
			return_list.append(close_neighboursIndx[np.argmin(abs(close_neighbour_y - average_y))]) # closest to average
			return return_list 
		else: 
			if len(close_neighboursIndx) != 1:
				distant_y_position = np.argmax(abs(close_neighbour_y - average_y)) # remove the furthest from the average point
				return_list.append(close_neighboursIndx[distant_y_position]) # add to the return list
				close_neighboursIndx = np.delete(close_neighboursIndx, distant_y_position) # remove it for recurson
			else:
				return_list.append(close_neighboursIndx[0])
				return return_list

		return SUS.datadecision(close_neighboursIndx, y_major, return_list, spread_tr)  



##########################################################################################
