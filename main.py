'''Balancing PPI dataset with SUS.
'''
import numpy as np
import pandas as pd
import torch
import time
from sampling import *



def main(config, dataframe):

	# create instance of a class
	data = DataProcess(df, config["cutoff"])   #include tokenization
	print("Data Processed")

	# process dataset and obtain changed - sampled dataset
	start_time = time.time()
	X_sampled, y_sampled = SUS(data, config["n_neighbors"], config["blob_threshold"], config["spread_threshold"]).sample()
	end_time = time.time()
	print(f"Time for sampling: {end_time - start_time}")
	combined_array = np.hstack((y_sampled, X_sampled))
	new_df = pd.DataFrame(combined_array, columns=['logFC', 'seq1', 'seq2'])
	new_df['label'] = new_df['logFC'].apply(lambda x: 1 if x>=1 else 0)
	new_df.to_csv("SUS_data.txt", index=False)




if __name__=="__main__":
		
	dataset = "../data/ppiData.txt" # 
	df = pd.read_csv(dataset, sep="\t", header=0) 
	df.drop(columns=['FragPair'], axis=0, inplace=True)
	print(df)
	


	config = { 
			# sus parameters
			"n_neighbors": 7,
			"blob_threshold": 75,
			"spread_threshold": 0.5,

			# relevance cutoff
			"cutoff": 1  #interactions
			}


	main(config, df)

	
					
							







