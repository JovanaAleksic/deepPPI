'''Balancing PPI dataset with SUS.
'''
import numpy as np
import pandas as pd
from susppi import SUSppi
from tokenization_proteinBERT import *


df = pd.read_csv("../data/seqUniqFragPairslogFC_filt2libs.txt", sep="\t", header=0) 
df.drop(columns=['FragPair'], axis=0, inplace=True)
df['F1:F2'] =df['F1_AA'] + df['F2_AA']
df['F1:F2'] = df['F1:F2'].apply(lambda x: np.array(tokenize_seq(x)))
print(df)

X = np.array(np.vstack(df.iloc[:, 3].values))  # feature values
y = df.iloc[:, 0].values

sus = SUSppi(k=7, blobtr=0.75, spreadtr=0.5) 
X_sampled, y_sampled = sus.fit_resample(X, y)
print(f"SUS done")

combined_array = np.hstack((y_sampled, X_sampled))
new_df = pd.DataFrame(combined_array, columns=['logFC', 'seq1', 'seq2'])
new_df['label'] = new_df['logFC'].apply(lambda x: 1 if x>=1 else 0)
new_df.to_csv("SUS_PPIdata_v2.txt", index=False)




	
					
							







