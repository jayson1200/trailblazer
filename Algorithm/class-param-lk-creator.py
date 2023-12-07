"""
Idea:
Create a naive bayes parameter matrix for Y = 1 and Y = 0
"""

import numpy as np
import pandas as pd
from naivebayes import NaiveBayesClassifier

pop_file = "/home/meribejayson/Desktop/Projects/trailblazer/Algorithm/cs_population.csv"
csv_dest = "/home/meribejayson/Desktop/Projects/trailblazer/Algorithm/alg_data/"
pop_df = pd.read_csv(pop_file)

PARAMS_DIM = len(pop_df.columns)

X_I_Y_1_MAT = np.ndarray((PARAMS_DIM, PARAMS_DIM))
X_I_Y_0_MAT = np.ndarray((PARAMS_DIM, PARAMS_DIM))

Y_1_MAT = np.ndarray((1, PARAMS_DIM))
col_names = list(pop_df.columns)

# Yes I know this is terribly inefficient, but I have no time to debug errors so here is the simplest solution
for i in range(PARAMS_DIM):
    Y_TRAIN = pop_df.loc[:, col_names[i]]
    X_TRAIN = pop_df.drop(columns=[col_names[i]])

    classifier = NaiveBayesClassifier(X_TRAIN.to_numpy(), Y_TRAIN.to_numpy())

    Y_1_MAT[0, i] = classifier.get_proby_one()

    prob_x_i_given_y_1_vec = classifier.get_prob_x_i_given_y_1_vec()
    prob_x_i_given_y_0_vec = classifier.get_prob_x_i_given_y_0_vec()
    
    X_I_Y_0_MAT[i,:] = np.insert(prob_x_i_given_y_0_vec, i, 0)
    X_I_Y_1_MAT[i,:] = np.insert(prob_x_i_given_y_1_vec, i, 1)

df_x_i_y_0 = pd.DataFrame(X_I_Y_0_MAT, columns=col_names, index=col_names)
df_x_i_y_1 = pd.DataFrame(X_I_Y_1_MAT, columns=col_names, index=col_names)
df_y_1 = pd.DataFrame(Y_1_MAT, columns=col_names)

df_x_i_y_0.to_csv(csv_dest + "px_i_y_zero.csv")
df_x_i_y_1.to_csv(csv_dest + "px_i_y_one.csv")
df_y_1.to_csv(csv_dest + "p_y_one.csv")