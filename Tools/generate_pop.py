"""
Constants:
There are 15 categories
The z score bias is 0.1
The number of CS graduates in 2021-22 year was 656
    - I will use 2000 samples

Population Generation Plan:

1. Create a dictionary out of the class categories
2. Pull in the Z file as a dataframe

============================

Repeat: 2000
3. Create a copy of the Z file 
4. Pick a number between [0, 15)
5. The corresponding class cateogry that that belongs add the z score bias to all classes that are attached to that class
6. Apply the sigmoid function to the vector of z scores
7. For each class, using the obtained probability values as bernoullis, random assign the corresponding entry a 1 
8. Transpose the example vector and append to the population numpy matrix 

============================

Convert the numpy matrix to CSV with column labels for each class

"""
import numpy as np
import pandas as pd
import csv



# Create Dicitionary
def csv_to_dict(filename):
    result = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = row[0]
            values = np.array([int(i) for i in row[1:]])
            result[key] = values
    return result

def sig(z):
    return 1 / (1 + np.exp(-z))

def sing_to_many(orig_cat):
    if(orig_cat == 0):
        return [6, 10, 0]
    elif(orig_cat == 1):
        return [9, 11, 1]
    elif(orig_cat == 2):
        return [2, 3, 4]
    elif(orig_cat == 3):
        return [2, 4, 3]
    elif(orig_cat == 4):
        return [3, 2, 8, 4]
    elif(orig_cat == 5):
        return [5, 3, 4]
    elif(orig_cat == 6):
        return [0, 4, 6]
    elif(orig_cat == 7):
        return [7, 11, 12]
    elif(orig_cat == 8):
        return [4, 9, 8]
    elif(orig_cat == 9):
        return [1, 14, 8, 9]
    elif(orig_cat == 10):
        return [0, 12, 7, 10]
    elif(orig_cat == 11):
        return [1, 3, 9, 11]
    elif(orig_cat == 12):
        return [7, 12]
    elif(orig_cat == 13):
        return [13]
    elif(orig_cat == 14):
        return [9, 8, 14]


filename = '/home/meribejayson/Desktop/Projects/trailblazer/partial-joint/class-cat'
z_file = '/home/meribejayson/Desktop/Projects/trailblazer/partial-joint/z_like.csv'
PREF_BIAS = 0.125
POP_SIZE = 17000

cat_dict = csv_to_dict(filename)
z_file = pd.read_csv(z_file, index_col=0)
cat_nums = np.arange(0, 15, 1)


POP_MATRIX = np.ndarray((POP_SIZE, len(cat_dict)), dtype=np.uint8)



for usr_idx in range(POP_SIZE):
    user_pref_cat = np.random.choice(cat_nums)
    user_probs = z_file.copy(deep = True)

    for key, value in cat_dict.items():
        if(np.any(np.in1d(np.array(sing_to_many(user_pref_cat)), cat_dict[key]))):
            user_probs.loc[key,"Z"] = user_probs.loc[key,"Z"] + PREF_BIAS
        else:
            user_probs.loc[key,"Z"] = user_probs.loc[key,"Z"] - PREF_BIAS
    
    user_probs["Z"] = sig(user_probs["Z"])

    user_vec = np.array([int(np.random.binomial(1, p)) for p in user_probs["Z"]])

    POP_MATRIX[usr_idx, :] = user_vec

pd.DataFrame(POP_MATRIX).to_csv("cs_population.csv", index=False, header=list(cat_dict.keys()))



