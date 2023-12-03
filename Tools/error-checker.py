import os
import codecs
import numpy as np

root_dir = f"/home/meribejayson/Desktop/Projects/trailblazer/partial-joint"
problem_paths = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:   
                prob_list = np.array(f.read().split("\n"), dtype = float)
                sum_prob = np.sum(prob_list)

                if(prob_list.len != 5):
                    problem_paths.append("List length was greater than 5 at: \n" +file_path + "\n")
                elif(np.abs(sum_prob - 1) > 0.02 or sum_prob == 0):
                    problem_paths.append("False probability values: \n" +file_path + "\n")

        except Exception as e:
            problem_paths.append("Couldn't Read File: \n" +file_path + "\n")


for path in problem_paths:
    print(path + "\n")