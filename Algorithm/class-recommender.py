import numpy as np
import pandas as pd

x_i_y_one_file = "/home/meribejayson/Desktop/Projects/trailblazer/Algorithm/alg_data/px_i_y_one.csv"
x_i_y_zero_file = "/home/meribejayson/Desktop/Projects/trailblazer/Algorithm/alg_data/px_i_y_zero.csv"
y_one_file = "/home/meribejayson/Desktop/Projects/trailblazer/Algorithm/alg_data/p_y_one.csv"
instructions = """



=====================================================================================================================
Trailblazer is program to help CS undergrad, masters, and phd students find CS classes that they might be 
interested in

The instructions on how to use the program are below 
=====================================================================================================================

=====================================================================================================================
Type the name of the classes that you like in the following format:
CS-229 CS-224N CS-109 CS-107
=====================================================================================================================

Type here: """


PX_I_Y_ONE_df = pd.read_csv(x_i_y_one_file,index_col=0)
PX_I_Y_ZERO_df = pd.read_csv(x_i_y_zero_file, index_col=0)
PY_ONE_df = pd.read_csv(y_one_file, index_col=0)

input_classes = input(instructions).upper().split(" ")

classes_in_mat = list(PX_I_Y_ONE_df.columns)
classes = []

for inp_class in input_classes:
    if(inp_class in classes_in_mat):
        classes.append(inp_class)

classes = np.array(classes)

usr_class_odds = []
class_titles = []


for class_name in classes_in_mat:
    if(class_name in classes):
        continue
    
    class_titles.append(class_name)

    log_prob_1 = np.log(PY_ONE_df.loc[0,class_name]) 
    log_prob_0 = np.log(1 - PY_ONE_df.loc[0,class_name])

    for usr_class in classes:
        log_prob_1 += np.log(PX_I_Y_ONE_df.loc[class_name, usr_class])
        log_prob_0 += np.log(PX_I_Y_ZERO_df.loc[class_name, usr_class])

    
    usr_class_odds.append(log_prob_1 - log_prob_0)

prob_like_class_dict = {
    "Classes" : class_titles,
    "Odds": usr_class_odds
}


likable_class_desc_down = list(pd.DataFrame(prob_like_class_dict).sort_values(by="Odds", ascending=False)["Classes"])

output_str = f"""

Since you liked {' '.join(map(str, classes))} you will probably like
{(' '.join(map(str ,likable_class_desc_down[:15]))).replace(' ', "\n")}


and you probably won't like
{(' '.join(map(str ,likable_class_desc_down[-15:]))).replace(' ', "\n")}

"""


print(output_str)

