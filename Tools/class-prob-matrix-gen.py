import os
import numpy as np

root_dir = f"/home/meribejayson/Desktop/Projects/trailblazer/partial-joint/CS"

with open("prob-full-matrix", "x+") as file_write:
    with open("prob-full-matrix", 'w') as file_write:
        for root, dirs, files in os.walk(root_dir):
            for dir in dirs:
                file_write.write(dir + ",")
                for idx in range(4):
                    file_path = root_dir + "/" + str(idx) + "-" + dir
                    with open(file_path, "r") as curr_file:
                        curr_file .read().split("\n")
                        file_write.write(dir + "," + "\n")