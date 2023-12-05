import os
import numpy as np

root_dir = f"/home/meribejayson/Desktop/Projects/trailblazer/partial-joint"

with open("class-prob", "x+") as file_write:
    with open("class-prob", 'w') as file_write:
        for root, dirs, files in os.walk(root_dir):
            for dir in dirs:
                if(dir != "ATHLETIC" and dir != "CS" and dir != "OUTDOOR" and dir != "PHYSWELL"):
                    file_write.write(dir + "," + "\n")
