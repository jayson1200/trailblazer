import os 
import sys


def create_course(course_name):
    print(course_name)
    os.mkdir(course_name)

    for i in range(4):
        new_file = course_name + "/" + str(i) + "-" + course_name
        open(new_file,"x")


for course_id in range(1, len(sys.argv)):
    create_course(sys.argv[course_id])


print("""
      
      0 - How well did you achieve the learning goals of this course?
      
      1 - How much did you learn from this course?
      
      2 - Overall, how would you describe the quality of the instruction in this course?
      
      3 - How organized was this course?
      """)