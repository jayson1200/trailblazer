# **Trailblazer** 🔥

**Trailblazer is a course recommender that uses the Naive Bayes machine learning algorithm to recommend CS classes to CS students based on classes they have taken before. In order to do this, it uses course evaluations and details about course content to intelligently recommend classes to students given classes they have taken and liked**

## Implementation

### **Acquiring Data**

### Course Evaluations

I was able to obtain course evaluations for each class on
[Stanford's course evaluations website](https://evals.stanford.edu/new-course-evaluation-system?utm_source=pocket_saves).
I decided to take 4 questions from each course evaluation. Those questions were

- How well did you achieve the learning goals of this course?
- How much did you learn from this course?
- Overall, how would you describe the quality of the instruction in this course?
- How organized was this course?

I thought these question would best represent how likely someone who had taken class was to report that they enjoyed the class.

All of these questions had 5 options with which a student, who had taken the class, could choose from and the course evaluation document
for a class listed the percentage of students who chose a specifc option.

To get the data in a computer readable format, I manually typed in the percentages for all CS classes for each question for each option.
In order to condense these percentages into a z-value for each class, that I could throw into a $\sigma$ function to get a probability value,
I did the following:

As you can see, for a particular class, I just append the probability values to a vector then add that vector to a matrix.

```python
import os
import numpy as np

root_dir = f"PATH"

with open("prob-full-matrix", "x+") as file_write:
    with open("prob-full-matrix", 'w') as file_write:
        for root, dirs, files in os.walk(root_dir):
            for dir in dirs:
                file_write.write(dir)
                for idx in range(4):
                    file_path = root_dir + "/" + dir + "/" + str(idx) + "-" + dir
                    with open(file_path, "r") as curr_file:
                        file_write.write("," + curr_file.read().replace("\n", ","))

                file_write.write("\n")
```

Next, I used PCA to condense the each class vector into a z value which would be plugged into a $\sigma$ function to obtain a probability:

It turned out that the first principle component explained 99.64% of the variance between class vectors, so in order to get the first
principle component's linear combination of each of the questions option percentage, I took the dot product of each class vector and
each the first principle component, the result of which I used as a z-value for the probability that a student who takes a class likes
a class.

```python
import pandas as pd
import numpy as np


df = pd.read_csv("./prob-full-matrix", index_col=0, names=np.arange(0, 20, 1, dtype=int))
PROB_MATRIX = df.to_numpy()

for i in range(PROB_MATRIX.shape[1]):
    mean = np.sum(PROB_MATRIX[:, i]) / PROB_MATRIX.shape[1]
    PROB_MATRIX[:, i] = PROB_MATRIX[:, i] - mean

COV_MATRIX = (1/(PROB_MATRIX.shape[0]-1)) * (PROB_MATRIX.T @ PROB_MATRIX)
U, S, V = np.linalg.svd(COV_MATRIX)

print(f"{S[0]/np.sum(S)}% variance is exaplained by the first principal component.")

first_pc = U[:, 1]
first_pc
relative_probs = (first_pc.reshape(1, 20) @ PROB_MATRIX.T)
dict = {"Class":df.index.to_numpy(), "Z": relative_probs.flatten()}
z_df = pd.DataFrame(dict)

def sig(z):
    return 1 / (1 + np.exp(-z))

dict = {"Class":df.index.to_numpy(), "Prob": sig(relative_probs.flatten())}

prob_df = pd.DataFrame(dict)
z_df.to_csv("z_like.csv", index=False)
```

### Simulating CS Population

I was unable to get anonymized transcripts or any other data from Stanford University, so I was forced to simulate a population.
To do this I manually put Stanford CS classes into 14 buckets:

- Art
- Computer Engineering
- Theory
- Artificial Intelligence
- Algorithms
- Computational Biology
- Graphics
- Human Computer Interaction
- Information
- Systems
- Software Engineering
- Robotics
- Ethics
- Not Specific
- Cyber Security

To create the population, I first made a copy of the z-value which would eventually represent the probability the current member
of the population would like a class. I then sampled 1 value from the uniform multinomial distribution that is the 14 buckets and
based on that I gave all classes that were in the same category a bias that would increase the probability that that particular person
would like classes in the cateogry. Then for that particular person, I sampled from a bernoulli distribution for each class. The result
of each iteration was a vector of 0s and 1s representing whether the current member would like a class or not. I appended this to a population
matrix and repeated the aforementioned steps 17000 times.

Here is a snippet for some details

```python
for usr_idx in range(POP_SIZE):
    user_pref_cat = np.random.choice(cat_nums)
    user_probs = z_file.copy(deep = True)

    for key, value in cat_dict.items():
        if(np.any(np.in1d(np.array(sing_to_many(user_pref_cat)), cat_dict[key]))):
            user_probs.loc[key,"Z"] = user_probs.loc[key,"Z"] + PREF_BIAS
        else:
            user_probs.loc[key,"Z"] = user_probs.loc[key,"Z"] - NOT_PREF_BIAS

    user_probs["Z"] = sig(user_probs["Z"])

    user_vec = np.array([int(np.random.binomial(1, p)) for p in user_probs["Z"]])

    POP_MATRIX[usr_idx, :] = user_vec
```

### **Training**

To train, from my newly acquired population matrix of 0s and 1s, I picked a class to be the truth value and removed it from the population matrix. The new
population matrix was deemed the training data I then calculated the parameters i.e. $(P(X_i \mid Y = 1) \text{ and } P(X_i \mid Y = 0))$, according to Naive Bayes,
and appended them to two paramter matrices—one representing of $X_i$ if $Y = 1$ and the other represent $Y = 0$. At this point, I had two matrices with the
following structure:

$PARAMMATRIXYone_{i,j}$ represents the probability that a student would like class $X_j$ given that they like $X_i$

The other matrix represents the opposite. that is:

$PARAMMATRIXYzero_{i,j}$ represents the probability that a student would like class $X_j$ given that they did not like $X_i$

### **Prediction**

To give suggestions to a student based on classes that they reported that they like, I looped over all classes that they had
not taken and calculated the probability that they would like the class given the classes that they said they like using the
naive bayes assumption. I did the same for whether they would not like the class they haven't taken before. I then combined both
value into a log odds and appended it to a list. After looping, I sorted the log odds and reported ones that were the largest and
smallest.

### To try it

I didn't have enough time to make a website, so the code is available [here](https://github.com/jayson1200/trailblazer.git).
To run it, all you have to do is change all of the paths at the top of the file to correct paths for your system and run
lass-recommender.py the Algorithm folder via python in an environment that has pandas and numpy. If you have any suggestions,
please contact me at jmeribe@stanford.edu. I am very interested in improving recommendations.
