'''
Naive Bayes Classifier 

The file implements a fully functioning Naive Bayes Classifier
'''

import numpy as np

class NaiveBayesClassifier:
    
    """
        Initializes the training and testing sets and trains the classifier 
        via the training set
    """
    def __init__(self, X_TRAIN, Y_TRAIN):
        self.X_TRAIN = X_TRAIN
        self.Y_TRAIN = Y_TRAIN

        self.proby_one = (np.sum(Y_TRAIN) + 1) / (Y_TRAIN.size + 2)
        self.proby_zero = 1 - self.proby_one

        self.x_i_given_y_1 = []
        self.x_i_given_y_0 = []

        # populates the parameter list with laplace smoothing for each probability vale 
        for col_idx in range(X_TRAIN.shape[1]):
            self.x_i_given_y_1.append((np.sum(X_TRAIN[Y_TRAIN == 1,col_idx]) + 1) / (Y_TRAIN[Y_TRAIN == 1].size + 2))
            self.x_i_given_y_0.append((np.sum(X_TRAIN[Y_TRAIN == 0,col_idx]) + 1) / (Y_TRAIN[Y_TRAIN == 0].size + 2))
    
    """
        Given an example, exp, predict predicts what class the example belongs to
        using the parameters calculated during the training step
    """
    def predict(self, exp):
        proby_zero = np.log(self.proby_zero)
        proby_one = np.log(self.proby_one)

        for idx, val in enumerate(exp):
            if(val == 0):
                proby_zero += np.log(1 - self.x_i_given_y_0[idx])
                proby_one += np.log(1 - self.x_i_given_y_1[idx])
            else:
                proby_zero += np.log(self.x_i_given_y_0[idx])
                proby_one += np.log(self.x_i_given_y_1[idx])

        return 1 if proby_one > proby_zero else 0

    """
        Given a test set, returns the accuracy of the model trained with the given training data
    """
    def test(self, X_TEST, Y_TEST):
        num_success = 0

        for idx in range(Y_TEST.size):
            if(self.predict(X_TEST[idx, :]) == Y_TEST[idx]): 
                num_success += 1

    
        return num_success / Y_TEST.size
    
    """
        Returns P(Y = 1)
    """
    def get_proby_one(self):
        return self.proby_one
    
    """
        returns P(Y = 0)
    """
    def get_proby_zero(self):
        return self.proby_zero
    
    """
        returns P(X_i = 1 | Y = 1) in vector format where
        the index corresponds with i
    """
    def get_prob_x_i_given_y_1_vec(self):
        return np.array(self.x_i_given_y_1)
    
    """
        returns P(X_i = 1 | Y = 0) in vector format where
        the index corresponds with i
    """
    def get_prob_x_i_given_y_0_vec(self):
        return np.array(self.x_i_given_y_0)

