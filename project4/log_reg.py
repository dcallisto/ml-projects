import numpy as np

class LogisticRegression:
    def __init__(self, classname, alpha, num_iter):
        self.alpha = alpha
        self.num_iter = num_iter
        self.classname = classname
    

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    
    def fit(self, features, labels):
        
        # weights initialization
        self.theta = np.zeros(features.shape[1])
        self.best_theta = np.zeros(features.shape[1])
        self.best_accuracy = 0
        
        for i in range(0, self.num_iter):
            z = np.dot(features, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(features.T, (np.round(h) - labels)) / labels.size
            self.calculate_accuracy(features, labels)
            if self.accuracy > self.best_accuracy:
                self.best_accuracy = self.accuracy
                self.best_theta = np.copy(self.theta)
            self.theta -= self.alpha * gradient
        print("Class " + self.classname + " fit")
        self.theta = self.best_theta

       # print(self.theta[0])
       # print(self.theta[1])

            

    

    def calculate_prob_for_given_model(self, features):
        z = np.dot(features, self.theta)
        h = self.sigmoid(z)
        print("Done calculating for class " + self.classname)
        return h

        

    def calculate_accuracy(self, features, labels):

        z = np.dot(features, self.theta)
        h = self.sigmoid(z)
        #if rounded h != label, misclassified
        rounded = np.round(h)
        accuracy_matrix = rounded - labels
        num_misclassified = np.count_nonzero(accuracy_matrix)
        self.accuracy = 1 - (num_misclassified / features.shape[0])
        #print("Class " + self.classname + " accuracy: ")
        #print(self.accuracy)
