import numpy as np
import csv

class SoftmaxRegression:
    '''
    A simple class to implement all the methods required for a Softmax Regression Classifier
    '''
    def __init__(self, learning_rate=0.1):
        '''
        Constructor to allow changing the learning rate of classifier.
        '''
        self.learning_rate = learning_rate 
        self.theta = None
        self.biases = None
    
    def read_file(self, file_name):
        '''
        Method to extract data from a given csv file
        '''
        X = []
        y = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                y.append(int(row.pop()))  # Remove the pre-classified label info
                X.append([float(i) for i in row])
        # Convert the arrays to numpy arrays for further analysis
        X = np.array(X)
        y = np.array(y)
        return [X, y]
    
    def one_hot_vector(self, y):
        '''
        Method to find one hot vector
        '''
        return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
    
    def cross_entropy(self, output, y_target):
        '''
        Method to find cross entropy
        '''
        return - np.sum(np.log(output) * (y_target), axis=1)

    def softmax_probability(self,z):
        '''
        Method to calculate probabilities of belonging to any of the k classes
        '''
        return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)
    
    def cost_theta(self, probs, n_samples, X, y):
        '''
        Method to calculate cost for theta matrix
        '''
        probs[np.arange(n_samples),y] -= 1
        probs /= n_samples
        theta_update = np.transpose(X).dot(probs)
        return theta_update
    
    def cost_bias(self, probs, n_samples, y):
        '''
        Method to calculate cost for bias vector
        '''
        probs[np.arange(n_samples),y] -= 1
        probs /= n_samples
        bias_update = np.sum(probs, axis=0, keepdims=True) 
        return bias_update

    def calculate_cost(self, X, y, n_features, n_samples, n_classes):
        '''
        Method to implement cost function
        '''
        z = np.dot(X, self.theta) + self.biases
        probs = self.softmax_probability(z)
        probs_copy = probs.copy()
        theta_update = self.cost_theta(probs, n_samples, X, y)
        bias_update = self.cost_bias(probs_copy, n_samples, y)
        return theta_update, bias_update
    
    def train(self, train_file, learning_rate=1e-2, epochs=1000):
        '''
        Method to train the model using data from a file
        '''
        X, y = self.read_file(train_file)
        # Get useful parameters
        n_features, n_samples, n_classes = X.shape[1], X.shape[0], len(np.unique(y))   
        # Initialize theta matrix and bias vector as zeroes
        self.theta = np.zeros((n_features, n_classes))
        self.biases = np.zeros((1, n_classes))
        # Stopping criterion is when the model reaches the fixed number of epochs
        for _ in range(epochs):
          # Get loss and gradients
            theta_update, bias_update = self.calculate_cost(X, y, n_features, n_samples, n_classes)
          # update weight matrix and bias vector
            self.theta -= self.learning_rate*theta_update
            self.biases -= self.learning_rate*bias_update

    def predict(self, X):
        '''
        Method to predict the class of a given input vector
        '''
        y_pred = np.dot(X, self.theta)+self.biases
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def accuracy(self, file_name):
        '''
        Method to calculate the accuracy of the classifier
        '''
        X, y = self.read_file(file_name)
        return np.mean(self.predict(X)==y)

if __name__ == "__main__":
    classifier = SoftmaxRegression(learning_rate=0.1)
    classifier.train('training1.csv', epochs=500)
    print("Train Set Accuracy: ", classifier.accuracy('training1.csv'))
    print("Test Set Accuracy: ", classifier.accuracy('test1.csv'))