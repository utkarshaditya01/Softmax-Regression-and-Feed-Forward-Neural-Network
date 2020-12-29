import numpy as np
import pandas as pd

class NeuralNetwork:
    '''
    A simple class to implement all the methods required for Neural Nework
    '''
    def __init__(self, filename , learning_rate=0.45):
        '''
        Constructor to allow changing the learning rate of classifier and initialize input and label array.
        '''
        x, y = self.readfile(filename)
        np.random.seed(4)
        self.input = x
        self.learning_rate = learning_rate
        self.w1 = np.random.randn(self.input.shape[1], 251)*0.01
        self.w2 = np.random.randn(251, 10)*0.01
        self.y = self.One_Hot_Encoder(y, x)
        self.output = np.zeros(self.y.shape)
    
    def readfile(self, filename):
        '''
        Method to extract data from a given csv file
        '''
        df = pd.read_csv(filename)
        ydf = df.iloc[:, -1:]
        xdf = df.iloc[:, : -1]
        label = ydf.to_numpy()
        x = xdf.to_numpy()
        return x, label

    def feedforward(self):
        '''
        Method to formulate forward propagation
        '''
        x1 = self.sigmoid(np.dot(self.input, self.w1))
        x2 = self.sigmoid(np.dot(x1, self.w2))
        return x1, x2

    def train_backprop(self, epochs = 300):
        '''
        Method to backpropagate and balance weights
        '''
        # Applying Chain Rule
        for _ in range(epochs):
            a1, a2 = self.feedforward()
            d_w2 = self.learning_rate * \
                np.dot(
                    a1.T, ((1/self.input.shape[0])*(a2 - self.y) * self.sigmoid_derivative(a2)))
            d_w1 = self.learning_rate*np.dot(self.input.T,  ((1/self.input.shape[0])*np.dot(
                (a2 - self.y) * self.sigmoid_derivative(a2), self.w2.T) * self.sigmoid_derivative(a1)))

            # Updating Weights
            self.w1 -= d_w1
            self.w2 -= d_w2

    def sigmoid(self, x):
        '''
        Method to implement activation fuction i.e sigmoid function
        '''
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        '''
        Method to implement sigmoid derivative function
        '''
        vec = self.sigmoid(x)
        return vec*(1-vec)

    def One_Hot_Encoder(self, y, x):
        '''
        Method to find one hot vector
        '''
        target = np.zeros((x.shape[0], 10))
        row, col = x.shape
        for i in range(row):
            target[i][y[i]] = 1
        return target
    
    def score(self, filename):
        '''
        Method to predict accuracy given the dataset(file)
        '''
        x, y = self.readfile(filename)
        self.input = x
        self.y = self.One_Hot_Encoder(y, x)
        self.output = np.zeros(self.y.shape)
        a1, y_pred = train_net.feedforward()
        y_pred = np.argmax(y_pred, axis=1)
        print("accuracy score {:.2f}".format(self.accuracy_score(y, y_pred)*100))

    def accuracy_score(self, X, Y, *, normalize=True):
        '''
        Method to calculate the accuracy of the classifier
        '''
        return sum([1 if x == y else 0 for (x,y) in zip(X,Y)]) / (len(X) if normalize else 1)

if __name__ == "__main__":
    train_net = NeuralNetwork("training1.csv", learning_rate = 0.45)
    train_net.train_backprop(epochs=280)
    train_net.score("training1.csv")
    train_net.score("test1.csv")
