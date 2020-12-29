<center><h1>Machine Learning</h1></center>

Implementation of 

1) Softmax Regression  

2) Feed Forward Neural Network 

for classifying OCR hand-written Digits.

> # Softmax Regression

## *Overview*

------

*Softmax Regression* is a generalization of logistic regression that we can use for multi-class classification. In *Softmax Regression* (SMR), we replace the sigmoid logistic function by the so-called *softmax* function. We use the *Logistic Regression* model in binary classification tasks

![](img\Capture.PNG)

## *Understanding*

------

The softmax function computes the probability of the training sample x<sup>(i)</sup> belonging to class j given the weight and net input z<sup>(i)</sup> . So, we compute the probability p(y=j∣x<sup>(i)</sup> ;w(j) for each class label in j=1,…,k. The normalization term in the denominator which causes these class probabilities to sum up to one.

### *Formulas Used*

1. **Update Rule for weight vector**

   ![](img\Capture2.PNG)

   where η is the learning rate, w<sub>j</sub> is the weight matrix and, the cost derivative is 

   ![](img\Capture.3PNG.PNG)

   where O<sub>i</sub> is the output vector and T<sub>i</sub> is the target vector.

   and for updating bias vector, the following formula is user 

   ![](img\Capture4.PNG)

2. **Cross Entropy function**

   ![](img\Capture5.PNG)

## *Procedure*

***Libraries Used* :** Numpy and csv

1. **Reading the data : ** The train.csv is read and the data is split into feature vectors array and respective class labels array.
2. **Training :** The theta matrix is started off as a zero matrix. Using the update formula the theta matrix is kept on updating until a certain number of epochs are reached.
3. **Prediction :**  Given the input feature vector, select the class which has the maximum probability among the probabilities  vector returned by the softmax function.
4. **Testing(Accuracy) :** If the prediction matches with the class of the feature vector in the test set, the prediction is correct. The performance of the classifier is tested by taking the number of correct predictions and dividing them by the total number of predictions. This is critical in understanding the veracity of the model.

## *Results*

**Stopping Criterion** : Number of epochs

1. **Training Set Accuracy** : 0.94

2. **Test Set Accuracy** : 0.91

   

> # Feed Forward Neural Network

## *Overview*

A Feed Forward Neural Network is an artificial neural network in which the connections between nodes does not form a cycle. The feed forward model is the simplest form of neural network as information is only processed in one direction.

## *Understanding*

The feedforward model is so called because information ﬂows through the function being evaluated from **x**, through the intermediate computations used to deﬁne f, and ﬁnally to the output y. There are no feedback connections in which outputs of the model are fed back into itself.

### *Formulas Used*

1. Cost Error

   ![](img\Capture6.PNG)

   Backpropagation implementation using chain rule

   ![](img\unknown (1).png)

2. Activation Function

   ![](img\unknown.png)

3. Hypothesis Function

   <img src="img\Capture7.PNG" height=200>

## *Procedure*

***Libraries Used* :** Numpy and panda

1. **Reading the data : ** The train.csv is read and the data is split into feature vectors array and respective class labels array.
2. **Training :** First the input is multiplied with weights and then they are passed to the activation function and then the output is feed forwarded to the next layer. Then according to the error the weights are balanced using the backpropagation algorithm.
3. **Prediction :**  Given the input feature vector, the class cell with the highest score in the output layer is selected.
4. **Testing(Accuracy) :** If the prediction matches with the class of the feature vector in the test set, the prediction is correct. The performance of the classifier is tested by taking the number of correct predictions and dividing them by the total number of predictions. This is critical in understanding the veracity of the model.

## *Results*

**Stopping Criterion** : Number of epochs

1. **Training Set Accuracy** : 87.91 %
2. **Test Set Accuracy** : 87.84 %



> # Comparison

**Softmax Regression** 

------

*Performance* :>  Training accuracy : 0.94       Testing Accuracy : 0.91

*Stopping Criterion* :> Number Of Epochs = 500 		Learning Rate = 0.1

------

**Feed Forward Neural Network**

------

*Performance* :>  Training accuracy : 0.94       Testing Accuracy : 0.91

*Stopping Criterion* :> Number Of Epochs = 280 		Learning Rate = 0.45

------

For the given dataset softmax regression classifier is performing better than the feed forward neural network. The Number of epochs for Softmax is more so it fits better to the given data compared to the Feed forward neural network, but we know in general FFNN should perform better.  Better performance of FFNN can be achieved if number of epochs is increased, but it will increase the training time.

------



