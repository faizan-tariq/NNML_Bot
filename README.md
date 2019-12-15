## NNML_Bot (For Beginners)

### Goal
The goal is to build an AI based Machine Learning bot using Neutral Network which has the capability of rating an interviewee as Recommendded or Not Recommended based on the individual scoring of different departments as input. 

### Implementing a basic neural network
Neuron is the basic building block of a neural network. Multiple neurons can be connected which form a neural network. Output of one neuron is treat as input for other neuron. The basic level implementation of neural network can be started with one neuron based on the complexity of the problem. 

<img src="https://faizan-tariq.github.io/NNML_Bot/NNML.png"/>

A highlevel diagram of a basic neuron is shown in the above picture. 
x1,x2,x3.. are the inputs to this neuron. Ypred is the predicted value which is the output of this neuron. Your input goes into the progression logic function, which returns an unbound output. Then unbounded output is passed through an activation function. 

### Sigmoid Activation function
The activation function is used to turn an unbounded input into an output that has a predictable form. A commonly used activation function is the sigmoid function. The sigmoid function only outputs numbers in the range (0,1). You can think of it as compressing (−∞,+∞) to (0,1). Big negative numbers become approx. 0, and big positive numbers become  approx. 1.

<img src="https://faizan-tariq.github.io/NNML_Bot/sigmoid.png"/>

````
def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
````

### Feed Forward
Inputs to neurons are actually feed, and forwarding feed to other neurons in the network or fowarding feed to the same neuron while progressions is feed forward. Once your neuron / neural netowrk is trained, then you can feed forward to predict the output.
````
def feedforward(self, x):
        # x is a numpy array with 3 elements.
        n1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        return n1
````
*As you can see above we have w1, w2, w3 and x0, x1, x2. Since for now we have limited our scope to only 3 departments i.e. OOP, DB and OS, so we have 3 weightages and input defined in our neuron. If you want to add more departments then you need same amount of weightages and input variables in your system.*

### Loss
Training a neural network is actually trying to minimize its loss.
We need a way to quantify how good it is so that it can try to do better. We will use the mean squared error (MSE) for this.
````
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()
````

### Stochastic Gradient Descent
In order to train neural network we need an optimization algorithm called stochastic gradient descent (SGD) that tells us how to change our weights and biases to minimize loss which is based on the principle of partial derivatives. You need to calculate the partial derivatives of sigmoid function against each entity to get the rate of change.  
````
def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)
````


### Traning
The exammple consist of 1 neuron, which predicts the status of a candidate in an interview, i.e. either Recommended or Not Recommended based on 3 departments OOP, DB and OS by marking score out of 5 in each. The data is more biased towards OOP which has high priority among other departments. Based on the data our neural network will get trainned in a way that it will learn that OOP has highest weightage among others (Do notice there is no hardcoded checks or conditonal logic defined in the code to prioritize OOP over others but still our system will learn this behavior based on the given data)

````
# [oop, db, os]
    [5, 5, 5],     # Recommended
    [4, 4, 4],     # Recommended
    [2, 2, 2],     # Not Recommended
    [1, 1, 1],     # Not Recommended
    [0, 0, 0],     # Not Recommended
    [0, 5, 5],     # Not Recommended
    [5, 3, 3],     # Recommended
    [4, 1, 1],     # Not Recommended
````

* Entry [0, 5, 5] and [5, 3, 3] are the ones which are deciding that OOP has highest priority over other department.* 

Following is the list of output of training samples above, this output is also part of traning data for our nerual network and it will be compared with the predicted value and help in loss calculation and refinement. 
0 means not recommend and 1 means recommend.
````
[
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0
]
````


### Predictions
Input to test is as follows:
````
[2, 2, 3] # Not Recommended
[1, 3, 3] # Not Recommended
[4, 3, 3] # Recommended
[5, 4, 3] # Recommended
````
Output approx. near to 0 (means not recommend) and output apporaching 1 (means recommend).
````
Not Recommended: 0.007
Not Recommended: 0.008
Recommended: 0.752
Recommended: 0.995
````
*The prediction is 100% perfect. But you will notice that 0.752 and 0.995 both lies in Recommended bracket, the reason to this is because based on their respective input data [5,4,3] is better than [4,3,3] based on the fact that OOP has highest weightage over others which our system induced based on the training data. If you want this weightage to be changed you can change your training data and your system will learn it and will change its behavior accordingly. Whatever the training you give to your neuron, it will learn and develop itself based on that.*

## Future Goal
**The Ultimate Goal is to automate an interview activity for basic screening.** Notice that this neural network accepts individual score of an interviewee in 3 different departments. We still need a manual effort of asking questions and deciding which answer is right or wrong to give score in each deptartment. I am planning to automate this question asking part of an interview and **connect this Neural Network** created above with **NLP based Expert System** which has a **Knowledge Pool** of some true/false questions and will ask the questions and give score which will act as input to this neural netowrk which gives the final Recommendation. 

*Stay Tunned :)* 
