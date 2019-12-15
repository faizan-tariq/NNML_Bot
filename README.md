### NNML_Bot (For Beginners)

### Goal
The goal is to build an AI based machine learning bot using Neutral Network which has the capability of rating an interviewee as Recommendded or Not Recommended based on the individual scoring of different departments as input. 

### Implementing a basic neural network
Neuron is the basic building block of a neural network. Multiple neurons can be connected which form a neural network. Output of one neuron is treat as input for other neuron. The basic level implementation of neural network can be started with one neuron based on the complexity of the problem. 

<img src="https://faizan-tariq.github.io/NNML_Bot/NNML.png"/>

A highlevel diagram of a basic neuron is shown in the above picture. 
x1,x2,x3.. are the inputs to this neuron. Ypred is the predicted value which is the output of this neuron. Your input goes into the progression logic function, which returns an unbound output. Then unbounded output is passed through an activation function. 

### Sigmoid Activation function
The activation function is used to turn an unbounded input into an output that has a predictable form. A commonly used activation function is the sigmoid function. The sigmoid function only outputs numbers in the range (0,1). You can think of it as compressing (−∞,+∞) to (0,1). Big negative numbers become approx. 0, and big positive numbers become  approx. 1.

<img src="https://faizan-tariq.github.io/NNML_Bot/sigmoid.png"/>


### Feed Forward
Inputs to neurons are actually feed, and forwarding feed to other neurons in the network or fowarding feed to the same neuron while progressions is feed forward. Once your neuron / neural netowrk is trained, then you can feed forward to predict the output.

### Loss
Training a neural network is actually trying to minimize its loss.
We need a way to quantify how good it is so that it can try to do better. We will use the mean squared error (MSE) for this.

### Stochastic Gradient Descent
In order to train neural network we need an optimization algorithm called stochastic gradient descent (SGD) that tells us how to change our weights and biases to minimize loss which is based on the principle of partial derivatives. 

### Traning
The exammple consist of 1 neuron, which predicts the status of a candidate in an interview, i.e. either Recommended or Not Recommended based on 3 departments OOP, DB and OS by marking score out of 5 in each. The data is more biased towards OOP which has high priority among other departments. Based on the data our neural network will get trainned in a way that it will learn that OOP has high weightage among others (Do notice there is no hardcoded checks or conditonal logic defined in the code to prioritize OOP over others but still our system will learn this based on the given data)

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

### Future Goal
Notice that this neural network accepts individual score of an interviewee in 3 different department. We still need an manual effort of asking questions and deciding which answer is right or wrong to give score in each deptartment. I am planning to automate this question asking part of an interview and connect this neural network created above with NLP based expert system which has a knowledge pool of some true/false questions and will ask the questions and give rating. 

*Stay Tunned :)* 
