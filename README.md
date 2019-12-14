### NNML_Bot

### Implementing a basic neural network
Neuron is the basic building block of a neural network. Multiple neurons can be connected which form a neural network. Output of one neuron is treat as input for other neuron. The basic level implementation of neural network can be started with one neuron based on the complexity of the problem. 

<img src="https://faizan-tariq.github.io/NNML_Bot/NNML.png"/>

A highlevel diagram of a basic neuron is shown in the above picture. 
x1,x2,x3.. are the inputs to this neuron. Ypred is the predicted value which is the output of this neuron. Your input goes into the progression logic function, which returns an unbound output. Then unbounded output is passed through an activation function. 

### Sigmoid Activation function

The activation function is used to turn an unbounded input into an output that has a predictable form. A commonly used activation function is the sigmoid function. The sigmoid function only outputs numbers in the range (0,1). You can think of it as compressing (−∞,+∞) to (0,1). Big negative numbers become approx. 0, and big positive numbers become  approx. 1.

<img src="https://faizan-tariq.github.io/NNML_Bot/sigmoid.png"/>

