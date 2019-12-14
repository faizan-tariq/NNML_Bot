import numpy as np

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

class BasicNeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        n1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        return n1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        revisions = 5000

        for revision in range(revisions):
            for x, y_true in zip(data, all_y_trues):
                sum_n1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                n1 = sigmoid(sum_n1)

                y_pred = n1

                # --- Calculate partial derivatives.
                d_L_d_ypred = -2 * (y_true - y_pred)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_n1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_n1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_n1)
                d_h1_d_b1 = deriv_sigmoid(sum_n1)

                # --- Update weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_h1_d_w3
                self.b1 -= learn_rate * d_L_d_ypred * d_h1_d_b1

                # --- Calculate total loss at the end of each epoch

                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Progression %d loss: %.3f" % (revision, loss))

# [oop, db, os]

# Define dataset
data = np.array([
    [5, 5, 5],     # Recommended
    [4, 4, 4],     # Recommended
    [2, 2, 2],     # Not Recommended
    [1, 1, 1],     # Not Recommended
    [0, 0, 0],     # Not Recommended
    [0, 5, 5],     # Not Recommended
    [5, 3, 3],     # Recommended
    [4, 1, 1],     # Not Recommended
])
all_y_trues = np.array([
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0
])

# Train neural network!
network = BasicNeuralNetwork()
network.train(data, all_y_trues)


not_recommended = np.array([2, 2, 3])
not_recommended1 = np.array([1, 3, 3])
recommended = np.array([4, 3, 3])
recommended1 = np.array([5, 4, 3])

print("Not Recommended: %.3f" % network.feedforward(not_recommended))
print("Not Recommended: %.3f" % network.feedforward(not_recommended1))
print("Recommended: %.3f" % network.feedforward(recommended))
print("Recommended: %.3f" % network.feedforward(recommended1))
