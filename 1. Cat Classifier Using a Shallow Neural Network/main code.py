'''
A simple Cat Classifier using a 3-layer neural network

The model is trained on the dataset provided in course-1 of
the Deep Learning Specialization taught by Andrew Ng

the training set contains 209 images, 64 X 64 X 3 in size

the model, after training, averages 98% accuracy on the training set,
and varies between 56% - 80% on the test set, averaging to a 72% accuracy

By - Kushagra Agarwal
'''


import matplotlib.pyplot as plt
import numpy as np
import h5py 

np.random.seed(2)

def load_dataset():
    with h5py.File('train_catvnoncat.h5', 'r') as f:
        x_train_orig = np.array(f['train_set_x'])
        y_train_orig = np.array(f['train_set_y'])
    with h5py.File('test_catvnoncat.h5', 'r') as f:
        x_test_orig = np.array(f['test_set_x'])
        y_test_orig = np.array(f['test_set_y'])
    
    x_train = x_train_orig.reshape(x_train_orig.shape[0], -1).T / 255
    y_train = y_train_orig.reshape(-1, y_train_orig.shape[0])
    x_test = x_test_orig.reshape(x_test_orig.shape[0], -1).T / 255
    y_test = y_test_orig.reshape(-1, y_test_orig.shape[0])

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_dataset()
m = x_train.shape[1]
n_x = x_train.shape[0]
n_h1 = 10
n_h2 = 6
n_h3 = 4
n_y = y_train.shape[0]
cache = {}
dims = [n_x, n_h1, n_h2, n_h3, n_y]
learning_rate = 0.001
lambd = 0.01
L = len(dims)

def relu(input):
    return np.maximum(0, input)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params():
    for l in range(1, L):
        cache['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2 / dims[l-1])
        cache['b' + str(l)] = np.zeros((dims[l], 1))

def for_prop():
    cache['A0'] = x_train

    for l in range(1, L):
        Z = np.dot(cache['W' + str(l)], cache['A' + str(l-1)]) + cache['b' + str(l)]
        cache['Z' + str(l)] = Z
        if l == L-1:
            cache['A' + str(l)] = sigmoid(Z)
        else:
            cache['A' + str(l)] = relu(Z)

def compute_cost(AL, Y):
    cc_cost = (- (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)) / m)
    l2_sum = 0
    for l in range(1, L):
        l2_sum += np.sum(np.square(cache['W' + str(l)]))
    l2_cost = (lambd / (2 * m)) * l2_sum

    return cc_cost + l2_cost

def back_prop_step(A_prev, dZ, W):
    dW = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev

def back_prop(AL, Y):
    dZ = AL - Y
    for l in range(L - 1, 0, -1):
        dW, db, dA_prev = back_prop_step(cache['A' + str(l - 1)], dZ, cache['W' + str(l)])
        cache['W' + str(l)] -= learning_rate * dW
        cache['b' + str(l)] -= learning_rate * db
        if l > 1:
            dZ = dA_prev * (cache['Z' + str(l-1)] > 0)

def model(iters = 10000):
    initialize_params()
    costs = []
    for i in range(iters):
        for_prop()
        AL = cache['A' + str(L-1)]
        cost = compute_cost(AL, y_train)
        costs.append(np.squeeze(cost))
        back_prop(AL, y_train)
        if(i % 100 == 0):
            print("cost at iteration ", i , " : ", cost)
    return costs

costs = model()

plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Training cost over iterations")
plt.show()

def predict(X, Y):
    m_examples = X.shape[1]
    A_prev = X
    
    for l in range(1, L):
        Z = np.dot(cache['W' + str(l)], A_prev) + cache['b' + str(l)]
        if l == L - 1:
            A_prev = sigmoid(Z)
        else:
            A_prev = relu(Z)
    
    AL = A_prev
    predictions = (AL > 0.5).astype(int)
    accuracy = np.mean(predictions == Y) * 100
    
    return predictions, accuracy

train_preds, train_acc = predict(x_train, y_train)
test_preds, test_acc = predict(x_test, y_test)

print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%") 
