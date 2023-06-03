from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

print(X)
print(X.shape)
print(y.shape)

data = X.tolist()
for x in data:
  x.append(1)

X = np.array(data)

# randomly shuffle the dataset
np.random.shuffle(X)
data = X.tolist()

# calculating the size of split of each data set
train_size = int(0.7*len(data))
val_size = int(0.15*len(data))
test_size = len(data) - (train_size + val_size)
print(f"Size of Train_data,Val_data,Test_data:{train_size,val_size,test_size}")

# split the data into train_data,val_data,test_data
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)

train_data = np.array(train_data)
theta = np.random.rand(train_data.shape[1], 1)
#train_data = train_data.tolist()

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
  
import numpy as np

def log_loss(y_true, y_pred):
    epsilon = 1e-15 # to prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # clip the predicted values to prevent log(0)
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return loss.mean()
  
  
 
train_loss = []
for i in range(1000):
    TJ = 0
    for sample, label in zip(train_data, y_train):
        # Compute the predicted output and the loss
        z = np.dot(sample, theta)
        h = sigmoid(z)
        J = log_loss(label, h)
        TJ += J
        
        # Compute the derivative of the loss with respect to theta
        dv = np.dot(sample.reshape(-1, 1), (h - label).reshape(1, -1))
        
        # set learing rate
        lr = 0.001

        # Update theta using gradient descent
        theta -= lr * dv
        
    TJ /= len(train_data)
    
    # Print the average loss for this iteration
    #print(f"Iteration {i + 1}: TJ = {TJ:.6f}")
    train_loss.append(TJ)
    
    
import matplotlib.pyplot as plt

# train loss function
iterations = range(1, 1001)
plt.plot(iterations, train_loss)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Iteration')
plt.show()


# validation
correct = 0
for V, y_true in zip(val_data, y_val):
    z = np.dot(V, theta)
    h = sigmoid(z)

    if h >= 0.5:
        h = 1
    else:
        h = 0
    
    if h == y_true:
        correct += 1
        
        
# validation accuracy
val_acc = (correct/len(val_data)) * 100
print(f"Validation Accuracy:{val_acc}")

# test
correct = 0
for V, y_true in zip(test_data, y_test):
    z = np.dot(V, theta)
    h = sigmoid(z)

    if h >= 0.5:
        h = 1
    else:
        h = 0
    
    if h == y_true:
        correct += 1
        
        
# test accuracy
val_acc = (correct/len(test_data)) * 100
print(f"Test Accuracy:{val_acc}")
