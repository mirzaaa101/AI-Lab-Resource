# connecting google colab with the google drive
from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
# randomly shuffle the dataset
np.random.shuffle(my_data)
data = my_data.tolist()

# calculating the size of split of each data set
train_size = int(0.7*len(data))
val_size = int(0.15*len(data))
test_size = len(data) - (train_size + val_size)
print(f"Size of Train_data,Val_data,Test_data:{train_size,val_size,test_size}")

# split the data into train_data,val_data,test_data
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

import numpy as np
# function to calculate Euclidean distance
def calc_euclidean_distance(lst1,lst2):
  # Convert the lists to NumPy arrays
  arr1 = np.array(lst1)
  arr2 = np.array(lst2)

  squared_distance = np.square(arr1-arr2)  # Calculate the squared distance between each corresponding coordinate
  distance = np.sqrt(np.sum(squared_distance)) # Sum the squared differences and take the square root
  return distance
  
  
 # validation
import math

Error = 0
train_data_copy = []

for V in val_data:
    for T in train_data:
        Vx = V.copy() # making copy of val_set with N-1 elements
        Tx = T.copy() # making copy of test_set with N-1 elements
        D = calc_euclidean_distance(Vx, Tx) # passing value of Vx, Tx with N-1 elements to calculate Euclidean distance
        T_new = T.copy() # copy T list into another new list
        T_new.append(D) # making new list with the distance value
        train_data_copy.append(T_new) # making new train data set with distance value

    T_sorted = sorted(train_data_copy, key=lambda x: x[-1])  # sorting the list based on the distance
    k = 5 # set the value of k
    T_KNN = T_sorted[:k] # taking first k neighbours
    determined_output = sum(x[10] for x in T_KNN)/k  # Take the average output of the K samples
    error = (V[10] - determined_output)**2  # Calculate the squared difference between true output and determined output
    Error += error  # Add error to the running total

mean_squared_error = Error/len(val_data)  # Divide the total error by the number of validation samples
validation_accuracy = math.sqrt(mean_squared_error)
print(f"Validation Accuracy: {validation_accuracy}")


# Test
import math

Error= 0
train_data_copy = []
for V in test_data:
  for T in train_data:
    Vx = V.copy() # making copy of val_set with N-1 elements
    Tx = T.copy() # making copy of test_set with N-1 elements
    D = calc_euclidean_distance(Vx,Tx) # passing value of Vx,Tx with N-1 elements to calculate Euclidean distance
    T_new = T.copy() # copy T list into another new list
    T_new.append(D) # making new list with the distance value
    train_data_copy.append(T_new) # making new train data set with distance value

  T_sorted = sorted(train_data_copy, key=lambda x: x[-1])  # sorting the list based on the distance
  k = 5 # set the value of k
  T_KNN = T_sorted[:k] # taking first k neighbours
  determined_output = sum(x[10] for x in T_KNN)/k  # Take the average output of the K samples
  error = (V[10] - determined_output)**2  # Calculate the squared difference between true output and determined output
  Error += error  # Add error to the running total

mean_squared_error = Error/len(test_data)  # Divide the total error by the number of validation samples
print(f"Validatio Accuracy = {math.sqrt(mean_squared_error)}")
