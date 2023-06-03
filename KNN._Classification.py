# connecting google colab with the google drive
from google.colab import drive
drive.mount('/content/gdrive')

# importing data from the google drive
from numpy import genfromtxt
data_path = '/content/gdrive/MyDrive/Trimester-09/AI Lab/Assignments/KNN/Dataset/iris.csv'
my_data = genfromtxt(data_path, delimiter=',')

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

def find_majority_class(lst):
    # variable to count max class
    class0_count = 0
    class1_count = 0
    class2_count = 0
    
    for sublist in lst:
        if sublist[4] == 0:
            class0_count += 1
        elif sublist[4] == 1:
            class1_count += 1
        else:
            class2_count += 1

    # Check which class has the maximum count
    if class0_count >= class1_count and class0_count >= class2_count:
        return 0.0
    elif class1_count >= class0_count and class1_count >= class2_count:
        return 1.0
    else:
        return 2.0
      
     
# validation
correct_result = 0
train_data_copy = []
for V in val_data:
  for T in train_data:
    Vx = V[:len(V)-1] # making copy of val_set with N-1 elements
    Tx = T[:len(T)-1] # making copy of test_set with N-1 elements
    D = calc_euclidean_distance(Vx,Tx) # passing value of Vx,Tx with N-1 elements to calculate Euclidean distance
    T_new = T.copy() # copy T list into another new list
    T_new.append(D) # making new list with the distance value
    train_data_copy.append(T_new) # making new train data set with distance value
    # print(train_data_copy)

  T_sorted = sorted(train_data_copy, key=lambda x: x[-1])  # sorting the list based on the distance
  k = 15 # set the value of k
  T_KNN = T_sorted[:k] # taking first k neighbours
  result = find_majority_class(T_KNN)  # finding which class gets more count
  # counting correct answer
  if T[4] == result:
      correct_result += 1

validation_accuracy = (correct_result/len(val_data))*100
print(f"Validatio Accuracy = {validation_accuracy}%")


# Test
correct_result = 0
train_data_copy = []
for V in test_data:
  for T in train_data:
    Vx = V[:len(V)-1] # making copy of val_set with N-1 elements
    Tx = T[:len(T)-1] # making copy of test_set with N-1 elements
    D = calc_euclidean_distance(Vx,Tx) # passing value of Vx,Tx with N-1 elements to calculate Euclidean distance
    T_new = T.copy() # copy T list into another new list
    T_new.append(D) # making new list with the distance value
    train_data_copy.append(T_new) # making new train data set with distance value
    # print(train_data_copy)

  T_sorted = sorted(train_data_copy, key=lambda x: x[-1])  # sorting the list based on the distance
  k = 15 # set the value of k
  T_KNN = T_sorted[:k] # taking first k neighbours
  result = find_majority_class(T_KNN)  # finding which class gets more count
  # counting correct answer
  if T[4] == result:
      correct_result += 1

validation_accuracy = (correct_result/len(test_data))*100
print(f"Test Accuracy = {validation_accuracy}%")
