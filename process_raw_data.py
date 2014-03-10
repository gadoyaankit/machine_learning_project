import math 

train_file = open('train.csv','r')
test_file = open('test.csv','r')
sample_prediction = open("sample_prediction",'r')

# Strip the header from the training file
header = train_file.next().rstrip().split(',')

# Strip header from the testing file
header = test_file.next().rstrip().split(',')

# There are two users per data point.

# We divide data into two parts. First part is all users on first side of data point
# Second part is all users on second side of data point
 
# X_train_A =  list
# It list all users from first part
X_train_A = []

# X_train_B = list
# It list all users from second part
X_train_B = []

# y_train = list
# It list all the labels of the data
y_train = []

# Populate the data
for line in train_file:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train_A.append(A_features)
    X_train_B.append(B_features)

# close the file
train_file.close()

number_of_training_point = len(y_train)
number_of_feature = 11

# return the log of the number
# we add one to each number, to handle a case when number is zero

def get_log(number):
    return math.log(1+number)

# Now we have data such that, for each data point, we have two user and a single label.
# Label defines who is more influential among the given two user.

# This means, there is some difference between two users which makes one user more
# influential than other.

# Hence we take the difference of feature values of two users.

# We convert all the feature values to log scale.
# This makes processing some of the large feature value easier.

# X_train = list
# This contains the difference of feature value of two users 
X_train = []

# Populate X_train
for i in range(number_of_training_point):
    temp = []
    for j in range(number_of_feature): 
        temp.append(get_log(X_train_A[i][j]) - get_log(X_train_B[i][j]))
    X_train.append(temp)
    
# Write back the processed data to the file
processed_train_data = open('processed_train.txt','w')

for i in range(number_of_training_point):
    temp = str(X_train[i])
    temp = temp.strip("[")
    temp = temp.strip("]")
    processed_train_data.write(temp)
    processed_train_data.write(","+str(y_train[i]))
    processed_train_data.write("\n")

# ---------------------------------------------------------------------------------------------------------------------

# Now we repeat the same process for testing data.

# X_test_A =  list
# It list all users from first part
X_test_A = []

# X_test_B = list
# It list all users from second part
X_test_B = []

# y_test = list
# It list all the labels of the data
y_test = []


# Populate the data
for line in test_file:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
    
# Populate the testing labels
for label in sample_prediction:
    y_test.append(label)

test_file.close()
sample_prediction.close()

number_of_test_point = len(y_test)
print number_of_test_point 
# X_test = list
# This contains the difference of feature value of two users 

X_test = []

# Populate X_test
for i in range(number_of_test_point):
    temp = []
    for j in range(number_of_feature): 
        temp.append(get_log(X_test_A[i][j]) - get_log(X_test_B[i][j]))
    X_test.append(temp)
    
# Write back the processed data to the file
processed_test_data = open('processed_test.txt','w')

for i in range(number_of_test_point):
    temp = str(X_test[i])
    temp = temp.strip("[")
    temp = temp.strip("]")
    processed_test_data.write(temp)
    processed_test_data.write(","+str(y_test[i]))


