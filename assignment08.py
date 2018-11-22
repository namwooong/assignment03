import matplotlib.pyplot as plt
import numpy as np

file_data_train = "mnist_train.csv"
file_data_test  = "mnist_test.csv"

h_data_train    = open(file_data_train, "r")
h_data_test     = open(file_data_test, "r")

data_train      = h_data_train.readlines()
data_test       = h_data_test.readlines()

h_data_train.close()
h_data_test.close()

size_row    = 28    # height of the image
size_col    = 28    # width of the image

num_train   = len(data_train)   # number of training images
num_test    = len(data_test)    # number of testing images

#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)

#
# example of distance function between two vectors x and y
#
def distance(x, y):

    d = (x - y) ** 2
    s = np.sum(d)
    # r = np.sqrt(s)

    return(s)

#
# make a matrix each column of which represents an images in a vector form 
#
list_image_train    = np.empty((size_row * size_col, num_train), dtype=float)
list_label_train    = np.empty(num_train, dtype=int)

list_image_test     = np.empty((size_row * size_col, num_test), dtype=float)
list_label_test     = np.empty(num_test, dtype=int)

count = 0

for line in data_train:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label_train[count]     = label
    list_image_train[:, count]  = im_vector    

    count += 1

count = 0

for line in data_test:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label_test[count]      = label
    list_image_test[:, count]   = im_vector    

    count += 1

# 
# plot first 150 images out of 10,000 with their labels
# 
f1 = plt.figure(1)

for i in range(150):

    label       = list_label_train[i]
    im_vector   = list_image_train[:, i]
    im_matrix   = im_vector.reshape((size_row, size_col))

    plt.subplot(10, 15, i+1)
    plt.title(label)
    plt.imshow(im_matrix, cmap='Greys', interpolation='None')

    frame   = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

#plt.show()


#예제코드 수정 

f2 = plt.figure(2)

im_average = np.zeros((size_row * size_col, 2), dtype=float)
im_count = np.zeros(2, dtype=int)

for i in range(num_train):
    if list_label_train[i] == -1:
        im_average[:, 1] += list_image_train[:, i]
        im_count[1] += 1
    else:
        im_average[:, 0] += list_image_train[:, i]
        im_count[0] += 1
    
for i in range(2):
    im_average[:, i] /= im_count[i]
    
    plt.subplot(1, 2, i+1)
    plt.title(i)
    plt.imshow(
        im_average[:, i].reshape((size_row, size_col)), 
        cmap='Greys', interpolation='None'
    )
    
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    
plt.show()

def norm(v1):
    sum = 0
    for i in range(len(v1)):
        sum += v1[i]**2
    return np.sqrt(sum)

def proj(v1, v2):
    return (np.matmul(v1.T, v2) / np.matmul(v1.T, v1))*v1


def ans(matrix,b):
    matrix_0 = np.zeros([matrix.shape[0], matrix.shape[1]])
    matrix_1 = np.zeros([matrix.shape[1], matrix.shape[1]])
    matrix_2 = np.zeros([matrix.shape[1], matrix.shape[0]])
    
    for i in range(matrix.shape[1]):
        sum = 0
        matrix_i = matrix[:,i]
        for j in range(i):
            sum += proj(matrix_2[j], matrix_i)
        matrix_2[i] = matrix_i - sum
        
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            matrix_0[j][i] = matrix_2[i][j]/norm(matrix_2[i])
            
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            matrix_1[i][j] = np.dot(matrix_0[:,i], matrix[:,j])


    sol = np.matmul(matrix_0.T, b)
    ans = np.zeros(sol.shape)
    for i in reversed(range(sol.shape[0])):
        a = sol[i]
        for j in reversed(range(i+1, sol.shape[0])):
            a -= ans[j]*matrix_1[i][j]
        if matrix_1[i][i] == 0:
            ans[i] = 0
        else:
            ans[i] = a / matrix_1[i][i]
    return ans



answer=ans(list_image_train.T,list_label_train)


pred = np.sign(np.matmul(list_image_test.T, answer))

print("make list")
list_TF = [None for i in range(num_test)]
for i in range(num_test):
    if pred[i] == 1.0 and list_label_test[i] == 1:
        list_TF[i] = 'TP'
    elif pred[i] == -1.0 and list_label_test[i] == 1:
        list_TF[i] = 'FN'
    elif pred[i] == -1.0 and list_label_test[i] == -1:
        list_TF[i] = 'TN'
    elif pred[i] == 1.0 and list_label_test[i] == -1:
        list_TF[i] = 'FP'

print("(TP, FP, TN, FN) : ("
      +str(list_TF.count('TP'))+", "
      +str(list_TF.count('FP'))+", "
      +str(list_TF.count('TN'))+", "
      +str(list_TF.count('FN'))+")"
     )