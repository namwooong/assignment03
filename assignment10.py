import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla


def normalize(data):
    data_normalized = (data  - min(data)) / (max(data) - min(data))
    return data_normalized

def norm(vector):
    sum = 0
    for i in range(len(vector)):
        sum += vector[i]**2
    return np.sqrt(sum)

def proj(e, a):
    return (np.matmul(e.T, a) / np.matmul(e.T, e))*e

def randomExtractor(p):
    extractor = np.random.normal(0, 1, (p, 784))
    return extractor


file_data_train = "mnist_train.csv"
file_data_test = "mnist_test.csv"

h_data_train = open(file_data_train, 'r')
h_data_test = open(file_data_test, 'r')

data_train = h_data_train.readlines()
data_test = h_data_test.readlines()

h_data_train.close()
h_data_test.close()

size_row = 28 # height of the image
size_col = 28 # width of the image

num_train = len(data_train) # number of training images
num_test = len(data_test) # number of testing images




def makeTrainTestSet(i, data_train, data_test):
    #
    # make a matrix each column of which represents an images in a vector form
    #

    list_image_train = np.empty((size_row * size_col, num_train), dtype=float)
    list_label_train = np.empty(num_train, dtype=int)

    list_image_test = np.empty((size_row * size_col, num_test), dtype=float)
    list_label_test = np.empty(num_test, dtype=int)

    count = 0

    for line in data_train:
        line_data = line.split(',')
        label = line_data[0]

        if label == str(i):
            label = 1
        else:
            label = -1

        im_vector = np.asfarray(line_data[1:])
        im_vector = normalize(im_vector)

        list_label_train[count] = label
        list_image_train[:, count] = im_vector

        count += 1

    count = 0

    for line in data_test:
        line_data = line.split(',')
        label = line_data[0]

        if label == str(i):
            label = 1
        else:
            label = -1

        im_vector = np.asfarray(line_data[1:])
        im_vector = normalize(im_vector)

        list_label_test[count] = label
        list_image_test[:, count] = im_vector

        count += 1
        
    return list_image_train, list_label_train, list_image_test, list_label_test


def findX(Q, R, b):
    Rsol = np.matmul(Q.T, b)
    sol = np.zeros(Rsol.shape)
    for i in reversed(range(Rsol.shape[0])):
        a = Rsol[i]
        for j in reversed(range(i+1, Rsol.shape[0])):
            a -= sol[j]*R[i][j]
        if R[i][i] == 0:
            sol[i] = 0
        else:
            sol[i] = a / R[i][i]
    return sol

def F1score(list_TFcount):
    precision = list_TFcount[0] / (list_TFcount[0] + list_TFcount[3])
    recall = list_TFcount[0] / (list_TFcount[0] + list_TFcount[1])
    return 2*((precision * recall) / (precision + recall))

def TF(RE, sol, image_test, label_test):
    Test = np.matmul(RE, image_test)
    pred = np.sign(np.matmul(Test.T, sol))

    list_TF = [None for i in range(num_test)]
    for i in range(num_test):
        if pred[i] == 1.0 and label_test[i] == 1:
            list_TF[i] = 'TP'
        elif pred[i] == -1.0 and label_test[i] == 1:
            list_TF[i] = 'FN'
        elif pred[i] == -1.0 and label_test[i] == -1:
            list_TF[i] = 'TN'
        elif pred[i] == 1.0 and label_test[i] == -1:
            list_TF[i] = 'FP'

    TFcount = [
        list_TF.count('TP'), list_TF.count('FP'), 
        list_TF.count('TN'), list_TF.count('FN')
    ]
    print("(TP, FP, TN, FN) :", TFcount)
    
    F1 = F1score(TFcount)
    print("F1 score :", F1)

    return list_TF, F1



preds = []

for i in range(10):
    img_tr, lbl_tr, img_te, lbl_te = makeTrainTestSet(i, data_train, data_test)
    
    extractor = randomExtractor(818)
    A = np.matmul(extractor, img_tr)
    Q, R = np.linalg.qr(A.T)
    sol = findX(Q, R, lbl_tr)
    list_TF, F1 = TF(extractor, sol, img_te, lbl_te)
    
    Test = np.matmul(extractor, img_te)
    p = np.matmul(Test.T, sol)
    preds.append(p)
    print("Finished", i)