import numpy as np
import matplotlib.pyplot as plt

num     = 1001
std     = 5 

# x  : x-coordinate data
# y1 : (noisy) y-coordinate data
# y2 : (clean) y-coordinate data 

def fun(x):
 
	# f = np.sin(x) * (1 / (1 + np.exp(-x))) 
	f = np.abs(x) * np.sin(x)

	return f

def norm(v1):
    sum = 0
    for i in range(len(v1)):
        sum += v1[i]**2
    return np.sqrt(sum)

def proj(v1, v2):
    return (np.matmul(v1.T, v2) / np.matmul(v1.T, v1))*v1

def ans(x, p, y1):
    matrix_0 = np.array([[x1**pi for pi in range(p)] for x1 in x])
    matrix_1 = np.zeros([matrix_0.shape[0], matrix_0.shape[1]])
    matrix_2 = np.zeros([matrix_0.shape[1], matrix_0.shape[1]])
    matrix_3 = np.zeros([matrix_0.shape[1], matrix_0.shape[0]])
    
    for i in range(matrix_0.shape[1]):
        sum = 0
        matrix_i = matrix_0[:,i]
        for j in range(i):
            sum += proj(matrix_3[j], matrix_i)
        matrix_3[i] = matrix_i - sum
        
    for i in range(matrix_0.shape[1]):
        for j in range(matrix_0.shape[0]):
            matrix_1[j][i] = matrix_3[i][j]/norm(matrix_3[i])
            
    for i in range(matrix_0.shape[0]):
        for j in range(i, matrix_0.shape[1]):
            matrix_2[i][j] = np.dot(matrix_1[:,i], matrix_0[:,j])

    Rsol = np.matmul(matrix_1.T, y1)
    sol = np.zeros(Rsol.shape)
    for i in reversed(range(Rsol.shape[0])):
        a = Rsol[i]
        for j in reversed(range(i+1, Rsol.shape[0])):
            a -= sol[j]*matrix_2[i][j]
        sol[i] = a / matrix_2[i][i]
    
    f = 0
    for i, j in enumerate(sol):
        f += j * (x**i)
    return f
  

def calLoss(f, y):
    if f.shape != y.shape: 
        return np.inf
    sum = 0
    for (y1, y2) in zip(f, y):
        sum += (y1 - y2)**2
    return sum


n       = np.random.rand(num)
nn      = n - np.mean(n)
x       = np.linspace(-10,10,num)
y1      = fun(x)
y2      = y1 + nn * std



for i in range(1, 11):
   
    f = ans(x,i,y1)
    
    plt.xlim(-11,11)
    plt.ylim(-11,11)
    plt.plot(x, f, 'b.', x, y2, 'k.')
    plt.title(str(i))
    plt.show()
    
    print("Loss : "+str(calLoss(f, y2)))

