import numpy as np 
import pandas as pd 
import scipy.optimize as opt
import matplotlib.pyplot as plt 

path = r"C:\Users\aboub\OneDrive\Desktop\ML_class\ex2\ex2data1.txt"

data = pd.read_csv(path,header=None,names=['First Exam','Second Exam','Admitted'])

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# print(f"Admitted = \n {positive} \n")
# print(f"Nonadmitted = \n {negative} \n")

fig , ax = plt.subplots(figsize=(8,5))

ax.scatter(positive['First Exam'],positive['Second Exam'],
          s=50, c='g',marker='x',label="Admitted")
ax.scatter(negative['First Exam'],negative['Second Exam'],
          s=50, c='b',marker='o',label="Not Admitted")

ax.legend()
ax.set_xlabel('First exam score')
ax.set_ylabel('Second exam score')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

nums = np.arange(-10,10,step=1)

fig , ax = plt.subplots(figsize=(8,5))
ax.plot(nums,sigmoid(nums),'r')

data.insert(0,'ONES',1)
cols = data.shape[1]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

x = np.array(x.values)
y = np.array(y.values)
theta = np.zeros(data.shape[1]-1)

def cost_function(theta,x,y):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    first  = np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    return np.sum(first - second) / len(x)

cost = cost_function(theta,x,y)
print(f"cost : \n {cost}")

def gradient_descent(theta,x,y):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(x*theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error,x[:,i])
        grad[i] = np.sum(term) / len(x)
        
    return grad

result = opt.fmin_tnc(func=cost_function,x0=theta,fprime=gradient_descent,args=(x,y))
print(result)

cost_function_after = cost_function(result[0],x,y)
print(cost_function_after)

def prediction(theta,x):
    probability = sigmoid(x*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = prediction(theta_min,x)

print(f"The prediction : \n {predictions} \n")

correct = [
            1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 
            for (a,b) in zip(predictions,y)
          ]
accuracy = (sum(map(int,correct))%len(correct))
print(f'Accuracy is {accuracy}%')