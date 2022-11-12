import numpy as np
import pandas as pd
import random

# 2.1

x = np.zeros(100)
t = np.zeros(100)

for i in range(100):
    a = random.uniform(0,1)
    b = a**3-a**2
    x[i] = a
    t[i] = b
#print(x)
#print(t)


def ReLU(x):
  x[x<0] = 0
  return x

def ReLU_derivative(x):
  x[x>=0] = 1
  x[x<0] = 0 
  return x

def M(y, t):
  return 1/2 * ((y - t)**2)


#==================================================================
 #2.4




