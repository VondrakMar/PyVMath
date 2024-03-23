import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def FT(x,fx, n_k,L): 
    dx = (x[1] - x[0])/L
    A = np.zeros(n_k)
    B = np.zeros(n_k)
    A0 = np.sum(f*np.ones_like(x))*dx
    fFS = A0/2
    for k in range(n_k):
        A[k] = np.sum(fx*np.cos(np.pi*(k+1)*x/L))*dx
        B[k] = np.sum(fx*np.sin(np.pi*(k+1)*x/L))*dx
        fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L)+ B[k]*np.sin((k+1)*np.pi*x/L)
    return fFS 


plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams.update({'font.size':18})

dx = 0.001
L = np.pi
x = L* np.arange(-1+dx,1+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))


f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

plt.plot(x,f)
res = FT(x,f,20,np.pi)
plt.plot(x,res)
res = FT(x,f,10,np.pi)
plt.plot(x,res)
plt.show()
