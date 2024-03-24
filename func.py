import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams.update({'font.size':18})
plt.rcParams["figure.figsize"] = [8,8]


def FT(x,fx, n_k,L,dx): 
#    dx = (x[1] - x[0])/L
    A = np.zeros(n_k)
    B = np.zeros(n_k)
    A0 = np.sum(fx*np.ones_like(x))*dx
    fFS = A0/2
    for k in range(n_k):
        A[k] = np.sum(fx*np.cos(np.pi*(k+1)*x/L))*dx
        B[k] = np.sum(fx*np.sin(np.pi*(k+1)*x/L))*dx
        fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L)+ B[k]*np.sin((k+1)*np.pi*x/L)
    return fFS 



dx = 0.001
L = np.pi
x = L* np.arange(-1+dx,1+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))
f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

plt.plot(x,f)
res = FT(x,f,20,np.pi,dx)
plt.plot(x,res)
res = FT(x,f,10,np.pi,dx)
plt.plot(x,res)
# plt.show()
plt.close()


def some_function(x):
    A = np.sin(3*x)
    B = np.cos(4*x)
    C = 2*np.sin((2*x)-(np.pi/4))
    D = np.cos((12.0*x)+(np.pi/3))
    return A + B + C + D

print(some_function(0))
print(some_function(2*np.pi))
# x_trig = np.arange(-2*np.pi,2*np.pi,0.001)
dx = 0.00001
x_trig = 2*np.pi*np.arange(-1+dx,1-dx,0.00001)
y_trig = some_function(x_trig)
print(len(x_trig),len(y_trig))


plt.plot(x_trig,y_trig)
res = FT(x_trig,y_trig,5,2*np.pi,dx)
plt.plot(x_trig,res)
res = FT(x_trig,y_trig,10,2*np.pi,dx)
plt.plot(x_trig,res)
res = FT(x_trig,y_trig,15,2*np.pi,dx)
plt.plot(x_trig,res)
res = FT(x_trig,y_trig,30,2*np.pi,dx)
plt.plot(x_trig,res)
res = FT(x_trig,y_trig,50,2*np.pi,dx)
plt.plot(x_trig,res)
res = FT(x_trig,y_trig,100,2*np.pi,dx)
plt.plot(x_trig,res)
plt.show()

print(np.cos(np.pi))
