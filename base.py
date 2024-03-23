import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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

fig, ax = plt.subplots()
ax.plot(x,f,"-",color="k")

# ax.scatter(x,f,color="k")

name = "Accent"
cmap = get_cmap("tab10")
colors = cmap.colors
ax.set_prop_cycle(color=colors)
A0 = np.sum(f*np.ones_like(x))*dx
fFS = A0/2

n_k = 10
A = np.zeros(n_k)
B = np.zeros(n_k)

for k in range(n_k):
    A[k] = np.sum(f*np.cos(np.pi*(k+1)*x/L))*dx
    B[k] = np.sum(f*np.sin(np.pi*(k+1)*x/L))*dx
    fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L)+ B[k]*np.sin((k+1)*np.pi*x/L)
ax.plot(x,fFS,'-')

plt.show()

