import numpy as np
import math as m
import matplotlib.pyplot as plt


a = 50
h = 0.1
N = int(a / h - 1)
V = 1
sig = 1
to = 0.0025

alpha = V * to / (2 * h)

xe = a / 2
c = V * to / h
print(c)
T = 5

x = np.linspace(0, a, num = N)

def f(x):
    u = np.zeros(len(x))
    for i in range(len(x)):
        u[i] = (1/(sig * m.sqrt(2 * m.pi))) * m.exp(-((x[i] - xe)**2)/ (2*sig**2)) # Attention c'est bizarre 
    return u

u0 = f(x)
def EE(N):
  
    M = np.zeros((N, N))
    M[-1, -1] = 1
    for i in range(N - 1):
        M[i ,i] = 1
        M[i, i + 1] = - c / 2
        M[i + 1, i] = c / 2

    return M


n_final = int(T / to)
u = u0
for t in range(n_final):
    u = np.dot(EE(N), u)


plt.plot(x,u0)
plt.plot(x, u)
plt.show()

Me = EE(4)

M = (np.diag(np.ones(4)) - Me)*2/to

def CNG(N):
    M = np.zeros((N, N))
    M[-1, -1] = 1
    for i in range(N - 1):
        M[i ,i] = 1
        M[i, i + 1] = c / 2
        M[i + 1, i] = - c / 2

    return M

def CND(N):
    M = np.zeros((N, N))
    M[-1, -1] = 1
    for i in range(N - 1):
        M[i ,i] = 1
        M[i, i + 1] = - c / 2
        M[i + 1, i] = c / 2

    return M

for t in range(n_final):
    u = np.dot(CNG(N),CND(N)@u)

plt.plot(x,u0)
plt.plot(x, u)
plt.show()

print(u)
