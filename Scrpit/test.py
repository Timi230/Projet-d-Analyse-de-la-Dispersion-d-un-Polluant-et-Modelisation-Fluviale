import numpy as np
from matplotlib import pyplot as plt

h = 0.1
tau = 0.0025
a = 50
T = 10
V = 1
N = int(a / h - 1)
sigma = 1
mu = 1


def f(N):
    print(T)
    x     = np.linspace(0,a,N+1)
    n_max = int(T/tau)
    print(n_max)
    t     = np.linspace(0,n_max*tau,n_max+1)
    U     = np.zeros((n_max+1,N+1))
    U[0,:] = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-a/2)**2)/(2*sigma**2))
    un = U[0,: N]

    return x, n_max, t, U, un 

def mat_I(N):
    return np.eye(N)



def mat_M(N):
    M = np.zeros((N,N))
    M += np.diagflat(-1*np.ones((N-1,1)),-1) + np.diagflat(1*np.ones((N-1,1)),1)
    M[:,-1] = M[:,-2]*h
    M = M*(1/(2*h))
    return M

def mat_A(N):
    A = -2 * np.eye(N)
    A += np.diagflat(-1*np.ones((N-1,1)),-1)  + np.diagflat(1*np.ones((N-1,1)),1)
    A[:,-1] = A[:,-2]*h
    A = A * (1/h**2)
    return A

def mat_F(N):
    F = np.zeros((n_max + 1, N))  # Initialisation de la matrice F
    x = np.linspace(0, a, N + 1)  # Grille des positions
    # Remplissage des coefficients de la matrice F
    for n in range(n_max):
        for j in range(N):
            f_nj = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x[j] - a/2)**2) / (2 * sigma**2))
            F[n, j] = f_nj
    
    return F


x, n_max, t, U, un = f(N)
c = (V * tau) / 2
d = (mu * tau) / 2

M = mat_M(N)
A = mat_A(N)
In = mat_I(N)

Fnj = mat_F(N)

for n in range(n_max):
    un = np.linalg.solve((In + c * M - d * A), (In - c * M + d * A) @ (np.add(un,Fnj[0,:])))
    U[n+1,:N] = un
    U[n+1,N] = U[n+1,0]


l_representee = np.linspace(0,5,20)


x = np.linspace(0,a,N+1)
u_ref = 1/ (sigma * np.sqrt(2*np.pi)) * np.exp(-(x - a/2)**2 / (2*sigma**2))
l_representee = np.linspace(0,5,100)
errors = []

for t in l_representee:
    num = int(t/tau)
    u = U[num,:]
    error = np.linalg.norm(u - u_ref*t)
    errors.append(error)

p = np.log(np.divide(errors[1:] , errors[:-1])) / np.log(np.divide(l_representee[1:] , l_representee[:-1]))
plt.plot(l_representee[1:],p)
plt.xlabel("t")
plt.ylabel("Vitesse de convergence")
plt.show()

