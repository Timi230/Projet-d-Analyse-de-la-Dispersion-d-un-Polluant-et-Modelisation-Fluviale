import numpy as np
from matplotlib import pyplot as plt

h = 0.1
tau = 0.0025
a = 50
T = 5
V = 1
N = int(a / h - 1)
sigma = 1
mu =1 

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


def SolutionCN_4(N):

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

    return x,t,U, Fnj


def SolutionCN_5(N):

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

    def mat_F1(N):
        F = np.zeros((n_max + 1, N))  # Initialisation de la matrice F
        x = np.linspace(0, a, N + 1)  # Grille des positions
        
        # Remplissage des coefficients de la matrice F
        for n in range(n_max):
            for j in range(N):

                if n%2 == 0 :
                    f_nj = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x[j] - a/2)**2) / (2 * sigma**2))
                    
                else :
                    f_nj = 0

                F[n, j] = f_nj
                
        return F
    

    x, n_max, t, U, un = f(N)
    c = (V * tau) / 2
    d = (mu * tau) / 2

    M = mat_M(N)
    A = mat_A(N)
    In = mat_I(N)

    Fnj = mat_F1(N)
    
    for n in range(n_max):
        un = np.linalg.solve((In + c * M - d * A), (In - c * M + d * A) @ (np.add(un,Fnj[0,:])))
        U[n+1,:N] = un 
        U[n+1,N] = U[n+1,0]

    return x,t,U, Fnj


xCN4,tCN4,uCN4, Fnj4 = SolutionCN_4(N)
xCN5,tCN5,uCN5, Fnj5 = SolutionCN_5(N)



# l= [0,0.5,1,1.5,2]
# l= [0,1,2,3,4,5,6,7,8,9,10]

# for i in tCN4:
#     plt.plot(xCN4,uCN4[int(i/tau),:],label="t="+str(tCN4))

# plt.title("Crank-Nicolson")
# plt.show()