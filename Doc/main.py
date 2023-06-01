from matplotlib import pyplot as plt
import numpy as np

h = 0.02
tau = 0.01
t_max = 2
V = 1/2
N = int(1/h)


#---------------- Explicite-décenté --------------------#

def ED(N,c):
    Me = np.zeros((N,N))
    uns = np.ones((N,1))
    uns_d = np.ones((N-1,1))
    Me += (1-c) * np.diagflat(uns) + c * np.diagflat(uns_d,-1)
    Me[0,N-1] = c   
    return Me

def SolutionED(N):
    x = np.linspace(0,1,N+1)
    n_max = int(t_max/tau)
    t = np.linspace(0,n_max*tau,n_max+1)
    U = np.zeros((n_max+1,N+1))
    U[0,:] = np.sin(np.pi*x)**6
    un = U[0,:N]
    c = V * tau / h
    Me = ED(N,c)
    for n in range(n_max):
        un = Me @ un
        U[n+1,:N] = un
        U[n+1,N] = U[n+1,0]

    return x,t,U


#---------------- Explicite-centré --------------------#

def EC(N,c):
    Me = np.zeros((N,N))
    uns = np.ones((N,1))
    d_c = -c/2 * np.ones((N-1,1))
    Me += np.diagflat(uns) - np.diagflat(d_c,-1) + np.diagflat(d_c,1)
    Me[0,N-1] = c/2
    Me[N-1,0] = -c/2
    print(Me)
    return Me

def SolutionEC(N):
    x = np.linspace(0,1,N+1)
    n_max = int(t_max/tau)
    t = np.linspace(0,n_max*tau,n_max+1)
    U = np.zeros((n_max+1,N+1))
    U[0,:] = np.sin(np.pi*x)**6
    un = U[0,:N]
    c = V * tau / h
    Me = EC(N,c)
    for n in range(n_max):
        un = Me @ un
        U[n+1,:N] = un
        U[n+1,N] = U[n+1,0]

    return x,t,U

#---------------- implicite-centré --------------------#

def SolutionIC(N):
    x = np.linspace(0,1,N+1)
    n_max = int(t_max/tau)
    t = np.linspace(0,n_max*tau,n_max+1)
    U = np.zeros((n_max+1,N+1))
    U[0,:] = np.sin(np.pi*x)**6
    un = U[0,:N]
    c = V * tau / h
    Mi = EC(N,-c)
    for n in range(n_max):
        un = np.linalg.solve(Mi,un)
        U[n+1,:N] = un
        U[n+1,N] = U[n+1,0]

    return x,t,U
#---------------- Lax-Wendrof --------------------#

def LW(N,c):
    Me = np.zeros((N,N))
    uns = np.ones((N,1))
    uns_d = np.ones((N-1,1))
    Me += (1-c**2) * np.diagflat(uns)
    Me += (c+c**2)/2 * np.diagflat(uns_d,-1) + (-c/2+c**2/2) * np.diagflat(uns_d,1)
    Me[N-1,0] = -c/2+c**2/2
    Me[0,N-1] = (c+c**2)/2
    return Me

def SolutionLW(N):
    x = np.linspace(0,1,N+1)
    n_max = int(t_max/tau)
    t = np.linspace(0,n_max*tau,n_max+1)
    U = np.zeros((n_max+1,N+1))
    U[0,:] = np.sin(np.pi*x)**6
    un = U[0,:N]
    c = V * tau / h
    Me = LW(N,c)
    for n in range(n_max):
        un = Me @ un
        U[n+1,:N] = un
        U[n+1,N] = U[n+1,0]

    return x,t,U

#---------------- Lax-Friedrichs --------------------#

def LXF(N,c):
    Me = np.zeros((N,N))
    uns = np.ones((N,1))
    uns_d = np.ones((N-1,1))
    Me += (1+c)/2 * np.diagflat(uns_d,-1) + (1-c)/2 * np.diagflat(uns_d,1)
    Me[0,N-1] = (1+c)/2
    Me[N-1,0] = (1-c)/2
    return Me

def SolutionLXF(N):
    x = np.linspace(0,1,N+1)
    n_max = int(t_max/tau)
    t = np.linspace(0,n_max*tau,n_max+1)
    U = np.zeros((n_max+1,N+1))
    U[0,:] = np.sin(np.pi*x)**6
    un = U[0,:N]
    c = V * tau / h
    Me = LXF(N,c)
    for n in range(n_max):
        un = Me @ un
        U[n+1,:N] = un
        U[n+1,N] = U[n+1,0]

    return x,t,U

#---------------- Affichage --------------------#

x,t,u = SolutionEC(int(1/0.02))

l= [0,0.1,0.2,0.5,1,1.5,2]

for i in l:
    plt.plot(x,u[int(i/0.01),:],label="t="+str(t))

plt.title("Lax-Wendroff, h = 0.02, tau = 0.01")
plt.show()






