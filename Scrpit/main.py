# Nom du projet : Analyse de la dispersion d’un polluant et application à la modélisation fluviale


# Date : 4 juin 2023

# Étudiants :     Guillaume Faivre 
#                 Emma Filippone
#                 Titouan Millet 
#                 Camille Pieussergues

# Utilisation du programme : 


# Nous avons fait le choix de séparer le code dans différentes fonction que nous appelons celon la partie ou la question demande. 

# Exemple pour accéder à la partie 2.2 et la question 2 nous nous rendons dans la fonction parti22() et nous appelons la fonction
# associé. Puis dans le main nous appelons la fonciton parti22(). Nous avons fait ce choix afin d'avoir un seul code. 

# Lien du gitHub : 

# https://github.com/Timi230/Pojet_Ma323


#----------------------------------------------------------
#                       Bibliothèques
#----------------------------------------------------------


import numpy as np
from matplotlib import pyplot as plt


#----------------------------------------------------------
#          Définition des constantes
#----------------------------------------------------------

h = 0.1
tau = 0.0025
a = 50
T = 5
V = 1
N = int(a / h - 1)
sigma = 1


def parti21():
   

    def f(N): # conditon initial pour le calul de u_0
        x     = np.linspace(0,a,N+1)
        n_max = int(T/tau)
        t     = np.linspace(0,n_max*tau,n_max+1)
        U     = np.zeros((n_max+1,N+1))
        U[0,:] = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-a/2)**2)/(2*sigma**2))
        un = U[0,: N]
        c  = V * tau / h

        return x, n_max, t, U, un, c

    def mat_id(N):
        return np.eye(N)

    def mat_M(N,c): # matrice M de explicite centré
        M = np.zeros((N,N))
        d_c = -c/2 * np.ones((N-1,1))
        M += np.diagflat(d_c,-1) + np.diagflat(-d_c,1)
        M[0,N-1] = -c/2
        M[N-1,0] = c/2
        return M

    def mat_M_CN(N,c): # matrice de M de Crank-Nicoloson
        M = np.zeros((N,N))
        d_c = -c/2 * np.ones((N-1,1))
        M += np.diagflat(d_c,-1) + np.diagflat(-d_c,1)
        return M


    def SolutionEC(N): # Fonction de résolution du schéma ecplicite centré
        x, n_max, t, U, un, c = f(N)
        
        Me = mat_M(N,c)
        In = mat_id(N)
        
        for n in range(n_max):
            un = (In-Me) @ un
            U[n+1,:N] = un
            U[n+1,N] = U[n+1,0]
        
        return x,t,U


    def SolutionCN(N): # FOcntion de résolution du schéma de Crank-Nicoloson 
        x, n_max, t, U, un, c = f(N)
        c = (V * tau) / (2*h)
        
        Me = mat_M_CN(N,c)
        In = mat_id(N)
        
        for n in range(n_max):
            un = np.linalg.solve(In + Me, (In - Me) @ un)
            U[n+1,:N] = un
            U[n+1,N] = U[n+1,0]

        return x,t,U
    

    #----------------------------------------------------------
    #          Affichage des courbes avec appel 
    #        de la fonction de la quesiton associé 
    #----------------------------------------------------------
    

    xEC,tEC,uEC = SolutionEC(N)
    xCN,tCN,uCN = SolutionCN(N)
    

    l= [0,0.5,1,1.5,2]

    for i in l:
        plt.plot(xEC,uEC[int(i/tau),:],label="t="+str(tEC))

    plt.title("Explicite centré")
    plt.show()

    for i in l:
        plt.plot(xCN,uCN[int(i/tau),:],label="t="+str(tCN))

    plt.title("Crank-Nicolson")
    plt.show()

    #----------------------------------------------------------
    #                  Brouillon question 5
    #----------------------------------------------------------


    """"
    l_representee = np.linspace(0,5,20)

    x = np.linspace(0,a,N+1)
    u_ref = 1/ (sigma * np.sqrt(2*np.pi)) * np.exp(-(x - a/2)**2 / (2*sigma**2))
    l_representee = np.linspace(0,5,100)
    errors = []

    for t in l_representee:
        a = int(t/tau)
        u = U[a,:]
        error = np.linalg.norm(u_ref * t - u)
        errors.append(error)

    p = np.log(np.divide(errors[1:] , errors[:-1])) / np.log(np.divide(l_representee[1:] , l_representee[:-1]))
    plt.plot(l_representee[1:],p)
    plt.xlabel("t")
    plt.ylabel("Vitesse de convergence")
    plt.show()
    """

    
def parti22():
    mu = 1
    T = 10


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
    

    #----------------------------------------------------------
    #                        Question 2
    #----------------------------------------------------------

    
    def SolutionCN_2(N):

        def mat_M(N):
            M = np.zeros((N,N))
            M += np.diagflat(-1*np.ones((N-1,1)),-1) + np.diagflat(1*np.ones((N-1,1)),1)
            M = M*(1/(2*h))
            return M
    
        def mat_A(N):
            A = -2 * np.eye(N)
            A += np.diagflat(-1*np.ones((N-1,1)),-1)  + np.diagflat(1*np.ones((N-1,1)),1)
            A = A * (1/h**2)
            return A
        
        x, n_max, t, U, un = f(N)
        c = (V * tau) / 2
        d = (mu * tau) / 2

        M = mat_M(N)
        A = mat_A(N)
        In = mat_I(N)
        
        for n in range(n_max):
            un = np.linalg.solve((In + c * M - d * A), (In - c * M + d * A) @ un)
            U[n+1,:N] = un
            U[n+1,N] = U[n+1,0]

        return x,t,U
    
    #----------------------------------------------------------
    #                        Question 3
    #----------------------------------------------------------
    
    def SolutionCN_3(N):

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

        x, n_max, t, U, un = f(N)
        c = (V * tau) / 2
        d = (mu * tau) / 2

        M = mat_M(N)
        A = mat_A(N)
        In = mat_I(N)

        
        for n in range(n_max):
            un = np.linalg.solve((In + c * M - d * A), (In - c * M + d * A) @ un)
            U[n+1,:N] = un
            U[n+1,N] = U[n+1,0]

        return x,t,U
    
    #----------------------------------------------------------
    #                        Question 4
    #----------------------------------------------------------
    
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

        return x,t,U
    
    #----------------------------------------------------------
    #                        Question 5
    #----------------------------------------------------------
    

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
            F = np.zeros((n_max + 1, N))  
            x = np.linspace(0, a, N + 1)  
            print(t)
            for n in range(n_max):
                print(n, "/", n_max)
                for j in range(N):
                    for z in t:
                        z = int(z)
                        if z%2 == 0 : #determine si t est paire ou impaire
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
        print(Fnj)
        
        for n in range(n_max):
            un = np.linalg.solve((In + c * M - d * A), (In - c * M + d * A) @ (np.add(un,Fnj[0,:])))
            U[n+1,:N] = un 
            U[n+1,N] = U[n+1,0]

        return x,t,U, Fnj
    
    
    #----------------------------------------------------------
    #          Affichage des courbes avec appel 
    #        de la fonction de la quesiton associé 
    #----------------------------------------------------------
    
   

    xCN,tCN,uCN, Fnj = SolutionCN_5(N)


    for i in tCN:
        plt.plot(xCN,uCN[int(i/tau),:],label="t="+str(tCN))

    plt.title("Crank-Nicolson")
    plt.show()


def parti31():

    def Question4(N) : 

        N = 10 # definiton de N car avec un N plus grand nous avions des temsp de calcul trop long
        
        def mat_M(N):
            N2 = N**2
            M = 4 * np.eye(N2)
            M += np.diagflat(-1*np.ones((N2-1,1)),-1) + np.diagflat(-1*np.ones((N2-1,1)),1) + np.diagflat(-1*np.ones((N2-3,1)),-3) + np.diagflat(-1*np.ones((N2-3,1)),3)
            M = M*(1/(h**2))
            return M

        def mat_U(U) :
            U[0, :] = 0
            U[-1, :] = 0
            U[:, 0] = 0
            U[:, -1] = 0
            U = np.concatenate(U)

            return U.reshape((-1,1))

        def mat_F(N):
                
                F = np.zeros((N, N)) 
                x = np.linspace(0, a, N + 1)  
                for n in range(N):
                    for j in range(N):
                        f_nj = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x[j] - a/2)**2) / (2 * sigma**2))
                        F[n, j] = f_nj
                
                F = np.concatenate(F)
            
                return F.reshape((-1,1))

        
        x = np.linspace(0, a, N**2)

        M = mat_M(N)
        F = mat_F(N)  

        U = np.linalg.solve(M , F) #resout MU=F


        #----------------------------------------------------------
        #        Ajout des condtion de Dirichlet
        #----------------------------------------------------------

        u = np.zeros((N,N))

        k = 0 
        j=0

        for i in range(len(U)):
            
            if k != 10 : 
                u[j,k] = U[i]
            else : 
                k = 0
                j +=1
        
            k +=1

        U = mat_U(u)

        #----------------------------------------------------------
        #         Traitement pour afficher la courbe
        #----------------------------------------------------------

        val_U = U.tolist() #tranformation du vecteur colone en liste 
        X = x.tolist()
        Y = x.tolist()

        val_U = [item for sublist in val_U for item in sublist]

        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(X, Y, val_U)


        # Afficher le graphe

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')

        #----------------------------------------------------------
        #                  Solution exact 
        #----------------------------------------------------------

        n = 7
        k = 7
        b = a
        s_exact = []
        for i in range(len(X)):
            s_exact.append(np.sin((n*np.pi*X[i])/a)*np.sin((k*np.pi*Y[i])/b))

        ax2.plot(X, Y, s_exact)
        plt.show()
    
    Question4(N)
    
def parti32():
    mu = 1
    a  = 50
    b = 50
    Tmin = 0
    Tmax = 10
    h = 0.5
    tau = 0.1
    xe = a/2
    ye = a/2

    def U0(N): # matrice de taille N^2 obtenue par concaténation des lignes de la matrice (uij) pour t=0 
        X     = np.linspace(0,a,N+1) # à changer car x et y appartiennent à ]0,a[ 
        Y     = np.linspace(0,a,N+1)
        U     = np.zeros((N**2,N**2))

        for x in range(0,a):
            for y in range(0,a):
                f = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-xe)**2+(y-ye)**2)/(2*(sigma**2)))
                U[x,y] = f
     
        return X,Y,U
   
    def mat_I(N):
        return np.eye(N**2)
    

    #----------------------------------------------------------
    #                Brouillon  Question 2
    #
    #
    #        Problème sur les déinitions de matrice 
    #----------------------------------------------------------

    def SolutionCN_2(N):

        def mat_Mx(N):
            Mx = np.zeros((N**2,N**2))
            Mx += np.diagflat(1*np.ones(((N**2)-3,1)),3) + np.diagflat(-1*np.ones(((N**2)-3,1)),-3)
            Mx = Mx*(1/(2*h))
            return Mx
        
        def mat_My(N):
            My = np.zeros((N**2,N**2))
            My += np.diagflat(-1*np.ones(((N**2)-1,1)),-1) + np.diagflat(1*np.ones(((N**2)-1,1)),1)
            My = My*(1/(2*h))
            return My
    
        def mat_A(N):
            A = -4 * np.eye(N**2)
            A += np.diagflat(1*np.ones(((N**2)-3,1)),-3) + np.diagflat(1*np.ones(((N**2)-1,1)),-1)  + np.diagflat(1*np.ones(((N**2)-1,1)),1) + np.diagflat(1*np.ones(((N**2)-3,1)),3)
            A = A * (1/h**2)
            return A
        
        print(N)
        N = 10


        print('je suis avant I ')
        In = mat_I(N)
        print('je suis après I')
        print('je suis avant Mx ')
        Mx = mat_Mx(N)
        print('je suis après Mx')
        print('je suis avant My ')
        My = mat_My(N)
        print('je suis après My ')
        print('je suis avant A ')
        A = mat_A(N)
        print('je suis après A ')
        x, y, U = U0(N)

        d = (mu*tau / 2)

        for n in range(Tmin, Tmax):
            un = np.linalg.solve((In - d * A + (tau/2) * (Mx + My) ), (In + d * A - (tau/2) * (Mx + My)) @ un)
            U[n+1,:N] = un
            U[n+1,N-1] = U[n+1,0]

        return x, y, U
        
# "------------- En construction jusque là -----------"
    
    xCN,yCN,uCN = SolutionCN_2(N)
  
    l= [0,0.5,1,1.5,2]

    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')

    for i in l:
        print(i)
        ax.plot(xCN,yCN, uCN[int(i/0.1),:])

    plt.title("Crank-Nicolson")
    plt.show()

    

    return xCN,yCN,uCN



#----------------------------------------------------------
#               Main du projet 
#----------------------------------------------------------



if __name__ == "__main__":


    #----------------------------------------------------------
    #                          Partie 2.1
    #----------------------------------------------------------
    
    parti21()

    #----------------------------------------------------------
    #                          Partie 2.2
    #----------------------------------------------------------

    #parti22()


    #test d'affichage de concentration max. 
    # P5 = parti22()

    # U = P5[-3]

    # tau = 0.025
    # for i in P5[1]:
    #     print(max(U[int(i/tau),:]))


    #----------------------------------------------------------
    #                          Partie 3.1
    #----------------------------------------------------------
    
   
  
    parti31()

    #----------------------------------------------------------
    #                          Partie 3.2
    #----------------------------------------------------------
    

    parti32()
    
    