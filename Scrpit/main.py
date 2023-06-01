from library import *
from pylab import *
from matplotlib import animation



"""

            PARTIE 2.1 PHÉNOMÈNE DE CONVECTION 1D

"""


# Q.1

x,t,u = SolutionEC(N)

#l= [0,0.5,1,1.5,2]
l= [0,1,2,3,4,5,6,7,8,9,10]


for i in l:
    plt.plot(x,u[int(i/tau),:],label="t="+str(t))
    plt.pause(0.5)

plt.show()

# Q.2


