import numpy as np

# Définir la taille du domaine discretisé
N = 3

# Créer une matrice vide pour stocker les valeurs de u
u_values = np.zeros((N, N))

# Remplir la matrice u_values avec les valeurs de u

# ... Calculer les valeurs de u ...
n = 0
for i in range(0,N):
    for j in range(0,N):
        n += 1
        u_values[i][j] = n

# Initialiser le vecteur U
U = np.zeros(N*N)

# Remplir le vecteur U à partir des valeurs de u
for i in range(N):
    for j in range(N):
        U[i*N + j] = u_values[i, j]

# Afficher le vecteur U
print(u_values)
print(U)
