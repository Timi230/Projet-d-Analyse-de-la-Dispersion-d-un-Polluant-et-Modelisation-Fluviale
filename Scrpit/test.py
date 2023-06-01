#----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def mat_id(N):
    return np.eye(N)

def mat_M(N, c):
    M = np.zeros((N, N))
    d_c = -c / 4 * np.ones((N - 1,))
    np.fill_diagonal(M[1:], d_c)
    np.fill_diagonal(M[:, 1:], -d_c)
    M[0, -1] = -c / 4
    M[-1, 0] = c / 4
    return M

def solve_decentre_amont(N, c, tau):
    M = mat_M(N, c)
    In = mat_id(N)
    U = np.zeros((N,))
    for _ in range(int(T / tau)):
        U = np.linalg.solve(In + tau * M, In @ U)
    return U

def solve_crank_nicolson(N, c, tau):
    M = mat_M(N, c)
    In = mat_id(N)
    U = np.zeros((N,))
    for _ in range(int(T / tau)):
        U = np.linalg.solve(In + 0.5 * tau * M, (In - 0.5 * tau * M) @ U)
    return U

def calculate_error(U, U_ref):
    return np.max(np.abs(U - U_ref))

def compute_orders_convergence(V, h_values, tau_values):
    orders_decentre_amont = []
    orders_crank_nicolson = []
    for h in h_values:
        errors_decentre_amont = []
        errors_crank_nicolson = []
        for tau in tau_values:
            N = int(a / h - 1)
            c = V * tau / h

            U_decentre_amont = solve_decentre_amont(N, c, tau)
            U_crank_nicolson = solve_crank_nicolson(N, c, tau)

            error_decentre_amont = calculate_error(U_decentre_amont, U_ref)
            error_crank_nicolson = calculate_error(U_crank_nicolson, U_ref)

            errors_decentre_amont.append(error_decentre_amont)
            errors_crank_nicolson.append(error_crank_nicolson)

        order_decentre_amont = np.polyfit(np.log(tau_values), np.log(errors_decentre_amont), 1)[0]
        order_crank_nicolson = np.polyfit(np.log(tau_values), np.log(errors_crank_nicolson), 1)[0]

        orders_decentre_amont.append(order_decentre_amont)
        orders_crank_nicolson.append(order_crank_nicolson)

    return orders_decentre_amont, orders_crank_nicolson

# Paramètres communs
a = 50
T = 5
sigma = 1
U_ref = solve_crank_nicolson(1000, 0, 0.0001)

# Paramètres de test
V = 0.25
h_values = [0.1, 0.05, 0.025, 0.0125]
tau_values = [0.0025, 0.00125, 0.000625, 0.0003125]

