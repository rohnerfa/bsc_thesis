import generate_matrix as gm
import numpy as np


def newton_step(omega_old,c_old, N, mu, epsilon, x):

    A_step = gm.create_A_matrix(N,mu,epsilon,omega_old,x)
    A_prime_step = gm.create_A_prime_matrix(N,mu,epsilon,omega_old,x)

    b_step = np.matmul(A_prime_step, c_old)
    y_step = np.linalg.solve(A_step, b_step)
    
    c_new = y_step/y_step[0]
    omega_new = omega_old - 1/y_step[0]
    return omega_new, c_new

def main():
    return

if __name__ == '__main__':
    main()