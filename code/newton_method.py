import generate_matrix as gm
import numpy as np


def newton_step(omega_old, structure, c_old):

    A_step = gm.create_A_matrix(omega_old, structure)
    A_prime_step = gm.create_A_prime_matrix(omega_old, structure)

    b_step = np.matmul(A_prime_step, c_old)
    y_step = np.linalg.solve(A_step, b_step)
    
    c_new = y_step/y_step[0]
    omega_new = omega_old - 1/y_step[0]

    # Compute difference 
    diff_vec = np.append(c_new, omega_new) - np.append(c_old, omega_old)
    diff = np.linalg.norm(diff_vec)

    return omega_new, c_new, diff

def newton_method(structure, omega_0, c_0, eps, max_steps=10e6):
    steps = 0
    is_converged = False

    omega_old = omega_0
    c_old = c_0
    while not is_converged:
        steps = steps+1

        omega_new, c_new, diff = newton_step(omega_old, structure, c_old)
        omega_old, c_old = omega_new, c_new

        # Check for convergence
        if diff <= eps:
            is_converged = True
            print(f'Newton Method converged after {steps} steps.')
            return omega_old, c_old

        elif steps > max_steps:
            print(f'Newton Method did not converge after reaching max_steps = {max_steps} steps.')
            break

def main():
    return

if __name__ == '__main__':
    main()