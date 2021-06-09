import numpy as np
from numpy.lib.function_base import gradient

class Structure:
    def __init__(self, r=0, m=0, e=0):
        # for simplicity also include x_1 and x_N
        self.r = r
        # also include mu_1 = 1 = mu_N+1
        self.m = m
        # also include epsilon_1 = 1 = epsilon_N+1 
        self.e = e
        self.n = np.size(r)

class Structure_Gradient:
    def __init__(self, r=0, m=0, e=0):
        # contains gradient w.r.t. jump points
        self.r = r
        # contains gradient w.r.t. mu
        self.m = m
        #contains gradient w.r.t. epsilon
        self.e = e

    def get_grad(self):
        return np.concatenate((self.r, self.m, self.e))

import compute_gradient
import newton_method

def gradient_ascent_step(omega_opt, structure_opt, p, coefficients):

    gradients = compute_gradient(structure_opt, coefficients, omega_opt)
    grad_omega_opt = gradients.get_grad()
    
    scal_prod = np.vdot(grad_omega_opt,np.imag(grad_omega_opt))
    epsilon = p * np.absolute(omega_opt/scal_prod)

    omega_approx_update = omega_opt + epsilon*scal_prod

    structure_opt.r = structure_opt.r + epsilon*np.imag(gradients.r)
    structure_opt.m = structure_opt.m + epsilon*np.imag(gradients.m)
    structure_opt.e = structure_opt.e + epsilon*np.imag(gradients.e)

    return structure_opt, omega_approx_update

def gradient_ascent(omega_0, start_structure, p, eps, steps):
    # what is initial coefficients c_0 for newton method?
    omega_opt, coeff = newton_method(start_structure, omega_0, c_0, eps)
    structure_opt = start_structure
    for k in range(steps):
        structure_opt, omega_approx_update = gradient_ascent_step(omega_opt, structure_opt, p, coeff)
        omega_opt, coeff = newton_method(structure_opt, omega_approx_update, coeff, eps)

    return structure_opt, omega_opt


def main():
    pass

if __name__ == '__main__':
    main()