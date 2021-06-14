import numpy as np
from numpy.lib.function_base import gradient
import matplotlib.pyplot as plt

class Structure:
    def __init__(self, r=0, m=0, e=0):
        # for simplicity also include x_1 and x_N
        self.r = np.array(r)
        # also include mu_1 = 1 = mu_N+1
        self.m = np.array(m)
        # also include epsilon_1 = 1 = epsilon_N+1 
        self.e = np.array(e)
        self.n = np.size(r)

    def plot_structure(self):
        steps = 50
        x = self.r
        x = np.append(x, self.r[-1]+.5)
        x = np.insert(x, 0, self.r[0]-.5)
        
        l = (len(x) - 1)*steps + 1          # total length after interpolation
        x = np.interp(np.arange(l), np.arange(l, step=steps), x)

        
        y_e = np.zeros(len(x))
        y_m = np.zeros(len(x))

        for i in range(self.n + 1):
            for k in range(steps):
                y_e[(steps*i)+k] = self.e[i]
                y_m[(steps*i)+k] = self.m[i]
        
        # fix last entry since epsilon, mu discontinuous at jump points always include left endpoint but not right one
        y_e[-1] = self.e[-1]
        y_m[-1] = self.m[-1]

        plt.subplot(2, 1, 1)
        plt.plot(x, y_e)
        plt.title('Plot of structure')
        plt.ylabel('epsilon')

        plt.show(block=False)
        # plt.subplot(2, 1, 2)
        # plt.plot(x, y_m)
        # plt.xlabel('x')
        # plt.ylabel('mu')
        
        return

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
    x = np.linspace(1- 21*(0.0324),1 + 24*0.0324 ,46)

    epsilon = np.ones(47)
    mu = np.ones(47)
    
    for k in range(len(epsilon)):
        if (k % 2) == 1:
            epsilon[k] = 1/2
    epsilon[23] = 1

    struc = Structure(x, mu, epsilon)
    struc.plot_structure()

    #print(epsilon)
    plt.show()

if __name__ == '__main__':
    main()