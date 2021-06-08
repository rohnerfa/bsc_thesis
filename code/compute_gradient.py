from numpy.lib.function_base import gradient
from steepest_ascent import Structure
import numpy as np
import scipy
from scipy.integrate import quad

def make_interval_function(c, omega, r_coef):
    def func_interval(z):
        x = c[0]*np.exp(1j*r_coef*omega*z) + c[1]*np.exp(-1j*r_coef*omega*z)
        return x
    return func_interval

def create_u_function(structure, coefficients, omega):
    def func_u(z):
        #jump_points include both endpoints x_1, x_N
        jump_points = structure.r
        N = structure.n
        r = np.sqrt(np.multiply(structure.m,structure.e))
        condlist = []
        funclist = []
        for j in range(1,N):
            current_coeff = [coefficients[2*j-1], coefficients[2*j]]
            current_r = r[j]
            condlist.append(jump_points[j-1] < z and z < jump_points[j])
            funclist.append(make_interval_function(current_coeff, omega, current_r))
        return np.piecewise(complex(z), condlist, funclist)
    return func_u

def create_mu_function(structure):
    def func_mu(z):

        jump_points = structure.r
        N = structure.n
        mu = structure.m

        condlist = []
        funclist = []

        for j in range(1,N):
            condlist.append(jump_points[j-1] < z and z < jump_points[j])
            funclist.append(lambda x: mu[j])
        return np.piecewise(complex(z), condlist, funclist)
    return func_mu

# https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def compute_gradient_mu(structure, coefficients, omega):
    jump_points = structure.r
    N = structure.n
    func_u = create_u_function(structure, coefficients, omega)
    func_mu = create_mu_function(structure)

    def func_u_squared_mu(z):
        return func_u(z)**2 * func_mu(z)

    def func_u_squared(z):
        return func_u(z)**2

    integral_denom = complex_quadrature(func_u_squared_mu, jump_points[0], jump_points[-1])

    ########################## CAREFUL func not defined at endpoints#########################
    denom = 1j*(func_u(jump_points[0])**2 + func_u(jump_points[1]**2)) + 2*omega*integral_denom

    gradient_mu = np.zeros(N-1, dtype=np.complex)
    for k in range(len(gradient_mu)):
        integral = complex_quadrature(func_u_squared, jump_points[k], jump_points[k+1])
        gradient_mu[k] = -omega**2 *integral/()
    return integral

def main():
    r = [0,1,2]
    m = [1,1,1,1]
    e = [1,1,1,1]
    struc = Structure(r,m,e)

    coeff = [0,1,0,0,0,0]
    omega = 1
    grad = compute_gradient_mu(struc, coeff, omega)
    print(grad)
    """ x = np.linspace(0,100,200)
    y = np.zeros(len(x))

    for i in range(100):
        y[i] = np.absolute(func_u(x[i]))

    import matplotlib.pyplot as plt
    plt.plot(x,y)
    plt.show() """

if __name__ == '__main__':
    main()