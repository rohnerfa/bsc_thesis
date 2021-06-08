import numpy as np

class Structure:
    def __init__(self, r=0, m=0, e=0):
        # for simplicity also include x_1 and x_N
        self.r = r
        # also include mu_1 = 1 = mu_N+1
        self.m = m
        # also include epsilon_1 = 1 = epsilon_N+1 
        self.e = e
        self.n = np.size(r)







#     grad_omega + ...

#     return grad_omega




""" def omptimize_structure(steps, omega_0, initial_structure):

    omega_opt = newton_method(eps, other inputs)
    opt_structure = Structure()

    for k in range(steps):

        grad_omega_opt = compute_gradient()
        scal_prod = np.vdot(grad_omega_opt,np.imag(grad_omega_opt))
        epsilon_new = p * np.absolute(omega_opt/scal_prod)
        omega_approx_update = omega_opt + epsilon_new*scal_prod

        opt_structure.r = 
        opt_structure.m =
        opt_structure.e = 

        omega_opt = newton_method(eps, updated inputs)

        return (r_opt, m_opt, e_opt), omega_opt """

def main():
    r_0 = np.zeros(2)
    m_0 = np.zeros(3)
    e_0 = np.zeros(4)

    initial_structure = Structure(r_0, m_0, e_0)
    struc = Structure()
    print(struc.n)
    print(initial_structure.n)
if __name__ == '__main__':
    main()