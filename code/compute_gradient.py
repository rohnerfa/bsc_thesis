import numpy as np

def function_u_interval(z, interval, c, omega, r):
    x = c[0]*np.exp(1j*r*omega*z) + c[1]*np.exp(-1j*r*omega*z)
    return np.where(interval[0] < z and z < interval[1], x, 0)

def function_u(structure):
    jump_points = 
    condlist = []
    funclist = []
    for j in range(N-1):
        condlist.append(jump_points[j-1] < x and x < jump_points[j])
        funclist.append(function_u_interval)


    return np.piecewise(x, condlist, funclist, (interval, c, omega, r))


#def compute_gradient_epsilon():
c = [1,1]
omega = np.pi 
r = 1
interval = [0,1]

x = 2.5
condlist = []
for j in range(4):
    condlist.append(j-1 < x and x < j)

    print(condlist)