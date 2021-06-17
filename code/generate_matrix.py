import numpy as np

def create_A_matrix(omega, structure):
    mu = structure.m
    epsilon = structure.e
    x = structure.r
    N = structure.n
    
    r = np.sqrt(np.multiply(mu,epsilon))
    A = np.zeros((2*N,2*N), dtype=np.cfloat)

    A[0,0] = np.exp(-1j *omega*x[0])
    A[0,1] = -np.exp(1j *r[1]*omega*x[0])
    A[0,2] = -np.exp(-1j *r[1]*omega*x[0])

    A[1,0] = -np.exp(-1j *omega*x[0])
    A[1,1] = -r[1]*1/epsilon[1]*np.exp(1j *r[1]*omega*x[0])
    A[1,2] = r[1]*1/epsilon[1]*np.exp(-1j *r[1]*omega*x[0])

    if N > 2:
        for j in range(2,N):
            A[2*(j-1),2*j-3] =  np.exp(1j*r[j-1]*omega*x[j-1])
            A[2*(j-1),2*j-2] =  np.exp(-1j*r[j-1]*omega*x[j-1])
            A[2*(j-1),2*j-1] =  -np.exp(1j*r[j]*omega*x[j-1])
            A[2*(j-1),2*j] =  -np.exp(-1j*r[j]*omega*x[j-1])

            #CHECK THESE AGAIN PROBABLY WRONG
            A[2*j-1,2*j-3] =  r[j-1]*1/epsilon[j-1]*np.exp(1j*r[j-1]*omega*x[j-1])
            A[2*j-1,2*j-2] =  -r[j-1]*1/epsilon[j-1]*np.exp(-1j*r[j-1]*omega*x[j-1])
            A[2*j-1,2*j-1] =  -r[j]*1/epsilon[j]*np.exp(1j*r[j]*omega*x[j-1])
            A[2*j-1,2*j] =  r[j]*1/epsilon[j]*np.exp(-1j*r[j]*omega*x[j-1])
    

    A[2*N-2,2*N-3] = np.exp(1j*r[N-1]*omega*x[N-1])
    A[2*N-2,2*N-2] = np.exp(-1j*r[N-1]*omega*x[N-1])
    A[2*N-2,2*N-1] = -np.exp(1j*omega*x[N-1])

    A[2*N-1,2*N-3] = r[N-1]*1/epsilon[N-1]*np.exp(1j*r[N-1]*omega*x[N-1])
    A[2*N-1,2*N-2] = -r[N-1]*1/epsilon[N-1]*np.exp(-1j*r[N-1]*omega*x[N-1])
    A[2*N-1,2*N-1] = -np.exp(1j*omega*x[N-1])

    return A

def create_A_prime_matrix(omega, structure):

    mu = structure.m
    epsilon = structure.e
    x = structure.r
    N = structure.n

    A_prime = np.zeros((2*N,2*N), dtype=np.cfloat)
    r = np.sqrt(np.multiply(mu,epsilon))

    A_prime[0,0] = -1j *x[0]*np.exp(-1j *omega*x[0])
    A_prime[0,1] = -1j *r[1]*x[0]*np.exp(1j *r[1]*omega*x[0])
    A_prime[0,2] = 1j *r[1]*x[0]*np.exp(-1j *r[1]*omega*x[0])

    A_prime[1,0] = 1j *x[0]*np.exp(-1j *omega*x[0])
    A_prime[1,1] = -1j *r[1]*x[0]*r[1]*1/epsilon[0]*np.exp(1j *r[1]*omega*x[0])
    A_prime[1,2] = -1j *r[1]*x[0]*r[1]*1/epsilon[0]*np.exp(-1j *r[1]*omega*x[0])

    if N > 2:
        for j in range(2,N):
            A_prime[2*(j-1),2*j-3] =  1j*r[j-1]*x[j-1]*np.exp(1j*r[j-1]*omega*x[j-1])
            A_prime[2*(j-1),2*j-2] =  -1j*r[j-1]*x[j-1]*np.exp(-1j*r[j-1]*omega*x[j-1])
            A_prime[2*(j-1),2*j-1] =  -1j*r[j]*x[j-1]*np.exp(1j*r[j]*omega*x[j-1])
            A_prime[2*(j-1),2*j] =  1j*r[j]*x[j-1]*np.exp(-1j*r[j]*omega*x[j-1])

            #CHECK THESE AGAIN PROBABLY WRONG
            A_prime[2*j-1,2*j-3] =  1j*r[j-1]*x[j-1]*r[j-1]*1/epsilon[j-1]*np.exp(1j*r[j-1]*omega*x[j-1])
            A_prime[2*j-1,2*j-2] =  -1j*r[j-1]*x[j-1]*r[j-1]*1/epsilon[j-1]*np.exp(-1j*r[j-1]*omega*x[j-1])
            A_prime[2*j-1,2*j-1] =  -1j*r[j]*x[j-1]*r[j]*1/epsilon[j]*np.exp(1j*r[j]*omega*x[j-1])
            A_prime[2*j-1,2*j] =  -1j*r[j]*x[j-1]*r[j]*1/epsilon[j]*np.exp(-1j*r[j]*omega*x[j-1])
    

    A_prime[2*N-2,2*N-3] = 1j*r[N-1]*x[N-1]*np.exp(1j*r[N-1]*omega*x[N-1])
    A_prime[2*N-2,2*N-2] = -1j*r[N-1]*x[N-1]*np.exp(-1j*r[N-1]*omega*x[N-1])
    A_prime[2*N-2,2*N-1] = -1j*x[N-1]*np.exp(1j*omega*x[N-1])

    A_prime[2*N-1,2*N-3] = 1j*r[N-1]*x[N-1]*r[N-1]*1/epsilon[N-1]*np.exp(1j*r[N-1]*omega*x[N-1])
    A_prime[2*N-1,2*N-2] = 1j*r[N-1]*x[N-1]*r[N-1]*1/epsilon[N-1]*np.exp(-1j*r[N-1]*omega*x[N-1])
    A_prime[2*N-1,2*N-1] = -1j*x[N-1]*np.exp(1j*omega*x[N-1])

    return A_prime

def main():
    mu = [1,1,1]
    epsilon = [1,1,1]
    x = [1,2,3]
    A = create_A_matrix(3, mu, epsilon, 1, x)
    print(A)

if __name__ == '__main__':
    main()