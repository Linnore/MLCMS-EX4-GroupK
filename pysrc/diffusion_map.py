import numpy as np
import scipy.spatial

class diffusion_map:
    def __init__(self, X):
        self.X = X
        self.eigenValues = None
        self.eigenVectors = None

        # Distance matrix
        D = scipy.spatial.distance_matrix(self.X, self.X)

        # epsilon
        self.epsilon = 0.05*np.max(D)

        # Kernel matrix W
        W = np.exp(-D**2/self.epsilon)

        # Diagonal normalization matrix P
        P = np.diag(np.sum(W, axis=1))

        # Normalized kernel matrix K
        P_inv = np.linalg.inv(P)
        K =  P_inv @ W @ P_inv

        # Diagonal normalization matrix Q
        Q = np.diag(np.sum(K, axis=1))

        # The symmetric matrix \hat{T}
        self.Q_sqrt_inv = np.linalg.inv(scipy.linalg.sqrtm(Q))
        self.T_hat = self.Q_sqrt_inv @ K @ self.Q_sqrt_inv


    def get_values_of_eigenfunctions(self, L):

        # L+1 largest eigenvalues al and associated eigenvectors vl of \hat{T}
        eigenValues, eigenVectors = np.linalg.eig(self.T_hat)
        sorted_idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[sorted_idx]
        eigenVectors = eigenVectors[:, sorted_idx]

        al = eigenValues[:L+1]
        vl = eigenVectors[:, :L+1]

        # Eigenvalues of \hat{T}^{1/epsilon}
        self.eigenValues = (al**(1/self.epsilon))**0.5

        # Eigenvectors of T=inv(Q)K
        self.eigenVectors = self.Q_sqrt_inv @ vl

        return self.eigenVectors * self.eigenValues






