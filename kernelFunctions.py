import numpy as np

def dot_product(x_i, x_j, hyperparameters):
    # Hyperparameter needed to comply with kernel interface
    return x_i.T@x_j

def polynomial_kernel(x_i, x_j, power):
    return (x_i.T@x_j) ** power

def gaussian_kernel(x_i, x_j, c):
    return np.e**(-c * np.linalg.norm(x_i.T-x_j) ** 2)