import numpy as np

def dot_product(x_i, x_j, hyperparameters):
    # Hyperparameter needed to comply with kernel interface
    return x_i.T@x_j

def polynomial_kernel(x_i, x_j, power):
    return (x_i.T@x_j) ** power

def gaussian_kernel(x_i, x_j, c):
    return np.e**(-c * np.linalg.norm(x_i.T-x_j) ** 2)

def calc_symetric_kernel_matrix(kernel, hyperparameters, data):
        # Kernels used are symetric so at training time so is the kernel matrix
        # Hence calculate top half of matrix and then reflect
        n_samples = len(data)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_matrix[i,j] = kernel(data[i], data[j], hyperparameters)
        # Reflection
        kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
        return kernel_matrix

def calc_kernel_matrix(kernel, hyperparameters, data_1, data_2=None):
    # Check if can shorcut calc due to symetries
    if data_2 is None:
        return calc_symetric_kernel_matrix(kernel, hyperparameters, data_1)

    kernel_matrix = np.zeros((len(data_1), len(data_2)))
    for i in range(len(data_1)):
        for j in range(len(data_2)):
            kernel_matrix[i,j] = kernel(data_1[i], data_2[j], hyperparameters)
    return kernel_matrix
