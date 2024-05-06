import numpy as np
import ipdb

def sherman_morrison_update(A_inverse, u, v):
    """
    Calculates the Sherman-Morrison update for (A + u v^T)^-1 given A_inverse, u, and v
    """
    # Calculate the numerator term
    numerator = A_inverse @ np.outer(u, v) @ A_inverse

    # Calculate the denominator term
    denominator = 1 + v.T @ A_inverse @ u

    # Calculate the updated inverse
    A_updated_inverse = A_inverse - numerator / denominator

    return A_updated_inverse

def schur_first_order_update(A_inv, b, c):
    """"
    update of the inverse of A_{t+1} = [[A_t, b], [b^T, c]]
    using the inverse of A_{t}
    params
    A_inv: np.array of size (n,n)
    u: np.array of size (n, 1)
    v = np.array of size (n, 1)
    """
    z = np.dot(A_inv, b)
    z2 = np.dot(b.T, A_inv)
    s = 1/(c - np.dot(z.T, b))
    Z_12 = - s * z
    Z_12_exp = np.expand_dims(Z_12, axis = 1)

    #Z_11 = A_inv + s * np.dot(z, z.T)
    Z_11 = A_inv + s * np.outer(z, z)

    return np.block([[Z_11, Z_12_exp], [Z_12_exp.T, s]])

def schur_first_order_update_fast(A_inv, A_inv_b, b, c):
    """"
    update of the inverse of A_{t+1} = [[A_t, b], [b^T, c]]
    using the inverse of A_{t}
    params
    A_inv: np.array of size (n,n)
    u: np.array of size (n, 1)
    v = np.array of size (n, 1)
    """
    z = A_inv_b
    z2 = np.dot(b.T, A_inv)
    s = 1/(c - np.dot(z.T, b))
    Z_12 = - s * z
    Z_12_exp = np.expand_dims(Z_12, axis = 1)
    #Z_11 = A_inv + s * np.dot(z, z.T)
    Z_11 = A_inv + s * np.outer(z, z2)

    return np.block([[Z_11, Z_12_exp], [Z_12_exp.T, s]])

