"""
https://github.com/schariya/exact-embeddings/blob/master/ExactEmbeddings.ipynb
"""
import numpy as np
import scipy as sp
import scipy.io, scipy.optimize
from scipy.special import expit
from src.common.common import time_wrapper


def lpca_loss(factors, adj_s, rank, V_fixed=None): # adj_s = shifted adj with -1's and +1's
    n_row, n_col = adj_s.shape
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = factors[n_row*rank:].reshape(rank, n_col) if V_fixed is None else V_fixed
    logits = U @ V[:, :n_row]
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V[:, :n_row].T# / n_element

    if V_fixed is not None:
        return loss, U_grad.flatten()
    
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element

    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten())) 

def lpca_loss_fixed(factors, adj_s, rank, S): # adj_s = shifted adj with -1's and +1's
    # S is the fixed matrix of size rank/rank
    n_row, _ = adj_s.shape # adj_s is square
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = S @ U.T
    logits = U @ V
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V.T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element
    print(loss)
    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))

clip_01 = lambda M : np.clip(M, a_min=0, a_max=1)

@time_wrapper
def decomposition_at_k(A, k, save_path=None, fixed_V=None, max_iter=2000, bounds=None):
    n, _ = A.shape

    # initalize uniformly on [-1,+1]
    size = 2*n*k if fixed_V is None else n*k
    factors = -1+2*np.random.random(size=size)
    
    res = scipy.optimize.minimize(lambda x, adj_s, rank: lpca_loss(x, adj_s, rank, fixed_V), x0=factors, 
                            args=(-1 + 2*np.array(A.todense()), k), jac=True, method='L-BFGS-B',
                            options={'maxiter':max_iter}, bounds=bounds)
    
    U = res.x[:n*k].reshape(n, k) 
    V = res.x[n*k:].reshape(k, n) if fixed_V is None else fixed_V
    A_reconstructed = clip_01(U@V)

    frob_error_norm = np.linalg.norm(A_reconstructed[:,:n] - A) / sp.sparse.linalg.norm(A)

    data = {'U':U, 'V':V, 'A': A}

    if save_path is not None:
        scipy.io.savemat(save_path, data)
    return frob_error_norm, res.nit, data


@time_wrapper
def decompose(A, max_k, max_iter=2000):
    n, _ = A.shape
    errors = []
    iters = []

    for k in range(1, max_k + 1):
        res = decomposition_at_k(A, k, max_iter)

        # if not res.success:
        #     # try with a lot higher num of iters if failed
        #     res = decomposition_at_k(A, k, 5000)
    
        U = res.x[:n*k].reshape(n, k)
        V = res.x[n*k:].reshape(k, n)
        frob_error_norm = np.linalg.norm(clip_01(U@V) - A) / sp.sparse.linalg.norm(A)

        errors.append(frob_error_norm)
        iters.append(res.nit)
    
    return errors, iters