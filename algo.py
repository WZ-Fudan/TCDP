import random
import numpy as np
import numba as nb
from utils import *
import time


@nb.jit(nopython=True) 
def update_CP(para, item, Delta_X, A, B, C, D, training_X, epsilon=10, is_gradient=False):
    """
    @para: a list contains learning rate and regularization term
    @item: index of extracted data
    @Delta_X: L1-sensitivity for tenosr X
    @A, B, C: factor matrices
    @D: rank of tensor X
    @training_X: training set of tensor X
    @epsilon: parameter budget
    @is_gradient: flag to determine whether to add noise to gradient.

    @return: gradient of factor matrix C, which is seleced as the example with noise.
    """
    alpha, lamb = para
    i, j, k = item

    # clipping constant
    Gan = 0.1

    diff = training_X[i, j, k] - np.sum(A[i] * B[j] * C[k])
    diff = 1 if diff > 1 else diff
    diff = -1 if diff < -1 else diff
    A[i] -= alpha * (diff * (- B[j] * C[k]) + lamb * A[i])
    B[j] -= alpha * (diff * (- A[i] * C[k]) + lamb * B[j])
    gradient_C = diff * (- A[i] * B[j]) + lamb * C[k]

    # clip using constant "Gan"
    gradient_C = gradient_C / max(1, np.sqrt(np.sum(np.square(gradient_C))) * Gan)
    if is_gradient:
        delta_r = 2 * Gan

        # generate noises sampled from exponential distribution
        vec = np.array([gauss(0, 1) for i in range(D)])
        mag = np.sqrt(np.sum(np.square(vec)))
        unit_ball_vectors = np.array([x/mag for x in vec])
        magnitude = np.random.gamma(D,  delta_r / epsilon)
        b = unit_ball_vectors * magnitude
        gradient_C += b
    C[k] -= alpha * gradient_C
    return gradient_C


# @nb.jit(nopython=True) 
def update_Tucker(para, item, Delta_X, A, B, C, E, D, training_X, epsilon=10, is_gradient=False):
    """
    @para: a list contains learning rate and regularization term
    @item: index of extracted data
    @Delta_X: L1-sensitivity for tenosr X
    @A, B, C: the factor matrices
    @E: the core tensor
    @D: rank of tensor X
    @training_X: training set of tensor X
    @epsilon: parameter budget
    @is_gradient: flag to determine whether to add noise to gradient.

    @return: gradient of factor matrix C, which is seleced as the example with noise.
    """
    alpha, lamb = para
    Gan = 0.1
    i, j, k = item
    pred = C[k] @ (B[j] @ (A[i] @ np.swapaxes(E, 0, 1)))
    diff = training_X[i, j, k] - pred
    diff = 1 if diff > 1 else diff
    diff = -1 if diff < -1 else diff
    A[i] -= alpha * (- diff * ((B[j] @ E) @ C[k]) + lamb * A[i])
    B[j] -= alpha * (- diff * (A[i] @ np.swapaxes(E, 0, 1) @ C[k]) + lamb * B[j])
    gradient_C = - diff * (B[j] @ (A[i] @ np.swapaxes(E, 0, 1))) + lamb * C[k]
    gradient_C = gradient_C / max(1, np.sqrt(np.sum(np.square(gradient_C))) * Gan)
    if is_gradient:
        delta_r = 2 * Gan
        vec = np.array([gauss(0, 1) for i in range(D)])
        mag = np.sqrt(np.sum(np.square(vec)))
        unit_ball_vectors = np.array([x/mag for x in vec])
        magnitude = np.random.gamma(D,  delta_r / epsilon)
        b = unit_ball_vectors * magnitude
        gradient_C += b
    C[k] -= alpha * gradient_C
    E -= alpha * (- diff * np.prod(np.ix_(A[i], B[j], C[k])) + lamb / 10 * E)
    return gradient_C


def print_temp(predicted_X, training_X, training_pos, test_X, test_pos, k, biscaler_list=None, real_data=False):
    """
    The function implemented for normative output of results.
    
    @k: iteration num
    @biscaler_list: the list used to biscale the result, and None represents no biscale.
    """
    if real_data:
        for t in range(predicted_X.shape[1]):
            slice_X1 = predicted_X[:, t, :]
            selected_rows, selected_columns, biscaler = biscaler_list[t]
            slice_X1[np.ix_(selected_rows, selected_columns)]= biscaler.inverse_transform(slice_X1[np.ix_(
                selected_rows, selected_columns)])
        training_rmse = calculate_rmse_mask(training_X, predicted_X, training_pos)
    else:
        training_rmse = calculate_rmse_mask(training_X, predicted_X, training_pos)
    test_rmse = calculate_rmse_mask(test_X, predicted_X, test_pos)
    print("Iteration: {}, Training_rmse: {} Test_rmse: {}".
        format(k, round(training_rmse, 4), round(test_rmse, 4)), end=' ')
    
    return training_rmse, test_rmse


def pure_sgd(paras, training_X, test_X, training_pos, test_pos, D, flag="CP", real_data=False):
    """
    Pure SGD for tensor decompositions.

    @para: a list contains learning rate maximum iteration and regularization term
    @training_X: training set of tensor X
    @test_X: test set of tensor X
    @training_pos: indexes of training set 
    @test_pos: indexes of test set 
    @D: rank of tensor X
    @flag: flag for "CP" or "Tucker"
    @real_data: whether to train real-world datasets

    @return: training_rmse, test_rmse
    """
    alpha, max_iteration, lamb = paras
    cp_training_X = deepcopy(training_X)
    N1, N2, N3 = cp_training_X.shape
    if real_data:
        biscaler_list, Delta_X = biscale(cp_training_X)
    else:
        biscaler_list = None
        Delta_X = np.max(training_X) - np.min(training_X)
    A = np.random.normal(0, 1, (N1, D))
    B = np.random.normal(0, 1, (N2, D))
    C = np.random.normal(0, 1, (N3, D))
    if flag == "Tucker":
        E = np.random.normal(0, 1, (D, D, D))
        
    last_rmse = 10
    for k in range(max_iteration):
        start = time.time()
        for item in training_pos:
            if flag == "CP":
                gradient_C = update_CP((alpha , lamb), item, Delta_X, A, B, C, D, 
                    cp_training_X, epsilon=10, is_gradient=False)    
            if flag == "Tucker":
                gradient_C = update_Tucker((alpha , lamb), item, Delta_X, A, B, C, E, D, 
                    cp_training_X, epsilon=10, is_gradient=False)    
        if flag == "CP":
            predicted_X = tensor_outer(A, B, C)
        if flag == "Tucker":
            predicted_X = np.einsum("ijk, mk ->ijm", 
              np.einsum("ijk, mj ->imk", 
              np.einsum("ijk, mi ->mjk", E, A), B), C) 
        training_rmse, test_rmse = print_temp(predicted_X, training_X, training_pos, test_X, test_pos, 
                                              k, biscaler_list=biscaler_list, real_data=real_data)
        end = time.time()
        print("Time Consuming: {}s".format(round(end-start), 2))
        if np.abs(training_rmse - last_rmse) < 1e-5 or test_rmse!=test_rmse:
            break
        else:
            last_rmse = training_rmse   
    return (training_rmse, test_rmse)


def input_sgd(paras, training_X, test_X, training_pos,
               test_pos, D, flag="CP", real_data=False):
    """
    Private input perturbation for tensor completion via SGD.

    @para: a list contains learning rate maximum iteration and regularization term
    @training_X: training set of tensor X
    @test_X: test set of tensor X
    @training_pos: indexes of training set 
    @test_pos: indexes of test set 
    @D: rank of tensor X
    @flag: flag for "CP" or "Tucker"
    @real_data: whether to train real-world datasets

    @return: a list of (training_rmse, test_rmse)
    """
    alpha, max_iteration, lamb = paras
    cp_training_X = deepcopy(training_X)
    N1, N2, N3 = cp_training_X.shape
    if real_data:
        biscaler_list, Delta_X = biscale(cp_training_X)
    else:
        biscaler_list = None
        Delta_X = np.max(training_X) - np.min(training_X)
            
    result = []
    if real_data:
        epsilons = [0.1, 10 ** (-0.5)] + [i for i in range(1, 11)] + [10 ** 0.5]
    else:
        epsilons = [0.1, 10 ** (-0.5)] + [i / 2 for i in range(1, 21)] + [10 ** 0.5]
    for epsilon in epsilons:
        XN = cp_training_X + add_laplace_noise(Delta_X, epsilon, cp_training_X.shape)  
        A = np.random.normal(0, 1, (N1, D))
        B = np.random.normal(0, 1, (N2, D))
        C = np.random.normal(0, 1, (N3, D))
        if flag == "Tucker":
            E = np.random.normal(0, 1, (D, D, D))
        last_rmse = 10
        for k in range(max_iteration):
            start = time.time()
            for item in training_pos:
                if flag == "CP":
                    gradient_C = update_CP((alpha , lamb), item, Delta_X, A, B, C, D, 
                        XN, epsilon=10, is_gradient=False)    
                if flag == "Tucker":
                    gradient_C = update_Tucker((alpha , lamb), item, Delta_X, A, B, C, E, D, 
                        XN, epsilon=10, is_gradient=False)    
            if flag == "CP":
                predicted_X = tensor_outer(A, B, C)
            if flag == "Tucker":
                predicted_X = np.einsum("ijk, mk ->ijm", 
                  np.einsum("ijk, mj ->imk", 
                  np.einsum("ijk, mi ->mjk", E, A), B), C) 
            training_rmse, test_rmse = print_temp(predicted_X, training_X, training_pos, test_X, test_pos, 
                                              k, biscaler_list=biscaler_list, real_data=real_data)
            if np.abs(training_rmse - last_rmse) < 1e-5 or training_rmse - last_rmse > 1e-4:
                break
            else:
                last_rmse = training_rmse
            end = time.time()
            print("Time Consuming: {}s".format(round(end-start), 2))
        print("\nEpsilon {}:".format(epsilon), round(training_rmse, 4), round(test_rmse, 4))
        result.append((round(training_rmse, 4), round(test_rmse, 4)))
    return result


def gradient_sgd(paras, training_X, test_X, training_pos, test_pos, D, flag="CP", real_data=False):
    """
    Private gradient perturbation for tensor completion via SGD.

    @para: a list contains learning rate maximum iteration and regularization term
    @training_X: training set of tensor X
    @test_X: test set of tensor X
    @training_pos: indexes of training set 
    @test_pos: indexes of test set 
    @D: rank of tensor X
    @flag: flag for "CP" or "Tucker"
    @real_data: whether to train real-world datasets

    @return: a list of (training_rmse, test_rmse)
    """
    alpha, max_iteration, lamb = paras
    cp_training_X = deepcopy(training_X)
    N1, N2, N3 = cp_training_X.shape
    if real_data:
        biscaler_list, Delta_X = biscale(cp_training_X)
    else:
        biscaler_list = None
        Delta_X = np.max(cp_training_X) - np.min(cp_training_X)
    result = []
    if real_data:
        epsilons = [0.1, 10 ** (-0.5)] + [i for i in range(1, 11)] + [10 ** 0.5]
    else:
        epsilons = [0.1, 10 ** (-0.5)] + [i / 2 for i in range(1, 21)] + [10 ** 0.5]
    for epsilon in epsilons:
        A = np.random.normal(0, 1, (N1, D))
        B = np.random.normal(0, 1, (N2, D))
        C = np.random.normal(0, 1, (N3, D))
        if flag == "Tucker":
            E = np.random.normal(0, 1, (D, D, D))
        last_rmse = 10
        for k in range(max_iteration):
            start = time.time()

            for item in training_pos:
                if flag == "CP":
                    gradient_C = update_CP((alpha , lamb), item, Delta_X, A, B, C, D, 
                        cp_training_X, epsilon=epsilon, is_gradient=True)    
                if flag == "Tucker":
                    gradient_C = update_Tucker((alpha , lamb), item, Delta_X, A, B, C, E, D, 
                        cp_training_X, epsilon=epsilon, is_gradient=True)    
            if flag == "CP":
                predicted_X = tensor_outer(A, B, C)
            if flag == "Tucker":
                predicted_X = np.einsum("ijk, mk ->ijm", 
                  np.einsum("ijk, mj ->imk", 
                  np.einsum("ijk, mi ->mjk", E, A), B), C) 
        
            training_rmse, test_rmse = print_temp(predicted_X, training_X, training_pos, test_X, test_pos, 
                                              k, biscaler_list=biscaler_list, real_data=real_data)
            if np.abs(training_rmse - last_rmse) < 1e-5 or training_rmse - last_rmse > 1e-4:
                break
            else:
                last_rmse = training_rmse
            end = time.time()
            print("Time Consuming: {}s".format(round(end-start), 2))
        print("\nEpsilon {}:".format(epsilon), round(training_rmse, 4), round(test_rmse, 4))
        result.append((round(training_rmse, 4), round(test_rmse, 4)))
    return result


def output_sgd(paras, training_X, test_X, training_pos, test_pos, D, flag="CP", real_data=False):
    """
    Output gradient perturbation for tensor completion via SGD.

    @para: a list contains learning rate maximum iteration and regularization term
    @training_X: training set of tensor X
    @test_X: test set of tensor X
    @training_pos: indexes of training set 
    @test_pos: indexes of test set 
    @D: rank of tensor X
    @flag: flag for "CP" or "Tucker"
    @real_data: whether to train real-world datasets

    @return: a list of (training_rmse, test_rmse)
    """   
    alpha, max_iteration, lamb = paras
    cp_training_X = deepcopy(training_X)
    N1, N2, N3 = cp_training_X.shape
    biscaler_list = None
    if real_data:
        biscaler_list, Delta_X = biscale(cp_training_X)
    else:
        biscaler_list = None
        Delta_X = np.max(training_X) - np.min(training_X)
        
    A = np.random.normal(0, 1, (N1, D))
    B = np.random.normal(0, 1, (N2, D))
    C = np.random.normal(0, 1, (N3, D))
    if flag == "Tucker":
        E = np.random.normal(0, 1, (D, D, D))
    last_rmse = 10
    result = []
    max_L = 0
    for k in range(max_iteration):
        start = time.time()
        for item in training_pos:
            if flag == "CP":
                gradient_C = update_CP((alpha , lamb), item, Delta_X, A, B, C, D, 
                    cp_training_X, epsilon=10, is_gradient=False)    
            if flag == "Tucker":
                gradient_C = update_Tucker((alpha , lamb), item, Delta_X, A, B, C, E, D, 
                    cp_training_X, epsilon=10, is_gradient=False)    
            max_L = max(max_L, linalg.norm(gradient_C))
        if flag == "CP":
            predicted_X = tensor_outer(A, B, C)
        if flag == "Tucker":
            predicted_X = np.einsum("ijk, mk ->ijm", 
              np.einsum("ijk, mj ->imk", 
              np.einsum("ijk, mi ->mjk", E, A), B), C) 
        training_rmse, test_rmse = print_temp(predicted_X, training_X, training_pos, test_X, test_pos, 
                                              k, biscaler_list=biscaler_list, real_data=real_data)
        if np.abs(training_rmse - last_rmse) < 1e-5 or training_rmse - last_rmse > 1e-4:
            break
        else:
            last_rmse = training_rmse
        end = time.time()
        print("Time Consuming: {}s".format(round(end-start), 2))

    Beta_for_C = linalg.norm((A.T.dot(A)) * (B.T.dot(B)) + lamb * np.eye(D))
    delta_r = 2 * 100 * max_L * alpha
    print(alpha, delta_r, Beta_for_C , alpha <= 2/Beta_for_C)
    if real_data:
        epsilons = [0.1, 10 ** (-0.5)] + [i for i in range(1, 11)] + [10 ** 0.5]
    else:
        epsilons = [0.1, 10 ** (-0.5)] + [i / 2 for i in range(1, 21)] + [10 ** 0.5]
    for epsilon in epsilons:
        perturbation_C = np.zeros_like(C) 
        for i in range(N3):
            perturbation_C[i] = C[i] + add_exponential_noise(D, delta_r, epsilon)
        if flag == "CP":
            predicted_X = tensor_outer(A, B, perturbation_C)
        if flag == "Tucker":
            predicted_X = np.einsum("ijk, mk ->ijm", 
              np.einsum("ijk, mj ->imk", 
              np.einsum("ijk, mi ->mjk", E, A), B), perturbation_C) 
        training_rmse, test_rmse = print_temp(predicted_X, training_X, training_pos, test_X, test_pos, 
                                              k, biscaler_list=biscaler_list, real_data=real_data)
        print()
        result.append((round(training_rmse, 4), round(test_rmse, 4)))
    return result
