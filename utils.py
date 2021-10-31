import numpy as np
import datetime
import random
from functools import reduce
from scipy import linalg
from random import gauss
import pandas as pd
from sktensor import nvecs
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill
)


def save_data(data, name):
    with open(name + '.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(name):
    with open(name + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def tensor_outer(A, B, C):
    return np.einsum("id, jd, kd->ijk", A, B, C)


def unfold(M, dim):
    if dim == 1:
        return np.hstack([M[:, :, i] for i in range(M.shape[2])])
    elif dim == 2:
        return np.hstack([M[:, :, i].T for i in range(M.shape[2])])
    elif dim == 3:
        return np.vstack([M[:, :, i].ravel() for i in range(M.shape[2])])
    else:
        raise Exception("Invalid Dim!")
        
        
def calculate_rmse(M1, M2):
    return np.sqrt(np.sum(np.square(M1.ravel() - M2.ravel())
                         ) / len(M1.ravel()))


def calculate_rmse_mask(M1, M2, pos):
    total = 0
    for i, j, k in pos:
        total += (M1[i, j, k] - M2[i, j, k]) ** 2
    return np.sqrt(total / len(pos))


def add_guassian_noise(Delta, delta, epsilon, k, shape):
    rho = epsilon * epsilon / (4 * k * np.log(1 / delta))
    sigma = np.sqrt(1 / (2 * rho)) * Delta
    return np.random.normal(0, sigma, shape)


def add_laplace_noise(Delta, epsilon, shape):
    return np.random.laplace(0, Delta / epsilon, shape)


def add_exponential_noise(D, delta_r, epsilon):
    unit_ball_vectors = make_rand_vector(D)
    magnitude = np.random.gamma(D,  delta_r / epsilon)
    b = unit_ball_vectors * magnitude
    return b


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])

def generate_X_CP(N1, N2, N3, D, p=1, SNR=1):
    A = np.random.normal(0, 1, (N1, D))
    B = np.random.normal(0, 1, (N2, D))
    C = np.random.normal(0, 1, (N3, D))
    
    # unit norm
    A = A / np.sqrt(np.sum(A**2, axis=0))
    B = B / np.sqrt(np.sum(B**2, axis=0))
    C = C / np.sqrt(np.sum(C**2, axis=0))
    
    N = np.random.normal(0, 1, (N1, N2, N3))
#     W = np.random.binomial(1, p, (N1, N2, N3))
    TX = tensor_outer(A, B, C) 
    TX = TX / np.abs(TX).max() 
    
    if SNR >= 0:
        SNR = 10**(SNR/10)
        xpower = np.sum(TX**2)/(N1 * N2 * N3)
        npower = xpower / SNR
        NX = TX + np.random.randn(N1, N2, N3) * np.sqrt(npower)
    return TX, NX


def rvs(dim):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H



def generate_X_Tucker(N1, N2, N3, D, p=1, SNR=1):
    O = rvs(dim=N1) 
    A = O[:, [i for i in range(D)]]
    O = rvs(dim=N2) 
    B = O[:, [i for i in range(D, 2*D)]]
    O = rvs(dim=N3) 
    C = O[:, [i for i in range(2*D, 3*D)]]
    E = np.random.normal(0, 1, (D, D, D))
#     W = np.random.binomial(1, p, (N1, N2, N3))
    TX = np.einsum("ijk, mk ->ijm", 
                  np.einsum("ijk, mj ->imk", 
                  np.einsum("ijk, mi ->mjk", E, A), B), C) 
    TX = TX / np.abs(TX).max()
    if SNR >= 0:
        SNR = 10**(SNR/10)
        xpower = np.sum(TX**2)/(N1 * N2 * N3)
        npower = xpower / SNR
        NX = TX + np.random.randn(N1, N2, N3) * np.sqrt(npower)
    return TX, NX


def split_X(NX, TX, mr=0.5, tr=0.8):
    N1, N2, N3 = NX.shape
    indexs = [_ for _ in range(N1 * N2 * N3)]
    random.shuffle(indexs)
    
    valid_indexs = indexs[int(mr * len(indexs)):]
    valid_pos = []
    for n in valid_indexs:
        p1 = n // (N2 * N3)
        p2 = (n - p1 * N2 * N3) // N3
        p3 = n - p1 * N2 * N3 - p2 * N3
        valid_pos.append([p1, p2, p3])

    training_num = int(tr * len(valid_indexs))
    training_indexs = valid_indexs[:training_num]
    training_pos = valid_pos[:training_num]
    test_indexs = valid_indexs[training_num:]
    test_pos = valid_pos[training_num:]
    
    training_X = deepcopy(NX)
    mask = np.ones_like(NX, dtype=bool)
    mask[tuple(zip(*training_pos))] = False    
    training_X[mask] = 0
    
    test_X = deepcopy(TX)
    mask = np.ones_like(TX, dtype=bool)
    mask[tuple(zip(*test_pos))] = False    
    test_X[mask] = 0

    return training_X, test_X, training_pos, test_pos


def biscale(training_X):
    biscaler_list = []
    for t in range(training_X.shape[1]):
        slice_y = training_X[:, t, :]
        biscaler = BiScaler(verbose=False)
        selected_rows = np.sum(slice_y, axis=1) != 0
        selected_columns = np.sum(slice_y, axis=0) != 0
        sub_slice_y = slice_y[np.ix_(selected_rows, selected_columns)]
        sub_slice_y[sub_slice_y == 0] = np.nan
        sub_slice_y = biscaler.fit_transform(sub_slice_y)
        slice_y[np.ix_(selected_rows, selected_columns)] = sub_slice_y
        biscaler_list.append((selected_rows, selected_columns, biscaler))
    training_X[np.isnan(training_X)] = 0
    Delta_X = np.max(training_X) - np.min(training_X)
    return biscaler_list, Delta_X


def extract_data_final(name='a'):
    names = ['user_id', 'movie_id', 'rating', 'timestamp']
    training_df = pd.read_csv('../dataset/ml-100k/u{}.base'.format(name), '\t', names=names, engine='python')
    test_df= pd.read_csv('../dataset/ml-100k/u{}.test'.format(name), '\t', names=names, engine='python')
    training_df["timestamp"] = training_df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    test_df["timestamp"] = test_df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    names = ["user id", "age", "gender", "occupation", "zip code"]
    info = pd.read_csv('../dataset/ml-100k/u.user', '|', names=names, engine='python')
    info["gender"] = info["gender"].apply(lambda x: int(x == 'M'))
    feature_matrix = info[["age", "gender"]].values
    user_count = training_df.user_id.nunique()
    movie_count = training_df.movie_id.nunique()
    time_count = training_df.timestamp.nunique()
    user_dict = dict(zip(training_df.user_id.unique(), range(user_count)))
    movie_dict = dict(zip(training_df.movie_id.unique(), range(movie_count)))
    time_dict = dict(zip(training_df.timestamp.unique(), range(time_count)))
    training_df["user_id"] = training_df["user_id"].apply(lambda x: user_dict[x])
    training_df["movie_id"] = training_df["movie_id"].apply(lambda x: movie_dict[x])
    training_df["timestamp"] = training_df["timestamp"].apply(lambda x: time_dict[x])
    test_df["user_id"] = test_df["user_id"].apply(lambda x: user_dict[x])
    test_df = test_df[test_df["movie_id"].isin(movie_dict.keys())]
    test_df["movie_id"] = test_df["movie_id"].apply(lambda x: movie_dict[x])
    test_df["timestamp"] = test_df["timestamp"].apply(lambda x: time_dict[x])
    training_rating = np.zeros((movie_count, time_count, user_count))
    test_rating = np.zeros((movie_count, time_count, user_count))
    training_pos = []
    test_pos = []
    valid_rows_cols = dict()
    for _, (k, i, r, j) in training_df.iterrows():
        i, j, k = int(i), int(j), int(k) 
        training_rating[i, j, k] = r
        training_pos.append((i, j, k))
        if j not in valid_rows_cols:
            valid_rows_cols[j] = (set([i]), set([k]))
        else:
            valid_rows_cols[j][0].add(i)
            valid_rows_cols[j][1].add(k)
    for _, (k, i, r, j) in test_df.iterrows():
        i, j, k = int(i), int(j), int(k) 
        if i in valid_rows_cols[j][0] and k in valid_rows_cols[j][1]:
            test_rating[i, j, k] = r
            test_pos.append((i, j, k))
    
    return training_rating, test_rating, training_pos, test_pos, feature_matrix



def plot_comparison(x, values, labels):
    markers = ['s-', 'o-', '^-', '<-', '>-', 'v-', '1-', '2-', '3-', '4-', '8-']
    for i in range(len(values)):
        plt.plot(x, values[i], markers[i], label=labels[i])
    plt.xlabel("Epsilon")
    plt.ylabel("RMSE")
    plt.xticks(x, rotation='vertical')
    plt.legend()
   