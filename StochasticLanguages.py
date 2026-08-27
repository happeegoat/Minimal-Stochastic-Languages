import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from itertools import product
from collections import Counter, defaultdict

# Language Classes

class Language:

    def __init__(self, alphabet, seq_fun):
        self.alphabet = alphabet
        self.seq_fun = seq_fun # for intended truncation length K, seq_fun must cover up to word length 2K + 1

    def get_hankel(self, row_word_length, col_word_length):
        all_row_words = get_all_words(self.alphabet, row_word_length)
        all_col_words = get_all_words(self.alphabet, col_word_length)
        row_dim = len(all_row_words)
        col_dim = len(all_col_words)
        hankel = np.zeros((row_dim, col_dim))

        for i, prefix in enumerate(all_row_words):
            for j, suffix in enumerate(all_col_words):
                hankel[i,j] = self.seq_fun.get(prefix + suffix, 0.0) # returns 0.0 if prefix + suffix is not in seq_fun

        return hankel

    def get_quasi_realisation(self, word_length, type='svd', glm=True):
        print('Obtaining quasi-realisation...')

        aug_hankel = self.get_hankel(word_length + 1, word_length)
        N = get_num_words(self.alphabet, word_length)
        hankel = aug_hankel[:N, :]
        r = np.linalg.matrix_rank(hankel)

        aug_basis = get_all_words(self.alphabet, word_length + 1)
        basis = aug_basis[:N]
        word_to_index = {word: index for index, word in enumerate(aug_basis)}

        if np.linalg.matrix_rank(aug_hankel) != r:
            print('Hankel matrix has insufficient rank! Please increase the word length.')

        # Rank-revealing factorisation of hankel matrix as H = PS
        if type == 'svd':
            U, Sig, Vt = np.linalg.svd(hankel, full_matrices=False)
            P = U[:, :r] @ np.diag(np.sqrt(Sig[:r])) # full col rank
            S = np.diag(np.sqrt(Sig[:r])) @ Vt[:r, :] # full row rank

        elif type == 'canonical': # technique follows the paper
            Q, R, piv = qr(hankel.T, pivoting=True, mode='economic') # pivot arranges the diagonal of R in nonincreasing order
            basis_rows = piv[:r] # up to word length K, not K + 1

            S = aug_hankel[basis_rows, :] # reduced Hankel matrix
            P = np.linalg.lstsq(S.T, hankel.T, rcond=None)[0].T # coefficient matrix s.t. P @ S = hankel

        # Define the shifted hankel matrix H_a[u, v] = H[ua, v]. We seek X^(a) s.t. H_a = P W^(a) S.
        # Since P has full col rank, P_pinv P = I_r. Likewise since S has full row rank, S S_pinv = I_r.
        # Hence, W^(a) := P_pinv H_a S_pinv.
        # So, H[*, x] = <*| P W^(x) S |*> = <*| P W^(x) S |*> := <p*| W^(x) |s*>

        P_pinv = np.linalg.solve(P.T @ P, P.T)
        S_pinv = np.linalg.solve(S @ S.T, S).T

        shifted_hankels = {}

        for a in self.alphabet:
            hankel_a = np.empty_like(hankel)

            for index, word in enumerate(basis):
                word_a = word + (a,)
                hankel_a[index, :] = aug_hankel[word_to_index[word_a], :]

            shifted_hankels[a] = hankel_a

        transition_matrices = {}

        for a in self.alphabet:
            transition_matrices[a] = P_pinv @ shifted_hankels[a] @ S_pinv

        left = P[0, :]
        right = S[:, 0]

        print('Quasi-realisation obtained.')

        if glm: # enforces right = |1>
            D = np.diag(right)
            D_pinv = np.diag([1/x if abs(x) > 1e-20 else 0 for x in right]) # tolerance may cause errors if not adjusted properly

            for a in self.alphabet:
                transition_matrices[a] = D_pinv @ transition_matrices[a] @ D

            left = left @ D
            right = D_pinv @ right

        return transition_matrices, left, right



class MPS:

    def __init__(self, transition_matrices, left_vector, right_vector):
        self.alphabet = list(transition_matrices.keys())
        self.transition_matrices = transition_matrices
        self.left_vector = left_vector
        self.right_vector = right_vector
        self.dim = len(left_vector)

    def get_seq_fun(self, word_length):
        states = {(): self.left_vector}
        seq_fun = {(): self.left_vector @ self.right_vector}
        frontier = [()]

        for _ in range(word_length):
            new_frontier = []

            for word in frontier:
                state = states[word]

                for a in self.alphabet:
                    child = word + (a,)
                    child_state = state @ self.transition_matrices[a]
                    states[child] = child_state
                    seq_fun[child] = child_state @ self.right_vector
                    new_frontier.append(child)

            frontier = new_frontier

        return seq_fun



# QHMM Functions

def get_transfer_matrix(kraus): # kraus is a list of operators, E^(x) = \sum_{j} ((K^(x)_j)^T \otimes (K^(x)_j)^\dag)
    dim = kraus[0].shape[0]
    transfer_matrix = np.zeros((dim*dim, dim*dim), dtype=np.complex128)
    for K in kraus:
        transfer_matrix += np.kron(K.T, K.conj().T)
    return transfer_matrix

def get_choi_matrix(kraus):
    dim = kraus[0].shape[0]
    choi_matrix = np.zeros((dim*dim, dim*dim), dtype=np.complex128)
    for K in kraus:
        vecK = K.reshape(-1, order='F')
        choi_matrix += np.outer(vecK, vecK.conj().T)
    return choi_matrix

def reshuffle(M): # converts between choi and transfer matrices
    d = int(np.sqrt(len(M)))
    tensor = M.reshape((d, d, d, d), order='C')
    permuted = tensor.transpose(0, 2, 1, 3)
    return permuted.reshape((d*d, d*d), order='C')

def gen_random_kraus_operators(dim, *seeds):
    kraus = []
    M = np.zeros((dim, dim), dtype=complex)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        kraus.append(A)
        M += A.conj().T @ A

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)

    sqrtMinv = (eigvecs / np.sqrt(eigvals)) @ eigvecs.conj().T

    for i in range(len(kraus)):
        kraus[i] = kraus[i] @ sqrtMinv

    return kraus

def gen_random_density_matrix(dim, seed): # generates a random density matrix
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = A @ A.conj().T
    return rho / np.trace(rho)



# General Utilities

def get_all_words(alphabet, word_length): # set of all sequences up to length K including null sequence
    words = [()]
    for l in range(1, word_length + 1):
        words.extend(product(alphabet, repeat=l))
    return words # returns tuples

def get_num_words(alphabet, word_length): # no. of words up to length K including null sequence
    m = len(alphabet)
    if m == 1:
        return word_length + 1
    return (m**(word_length+1) - 1)//(m - 1)

def disp_seq_fun(seq_fun, sep='', ShowZeros=True, tol=1e-15): # displays sequence function in a readable manner
    word_freq = {sep.join(word): freq for word, freq in seq_fun.items()}
    print('Sequence Function:')
    for word in word_freq:
        freq = word_freq.get(word)
        if np.real(freq) > tol or ShowZeros: # tolerance can be adjusted
            if word == '':
                word = '*'
            if np.imag(freq) > tol:
                print('Warning: Complex probability obtained.')
            print(f'{word}: {np.real(freq):.5g}')



# HMM Realisation

def find_HMM_realisation(transition_matrices, left_vector, right_vector):
    print('Finding HMM realisation...')
    dim = len(left_vector)

    for i in range(dim):
        X, loss = find_HMM_gauge(transition_matrices, left_vector, right_vector)

        if loss < 1e-15:
            print(f'HMM of rank {dim} found. Loss = {loss}')
            X_inv = np.linalg.inv(X)
            T = {symbol: X_inv @ W @ X for symbol, W in transition_matrices.items()}
            pi = left_vector @ X
            ones = X_inv @ right_vector
            return T, pi, ones

        elif i == dim - 1:
            print(f'HMM of rank {len(left_vector)} not found. Loss = {loss}')
            print('No HMM realisation is found.')
            return None, None, None

        else:
            print(f'HMM of rank {len(left_vector)} not found. Loss = {loss}')
            for a in transition_matrices.keys():
                transition_matrices[a] = np.block([[transition_matrices[a], np.zeros((dim+i, 1))],
                                      [np.zeros((1, dim+i)), np.array([1])]])
            left_vector = np.append(left_vector, 1)
            right_vector = np.append(right_vector, 1) # add a new state to the distribution

def calc_HMM_loss(x, transition_matrices, left_vector, right_vector):
    # We want to find X such that <left| X (X_inv W^(a) X) X_inv |right> = <prob dist| T^(a) |1>

    r = len(left_vector)
    X = x.reshape((r,r))

    try:
        X_inv = np.linalg.inv(X)
    except np.linalg.LinAlgError:
        return 1e20 # penalty for singular matrices

    loss = 0

    # Penalty for <left| X != <prob dist|
    new_left = left_vector @ X
    loss += (np.sum(new_left)-1)**2 # penalty for prob dist not summing to 1
    loss += np.sum(np.minimum(new_left, 0)**2 + np.maximum(new_left-1, 0)**2) # penalty for elements not within [0, 1]

    # Penalty for X_inv |right> != |1>
    new_right = X_inv @ right_vector
    loss += np.sum((new_right-1)**2)

    # Penalty for elements of X_inv W^(a) X not in [0, 1]
    for W in transition_matrices.values():
        T = X_inv @ W @ X
        loss += np.sum(np.minimum(T, 0)**2 + np.maximum(T-1, 0)**2)

    return loss

def find_HMM_gauge(transition_matrices, left_vector, right_vector):
    r = len(left_vector)
    X0 = []
    X0.append(np.eye(r).reshape(-1)) # flatten
    X0.append(np.ones((r, r)).reshape(-1))
    for i in range(3):
        stochastic_matrix = np.random.rand(r, r)
        stochastic_matrix = stochastic_matrix / stochastic_matrix.sum(axis=1, keepdims=True)
        X0.append(stochastic_matrix.reshape(-1))

    # Run through different initial x0
    result = None
    for x0 in X0:
        if result is None:
            result = minimize(fun=lambda x: calc_HMM_loss(x, transition_matrices, left_vector, right_vector), x0=x0, method="Powell")
        else:
            new_result = minimize(fun=lambda x: calc_HMM_loss(x, transition_matrices, left_vector, right_vector), x0=x0, method="Powell")
            if new_result.fun < result.fun:
                result = new_result

    return result.x.reshape(r, r), result.fun



# QHMM Realisation
# Work in progress

