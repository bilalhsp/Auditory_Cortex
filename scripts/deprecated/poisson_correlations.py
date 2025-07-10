import numpy as np
from utils_jgm.toolbox import r_pearson
rng = np.random.default_rng()


# Ns
N_experiments = 407
N_types = 10
N_tokens = 11
N_samples = 15

# rate parameters
l1 = 3
l2 = 6
l3 = 10

# derived params (see https://math.stackexchange.com/questions/244989/)
rho = l2/( (l1 + l2)*(l2 + l3) )**(1/2)
N_repeats = N_experiments*N_types

# independent Poisson RVs
Y1 = rng.poisson(l1, N_repeats*N_tokens*N_samples)
Y2 = rng.poisson(l2, N_repeats*N_tokens*N_samples)
Y3 = rng.poisson(l3, N_repeats*N_tokens*N_samples)

# dependent Poisson RVs
X1 = Y1 + Y2
X2 = Y2 + Y3

# helper functions
def reshape(Z, N, M):
    return Z.reshape(N*M, -1)
    
def r_split(Z1, Z2, N, M):
    R = r_pearson(reshape(Z1, N, M), reshape(Z2, N, M), MATRIX=False)
    return np.mean(R.reshape(N, M), axis=1)

def MSE(R):
    return np.mean( (np.mean(R.reshape(N_experiments, N_types), axis=1) - rho)**2 )


# compute correlations two ways: either "concatenating trials" or not
R_short_trials = r_split(X1, X2, N_repeats, N_tokens)
R_long_trials = r_split(X1, X2, N_repeats, 1)
R_super_long = r_split(X1, X2, 1, 1)

# plt.hist(R_a, alpha=0.5)
# plt.hist(R_b, alpha=0.5)

print('rho: %.3f' % rho)
print('MSE: %0.3g' % MSE(R_short_trials))
print('MSE: %0.3g' % MSE(R_long_trials))
print('MSE: %0.3g' % ((R_super_long - rho)**2 ))
